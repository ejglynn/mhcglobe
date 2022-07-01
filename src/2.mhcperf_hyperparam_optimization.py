import numpy as np
import pandas as pd
import multiprocessing as mp

from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import train_functions

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



#from mhcperf_foldsplit import balancefolds_by_alleleppv

class OptimizeHypers():
    def __init__(self, feature_col_names, n_cpus=mp.cpu_count()):
        self.n_cpus = n_cpus
        self.feature_col_names = feature_col_names
    
    def get_XY(self, df):
        X = df.loc[:, self.feature_col_names]
        y = df['PPV']
        return X, y
        
    def cv_by_alleleppv(self, df, k):
        """
        Split by allele with approximately balanced folds by PPV.
        
        Return a list of tuples containing (df_train, df_test) for
        each fold split.
        """
        # define fold splits
        df = balancefolds_by_alleleppv(df, k) 
        
        assert 'fold' in list(df.columns)
        folds = list(set(df['fold']))
        train_test_splits = []
        for test_fold in folds:
            df_test = df[df.fold==test_fold]
            df_train = df[df.fold!=test_fold]
            train_test_splits.append((df_train, df_test))
        assert len(train_test_splits) == k
        return train_test_splits
            
    def eval_hyperparam(self, input_tuple):
        """
        Input: Hyperparameters from defined grid,
        regressor model to be parameterized,
        df for performance evaluation, number of folds.

        Output: Hyperparameters and mean performance
        across folds.
        """
        hypers, df, kfolds, tmp_ann_savepath = input_tuple
        rgr = train_functions.get_compiled_mhcperf_model(hypers)  
        # Cross Validation
        scores = []
        for tr, ts in self.cv_by_alleleppv(df, kfolds):
            
            # Standard Scale Features
            scale_obj = ScaleData(tr, self.feature_col_names)
            tr = scale_obj.train_scaled
            ts = scale_obj.scale_transform_df(ts)

            # Split to Input/Output
            X_tr, y_tr = self.get_XY(tr)
            X_ts, y_ts = self.get_XY(ts)
            assert np.sum(y_ts) > 0

            # Fit and Predict
            rgr = train_functions.train_mhcperf_model(
                batch_size=hypers['batch_size'],
                epochs=hypers['epochs'],
                model=rgr,
                X_train=X_tr,
                Y_train=y_tr,
                savepath=tmp_ann_savepath,
                verbose=0)
            
            # Evaluate fold performance
            y_hat = rgr.predict(X_ts, verbose=0)
            y_hat = y_hat.flatten()
            score = mean_squared_error(y_true=y_ts, y_pred=y_hat)
            scores.append(score)
        return hypers, np.mean(scores)
    
    def gridsearch_in_series(self, df, kfolds, param_grid, tmp_ann_savepath):
        """
        Serial grid search.
        """
        results = []
        for g in tqdm(ParameterGrid(param_grid)):
            results.append(self.eval_hyperparam((g, df, kfolds, tmp_ann_savepath)))
        return results

    def hyperparameter_selection(self, df, kfolds, param_grid, tmp_ann_savepath):
        """
        Return hyperparameter settings with the best mean performance
        across kfolds.
        """
        print('Running grid search for hyperparameter selection using 3-fold CV.')
        # Initialize.
        best_score = 100 # Initialize with poor MSE
        best_params = ''
        best_rgr = ''
        # Get MSE for all parameters combinations in param_grid
        results = self.gridsearch_in_series(df, kfolds, param_grid, tmp_ann_savepath)
        for hypers, mean_score in results:
            if mean_score < best_score:
                best_score = mean_score
                best_params = hypers
        return best_params

    
class ScaleData():
    # Feature Scaling Functions
    def __init__(self, train_df, feature_names):
        self.feature_names = feature_names
        self.col_names_order = ['allele', 'PPV'] + list(feature_names)
        self.train_scaled, self.scaler = self.scale_fit_df(train_df)

    def scale_fit_df(self, train_df):
        """
        Scale training set. Return scaled df and scaler object.
        """
        train_df = train_df.reset_index(drop=True)
        scaler = MinMaxScaler().fit(train_df.loc[:,self.feature_names])
        train_scaled = scaler.transform(train_df.loc[:,self.feature_names])
        train_scaled = pd.DataFrame(train_scaled, columns=self.feature_names) # Add column names
        
        # Insert columns which are needed but are not scaled.
        train_scaled.loc[:, 'allele'] = train_df['allele']
        train_scaled.loc[:, 'PPV'] = train_df['PPV']
        train_scaled = train_scaled.loc[:, self.col_names_order] 
        self.has_NA(train_scaled) # code check
        return train_scaled, scaler

    def scale_transform_df(self, test_df):
        # Apply an already fit scaler object to a test set.
        test_df = test_df.reset_index(drop=True)
        test_scaled = self.scaler.transform(test_df.loc[:,self.feature_names])
        test_scaled = pd.DataFrame(test_scaled, columns=self.feature_names) # Add column names
        
        # Add expected columns
        test_scaled.loc[:, 'allele'] = test_df['allele']
        if 'PPV' in list(test_df.columns):
            test_scaled.loc[:, 'PPV'] = list(test_df.PPV)
        else: # Place holder column
            test_scaled.loc[:, 'PPV'] = 0
        test_scaled = test_scaled.loc[:, self.col_names_order]
        self.has_NA(test_scaled) # code check
        return test_scaled
    
    def has_NA(self, df):
        assert not df.isnull().values.any() # No NA in data frame.
        

def balancefolds_by_alleleppv(df, k):
    """
    Balance k folds by similar mean-PPV per allele.
    Mean PPV is computed over 10 trials per allele. 
    Rank alleles by mean PPV. 
    Iteratively assign each allele into one of 
    k folds to balance folds. 
    """
    # K is number of folds.
    df_mean = (
        df
        .groupby('allele')['PPV']
        .mean()
        .reset_index()
        .rename(columns={'PPV':'mean_allele_PPV'})
        .sort_values('mean_allele_PPV')
    )
    df_mean.loc[:,'meanPPV_rank'] = rankdata(df_mean['mean_allele_PPV'])
    df_mean = df_mean.sort_values('meanPPV_rank').reset_index(drop=True)

    group_assignment = []
    group = 1
    for i in range(df_mean.shape[0]):
        if group > k:
            group = 1
        group_assignment.append(group)
        group += 1
    df_mean.loc[:,'fold'] = group_assignment
    df = df_mean.merge(df)
    df = df.sort_values(['meanPPV_rank', 'PPV'])
    df.drop(['meanPPV_rank', 'mean_allele_PPV'], axis=1, inplace=True)
    return df
