import numpy as np
import multiprocessing as mp
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import train_functions # for ANN

from scipy.stats import rankdata

class OptimizeLevel2Hypers():
    def __init__(self, feature_col_names, n_cpus=mp.cpu_count()):
        #assert n_cpus <= mp.cpu_count(); print(f'CPUs Available {mp.cpu_count()}') 
        self.n_cpus = n_cpus
        self.feature_col_names = feature_col_names
        
    def get_XY(self, df):
        X = df.loc[:, self.feature_col_names]
        y = df['PPV']
        return X, y
    
    def cv_by_allele(self, df, k):
        """
        Split df into k cross validation splits on distinct alleles.
        Return list of k train-test tuples.
        """
        allele_list = pd.Series(list(set(df['allele']))) # must be converted to list then series.
        if k > 1:
            kf = KFold(n_splits=k, shuffle=True, random_state=1)
        elif k == 1:
            kf = KFold(n_splits=10, shuffle=True, random_state=1)
            
        train_test_splits = []
        #for train_index, test_index in kf.split(allele_list): # split by allele.
        for train_index, test_index in kf.split(df): # splitting by row not allele. 
            alleles_train, alleles_test = allele_list[train_index], allele_list[test_index]
            #df_train = df[df.allele.isin(alleles_train)]
            #df_test = df[df.allele.isin(alleles_test)]
            
            df_train = df.iloc[train_index,:]
            df_test = df.iloc[test_index,:]

            cv_split = (df_train, df_test)
            train_test_splits.append(cv_split)
        
        if k > 1:
            return train_test_splits
        if k == 1:
            #print(train_test_splits[0])
            return [train_test_splits[0]]
        
    def cv_by_row(self, df, k):
        """
        Split df into k cross validation splits on distinct alleles.
        Return list of k train-test tuples.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
        train_test_splits = []
        for train_index, test_index in kf.split(df): # splitting by row not allele.
            df_train = df.iloc[train_index,:]
            df_test = df.iloc[test_index,:]

            cv_split = (df_train, df_test)
            train_test_splits.append(cv_split)
        
        if k > 1:
            return train_test_splits
        if k == 1:
            #print(train_test_splits[0])
            return [train_test_splits[0]]
        
    def balance_ppv_by_allele(self, df, k):
        df_allele_mean = df.groupby('allele')['PPV'].mean().reset_index().rename(columns={'PPV':'mean_allele_PPV'})
        df_allele_mean = df_allele_mean.sort_values('mean_allele_PPV')
        df_allele_mean.loc[:,'meanPPV_rank'] = rankdata(df_allele_mean['mean_allele_PPV'])
        df_allele_mean = df_allele_mean.sort_values('mean_allele_PPV').reset_index(drop=True)

        group_assignment = []
        group = 1
        for i in range(df_allele_mean.shape[0]):
            if group > k:
                group = 1
            group_assignment.append(group)
            group += 1
        df_allele_mean.loc[:,'fold'] = group_assignment
        df = df_allele_mean.merge(df.drop('fold', axis=1))
        df = df.sort_values(['meanPPV_rank', 'PPV'])
        df.drop(['meanPPV_rank', 'mean_allele_PPV'], axis=1, inplace=True)
        return df
        
    def cv_by_allele_2(self, df, k):
        if True:
            df = self.balance_ppv_by_allele(df, k) # re-balance PPV by allele
        
        assert 'fold' in list(df.columns)
        folds = list(set(df['fold']))
        print(folds)
        train_test_splits = []
        for test_fold in folds:
            df_test = df[df.fold==test_fold]
            df_train = df[df.fold!=test_fold]
            cv_split = (df_train, df_test)
            train_test_splits.append(cv_split)
        return train_test_splits
            
    def eval_hyperparam(self, input_tuple):
        """
        Input: Hyperparameters from defined grid,
        regressor model to be parameterized,
        df for performance evaluation, number of folds.

        Output: Hyperparameters and mean performance
        across folds.
        """
        method, hypers, rgr, df, kfolds, scale, tmp_save_path = input_tuple
        
        if method == 'ANN':
            rgr = train_functions.get_compiled_model(hypers)
        else:
            rgr.set_params(**hypers)

        
        # Cross Validation
        scores = []
        print('kfolds: ', kfolds)
        #for tr, ts in self.cv_by_allele(df, k=kfolds):
        #for tr, ts in self.cv_by_row(df, k=kfolds):
        for tr, ts in self.cv_by_allele_2(df, kfolds):
            # Standard Scale Features
            #scale = False
            if scale:
                scale_obj = ScaleData(tr, self.feature_col_names)
                tr = scale_obj.train_scaled
                ts = scale_obj.scale_transform_df(ts)

            # Split to Input/Output
            X_tr, y_tr = self.get_XY(tr)
            X_ts, y_ts = self.get_XY(ts)

            # Fit and Predict
            if method == 'ANN':
                #tmp_save_path = '/home/eric/MHCGLOBE/src/Analysis/LVL2_ANN/lvl2_ann'
                rgr = train_functions.train_model_tfdata(500, rgr, X_tr, y_tr, tmp_save_path, verbose=0)
                y_hat = rgr.predict(X_ts)
                y_hat = y_hat.flatten()
            else:
                rgr.fit(X_tr, y_tr)
                y_hat = rgr.predict(X_ts)

            score = mean_squared_error(y_true=y_ts, y_pred=y_hat)
            scores.append(score)
        return hypers, np.mean(scores)

    def gridsearch_in_parallel(self, df, kfolds, rgr, param_grid, scale, tmp_ann_savepath):
        """
        Parallelize grid search using multiprocessing module.
        """
        pool = mp.Pool(self.n_cpus)
        results = pool.map(self.eval_hyperparam,
                           [(g, rgr, df, kfolds, scale, tmp_ann_savepath) for g in ParameterGrid(param_grid)])
        pool.close()
        return results
    
    def gridsearch_in_serial(self, method, df, kfolds, rgr, param_grid, scale, tmp_ann_savepath):
        """
        Serial grid search.
        """
        results = []
        for g in ParameterGrid(param_grid):
            results.append(self.eval_hyperparam((method, g, rgr, df, kfolds, scale, tmp_ann_savepath)))
        return results

    def hyperparameter_selection(self, method, df, kfolds, rgr, param_grid, scale, tmp_ann_savepath):
        """
        Return hyperparameter settings with the best mean performance
        across kfolds.
        """
        print(param_grid)
        # Initialize.
        best_score = 100 # Initialize with poor MSE
        best_params = ''
        best_rgr = ''
        # Get MSE for all parameters combinations in param_grid
        results = self.gridsearch_in_serial(method, df, kfolds, rgr, param_grid, scale, tmp_ann_savepath)
        for hypers, mean_score in results:
            if mean_score < best_score:
                best_score = mean_score
                best_params = hypers
        return best_params

class ScaleData():
    # Feature Scaling Functions
    def __init__(self, train_df, feature_names):
        self.feature_names = feature_names
        self.col_names = ['allele', 'PPV'] + list(feature_names)
        self.train_scaled, self.scaler = self.scale_fit_df(train_df)

    def scale_fit_df(self, train_df):
        # Scale training set, and return scaler object.
        train_df = train_df.reset_index(drop=True)
        #scaler = StandardScaler().fit(train_df.loc[:,self.feature_names])
        scaler = MinMaxScaler().fit(train_df.loc[:,self.feature_names])

        
        train_scaled = scaler.transform(train_df.loc[:,self.feature_names])
        train_scaled = pd.DataFrame(train_scaled, columns=self.feature_names) # Add column names
        # Insert columns which are needed but are not scaled.
        #train_scaled.loc[:, 'allele'] = train_df.allele
        #train_scaled.loc[:, 'group'] = train_df.group
        #if self.scale_ppv==False: # Addback PPV if not a scaled feature.
        train_scaled.loc[:, 'allele'] = train_df['allele']
        if 'PPV' in list(train_scaled.columns):
            train_scaled.loc[:, 'PPV'] = train_df['PPV']
        else:
            train_scaled.loc[:, 'PPV'] = 0
        train_scaled = train_scaled.loc[:, self.col_names] # prefered col name order for convienience
        self.has_NA(train_scaled)
        return train_scaled, scaler

    def scale_transform_df(self, test_df):
        # Apply an already fit scaler object to a test set.
        test_df = test_df.reset_index(drop=True)
        test_scaled = self.scaler.transform(test_df.loc[:, self.feature_names])
        test_scaled = pd.DataFrame(test_scaled, columns=self.feature_names) # Add column names
        test_scaled.loc[:, 'allele'] = test_df['allele']
        if 'PPV' in list(test_scaled.columns):
            test_scaled.loc[:, 'PPV'] = test_df['PPV']
        else:
            test_scaled.loc[:, 'PPV'] = 0
        test_scaled = test_scaled.loc[:, self.col_names] # prefered col name order for convienience
        self.has_NA(test_scaled)
        return test_scaled
    
    def has_NA(self, df):
        assert not df.isnull().values.any() # No NA in data frame.