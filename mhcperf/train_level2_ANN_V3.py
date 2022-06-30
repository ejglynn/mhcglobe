import train_functions
import sys
import argparse
import json
import pandas
import pickle
import level2_hyperparameter_optimization as level2_hyperopt
from scipy.stats import rankdata
import random
from feature_column_names import get_feature_cols

from sklearn.model_selection import ParameterGrid


def get_XY(df):
    global feature_col_names
    y = df['PPV']
    X = df.loc[:,feature_col_names]
    return X, y

def get_params():
    parameters = {
            'input_shape':[(len(feature_col_names),)],
            'dense_layers': [1],
            'dense_units_1': [10, 20],
            'activation': ['relu'],
            'L1_reg':[0],
            'dropout_rate': [0.25],
            'skip_1':[False],
            'rms_learning_rate': [1e-3, 1e-4, 1e-5],
            'rms_momentum': [0.8, 0.9],
            'rms_epsilon': [1e-8, 1e-7],
            'loss_type':['mae'],
            'batch_size':[5, 10],
            'epochs':[1000],
        }
    return parameters


def balance_ppv_by_allele(df, k):
    df_allele_mean = (
        df
        .groupby('allele')['PPV']
        .mean()
        .reset_index()
        .rename(columns={'PPV':'mean_allele_PPV'}))
    df_allele_mean.sort_values('mean_allele_PPV', inplace=True)
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
    if 'fold' in list(df.columns):
        df = df.drop('fold', axis=1)
    df = df_allele_mean.merge(df)
    df = df.sort_values(['meanPPV_rank', 'PPV'])
    df.drop(['meanPPV_rank', 'mean_allele_PPV'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    df_train_path = sys.argv[1]
    model_savepath = sys.argv[2]
    model_logpath = sys.argv[3]

    df_train = pandas.read_csv(df_train_path)
    
    feature_col_names = get_feature_cols()
    kfolds = 3  # Used for selection best hyperparameters in k-fold cross validation.
    param_grid = get_params()
    print('PARAMS', len(param_grid))
    hyper_obj = level2_hyperopt.OptimizeLevel2Hypers(feature_col_names, n_cpus=1)
    tmp_ann_savepath = model_savepath+'_TMP'
    best_params = hyper_obj.hyperparameter_selection(df_train, kfolds, 'rgr', param_grid, tmp_ann_savepath)

    print('~~~~~~ BEST ~~~~', '\n', best_params)
         
    rgr = train_functions.get_compiled_model(best_params)
    
    scale_obj = level2_hyperopt.ScaleData(df_train, feature_col_names)
    train_scaled = scale_obj.train_scaled
    
    X, Y = get_XY(train_scaled)
        
    rgr = train_functions.train_model_tfdata(
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],
        model=rgr,
        X_train=X,
        Y_train=Y,
        savepath=model_savepath,
        verbose=0,
    )
    rgr.load_weights(model_savepath)
    rgr.save(model_savepath)
    
    log_data = {
        'kfolds':kfolds,
        'param_grid':param_grid,
        'best_params':best_params,
        'train_data':df_train,
    }
    with open(model_logpath, 'wb') as pfile:
        pickle.dump(log_data, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    print(df_train_path, model_savepath, model_logpath)
