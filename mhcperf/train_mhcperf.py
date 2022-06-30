import train_functions
import sys
sys.path.append('./mhcperf_fairmhc/')

import argparse
import pandas
from mhcperf_hyperparam_optimization import OptimizeHypers, ScaleData
from scipy.stats import rankdata
from feature_column_names import get_feature_cols

from sklearn.model_selection import ParameterGrid


def get_XY(df):
    global feature_col_names
    y = df['PPV']
    X = df.loc[:,feature_col_names]
    return X, y

def get_params():
    parameters = {
            'input_shape'   : [(63,)],
            'dense_layers'  : [1],
            'dense_units_1' : [10, 20],
            'activation'    : ['relu'],
            'L1_reg'        : [0],
            'dropout_rate'  : [0.25],
            'skip_1'        : [False],
            'opt'           : ['RMSprop'],
            'rms_learning_rate': [1e-3, 1e-4, 1e-5],
            'rms_momentum'  : [0.8, 0.9],
            'rms_epsilon'   : [1e-08, 1e-07],
            'rms_centered'  : [True],
            'loss_type'     : ['mae'],
            'batch_size'    : [5, 10],
            'epochs'        : [1000]
        }
    return parameters


class TrainMHCPerf():
    def __init__(self, df_train_path: str, model_savepath : str, verbose : int=0):
        """
        Everytime mhcperf is trained hyperparameter optimization occurs.
        """
        self.model_savepath = model_savepath
        self.feature_col_names = get_feature_cols()
        self.df_train_path = df_train_path
        self.df_train = pandas.read_csv(df_train_path)
    
    def hyperparameter_search(self):
        # MHCPerf Hyperparameter Optimization
        kfolds = 3
        param_grid = get_params()
        tmp_ann_savepath = self.model_savepath+'_TMP'
        hyper_obj = OptimizeHypers(self.feature_col_names, n_cpus=1)
        self.best_params = hyper_obj.hyperparameter_selection(
            self.df_train, kfolds, param_grid, tmp_ann_savepath)
        
    def process_train_df(self):
        # Train MHCPerf using chosen/best hyperparameters
        self.model = train_functions.get_compiled_model(self.best_params)
        scale_obj = ScaleData(self.df_train, self.feature_col_names)
        train_scaled = scale_obj.train_scaled
        X, Y = get_XY(train_scaled)   
        return X, Y
    
    def train_mhcperf(self):
        self.hyperparameter_search()
        
        self.model = train_functions.train_model_tfdata(
            batch_size=self.best_params['batch_size'],
            epochs=self.best_params['epochs'],
            model=self.model,
            X_train=X,
            Y_train=Y,
            savepath=self.model_savepath,
            verbose=0)
        self.model.load_weights(self.model_savepath)
        self.model.save(self.model_savepath)
        return self.model


if __name__ == '__main__':
    # Inputs
    df_train_path = sys.argv[1]
    model_savepath = sys.argv[2]
    verbose = sys.argv[3]
    
    TrainMHCPerf(df_train_path, model_savepath, verbose).train_mhcperf()