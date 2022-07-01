import tensorflow as tf
import pandas as pd

import mhc_data
import train_functions
import featurize_mhcperf
from feature_column_names import get_feature_cols
from mhcperf_hyperparam_optimization import OptimizeHypers, ScaleData

def featurize_from_binding(binding_data_path : str, query_alleles : list):
    query_df = pd.DataFrame(query_alleles, columns=['allele'])
    ba_data_obj = mhc_data.pMHC_Data(only_EL=False)
    binding_data = ba_data_obj.positives
    data_dict = ba_data_obj.mk_data_dict(binding_data)
    data_alleles, data_alleles_50 = ba_data_obj.get_data_alleles(data_dict)
    
    binding_data_features = featurize_mhcperf.get_level2_features(
        query_df,
        data_alleles, data_dict, has_left_out_alleles=False)
    
    query_df = pd.concat([df.reset_index(drop=True) for df in [query_df, binding_data_features]], axis=1)
    return query_df


class model():
    """
    Utilize fully trained MHCGlobe-PPV Model.
    """
    def __init__(self, model_savepath=None, train_df_path=None):
        if (model_savepath == None) or (train_df_path==None):
            self.model_savepath= '/tf/mhcglobe/mhcglobe_ppv/mhcglobe/mhcglobe_ppv_AD/10foldCV/models/mhcglobe-ppv_FULL'
            self.train_df_path = '/tf/mhcglobe/mhcglobe_ppv/mhcglobe/src/df_update_withduplicates_newdistbins_PPVbalanced.csv'
        else:
            self.model_savepath= model_savepath
            self.train_df_path = train_df_path

        self.model = self.get_mhcglobe_ppv()
        
        self.feature_col_names = get_feature_cols()
        self.scale_object = self.get_scale_object()
        
    def get_mhcglobe_ppv(self):
        # MHCGlobe-Level 2
        tf.keras.backend.clear_session()
        mhcglobe_ppv = tf.keras.models.load_model(self.model_savepath)
        return mhcglobe_ppv
    
    def get_scale_object(self):
        df = pd.read_csv(self.train_df_path)
        scale_obj = ScaleData(df, self.feature_col_names)
        return scale_obj
    
    def scale_df(self, df_test):
        df_test_scaled = self.scale_object.scale_transform_df(df_test.loc[:, ['allele'] + self.feature_col_names].reset_index(drop=True))
        return df_test_scaled

    def predict_ppv(self, df_test):
        df_test_scaled = self.scale_df(df_test.reset_index(drop=True))
        ppv_est = self.model.predict(
            df_test_scaled.loc[:, self.feature_col_names],
            verbose=0
        ).flatten()
        return ppv_est


class train():
    def __init__(self, df_train_path: str, model_savepath : str, verbose : int=0):
        """
        Everytime mhcperf is trained hyperparameter optimization occurs.
        """
        self.model_savepath = model_savepath
        self.feature_col_names = get_feature_cols()
        self.df_train_path = df_train_path
        self.df_train = pd.read_csv(df_train_path)
        self.model = self.train_mhcperf()
        
    def get_params(self):
        parameters = {
                'model_name'    : ['mhcperf']
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
    
    def hyperparameter_search(self):
        # MHCPerf Hyperparameter Optimization
        kfolds = 3
        param_grid = self.get_params()
        tmp_ann_savepath = self.model_savepath+'_TMP'
        
        hyper_obj = OptimizeHypers(self.feature_col_names, n_cpus=1)
        self.best_params = hyper_obj.hyperparameter_selection(
            self.df_train, kfolds, param_grid, tmp_ann_savepath)
        
    def get_XY(self, df):
        y = df['PPV']
        X = df.loc[:,self.feature_col_names]
        return X, y
        
    def process_train_df(self):
        # Train MHCPerf using chosen/best hyperparameters
        scale_obj = ScaleData(self.df_train, self.feature_col_names)
        train_scaled = scale_obj.train_scaled
        X, Y = self.get_XY(train_scaled)   
        return X, Y
    
    def train_mhcperf(self):
        self.hyperparameter_search()
        self.model = train_functions.get_compiled_model(self.best_params)
        X, Y = self.process_train_df()
        
        self.model = train_functions.train_mhcperf_model(
            batch_size=self.best_params['batch_size'],
            epochs=self.best_params['epochs'],
            model=self.model,
            X_train=X,
            Y_train=Y,
            savepath=self.model_savepath,
            verbose=0)
        self.model.load_weights(self.model_savepath)
        self.model.save(self.model_savepath, save_format="tf")
        
        


if __name__ == '__main__':
    # Inputs
    df_train_path = sys.argv[1]
    model_savepath = sys.argv[2]
    
    TrainMHCPerf().train_mhcperf(df_train_path, model_savepath)
