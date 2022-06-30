import tensorflow as tf
import pandas as pd

#import sys
#sys.path.append('./mhcperf_fairmhc/')
import mhc_data
import featurize_mhcperf
from feature_column_names import get_feature_cols
import mhcperf_hyperparam_optimization as hyperopt

class MHCPerf():
    """
    Utilize fully trained MHCGlobe-PPV Model.
    """
    def __init__(self):
        self.model_savepath= '/tf/mhcglobe/mhcglobe_ppv/mhcglobe/mhcglobe_ppv_AD/10foldCV/models/mhcglobe-ppv_FULL'
        self.train_df_path = '/tf/mhcglobe/mhcglobe_ppv/mhcglobe/src/df_update_withduplicates_newdistbins_PPVbalanced.csv'

        self.model = self.get_mhcglobe_ppv()
        
        self.feature_col_names = get_feature_cols(include_distbin8=False)
        self.scale_object = self.get_scale_object()
        
    def get_mhcglobe_ppv(self):
        # MHCGlobe-Level 2
        tf.keras.backend.clear_session()
        mhcglobe_ppv = tf.keras.models.load_model(self.model_savepath)
        return mhcglobe_ppv
    
    def get_scale_object(self):
        df = pd.read_csv(self.train_df_path)
        scale_obj = hyperopt.ScaleData(df, self.feature_col_names)
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
    
    
def featurize(binding_data_path : str, query_alleles : list):
    query_df = pd.DataFrame(query_alleles, columns=['allele'])
    ba_data_obj = mhc_data.pMHC_Data(only_EL=False, drop_duplicate_records=False, data_path=binding_data_path)
    binding_data = ba_data_obj.positives
    data_dict = ba_data_obj.mk_data_dict(binding_data)
    data_alleles, data_alleles_50 = ba_data_obj.get_data_alleles(data_dict)
    
    binding_data_features = featurize_mhcperf.get_level2_features(
        query_df,
        data_alleles, data_dict, include_distbin8=False, has_left_out_alleles=False)
    
    query_df = pd.concat([df.reset_index(drop=True) for df in [query_df, binding_data_features]], axis=1)
    return query_df
