import os

mhcglobe_dir = '/home/dghersi/Programs/mhcglobe'

class DataPaths():
    def __init__(self):
        self.user_main_dir       = '/tf/mhcglobe/'
        
        #### Models ####
        self.model_dir           = mhcglobe_dir + '/model'
        
        # MHCGlobe
        self.mhcglobe_init       = f'{self.model_dir}/mhcglobe/init/'
        self.mhcglobe_full       = f'{self.model_dir}/mhcglobe/full/'
        
        # MHCPerf
        self.mhcperf_full        = f'{self.model_dir}/mhcperf/mhcperf-full'
        
        #### Data ####
        self.data_dir            = mhcglobe_dir + '/data'
           
        # General
        self.allele_sequences    = f'{self.data_dir}/allele_sequences_seqlen34.csv'
       
        # MHCPerf
        self.mhcperf_train_data  = f'{self.data_dir}/mhcperf_train_data.csv'
        self.mhcperf_all_data    = f'{self.data_dir}/mhcperf_all_alleles_features.csv.gz'
        self.mhc_pairwise_dist   = f'{self.data_dir}/distB62_unique_pseudpsequences.pkl'
        self.dist_seq_index_dict = f'{self.data_dir}/seq_index_dict.pkl'
        
        # MHCGlobe
        self.mhcglobe_full_training_data = f'{self.data_dir}/mhcglobe_full_train_data.csv'
