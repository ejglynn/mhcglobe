class DataPaths():
    def __init__(self):
        self.user_main_dir    = '/tf/mhcglobe/'
        
        #### Models ####
        self.model_dir          = '/tf/mhcglobe_docker/model'
        
        # MHCGlobe
        self.mhcglobe_init      = self.model_dir + '/mhcglobe/init/'
        self.mhcglobe_full      = self.model_dir + '/mhcglobe/full/'
        
        # MHCPerf
        self.mhcperf_full       = self.model_dir + '/mhcperf/full'
        
        #self.models_dir = '/tf/natmtd/models_mhcglobe-natmtd'
        #self.mhcglobe_src = '/tf/natmtd/mhcglobe-natmtd/mhcglobe'
        
        #### Data ####
        self.data_dir           = '/tf/mhcglobe_docker/data'
           
        self.allele_sequences   = self.data_dir  + 'allele_sequences_seqlen34.csv'
        self.mhcglobe_full_training_data = self.data_dir + 'mhcglobe_full_train_data.csv'
        
        # MHCPerf Featurized DF
        self.mhcperf_train_data = self.data_dir  + 'mhcperf_train_data.csv'
        self.mhcperf_all_data   = self.data_dir  + 'mhcperf_all_alleles_features.csv.gz'
        
        self.mhc_pairwise_dist  = self.data_dir  + 'distB62_unique_pseudpsequences.pkl'
        self.dist_seq_index_dict = self.data_dir + 'seq_index_dict.pkl'
        