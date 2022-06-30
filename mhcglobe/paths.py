class DataPaths():
    def __init__(self):
        self.data_dir = '/tf/data_mhcglobe-natmtd'
        self.models_dir = '/tf/natmtd/models_mhcglobe-natmtd'
        self.mhcglobe_src = '/tf/natmtd/mhcglobe-natmtd/mhcglobe'
        
        self.allele_sequences = '/tf/natmtd/data/allele_sequences_seqlen34.csv'
        self.mhcglobe_full_training_data = '/home/eric/natmtd/data/mhcglobe_data/mhcglobe_full_train_data.csv'
        
        
        # MHCPerf Featurized DF
        self.mhcperf_train_data = '/home/eric/natmtd/data/mhcperf_data/mhcperf_train_data.csv'
        self.mhcperf_all_data = '/tf/natmtd/data/mhcperf_data/mhcperf_all_alleles_features.csv.gz'
    