import pickle
import numpy
import pandas
from scipy.stats import rankdata

from paths import DataPaths
import my_functions as mf
from sequence_functions import PseudoSeq


class pMHC_Data():
    def __init__(self, only_EL, drop_duplicate_records=False):
        pMHC_data_path = DataPaths().mhcglobe_full_training_data
        self.data = pandas.read_csv(pMHC_data_path)
            
        # Subset Data–Binding Affinity and Elution or ONLY Elution
        if not only_EL:
            self.positives = self.data[
                (self.data['measurement_value']<=500) &
                (self.data['measurement_inequality'].isin(['<', '=']))]
        else: # EL ONLY
            self.positives = self.data[
                (self.data['measurement_type']!='BA') &
                (self.data['measurement_value']<=500) &
                (self.data['measurement_inequality'].isin(['<', '=']))]
            
        if drop_duplicate_records:
            self.data = self.data.drop_duplicates(keep='first', subset=['allele', 'peptide'])
            self.positives = self.positives.drop_duplicates(keep='first', subset=['allele', 'peptide'])
            
        #self.positives.loc[:, 'measurement_type'] = 'positive' Don't think this is needed. ###############################
        
        # Used only for choosing test-MHC alleles.
        self.positives_noduplicates = self.positives.drop_duplicates(keep='first', subset=['allele', 'peptide'])
        
        self.pseudoseq = PseudoSeq().pseudoseq
        self.allele2seq = PseudoSeq().allele2seq
        
    
    def add_noData_MHC(self, pseudoseq, data_count_Dict, pos_neg='positive'):
        """
        Add alleles to dictionary not in the
        MHCflurry dataset to the data_count_Dict
        with value of 0.
        """ 
        for a in list(set(pseudoseq['allele'])):
            if a not in data_count_Dict:
                data_count_Dict[a] = {'peptide': 0}
            if a == 'BoLA-100901':
                print(a)
        return data_count_Dict

    def mk_data_dict(self, df):
        """Make dictionary with number of datapoints available for each MHC allelic variant.
        """
        #data_count_Dict = self.positives.groupby(['allele']).count()
        data_count_Dict = (
            df
            .groupby(['allele'])
            .count()
            .reset_index())
        data_count_Dict = (
            data_count_Dict[['allele', 'peptide']]
            .set_index('allele')
            .to_dict('index'))
        # Not all alleles have data, so wont be in the dict. Add 0 value to those w/o data
        data_count_Dict = self.add_noData_MHC(self.pseudoseq, data_count_Dict, 'positive')
        
        for allele in data_count_Dict:
            data_count_Dict[allele] = data_count_Dict[allele]['peptide'] # Remove useless level.
        return data_count_Dict
  
    def get_data_alleles(self, data_count_Dict):
        data_alleles = []
        data_alleles_50 = []
        #for allele in self.data_count_Dict:
        for allele in data_count_Dict:
            if data_count_Dict[allele] >= 1:
                data_alleles.append(allele)
            if data_count_Dict[allele] >= 50:
                data_alleles_50.append(allele)
        return data_alleles, data_alleles_50
    

    def update_data_dict(self, allele_gets_data : str, add_n_data : int):
        """
        Retrieve dictionary with (key) MHC allele names and
        (value) number of positive binding instance observed in
        MHCGlobe binding dataset. Add number of binding instance 
        (add_n_data) to dict for allele (allele_gets_data)

        Arguments
        _____

        allele_gets_data : str
            MHC allele name which recieved additional data beyond the current MHCGlobe training data

        add_n_data : int
            Number of new peptide-MHC positive binding instances recorded for allele_gets_data

        Outputs
        _____

        new_data_dict : dict
            Dictionary updated data count

        new_data_alleles : list
            List of allele names which are associated with positve binding data
            in new_data_dict.
        """
        new_data_dict = self.mk_data_dict(self.positives).copy()
        new_data_dict[allele_gets_data] += add_n_data
        new_data_alleles = self.get_data_alleles(new_data_dict)[0]
        return new_data_dict, new_data_alleles
    
'''
def balance_ppv_by_allele(df, k):
    """ Given df from LNO, assign alleles into fold groups using a `fold` column"""
    df_allele_mean = (
        df
        .groupby('allele')['PPV']
        .mean()
        .reset_index()
        .rename(columns={'PPV':'mean_allele_PPV'})
        .sort_values('mean_allele_PPV')
    )
    
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
    df = (
        df_allele_mean
        .merge(df)
        .sort_values(['meanPPV_rank', 'PPV'])
        .drop(['meanPPV_rank', 'mean_allele_PPV'], axis=1)
    )
    return df
'''
