import pickle
import numpy
import pandas
import pseudosequence_functions


class pMHC_Data():
    def __init__(self, only_EL, drop_duplicate_records, data_path=None):
        #data_path ='/home/eric/ejglynn@princeton.edu/mhc_share/src/22Oct2020/data_preprocessed_22Oct2020.csv'
        #data_path = '/home/eric/PHD/final/data/IEDB_S3_S1hits_22Oct2020.csv' #used to train ensmeble
        
        #data_path ='/home/eric/ejglynn@princeton.edu/mhc_share/src/22Oct2020/data_preprocessed_22Oct2020.csv'
        if not data_path:
            data_path ='/tf/data_mhcglobe-natmtd/data_preprocessed_22Oct2020.csv'
        
        self.data = pandas.read_csv(data_path)
            
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
            
        #self.positives['measurement_type'] = 'positive'
        
        # Used only for choosing test-MHC alleles.
        self.positives_noduplicates = self.positives.drop_duplicates(keep='first', subset=['allele', 'peptide'])
        
        self.pseudoseq = pseudosequence_functions.PseudoSequence().pseudoseq
        self.allele2seq = pseudosequence_functions.PseudoSequence().allele2seq
        
        #self.pseudoseq = pandas.read_csv('/home/eric/PHD/mhc-globe/data/Analysis/allele_sequences_seqlen34.csv')
        #self.pseudoseq = self.pseudoseq.rename(columns={'normalized_allele':'allele'})
        #self.allele2seq = self.get_allele2pseudoseq()
     
    #def get_allele2pseudoseq(self):
    #    allele_to_pseudoseq = self.pseudoseq.set_index('allele').to_dict()['sequence']
    #    return allele_to_pseudoseq
    
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
            .reset_index()
        )
        data_count_Dict = (
            data_count_Dict[['allele', 'peptide']]
            .set_index('allele')
            .to_dict('index')
        )
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


