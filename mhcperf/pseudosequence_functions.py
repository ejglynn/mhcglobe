import pandas as pd

class PseudoSequence():
    def __init__(self):
        self.pseudoseq = pd.read_csv('/tf/data_mhcglobe-natmtd/allele_sequences_seqlen34.csv')
        self.pseudoseq = self.pseudoseq.rename(columns={'normalized_allele':'allele'})
        self.allele2seq = self.get_allele2pseudoseq()
        
    def get_allele2pseudoseq(self):
        allele_to_pseudoseq = self.pseudoseq.set_index('allele').to_dict()['sequence']
        return allele_to_pseudoseq