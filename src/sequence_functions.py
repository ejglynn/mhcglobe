import sys
import numpy as np
import pandas as pd
import pickle

from paths import DataPaths
import binding_affinity as ba

#class MHCPseudoSeqDF():
#    def __init__(self):
#        self.pseudoseq_csv_path = DataPaths().allele_sequences
#        self.df = (
#            pd.read_csv(pseudoseq_csv_path)
#            .rename(columns={'normalized_allele':'allele'}))
        
class PseudoSeq():
    def __init__(self):
        pseudoseq_path = DataPaths().allele_sequences
        self.pseudoseq = (
            pd.read_csv(pseudoseq_path)
            .rename(columns={'normalized_allele':'allele'})
        )
        self.allele2seq = self.get_allele2pseudoseq()
        
    def get_allele2pseudoseq(self):
        allele_to_pseudoseq = (
            self.pseudoseq
            .set_index('allele')
            .to_dict()['sequence']
        )
        return allele_to_pseudoseq

        
class MHCName():
    def parse_allele(self, allele):
        """
        Parse allele name to match pseudosequence csv.
        """
        allele = allele.replace('*', '').replace(':', '')
        return(allele)

    def is_same(self, standard_allele_names, query_allele):
        """
        If allele2 parsed matched an allele name in standard alleles
        list, return the allele1 name (from standard alleles). Else
        return allele2 name.
        """
        a_q = self.parse_allele(query_allele.replace('HLA-', ''))
        for allele1 in standard_allele_names:
            a1 = self.parse_allele(allele1.replace('HLA-', '')) ####### Make this into a dict to avoid n * n comparisons.
            if a1==a_q:
                return allele1
        else: # Keep allele the same as input.
            print(a1, a_q)
            return query_allele 

    def standardize_allele(self, alleles_vct):
        """ Return vector of standardized alleles names
        if allele in alleles_vct is a parsed equivalent
        allele name from the pseudosequence allele names.
        """
        standard_allele_names = list(PseudoSeq().df['allele'].unique())
        
        # Creating mapping of allele input name to a standardized naming.
        allele_dict = {}
        for allele in alleles_vct:
            if allele not in allele_dict:
                allele_dict[allele] = self.is_same(standard_allele_names, allele)
        return [allele_dict[a] for a in alleles_vct]

    
class SeqRepresentation():
    def __init__(self, encode_type):
        self.enabled_residues = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
        self.BLOSUM = self.select_peptideencoding(encode_type)
        self.ALLELE2SEQ = self.allele2seq()

    def one_hot(self, residue):
        #// Make One-Hot matrix for peptide encoding
        aa_pos = self.enabled_residues.index(residue)
        out_vct = np.full((20,), 0.0)
        out_vct[aa_pos] = 1.0
        return out_vct

    def select_peptideencoding(self, encode_type):
        """ Make dictionary for encoding amino acid strings as BLOSUM62 or One Hot
        """
        encoding_dict = {}
        if encode_type=='BLOSUM62':
            for row in blosum62:
                row_vct = row.split()
                residue = row_vct[0]
                blosum_vct = map(int, row_vct[1:])
                if residue not in encoding_dict:
                    encoding_dict[residue] = list(blosum_vct)
        if encode_type=='ONE_HOT': # Make One-Hot matrix for peptide encoding
            for e1 in self.enabled_residues:
                encoding_dict[e1] = self.one_hot(e1)
        return encoding_dict

    def encode_peptide(self, peptide):
        return np.array([self.BLOSUM[aa] for aa in peptide])#.flatten()

    def encode_mhc(self, pseudoseq):
        return np.array([self.BLOSUM[aa] for aa in pseudoseq])

    def allele2seq(self):
        """ Dictionary mapping MHC allele name to
        numeric encoding of MHC pseudosequence
        """
        pseudoseq_df = PseudoSeq().pseudoseq
        pseudoseq_df.insert(0, 'mhcEncoded', list(map(self.encode_mhc, pseudoseq_df['sequence'])))
        allele2seq = (
            pseudoseq_df[['allele', 'mhcEncoded']]
            .set_index('allele')
            .to_dict()['mhcEncoded']
        )
        return allele2seq

    
def seq_to_15mer(encoded_pep):
    """
    Making input peptide representation in the style of MHCFlurry.
    Input will be formatted for GRU, where a 4-mer is represented in shape,
    [[], [], [], []].
    """
    aa_feature_len = 20
    X_vct = np.zeros((15, aa_feature_len), dtype='float32')
    X_vct[:4] = encoded_pep[:4] # replace first four amino acids
    X_vct[-4:] = encoded_pep[-4:] # replace last four ammino acids
    middle_residues = encoded_pep[4:-4]
    len_middle = len(middle_residues)
    # Index of middle amino acids in 15-mer to subsitute with X_vct.
    len_middle_start = {7: 4, 6: 5, 5: 5, 4: 6, 3: 6, 2: 7, 1: 7}
    if len_middle != 0:
        new_middle_start = len_middle_start[len_middle]
        new_middle_end = new_middle_start + len_middle
        X_vct[new_middle_start: new_middle_end] = middle_residues
    return X_vct


def get_XY(df, encode_type='ONE_HOT', get_Y=True):
    """ Feature Encoding.

    Represent all Peptides as 15-mer as in MHCFlurry 1.2 style encoding.
    """
    SeqObj = SeqRepresentation(encode_type)

    # MHC pseudoseq encoding
    X1 = np.array([SeqObj.ALLELE2SEQ[a] for a in df['allele']], dtype='float32')

    # Peptide encoding with 15-mer representation
    X2 = [SeqObj.encode_peptide(p) for p in df['peptide']]
    X2 = np.array([seq_to_15mer(p) for p in X2], dtype='float32')
    if get_Y:
        Y1 = np.array([ba.from_ic50(y) for y in df['measurement_value']], dtype='float32')
        return [X1, X2], Y1
    else:
        return [X1, X2]

    
blosum62 = ['A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0',
            'R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3',
            'N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3',
            'D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3',
            'C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1',
            'Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2',
            'E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2',
            'G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3',
            'H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3',
            'I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3',
            'L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1',
            'K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2',
            'M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1',
            'F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1',
            'P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2',
            'S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2',
            'T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0',
            'W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3',
            'Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1',
            'V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4']
