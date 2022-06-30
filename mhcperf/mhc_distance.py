import numpy
#mport pandas
import my_functions as mf
import pickle

class BLOSUM62():
    def __init__(self):
        self.blosum62_dict = mf.select_peptideencoding(encode_type='BLOSUM62')

    def substitution_scoreB62(self, seq1, seq2, weights=None):
        if weights == None:
            weights = numpy.ones(len(seq1))
        # Return BLOSUM62 distance (Weighted or Non-Weightedimport numpy
        # between two peptide sequences.
        AA = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
        a, b = list(seq1), list(seq2)
        sequence_score = 0
        for i in range(len(a)): # for each index position in the pseudo sequence.
            sub_score = self.blosum62_dict[ a[i] ][AA.index(b[i])]
            sub_score = sub_score * weights[i]
            sequence_score += sub_score
        return sequence_score

    def distance_score(self, seq1, seq2, weights=None):
        # Compute distance measurement based on NetMHCpan 2007 figure 2 legend equation.
        s1_s1 = self.substitution_scoreB62(seq1, seq1, weights)
        s2_s2 = self.substitution_scoreB62(seq2, seq2, weights)
        s1_s2 = self.substitution_scoreB62(seq1, seq2, weights)
        distance = 1 - (s1_s2/numpy.sqrt(s1_s1*s2_s2))
        return distance
    
    
# Precomputed matrix of BLOSUM62 distances between MHC pseudosequences.
similarity_path = '/tf/data_mhcglobe-natmtd/distB62_unique_pseudpsequences.pkl'
results_62 = pickle.load(open(similarity_path, 'rb'))
seq_index_dict_path = '/tf/data_mhcglobe-natmtd/seq_index_dict.pkl'
seq_index_dict = pickle.load(open(seq_index_dict_path, 'rb'))

def retrieve_distance(seq1, seq2):
    """ 
    Return BLOSUM62 distance between two protein sequences.
    Inputs must be protein sequences of equal length.
    'X' residue is not permitted.
    """
    global seq_index_dict, results_62
    return results_62[seq_index_dict[seq1]][seq_index_dict[seq2]]