import numpy
import pickle

def tup2dict(tup, di):
    for a, b in tup:
        di.setdefault(a, b)
    return di

def parse_allele(allele):
    """ Parse MHC alllele variant name into
    common format between datasets.
    """
    allele = allele.replace('*', '').replace(':', '')
    return allele

blosum62 = [
    'A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0', 
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

def get_vector(aa1):
    # Make One-Hot matrix for peptide encoding
    AA = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
    out_vct = []
    for e2 in AA:
        if e2 ==aa1: out_vct.append(1)
        else: out_vct.append(0)
    return out_vct

def select_peptideencoding(encode_type):
    # Make dictionary for encoding amino acid strings as BLOSUM62 or Braun2001 values from above arrays.
    aa_encoding_dict = {}
    if encode_type == 'BLOSUM50':
        for row in blosum50:
            row_vct = row.split()
            residue = row_vct[0]
            blosum_representation = map(int, row_vct[1:])
            if residue not in aa_encoding_dict:
                aa_encoding_dict[residue] = list(blosum_representation)
    if encode_type == 'BLOSUM62':
        for row in blosum62:
            row_vct = row.split()
            residue = row_vct[0]
            blosum_representation = map(int, row_vct[1:])
            if residue not in aa_encoding_dict:
                aa_encoding_dict[residue] = list(blosum_representation)
    if encode_type == 'ONE_HOT': # Make One-Hot matrix for peptide encoding
        AA = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
        for e1 in AA:
            if e1 not in aa_encoding_dict:
                aa_encoding_dict[e1] = get_vector(e1)
    return aa_encoding_dict


class MHCDistBLOSUM62():
    def __init__(self):
        self.blosum62_dict = select_peptideencoding(encode_type='BLOSUM62')
        
        # Precomputed matrix of BLOSUM62 distances between MHC pseudosequences.
        large_data_dir = '/tf/fairmhc/data_git_ignore'
        # Pairwise Distance Matrix
        similarity_path = f'{large_data_dir}/distB62_unique_pseudpsequences.pkl'
        # pseudosequence to matrix index mapping for distance retrieval
        seq_index_dict_path = f'{large_data_dir}/seq_index_dict.pkl'
        
        self.results_62 = pickle.load(open(similarity_path, 'rb'))
        self.seq_index_dict = pickle.load(open(seq_index_dict_path, 'rb'))

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
    
    def retrieve_distance(self, seq1, seq2):
        """ 
        Return BLOSUM62 distance between two protein sequences.
        Inputs must be protein sequences of equal length.
        'X' residue is not permitted.
        """
        #global seq_index_dict, results_62
        return self.results_62[self.seq_index_dict[seq1]][self.seq_index_dict[seq2]]