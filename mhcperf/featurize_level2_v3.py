import numpy as np
import pandas as pd
#from tqdm import tqdm

import multiprocessing as mp


#import sys
#sys.path.append('/home/eric/MHCGLOBE/src/General/')
import mhc_data
import mhc_distance
import pickle
import feature_column_names

# To compute distance from scratch.
BLOSUM62_OBJ = mhc_distance.BLOSUM62()

# Data count data
#data_obj_BAEL = mhc_data.pMHC_Data(only_EL=False) 
#data_dict = data_obj_BAEL.mk_data_dict(data_obj_BAEL.positives) # Data counts used when sorting neighbors.

#### pseudoseq stuff should not be in a data object. #####
import pseudosequence_functions
pseudoseq = pseudosequence_functions.PseudoSequence().pseudoseq
allele2seq = pseudosequence_functions.PseudoSequence().allele2seq

def make_seq2allele(pseudoseq):
    seq2allele = dict()
    for i, row in pseudoseq.iterrows():
        allele=row['allele']
        sequence=row['sequence']
        if sequence not in seq2allele:
            seq2allele[sequence] = []
        seq2allele[sequence].append(allele)
    return seq2allele

seq2allele = make_seq2allele(pseudoseq)

# FEATURE SET 1
class ResiduePositionData():
    """
    ***From MVP_MHC.py***
    Make df with each column being a pseudosequence position,
    and values are residue corresponding to each allele index.
    """
    def __init__(self, train_alleles, data_dict):
        #self.data_aa_pos_dict = self.get_mhc_pseudoseq_res_pos_df()
        self.cols = [f'data_aa_pos_{i+1}' for i in range(34)]
        self.data_aa_pos_dict = self.get_mhc_pseudoseq_res_pos_df(train_alleles, data_dict)
        
    def get_mhc_pseudoseq_res_pos_df(self, train_alleles, data_dict):
        """ Make df with each column being a pseudosequence position,
        and values are residue corresponding to each allele index.
        """
        global pseudoseq
        residues = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
        tmp = pseudoseq[pseudoseq.allele.isin(list(train_alleles))]
        tmp = pd.DataFrame([list(seq) for seq in tmp['sequence']], index=tmp.allele)
        tmp.loc[:,'n_data'] = [data_dict[a] for a in tmp.index]

        data_aa_pos_dict = {key: dict() for key in range(34)}
        for i in data_aa_pos_dict:
            data_aa_pos_dict[i] = tmp.loc[:,[i, 'n_data']].groupby(i)['n_data'].sum().to_dict()
            
            for aa in residues:
                if aa not in data_aa_pos_dict[i]:
                    data_aa_pos_dict[i][aa] = 0
        return data_aa_pos_dict

    def data_for_pos_residue_pair(self, query_allele):
        """
        N training data points matching residue-position pairs of query MHC allele.
        Returns DF.
        """
        # Turn a list into df columns.
        global allele2seq
        np_array = np.array([self.data_aa_pos_dict[i][aa] for i, aa in enumerate(allele2seq[query_allele])])
        return np.reshape(np_array, (1, 34))[0]    
    
class UpdateResiduePositionData():
    """
    Make df with each column being a pseudosequence position,
    and values are residue corresponding to each allele index.
    """
    def __init__(self, data_dict, train_alleles):
        self.data_dict = data_dict
        self.train_alleles = list(train_alleles)
        self.data_aa_pos_dict = self.get_mhc_pseudoseq_res_pos_df()
        self.cols = [f'data_aa_pos_{i}' for i in range(1, 34+1)]
        
    def get_mhc_pseudoseq_res_pos_df(self):
        """ Make df with each column being a pseudosequence position,
        and values are residue corresponding to each allele index.
        """
        global pseudoseq
        residues = 'A R N D C Q E G H I L K M F P S T W Y V'.split()
        tmp = pseudoseq[pseudoseq.allele.isin(self.train_alleles)]
        tmp = pd.DataFrame([list(seq) for seq in tmp['sequence']], index=tmp.allele)
        tmp.loc[:,'n_data'] = [self.data_dict[a] for a in tmp.index]

        data_aa_pos_dict = {key: dict() for key in range(34)}
        for i in data_aa_pos_dict:
            data_aa_pos_dict[i] = tmp.loc[:,[i, 'n_data']].groupby(i)['n_data'].sum().to_dict()
            
            for aa in residues:
                if aa not in data_aa_pos_dict[i]:
                    data_aa_pos_dict[i][aa] = 0
        return data_aa_pos_dict

    def data_for_pos_residue_pair(self, query_allele):
        """
        N training data points matching residue-position pairs of query MHC allele.
        Returns DF.
        """
        # Turn a list into df columns.
        global allele2seq
        np_array = np.array([self.data_aa_pos_dict[i][aa] for i, aa in enumerate(allele2seq[query_allele])])
        return np.reshape(np_array, (1, 34))[0]

    
    
def feat_1(df_all, data_dict, alleles_w_data, has_left_out_alleles=True):
    """
    Verified to work for df_classical and df (leave-x-out cross validation)
    Note: alleles_w_data needs to be updated when new alleles get data.
    has_left_out_alleles: boolean representing if left out alleles described in the
    left_out_alleles column.
    """
    # Remake data_aa_pos_dict
    df_all = df_all.reset_index()
    if not has_left_out_alleles: # faster if no alleles are left out from alleles_w_data input.
        AA_POS_Obj = ResiduePositionData(alleles_w_data, data_dict)
        df_vct = [AA_POS_Obj.data_for_pos_residue_pair(allele) for allele in df_all.allele]
    else: # use left_out_alleles col to determine MHCGlobe-BA training alleles used for each instance.
        df_vct = []
        for index, row in df_all.iterrows():
            allele = row['allele']
            left_out_alleles = row['left_out_alleles']
            train_alleles = set(alleles_w_data).difference(left_out_alleles.split(','))
            AA_POS_Obj = ResiduePositionData(train_alleles, data_dict)
            feature_vct = AA_POS_Obj.data_for_pos_residue_pair(allele) 
            df_vct.append(feature_vct)

    out_df = pd.DataFrame(df_vct, columns=[f'data_aa_pos_{i+1}' for i in range(34)])
    #out_df.index = df_all.allele
    return out_df

def updatefeat_1(df_all, data_dict, alleles_w_data):
    """
    Verified to work for df_classical and df (leave-x-out cross validation)
    Note: alleles_w_data needs to be updated when new alleles get data.
    has_left_out_alleles: boolean representing if left out alleles described in the
    left_out_alleles column.
    """
    # Remake data_aa_pos_dict
    df_all = df_all.reset_index()
    AA_POS_Obj = UpdateResiduePositionData(data_dict, alleles_w_data)
    df_vct = [AA_POS_Obj.data_for_pos_residue_pair(allele) for allele in df_all.allele]
    
    out_df = pd.DataFrame(df_vct, columns=[f'data_aa_pos_{i+1}' for i in range(34)])
    #out_df.index = df_all.allele
    return out_df

# FEATURE SET 3
class NeighborDistBins():
    
    #Verified to work for df_classical and lxo df.
    def __init__(self, data_dict, include_distbin8):
        """
        When include_distbin8==True use a binning strategy where
        dist_bin-0.0 only includes training allele neighbors with
        distance of 0 from test-MHC.
        When include_distbin8==False use binning strategy where
        dist_bin-0.0 includes training allele neighbors with distances
        0.0-0.1 but not including 0.1. This strategy does not produce 
        sufficient data to use dist_bin_0.8, and includes what would 
        have been there in dist_bin_0.7.
        """
        self.data_dict = data_dict
        self.include_distbin8 = include_distbin8
    
    # 2 Amount of positive data in neighboring bins.
    def dist_to_bin(self, distance):
        if self.include_distbin8: # [0.0-0.0], (0.1-0.2], ..., (0.8-0.9]
            n_bins =10
            bin_thresholds = [i/10 for i in range(0, n_bins+1, 1)]
            for thresh in bin_thresholds:
                if distance <= thresh:
                    return f'dist_bin_{thresh}'
        else: # (0.0-0.1], ..., (0.8-] # equal width bins, and 0.8 distbin hold everything greater than 0.8.
            n_bins=7
            bin_thresholds = sorted(range(0, n_bins+1), reverse=True)
            bin_thresholds = [t/10 for t in bin_thresholds]
            for thresh in bin_thresholds:
                if distance >= thresh:
                    return f'dist_bin_{thresh}'

    def add_dist_bin(self, dist_to_train_vct):
        return [self.dist_to_bin(dist) for dist in dist_to_train_vct]

    def get_dist_bin_data(self, target_allele, neighbor_alleles):
        n_bins = 8
        neighbors_df = neighbors_array(
            target_allele,
            neighbor_alleles,
            len(neighbor_alleles),
            self.data_dict,
            by_pseudoseq=True)
        neighbors_df.loc[:, 'dist_bin'] = self.add_dist_bin(neighbors_df['dist_to_train'])
        
        #tmp.loc[:,'train_pos_data'] = [self.data_dict[a] for a in tmp['train_allele']]
        tmp2 = (
            neighbors_df[['dist_bin', 'train_seq_data']]
            .groupby('dist_bin')['train_seq_data']
            .apply(np.sum)
            .reset_index())
        c_names = pd.DataFrame({'dist_bin':[f'dist_bin_{i/10}' for i in range(0, n_bins+1, 1)]})
        tmp2 = (
            pd.merge(tmp2, c_names, how='outer')
            .rename(columns={'train_seq_data':'bin_train_seq_data'}))
        tmp2.loc[:,'allele'] = target_allele
        tmp2 = (
            tmp2[['allele', 'dist_bin', 'bin_train_seq_data']]
            .pivot(index='allele',columns='dist_bin'))
        tmp2.columns = tmp2.columns.droplevel()
        tmp2 = (
            tmp2
            .reset_index()
            .fillna(0))
        tmp2.columns.name = None
        return tmp2

    def get_neighbor_features(self, df, data_alleles_BAEL, has_left_out_alleles=True):
        dist_bin_data = []
        if has_left_out_alleles:
            for index, row in df.iterrows():
                allele = row['allele']
                left_out_alleles = row['left_out_alleles']
                train_alleles = list(set(data_alleles_BAEL).difference(left_out_alleles.split(',')))
                feature_df = self.get_dist_bin_data(allele, train_alleles)
                dist_bin_data.append(feature_df)
        else: # Faster when no alleles are left out of MHCGlobe-BA training.
            dist_bin_data = [self.get_dist_bin_data(allele, data_alleles_BAEL) for allele in df['allele']]

        dist_bin_data = pd.concat(dist_bin_data, axis=0)
        if 'left_out_alleles' in list(df.columns):
            dist_bin_data.insert(1, 'left_out_alleles', list(df['left_out_alleles']))
        if 'trial' in list(df.columns):
            dist_bin_data.insert(1, 'trial', list(df['trial']))
        dist_bin_data.reset_index(drop=True, inplace=True)
        return dist_bin_data


# FEATURE SET 2
    
def neighbors_array(query_allele, train_alleles, K, data_dict, by_pseudoseq=True):
    global allele2seq
    query_seq = allele2seq[query_allele]
    train_allele_seqs = [allele2seq[tr_a] for tr_a in train_alleles]
    train_allele_data = [data_dict[tr_a] for tr_a in train_alleles]
        
    if by_pseudoseq==False:
        dist_to_train = np.array(
            [mhc_distance.retrieve_distance(query_seq, tr_seq) for tr_seq in train_allele_seqs],
            dtype='float64')
        
        neighbors_df = (
            pd.DataFrame({
                'allele': [query_allele for i in range(len(train_alleles))],
                'train_allele': train_alleles,
                'train_sequence':train_allele_seqs,
                'dist_to_train': dist_to_train,
                'train_allele_data':train_allele_data})
            .sort_values(['dist_to_train', 'train_allele_data'], ascending=[True, False]))
            
        neighbors_df = neighbors_df.iloc[:K]
        return neighbors_df
        
    if by_pseudoseq==True:
        neighbors_df = (
            pd.DataFrame({
                'train_sequence':train_allele_seqs,
                'train_allele_data': train_allele_data})
            .groupby('train_sequence')['train_allele_data']
            .sum() # Sum data that belongs to alleles with same pseudosequence.
            .reset_index() # Moves 'train_sequence' from index to a column.
            .rename(columns={'train_allele_data':'train_seq_data'})
        )
        
        # Use neighbors_df['train_sequence'] instead of train_allele_seqs because no more duplicate pseudoseqs.
        neighbors_df['dist_to_train']=[mhc_distance.retrieve_distance(query_seq, tr_seq) for tr_seq in neighbors_df['train_sequence']]
        neighbors_df.sort_values('dist_to_train', ascending=True, inplace=True)
        neighbors_df['allele']=[query_allele for i in range(neighbors_df.shape[0])]
        
        neighbors_df = neighbors_df.iloc[:K]
        return neighbors_df


def get_top_K_neighbor_features_mp(tup):
    """
    Blosum 62 distance and data between query allele and
    closest K neighbors in training set.
    Need neighbors by pseudosequence.
    """
    allele, data_dict, alleles_w_data = tup
    K=10
            
    neighbors_df = neighbors_array(allele, alleles_w_data, K, data_dict, by_pseudoseq=True) # Output not sorted
            
    #distances_sorted, data_sorted = neighbors_array(allele, alleles_w_data, K, data_dict, ouput_df=False)
    return np.concatenate([
        list(neighbors_df['dist_to_train']), # Top K distance features 
        list(neighbors_df['train_seq_data'])
        #[data_dict[allele] for allele in tr_alleles_sorted] # Top K data features 
    ])


def topK_features_mp(df, data_dict, data_alleles, has_left_out_alleles=True):
    """
    Inputs:
    
    df: contains allele names for which to get feature vectors
    data_dict: Current data available.
    data_alleles: list of alleles with data
    
    Output: 
    
    Data frame with distances and data counts for
    top K neighboring alleles by distance relative to the allele columns.
    
    """
    if has_left_out_alleles:
        input_tups = []
        for index, row in df.iterrows():
            allele = row['allele']
            left_out_alleles = row['left_out_alleles']
            train_alleles = list(set(data_alleles).difference(left_out_alleles.split(',')))
            tup = (allele, data_dict, train_alleles)
            input_tups.append(tup)
            
    else:
        input_tups = [(allele, data_dict, data_alleles) for allele in df.allele]
        
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    results = pool.map(
        get_top_K_neighbor_features_mp,
        input_tups
    )
    pool.close()
    
    k=10
    #dist_df.columns = [f'N{i}_dist' for i in range(1, k+1)]+[f'N{i}_data' for i in range(1, k+1)]
    
    dist_df = pd.DataFrame(
        results,
        columns=[f'N{i}_dist' for i in range(1, k+1)]+[f'N{i}_data' for i in range(1, k+1)]
    )
    #dist_df.insert(0, 'allele', df.allele)
    return dist_df


    
# FEATURE SET 4
    
def train_data_size(df, data_alleles_BAEL, data_dict, has_left_out_alleles=True):
    if has_left_out_alleles:
        train_data_size = []
        for index, row in df.iterrows():
            allele = row['allele']
            left_out_alleles = row['left_out_alleles']
            train_alleles = set(data_alleles_BAEL).difference(left_out_alleles.split(','))
            train_data_size.append(np.sum([data_dict[a] for a in train_alleles]))
    else:
        data_size_full = np.sum([data_dict[a] for a in data_alleles_BAEL])
        train_data_size = [data_size_full for i in df.index]
    return train_data_size


# COMBINE FEATURE SETS
def get_level2_features(df, alleles_with_data, data_dict, include_distbin8, has_left_out_alleles=True):
    """
    has_left_out_alleles==True will give the righ answer even if all alleles are included,
    but has_left_out_alleles==False avoids iterating df rows, when all data alleles
    are training alleles.
    """
    f1=feat_1(df, data_dict, alleles_with_data, has_left_out_alleles).reset_index(drop=True)
    f2=topK_features_mp(df, data_dict, alleles_with_data, has_left_out_alleles)
    f3=NeighborDistBins(data_dict, include_distbin8).get_neighbor_features(df, alleles_with_data, has_left_out_alleles)
    df_v2 = pd.concat([f1, f2, f3], axis=1)
    df_v2['data_size'] = train_data_size(df, alleles_with_data, data_dict, has_left_out_alleles)
    
    col_order = feature_column_names.get_feature_cols(include_distbin8=include_distbin8)
    df_v2 = df_v2.loc[:, col_order]
    return df_v2


# BELOW IF FOR UPDATING AN EXISTING FEATURE DF.

# Updating existing featurized df.
def get_dist_bin_data_tuples(target_allele, neighbor_alleles, data_dict, include_distbin8=False):
    # Used only when updating existing feature set for speed in MVP-MHC alg.
    # Does not need data_dict for anything at this step.
    global seq2allele
    # Should this be by allele or pseudosequence?
    neighbor_df = neighbors_array(
        target_allele,
        neighbor_alleles,
        len(neighbor_alleles),
        data_dict,
        by_pseudoseq=True
    )                          
    neighbor_df.loc[:,'dist_bin']=[NeighborDistBins(dict(), include_distbin8).dist_to_bin(dist) for dist in neighbor_df['dist_to_train']]
    tr_seqs_sorted, distance_bins_sorted = neighbor_df['train_sequence'], neighbor_df['dist_bin']
    return zip(tr_seqs_sorted, distance_bins_sorted)

def update_feat3(df_test, allele_gets_data, data_to_add, data_alleles_BAEL, classical_hlas, data_dict_new, include_distbin8):
    """
    Update data values for an existing df_test with bin features. 
    """
    classical_neighbors = list(set(data_alleles_BAEL).intersection(set(classical_hlas))) # only classical alleles can be possible MVP-MHCs. (Subset to decrease needed iterations).
    #f include_distbin8:
    #  distance_bins = 8
    #lse:
    if True:
        distance_bins = 7
     
    col_names = ['allele'] + [f'dist_bin_{i/10}' for i in range(0, distance_bins+1, 1)]
    df_test_copy = df_test.loc[:,col_names].copy()
    df_test_copy.set_index('allele', inplace=True)
     
    # Only compute dist bins for allele which is getting new data.
    # dist bins are reciprocal for two alleles. 
    # data_dict is not used in get_dist_bin_data_tuples() so use a empty dict to satify class object.
    tmp_tups = get_dist_bin_data_tuples(allele_gets_data, classical_neighbors, data_dict_new, include_distbin8)
    for train_seq, dist_bin in tmp_tups:
        for train_allele in seq2allele[train_seq]: 
            if train_allele not in list(df_test.allele): # Don't need to update features for alleles that have data, but are not in df_test.
                continue
            if dist_bin in ['dist_bin_0.8', 'dist_bin_0.9', 'dist_bin_1.0']:
                continue
            current_data = df_test_copy.loc[train_allele, dist_bin]
            df_test_copy.loc[train_allele, dist_bin] = current_data + data_to_add
    return df_test_copy.loc[:,'dist_bin_0.0':'dist_bin_0.7'].reset_index(drop=True)


def update_level2_features(df, allele_gets_data, data_to_add, alleles_with_data, classical_hlas, data_dict_new, include_distbin8):
    """
    Update an existing df (featurized?) with level 2 model features for speed increase when doing MVP_MHC algorithm.
    """
    
    for a in data_dict_new:
        if data_dict_new[a] >0:
            assert a in alleles_with_data
    for a in alleles_with_data:
        assert data_dict_new[a] >0
            
    # f3 is the only feature set I actualy update from input df.
    # The other feature sets are computed from scratch.
    f1=feat_1(df, data_dict_new, alleles_with_data, has_left_out_alleles=False)
    #updatefeat_1(df, data_dict_new, alleles_with_data) # This function was identica to feat_1 with a speed up I could put in feat_1 anyways.
    f2=topK_features_mp(df, data_dict_new, alleles_with_data, has_left_out_alleles=False)
    f3 = update_feat3(
        df,
        allele_gets_data,
        data_to_add, 
        alleles_with_data,
        classical_hlas,
        data_dict_new,
        include_distbin8
    ).reset_index(drop=True)
    
    df_v2 = pd.concat([f1, f2, f3], axis=1)
    df_v2['data_size'] = train_data_size(df, alleles_with_data, data_dict_new, has_left_out_alleles=False)

    col_order = feature_column_names.get_feature_cols(include_distbin8)
    df_v2 = df_v2.loc[:, col_order]
    return df_v2
