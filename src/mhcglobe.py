import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import train_functions as trainf
import sequence_functions as seqf
import binding_affinity as ba

from paths import DataPaths

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class LoadMHCGlobe():
    """Load tensorflow models that make up MHCGlobe. \
    MHCGlobe can vary by what data is was trained on.
    
    This code need not work for model suffixes outside of those enabled.
    """
    
    def __init__(self):
        # THESE DIR PATHS CAN ALSO BE IN A SEPERATE FILE AND READ IN. ##############
        #self.main_dir = '/home/eric'
        # THESE DIR PATHS CAN ALSO BE IN A SEPERATE FILE AND READ IN. ##############
        #self.init_model_dir          = DataPaths().models_dir + '/init/'
        #self.nonhuman_model_dir      = ''
        #self.full_model_dir          = DataPaths().models_dir + '/full/'
        #self.lno_model_dir           = ''
        #self.benchmark_netmhcpan_dir = '/home/eric/fairmhc/mhcglobe/models/noFRANK/'
        #self.bechmark_mhcflurry_dir  = DataPaths().models_dir + '/noS2/'

        self.modeldir_dict = {
            'init'                : DataPaths().mhcglobe_init,           # untrained
            'nonhuman'            : self.nonhuman_model_dir,      # trained on non-human mhc
            'full'                : DataPaths().mhcglobe_full,          # trained on full database
            #'lno'                : self.lno_model_dir,           # leave-n-out cross validation """A seperate class will be needed to coordinate lno paths and models."""
            'noFRANK' : self.benchmark_netmhcpan_dir, # benchmark excluded from training
            'noS2' : self.bechmark_mhcflurry_dir,  # benchmark excluded from training
        }
        self.hparam_ids = [(13, 14,'ONE_HOT'), (15, 37,'ONE_HOT'), (9, 79,'ONE_HOT')]
        
        
    def new_model_dir(self, train_type):
        """Check is a given model dir exists and return its path if True."""
        new_dir = DataPaths().models_dir + '/models_mhcglobe-natmtd/' + train_type
        return new_dir
        
    def paths(self, train_type):
        """Retrieve model paths specific to desired model_suffix label.
        
        Arguments
        _________

        train_type
            String specifying model type specific to how model was trained.

        Outputs
        ________


        ensemble_model_paths
            List of paths to each base model within the mhcglobe ensemble. 

        """
        self.train_type = train_type
        if train_type in self.modeldir_dict:
            self.model_dir = self.modeldir_dict[train_type]
        else:
            self.model_dir = self.new_model_dir(train_type)
        
        ensemble_model_paths = []
        for fold, model_number, protein_encoding in self.hparam_ids:
            model_filename = f'model{model_number}_fold{fold}_{protein_encoding}_{train_type}'
            model_path = os.path.join(self.model_dir, model_filename)
            ensemble_model_paths.append(model_path)
        return ensemble_model_paths
    
    def new_model_paths(self, model_save_path : str):
        new_model_paths = []
        for fold, model_number, protein_encoding in self.hparam_ids:
            model_filename = f'_model{model_number}_fold{fold}_{protein_encoding}'
            new_model_paths.append(model_save_path + model_filename)
        return new_model_paths

    def models(self, train_type):
        """ Load each neural network model in the mhcglobe ensemble.
        Return list of models and list of their paths.

        Outputs
        ________

        ensemble_models
            list of tensorflow models specific to the train_type class attribute..
        """
        
        ensemble_models = []
        ensemble_model_paths = self.paths(train_type)
        for model_save_path in ensemble_model_paths:
            model = trainf.load_trained_mhcglobe_model(model_save_path)
            ensemble_models.append(model)
        return ensemble_models


class MHCGlobe():
    """
    Ensemble of tensorflow neural networks used to predict binding affinity between
    a user defined mhc allele and short peptide fragment (8-15 amino acids in length).
    """
    def __init__(self, train_type=None, new_mhcglobe_path=None):
        """
        Arguments
        _________

        train_type
            String specifying model type specific to how model was trained.
        """
        K.clear_session()
        self.protein_encoding = 'ONE_HOT'
        self.hparam_ids = [(13, 14,'ONE_HOT'), (15, 37,'ONE_HOT'), (9, 79,'ONE_HOT')]
        assert set([tup[-1] for tup in self.hparam_ids]) == {self.protein_encoding}
        
        if train_type in ['init', 'full']:
            self.train_type = train_type        
            self.ensemble_base_models = LoadMHCGlobe().models(self.train_type)
            self.ensemble_model_paths = LoadMHCGlobe().paths(self.train_type)
        else:
            self.ensemble_model_paths = LoadMHCGlobe().new_model_paths(new_mhcglobe_path)
            self.ensemble_base_models = list(map(trainf.load_trained_mhcglobe_model, self.ensemble_model_paths))
            


    ##### TRAIN #####
    def setup_data_training(self, df_train):
        """Divide the training set for training and validation for early stopping.
        Convert allele-peptide instances into encodings, and return inputs and targets for each set.
              
        Arguments
        _________
        
        df_train
            Pandas data frame with training data
            
        Outputs
        _________
        
        X_tr: np.array
            Training inputs for MHCGlobe.
            
        Y_tr: np.array
            Training target values for MHCGlobe.
            
        X_es: np.array
            Validation/Early stopping inputs for MHCGlobe.
            
        Y_es: np.array
            Validation/Early stopping target values for MHCGlobe.
        """
        train, early_stopping = trainf.BalanceSplitData().get_train_val(df_train)
        X_tr, Y_tr = seqf.get_XY(
                train, encode_type=self.protein_encoding,
                get_Y=True)
        X_es, Y_es = seqf.get_XY(
                early_stopping, encode_type=self.protein_encoding,
                get_Y=True)
        return X_tr, Y_tr, X_es, Y_es
        
        
    def train_ensemble(self, df_train, new_mhcglobe_path, verbose=0):
        """
        train MHCGlobe on new data, `df_train`.
        
        Arguments
        _________
        
        df_train
            Pandas data frame with training data
        
        new_model_paths
            List of N full paths for each base MHCGlobe neural network to be saved. 
            N must be the same as number of MHCGlobe init models. 
        
        Load the init model, train it, then save trained model to new_model_path.
        """
        assert df_train.shape[0] >= 100
        print('Training...')
        new_model_paths = LoadMHCGlobe().new_model_paths(new_mhcglobe_path)
        assert len(new_model_paths) == len(self.ensemble_base_models)        
        assert np.sum(['init' in new_model_path for new_model_path in new_model_paths]) == 0, 'Cant create model init files.'
        assert self.train_type == 'init', 'Training requires "init" train_type.'
        
        # Train the MHCGlobe base neural networks iteratively.
        for init_model, new_model_path in zip(self.ensemble_base_models, new_model_paths):
            assert not os.path.exists(new_model_path), 'Already trained: {}'.format(new_model_path)
            
            # Data
            X_tr, Y_tr, X_es, Y_es = self.setup_data_training(df_train)
            
            # Train
            new_model = trainf.train_mhcglobe_model(init_model, X_tr, Y_tr, X_es, Y_es, new_model_path, verbose)

        print('Training complete.')
        return MHCGlobe(train_type=None, new_mhcglobe_path=new_mhcglobe_path)
        

    
    ##### PREDICT #####
    def ensemble_predict(self, X):
        """Run predictions with mhcglobe ensemble. Return pandas dataframe with predictions"""
        ensemble_predictions = pd.DataFrame()
        base_model_predictions = []
        for model in self.ensemble_base_models:
            predictions = pd.DataFrame(model.predict(X, verbose=0))
            base_model_predictions.append(predictions)
        ensemble_predictions = pd.concat(base_model_predictions, axis=1, ignore_index=True)
        ensemble_predictions.loc[:, 'mhcglobe_score'] = np.mean(ensemble_predictions, axis=1)
        ensemble_predictions.loc[:, 'mhcglobe_affinity'] = list(map(ba.to_ic50,ensemble_predictions['mhcglobe_score']))
        ensemble_predictions.columns = self.hparam_ids + ['mhcglobe_score', 'mhcglobe_affinity']
        return ensemble_predictions[['mhcglobe_affinity', 'mhcglobe_score']]

    def predict_on_dataframe(self, df):
        """Convert df columns into feature for model input `X` and
        run predictions. Return df with col predictions for each base model and
        a mean prediction across base models.
        
        Arguments
        _________

        df
            Pandas (pd) dataframe with columns containing mhc allele names and peptides to predict. 

        Output
        ________

        predictions
            Pandas dataframe with added columns containing predicted binding scores by
            each base model in mhcglobe ensemble and an aggregated prediction score
            (mean of base model predictions).
        """
        assert ('allele' in df.columns) & ('peptide' in df.columns)
        X = seqf.get_XY(df, encode_type=self.protein_encoding, get_Y=False)
        predictions = self.ensemble_predict(X)
        predictions.index = df.index
        #df.loc[:, 'mhcglobe_score'] = predictions
        return pd.concat([df, predictions], axis=1)
