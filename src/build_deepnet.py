import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Flatten

class BuildModel():
    """Create model architecture for MHCGlobe."""

    def __init__(self, model_name):
        K.clear_session() # reset tf session
        assert model_name in ['mhcglobe', 'mhcperf']
        self.model_name = model_name
        self.he_initializer = tf.compat.v1.initializers.he_normal()
        self.glorot_initializer = tf.compat.v1.keras.initializers.glorot_normal()

    def add_dense_block(self, tnsr, layer_number, hparams):
        dense = Dense(hparams[f'dense_units_{layer_number}'],
                   activation=hparams['activation'],
                   kernel_regularizer = tf.keras.regularizers.l1(l=hparams['L1_reg']),
                   kernel_initializer=self.he_initializer,
                   name=f'd{layer_number}')(tnsr)
        dense = Dropout(rate=hparams['dropout_rate'], name=f'dropout_d{layer_number}')(dense)
        if hparams['dense_layers']>=layer_number and hparams[f'skip_{layer_number}']:
            dense = Concatenate(axis=1, name=f'skip{layer_number}')([tnsr, dense])
        return dense

    def mhcglobe_encoder(self):
        aa_rep_len = 20
        mhc_len    = 34
        pep_len    = 15
        
        input_peptide_tnsr = Input(
            shape=(pep_len, aa_rep_len),
            name='peptide_input', dtype='float32')
        input_mhc_tnsr = Input(
            shape=(mhc_len, aa_rep_len),
            name='mhc_input', dtype='float32')

        peptide_tnsr = Flatten()(input_peptide_tnsr)
        mhc_tnsr = Flatten()(input_mhc_tnsr)

        tnsr = Concatenate(axis=1, name='concat')([mhc_tnsr, peptide_tnsr])
        return tnsr
    
    def mhcperf_encoder(self):
        input_tnsr = Input(
            shape=(63,),
            name='mhcperf_input', dtype='float32')
        
    def build_graph(self, hparams):
        """ Build Tensorflow graph for neural network model."""
        if self.model_name == 'mhcglobe':
            tnsr = mhcglobe_encoder()
        elif self.model_name == 'mhcperf':
            tnsr = mhcperf_encoder()
        
        # Hidden Layer 1
        tnsr = self.add_dense_block(tnsr, layer_number=1, hparams=hparams)
        if hparams['dense_layers'] >= 2:
            tnsr = self.add_dense_block(tnsr, layer_number=2, hparams=hparams)
        if hparams['dense_layers'] >= 3:
            tnsr = self.add_dense_block(tnsr, layer_number=3, hparams=hparams)

        output = Dense(1, activation='sigmoid', kernel_initializer=self.glorot_initializer, name='output')(tnsr)
        model = Model(inputs=[input_mhc_tnsr, input_peptide_tnsr], outputs=output, name='model')
        return model
