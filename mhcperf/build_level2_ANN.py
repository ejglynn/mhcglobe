import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Dropout, \
                                    GaussianNoise, Flatten, MaxPooling1D, \
                                    LocallyConnected1D, BatchNormalization, \
                                    ReLU, Concatenate

#initializer_He = tf.keras.initializers.HeNormal() #tf.compat.v1.initializers.he_normal()
initializer_GlorotNormal = tf.keras.initializers.GlorotNormal() #tf.compat.v1.keras.initializers.glorot_normal()

class BuildModel():
    def __init__(self):
        tf.keras.backend.clear_session() # reset tf session

    def add_Dense(self, tnsr, hparams):
        tnsr = Dense(hparams['dense_units'],
                     activation=hparams['activation'],
                     kernel_regularizer=tf.keras.regularizers.l1(l=hparams['L1_reg']),
                     )(tnsr)
        tnsr = GaussianNoise(stddev=hparams['gaussian_noise'])(tnsr)
        tnsr = Dropout(rate=hparams['dropout_rate'])(tnsr)
        return tnsr
    
    def build_graph(self, hparams):
        print(hparams)
        input_tnsr = Input(shape=hparams['input_shape'], name='level2_input', dtype='float32')

        # Hidden Layer 1
        d1 = Dense(hparams['dense_units_1'],
                   activation=hparams['activation'],
                   kernel_regularizer = tf.keras.regularizers.l1(l=hparams['L1_reg']),
                   kernel_initializer=initializer_GlorotNormal,
                   name='d1')(input_tnsr)
        d1 = Dropout(rate=hparams['dropout_rate'], name='dropout_d1')(d1)
        if hparams['dense_layers']>=1 and hparams['skip_1']:
            d1 = Concatenate(axis=1, name='skip1')([input_tnsr, d1])

        # Hidden Layer 2
        if hparams['dense_layers'] >= 2:
            d2 = Dense(hparams['dense_units_2'],
                       activation=hparams['activation'],
                       kernel_regularizer = tf.keras.regularizers.l1(l=hparams['L1_reg']),
                       kernel_initializer=initializer_GlorotNormal,
                       name='d2')(d1)
            d2 = Dropout(rate=hparams['dropout_rate'], name='dropout_d2')(d2)
            if hparams['dense_layers']>=2 and hparams['skip_2']:
                d2 = Concatenate(axis=1, name='skip2')([d1, d2])

        # Output Layer
        if hparams['dense_layers'] == 1:
            output = Dense(1, activation='sigmoid', kernel_initializer=initializer_GlorotNormal, name='output')(d1)
        elif hparams['dense_layers'] == 2:
            output = Dense(1, activation='sigmoid', kernel_initializer=initializer_GlorotNormal, name='output')(d2)

        model = Model(inputs=input_tnsr,
                      outputs=output,
                      name='model')
        return model
