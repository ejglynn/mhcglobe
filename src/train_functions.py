import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Callbacks
from tensorflow.keras import optimizers, losses
import build_deepnet#, #build_mhcperf
import inequality_loss as il

#def val_loss(Y_true, Y_predict, loss_type='mse_inequality'):
#    MSE = il.MSEWithInequalities().loss(
#            np.array(Y_true, dtype='float32'),
#            np.array(Y_predict, dtype='float32')).numpy()
#    return MSE

def get_mhcglobe_callbacks(model_savepath):
    callbacks = [
        Callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            baseline=1,
            min_delta=0.0001),
        Callbacks.ModelCheckpoint(
            filepath=model_savepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch')]
    return callbacks

def get_mhcperf_callbacks(model_savepath):
    callbacks = [
        Callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min'),
        Callbacks.ModelCheckpoint(
            model_savepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch')]
    return callbacks
"""

def build__mhcglobe_optimizer(hparams):
    optimizer = optimizers.RMSprop(
        learning_rate=hparams['rms_learning_rate'],
        momentum=hparams['rms_momentum'],
        epsilon=hparams['rms_epsilon'],
        centered=hparams['rms_centered'])
    return optimizer

def get_compiled_mhcglobe_model(hparams):
    # build tensorflow graph and optimizer
    K.clear_session()

    model = build_deepnet.BuildModel('mhcglobe').build_graph(hparams)
    #print(model.summary())
    optimizer = build_optimizer(hparams)

    inequality_loss = il.MSEWithInequalities().loss
    model.compile(loss=inequality_loss,
                       optimizer=optimizer)
    return model
"""

def build_optimizer(hparams):
    optimizer = optimizers.RMSprop(
        learning_rate=hparams['rms_learning_rate'],
        momentum=hparams['rms_momentum'],
        epsilon=hparams['rms_epsilon'],
        centered=hparams['rms_centered'])
    return optimizer

def get_compiled_model(hparams):
    # build tensorflow graph and optimizer
    K.clear_session()

    model_name = hparams['model_name']
    assert model_name in ['mhcglobe', 'mhcperf']
    #model = build_mhcperf.BuildModel().build_graph(hparams)
    model = build_deepnet.BuildModel(model_name).build_graph(hparams)
    
    optimizer = build_optimizer(hparams)
    
    if model_name == 'mhcglobe':
        loss = il.MSEWithInequalities().loss
    elif model_name == 'mhcperf':
        loss = hparams['loss_type']
        
    model.compile(loss, optimizer=optimizer)
    return model

def train_mhcglobe_model(model, X, Y, X_val, Y_val, savepath, verbose=0):
    callbacks = get_mhcglobe_callbacks(savepath)
    model.fit(
        X, Y,
        batch_size= 10000, #hparams['batch_size'], 300
        epochs=300,
        validation_data=(X_val, Y_val),
        shuffle=True,
        verbose=verbose,
        callbacks=callbacks,)
    model.load_weights(savepath)
    model.save(savepath, save_format="tf")
    if verbose > 0:
        print(savepath)
    return model

def train_mhcperf_model(batch_size, epochs, model, X_train, Y_train, savepath, verbose=0):
    callbacks = get_mhcperf_callbacks(savepath)
   
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              verbose=verbose,
              callbacks=callbacks,
             )
    model.load_weights(savepath)
    model.save(savepath, save_format="tf")
    if verbose > 0:
        print(savepath)
    return model


def load_trained_mhcglobe_model(model_path, loss_type='mse_inequality'):
    if loss_type == 'mse_inequality':
        tf.keras.utils.get_custom_objects().clear()
        tf.keras.utils.get_custom_objects()['loss'] = il.MSEWithInequalities().loss
        model = tf.keras.models.load_model(model_path)#, custom_objects={'loss': custom_loss})
    else:
        custom_loss = il.MSEWithInequalities().loss
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'loss': custom_loss})
    return model


def load_ensemble(model_dir : str, model_group : str):
    """
    model_group indicates if the model is an
    init, non-human trained, all mhc train, or
    excludes a particular benchmark dataset from
    training.
    """
    MODEL_IDS_TUPS
    ensemble_models =[]
    for fold, model_number, seq_encode in MODEL_IDS_TUPS:
        model_save_path = model_dir + f'{model_number}_{fold}_{seq_encode}_{model_group}'
        ensemble_models.append(
            load_trained_mhcglobe_model(model_save_path, loss_type='mse_inequality')
        )
    return ensemble_models

class BalanceSplitData():
    """
    Sample input df indices at random to be a second split. Splits are balanced by allele.
    """
    def assign_test_indicies(self, df):
        test_size = int(df.shape[0] * 1/5)
        test_indices = np.random.choice(df.index, size=test_size, replace=False)
        df.insert(0, 'test', [i in test_indices for i in df.index])
        return df

    def get_train_val(self, df):
        """
        Generate a split per allele.
        """
        df_with_split_col = []
        for allele in set(df['allele']):
            sub_df = self.assign_test_indicies(df[df['allele']==allele])
            df_with_split_col.append(sub_df)
            
        df_with_split_col = pd.concat(
            df_with_split_col,
            axis=0,
            sort=False).reset_index(drop=True)
        return df_with_split_col[~df_with_split_col['test']], df_with_split_col[df_with_split_col['test']]
