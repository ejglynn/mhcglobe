import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Callbacks
from tensorflow.keras import optimizers, losses
import build_level2_ANN

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

def build_mhcperf_optimizer(hparams):
    optimizer = optimizers.RMSprop(
        learning_rate=hparams['rms_learning_rate'],
        momentum=hparams['rms_momentum'],
        epsilon=hparams['rms_epsilon'],
        centered=True)
    return optimizer


def get_compiled_mhcperf_model(hparams):
    # build tensorflow graph and optimizer
    K.clear_session()

    model = build_level2_ANN.BuildModel().build_graph(hparams)
    optimizer = build_optimizer(hparams)
    
    model.compile(
        loss=hparams['loss_type'],
        optimizer=optimizer)
    return model

def train_model_tfdata(epochs, model, X_train, Y_train, savepath, verbose=0):
    callbacks = get_callbacks(savepath)
    model.fit(X_train, Y_train,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              verbose=verbose,
              callbacks=callbacks,
             )
    model.load_weights(savepath)
    model.save(savepath)
    print(savepath)
    return model
