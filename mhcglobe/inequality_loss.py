import os
import pandas
import numpy
from numpy import isnan, array
import tensorflow.keras.backend as K

def val_loss(Y_true, Y_predict, loss_type='mse_inequality'):
    MSE = il.MSEWithInequalities().loss(
            np.array(Y_true, dtype='float32'),
            np.array(Y_predict, dtype='float32')).numpy()
    return MSE

class MSEWithInequalities():
    """
    The code derived from MHCFlurry2.0.

    Supports training a regression model on data that includes inequalities
    (e.g. x < 100). Mean square error is used as the loss for elements with
    an (=) inequality. For elements with e.g. a (> 0.5) inequality, then the loss
    for that element is (y - 0.5)^2 (standard MSE) if y < 500 and 0 otherwise.
    This loss assumes that the normal range for y_true and y_pred is 0 - 1. As a
    hack, the implementation uses other intervals for y_pred to encode the
    inequality information.
    y_true is interpreted as follows:
    between 0 - 1
       Regular MSE loss is used. Penalty (y_pred - y_true)**2 is applied if
       y_pred is greater or less than y_true.
    between 2 - 3:
       Treated as a ">" inequality. Penalty (y_pred - (y_true - 2))**2 is
       applied only if y_pred is less than y_true - 2.
    between 4 - 5:
       Treated as a "<" inequality. Penalty (y_pred - (y_true - 4))**2 is
       applied only if y_pred is greater than y_true - 4.
    """
    name = "mse_with_inequalities"
    supports_inequalities = True
    supports_multiple_outputs = False

    @staticmethod
    def encode_y(y, inequalities=None):
        y = array(y, dtype="float32")
        if isnan(y).any():
            raise ValueError("y contains NaN", y)
        if (y > 1.0).any():
            raise ValueError("y contains values > 1.0", y)
        if (y < 0.0).any():
            raise ValueError("y contains values < 0.0", y)

        if inequalities is None:
            encoded = y
        else:
            offsets = pandas.Series(inequalities).map({
                '=': 0,
                '>': 2,
                '<': 4,
            }).values
            if isnan(offsets).any():
                raise ValueError("Invalid inequality. Must be =, <, or >")
            encoded = y + offsets
        assert not isnan(encoded).any()
        return encoded

    def loss(self, y_true, y_pred):
        # We always delay import of Keras so that mhcflurry can be imported
        # initially without tensorflow debug output, etc.
        #configure_tensorflow()
        #from tensorflow.keras import backend as K
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        # Handle (=) inequalities
        diff1 = y_pred - y_true
        diff1 *= K.cast(y_true >= 0.0, "float32")
        diff1 *= K.cast(y_true <= 1.0, "float32")

        # Handle (>) inequalities
        diff2 = y_pred - (y_true - 2.0)
        diff2 *= K.cast(y_true >= 2.0, "float32")
        diff2 *= K.cast(y_true <= 3.0, "float32")
        diff2 *= K.cast(diff2 < 0.0, "float32")

        # Handle (<) inequalities
        diff3 = y_pred - (y_true - 4.0)
        diff3 *= K.cast(y_true >= 4.0, "float32")
        diff3 *= K.cast(diff3 > 0.0, "float32")

        denominator = K.maximum(
            K.sum(K.cast(K.not_equal(y_true, 2.0), "float32"), 0),
            1.0)

        result = (
            K.sum(K.square(diff1)) +
            K.sum(K.square(diff2)) +
            K.sum(K.square(diff3))) / denominator

        return result


def val_loss(Y_true, Y_predict):
    MSE = MSEWithInequalities().loss(
            np.array(Y_true, dtype='float32'),
            np.array(Y_predict, dtype='float32')).numpy()
    return MSE
