"""
This module contains special loss functions that can
handle missing values.
"""
import keras
import keras.backend as K


def binary_crossentropy_mv(y_true, y_pred):
    """binary_crossentropy missing value aware."""
    return K.switch(K.tf.is_nan(y_true),
                    keras.losses.binary_crossentropy(y_true, y_pred),
                    0.0)


def categorical_crossentropy_mv(y_true, y_pred):
    """categorical_crossentropy missing value aware."""
    return K.switch(K.tf.is_nan(y_true),
                    keras.losses.categorical_crossentropy(y_true, y_pred),
                    0.0)
