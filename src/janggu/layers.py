"""Janggu specific network layers.

This module contains custom keras layers defined for
janggu.
"""
import numpy
import tensorflow as tf  # pylint: disable=import-error
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Constant

from janggu.utils import complement_permmatrix


class LocalAveragePooling2D(Layer):

    """LocalAveragePooling2D layer.

    This layer performs window averaging along the lead
    axis of an input tensor using a given window_size.
    At the moment, it assumes data_format='channels_last'.

    Parameters
    ----------
    window_size : int
        Averaging window size. Default: 1.
    """

    kernel = None
    bias = None

    def __init__(self, window_size=1, **kwargs):
        self.window_size = window_size
        super(LocalAveragePooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # This will only work with tensorflow at the moment
        # (filter_width, filter_height, input_channels, output_channels)
        kernel_shape = (self.window_size, 1, 1, 1)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=Constant(value=1./self.window_size),
                                      name='avg_filter',
                                      trainable=False)

        self.bias = None
        self.built = True

    def call(self, inputs):  # pylint: disable=arguments-differ

        pin = K.permute_dimensions(inputs, (0, 1, 3, 2))
        avg_conv = K.conv2d(pin,
                            self.kernel,
                            strides=(1, 1),
                            padding="valid",
                            data_format='channels_last',
                            dilation_rate=(1, 1))
        output = K.permute_dimensions(avg_conv, (0, 1, 3, 2))
        return output

    def get_config(self):
        config = {'window_size': self.window_size}
        base_config = super(LocalAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] -= self.window_size - 1
        return tuple(output_shape)


class Reverse(Layer):
    """Reverse layer.

    This layer can be used with keras to reverse
    a tensor for a given axis.

    Parameters
    ----------
    axis : int
        Axis which needs to be reversed. Default: 1.
    """

    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super(Reverse, self).__init__(**kwargs)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return K.reverse(inputs, self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Reverse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Complement(Layer):
    """Complement layer.

    This layer can be used with keras to determine
    the complementary DNA sequence in one-hot encoding
    from a given DNA sequences.
    It supports higher-order nucleotide representation,
    e.g. dinucleotides, trinucleotides.
    The order of the nucleotide representation is automatically
    determined from the previous layer. To this end,
    the input layer is assumed to hold the nucleotide representation
    dimension 3.
    The layer uses a permutation matrix that is multiplied
    with the original input dataset in order to evaluate
    the complementary sequence's one hot representation.

    Parameters
    ----------
    order : int
        Order of the one-hot representation.
    """
    rcmatrix = None

    def build(self, input_shape):

        # from the shape of the one-hot encoding (input_shape),
        # we determine the order of the encoding.
        self.rcmatrix = K.constant(
            complement_permmatrix(int(numpy.log(input_shape[2])/numpy.log(4))),
            dtype=K.floatx())
        super(Complement, self).build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return tf.einsum('ij,bsjc->bsic', self.rcmatrix, inputs)

    def get_config(self):
        base_config = super(Complement, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
