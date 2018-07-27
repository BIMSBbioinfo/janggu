
"""Janggu specific network layers.

This module contains custom keras layers defined for
janggu.
"""

from copy import copy

import numpy
import tensorflow as tf  # pylint: disable=import-error
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras.layers import Conv2D
from keras.layers import Wrapper

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
            complement_permmatrix(int(numpy.log(input_shape[-1])/numpy.log(4))),
            dtype=K.floatx())
        super(Complement, self).build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return tf.einsum('ij,bsdj->bsdi', self.rcmatrix, inputs)

    def get_config(self):
        base_config = super(Complement, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class DnaConv2D(Wrapper):
    """DnaConv2D layer.

    This layer is a special convolution layer for scanning DNA
    sequences. When using it with the default settings, it behaves
    identically to the normal keras.layers.Conv2D layer.
    However, when setting the flag :code:`scan_revcomp=True`
    the weight matrices are reverse complemented which allows
    you to scan the reverse complementary sequence for motif matches.

    All parameters are the same as for keras.layers.Conv2D except for scan_revcomp.

    Parameters
    ----------
    merge_mode : boolean
        Specifies how to merge information from both strands. Options:
        {"sum", "max", "ave", "concat"}
        Default: "max".

    Examples
    --------
    To scan both DNA strands for motif matches use

    .. code-block:: python

      conv = DnaConv2D(Conv2D(nfilters, filter_shape))
      # apply the normal convolution operation
      dnalayer = conv(input)

    """
    kernel = None

    def __init__(self, layer, merge_mode='max', **kwargs):

        if merge_mode not in ['sum', 'max', 'ave', 'concat']:
            raise ValueError('Invalid merge mode {}. '.format(merge_mode) + \
                             'Merge mode should be one of '
                             '{"sum", "max", "ave", "concat", None}')
        self.merge_mode = merge_mode
        self.forward_layer = layer
        self.rcmatrix = None
        self._trainable = True
        super(DnaConv2D, self).__init__(layer, **kwargs)


    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        self.forward_layer.trainable = value

    def get_weights(self):
        return self.forward_layer.get_weights()

    def set_weights(self, weights):
        self.forward_layer.set_weights(weights)

    def compute_output_shape(self, input_shape):
        output_shape = self.forward_layer.compute_output_shape(input_shape)
        if self.merge_mode == 'concat':
            output_shape = list(output_shape)
            output_shape[-1] *= 2
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [output_shape, copy(output_shape)]

    def build(self, input_shape):

        super(DnaConv2D, self).build(input_shape)

        self.rcmatrix = K.constant(
            complement_permmatrix(int(numpy.log(input_shape[-1])/numpy.log(4))),
            dtype=K.floatx())
        with K.name_scope(self.forward_layer.name):
            self.forward_layer.build(input_shape)

    def get_config(self):
        config = {'merge_mode': self.merge_mode}

        base_config = super(DnaConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        dna_layer = deserialize_layer(config.pop('layer'),
                                      custom_objects=custom_objects)

        layer = cls(dna_layer, **config)
        return layer

    def call(self, inputs):
        kernel = self.forward_layer.kernel

        # revert and complement the weight matrices

        # perform the convolution operation
        res1 = self.forward_layer.call(inputs)

        #tmp = self.kernel
        self.forward_layer.kernel = self.forward_layer.kernel[::-1, :, :, :]
        self.forward_layer.kernel = tf.einsum('ij,sdjc->sdic',
                                              self.rcmatrix,
                                              self.forward_layer.kernel)

        res2 = self.forward_layer.call(inputs)

        # restore the weights
        self.forward_layer.kernel = self.forward_layer.kernel[::-1, :, :, :]
        self.forward_layer.kernel = tf.einsum('ij,sdjc->sdic', self.rcmatrix,
                                              self.forward_layer.kernel)

        if self.merge_mode == 'concat':
            res = K.concatenate([res1, res2])
        elif self.merge_mode == 'max':
            res = K.maximum(res1, res2)
        elif self.merge_mode == 'sum':
            res = res1 + res2
        elif self.merge_mode == 'ave':
            res = (res1 + res2) /2
        elif self.merge_mode is None:
            res = [res1, res2]

        return res
