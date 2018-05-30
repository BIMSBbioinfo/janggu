
"""Janggu specific network layers.

This module contains custom keras layers defined for
janggu.
"""

from copy import copy
import numpy
import tensorflow as tf  # pylint: disable=import-error
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D
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


class DnaConv2D(Conv2D):
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
    scan_revcomp : boolean
        If True the reverse complement is scanned for motif matches.
        Default: False.

    Examples
    --------
    To scan both DNA strands for motif matches use

    .. code-block:: python

      conv = DnaConv2D(nfilters, filter_shape)
      # apply the normal convolution operation
      forward = conv(input)

      # obtain a copy of conv and scan the reverse compl. strand
      rcconv = conv.get_revcomp()
      reverse = rcconv(input)
    """
    kernel = None

    def __init__(self, filters,  # pylint: disable=too-many-locals
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 scan_revcomp=False, **kwargs):
        super(DnaConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.scan_revcomp = scan_revcomp
        self.rcmatrix = None

    def build(self, input_shape):
        super(DnaConv2D, self).build(input_shape)

        print(input_shape, input_shape[-1])
        self.rcmatrix = K.constant(
            complement_permmatrix(int(numpy.log(input_shape[-1])/numpy.log(4))),
            dtype=K.floatx())
        print(self.rcmatrix)

    def get_config(self):
        config = {'scan_revcomp': self.scan_revcomp}
        base_config = super(DnaConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        if self.scan_revcomp:
            print('using revcomp')
            # revert and complement the weight matrices
            tmp = self.kernel
            self.kernel = self.kernel[::-1, :, :, :]
            self.kernel = tf.einsum('ij,sdjc->sdic', self.rcmatrix, self.kernel)
        else:
            print('using conv2d')
        # perform the convolution operation
        res = super(DnaConv2D, self).call(inputs)
        if self.scan_revcomp:
            # restore the original kernel matrix
            self.kernel = tmp
        return res

    def get_revcomp(self):
        """Optain a copy of the layer that uses the reverse complement operation."""
        if not self.built:
            raise ValueError("Layer must be built before a copy can be obtained.")
        layer = copy(self)
        layer.name += '_rc'
        layer.scan_revcomp = True
        return layer
