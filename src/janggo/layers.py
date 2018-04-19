import numpy
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

from janggo.utils import complement_permmatrix


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

    def call(self, inputs):
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

    def __init__(self, **kwargs):
        super(Complement, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input_shape: {}".format(input_shape))

        # from the shape of the one-hot encoding (input_shape),
        # we determine the order of the encoding.
        self.rcmatrix = K.constant(
            complement_permmatrix(int(numpy.log(input_shape[2])/numpy.log(4))),
            dtype=K.floatx())
        print("input_shape: {}".format(input_shape))
        print("order: {}".format(int(numpy.sqrt(input_shape[2]))))
        print("mat.shape = {}".format(self.rcmatrix.shape))
        super(Complement, self).build(input_shape)

    def call(self, inputs):
        return tf.einsum('ij,bsjc->bsic', self.rcmatrix, inputs)

    def get_config(self):
        base_config = super(Complement, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
