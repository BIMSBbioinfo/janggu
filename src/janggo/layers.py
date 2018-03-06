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
    The layer can be used with arbitrary one-hot representation
    orders.

    Parameters
    ----------
    order : int
        Order of the one-hot representation.
    """
    rcmatrix = None

    def __init__(self, order, **kwargs):
        self.order = order
        super(Complement, self).__init__(**kwargs)

    def build(self, input_shape):
        self.rcmatrix = K.constant(complement_permmatrix(self.order),
                                   dtype=K.floatx())
        super(Complement, self).build(input_shape)

    def call(self, inputs):
        return tf.einsum('ij,bjsc->bisc', self.rcmatrix, inputs)

    def get_config(self):
        config = {'order': self.order}
        base_config = super(Complement, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
