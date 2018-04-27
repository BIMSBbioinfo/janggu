import numpy
from keras import Input
from keras import Model
from keras.layers import Dense
from keras.layers import Flatten

from janggo.data import NumpyWrapper
from janggo.evaluation import _input_dimension_match
from janggo.evaluation import _output_dimension_match


def test_input_dims():
    data = NumpyWrapper('testa', numpy.zeros((10, 10, 1)))
    xin = Input((10, 1), name='testy')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _input_dimension_match(m, data)

    xin = Input((20, 10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _input_dimension_match(m, data)
    # more input datasets supplied than inputs to models
    assert not _input_dimension_match(m, [data, data])

    xin = Input((10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _input_dimension_match(m, data)


def test_output_dims():
    data = NumpyWrapper('testa', numpy.zeros((10, 10, 1)))
    label = NumpyWrapper('testy', numpy.zeros((10, 1)))
    xin = Input(data.shape, name='asdf')
    out = Flatten()(xin)
    out = Dense(1)(out)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _output_dimension_match(m, label)

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(2, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _output_dimension_match(m, label)

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(1, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _output_dimension_match(m, label)

    assert _output_dimension_match(m, None)
