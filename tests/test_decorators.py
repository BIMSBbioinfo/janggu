import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from janggo import inputlayer
from janggo import outputconv
from janggo import outputdense


# ==========================================================
# Test without decorators
def make_dense_wo_decorator(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    layer = Dense(params)(input[0])
    output = [Dense(outshapes[name]['shape'][0], name=name,
                    activation=outshapes[name]['activation'])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without output decorator
@outputdense
def make_dense_w_top(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Dense(params)(input[0])
    return input, output


# ==========================================================
# Test without input decorator
@inputlayer
def make_dense_w_bottom(input, inshapes, outshapes, params):
    layer = Dense(params)(input[0])
    output = [Dense(outshapes[name]['shape'][0], name=name,
                    activation=outshapes[name]['activation'])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without input and output decorator
@inputlayer
@outputdense
def make_dense_w_topbottom(input, input_props, output_props, params):
    output = Dense(params)(input[0])
    return input, output


# ==========================================================
# Test without decorators
def make_conv_wo_decorator(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    layer = Conv2D(params, (1, 1))(input[0])
    output = [Conv2D(outshapes[name]['shape'][-1],
                     (6, 4),
                     name=name,
                     activation=outshapes[name]['activation'])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without output decorator
@outputconv
def make_conv_w_top(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Conv2D(params, (1, 1))(input[0])
    return input, output


# ==========================================================
# Test without input decorator
@inputlayer
def make_conv_w_bottom(input, inshapes, outshapes, params):
    input
    layer = Conv2D(params, (1, 1))(input[0])
    output = [Conv2D(outshapes[name]['shape'][-1],
                     (6, 4),
                     name=name,
                     activation=outshapes[name]['activation'])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without input and output decorator
@inputlayer
@outputconv
def make_conv_w_topbottom(input, input_props, output_props, params):
    output = Conv2D(params, (1, 1))(input[0])
    return input, output


def test_dense_decorators():
    inp = {'testin': {'shape': (10,)}}
    oup = {'testout': {'shape': (3,), 'activation': 'relu'}}

    funclist = [make_dense_w_top, make_dense_w_bottom, make_dense_w_topbottom]

    i, o = make_dense_wo_decorator(None, inp, oup, 30)
    ref_model = Model(i, o)
    for func in funclist:
        i, o = func(None, inp, oup, 30)
        model = Model(i, o)
        for i in range(len(model.layers)):
            np.testing.assert_equal(model.layers[i].input_shape,
                                    ref_model.layers[i].input_shape)
            np.testing.assert_equal(model.layers[i].output_shape,
                                    ref_model.layers[i].output_shape)


def test_conv_decorators():

    inp = {'testin': {'shape': (10, 4, 1)}}
    oup = {'testout': {'shape': (5, 1, 3), 'activation': 'relu'}}

    funclist = [make_conv_w_bottom, make_conv_w_top, make_conv_w_topbottom]

    i, o = make_conv_wo_decorator(None, inp, oup, 30)
    ref_model = Model(i, o)
    ref_model.summary()

    for func in funclist:
        i, o = func(None, inp, oup, 30)
        model = Model(i, o)
        for i in range(len(model.layers)):
            np.testing.assert_equal(model.layers[i].input_shape,
                                    ref_model.layers[i].input_shape)
            np.testing.assert_equal(model.layers[i].output_shape,
                                    ref_model.layers[i].output_shape)
