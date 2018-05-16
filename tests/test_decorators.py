import keras.backend as K
import matplotlib
import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from janggu import inputlayer
from janggu import outputconv
from janggu import outputdense

matplotlib.use('AGG')


# ==========================================================
# Test without decorators
def make_dense_wo_decorator(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    layer = Dense(params[0])(input[0])
    output = [Dense(outshapes[name]['shape'][0], name=name,
                    activation=params[1])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without output decorator, sigmoid as string
@outputdense('sigmoid')
def make_dense_w_top_str(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Dense(params[0])(input[0])
    return input, output


# ==========================================================
# Test without output decorator, sigmoid as string
@outputdense({'testout': 'sigmoid'})
def make_dense_w_top_dict(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Dense(params[0])(input[0])
    return input, output


# ==========================================================
# Test without output decorator, sigmoid as string
@outputdense(K.tanh)
def make_dense_w_top_func(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Dense(params[0])(input[0])
    return input, output


# ==========================================================
# Test without input decorator
@inputlayer
def make_dense_w_bottom(input, inshapes, outshapes, params):
    layer = Dense(params[0])(input[0])
    output = [Dense(outshapes[name]['shape'][0], name=name,
                    activation=params[1])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without input and output decorator
@inputlayer
@outputdense('sigmoid')
def make_dense_w_topbottom(input, input_props, output_props, params):
    output = Dense(params[0])(input[0])
    return input, output


# ==========================================================
# Test without decorators
def make_conv_wo_decorator(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    layer = Conv2D(params[0], (6, 4))(input[0])
    output = [Conv2D(outshapes[name]['shape'][-1],
                     (1, 1),
                     name=name,
                     activation=params[1])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without output decorator
@outputconv('sigmoid')
def make_conv_w_top_str(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Conv2D(params[0], (6, 4))(input[0])
    return input, output


# ==========================================================
# Test without output decorator
@outputconv({'testout': 'sigmoid'})
def make_conv_w_top_dict(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Conv2D(params[0], (6, 4))(input[0])
    return input, output


# ==========================================================
# Test without output decorator
@outputconv(K.tanh)
def make_conv_w_top_func(input, inshapes, outshapes, params):
    input = [Input(inshapes[name]['shape'], name=name)
             for name in inshapes]
    output = Conv2D(params[0], (6, 4))(input[0])
    return input, output


# ==========================================================
# Test without input decorator
@inputlayer
def make_conv_w_bottom(input, inshapes, outshapes, params):
    input
    layer = Conv2D(params[0], (6, 4))(input[0])
    output = [Conv2D(outshapes[name]['shape'][-1],
                     (1, 1),
                     name=name,
                     activation=params[1])(layer)
              for name in outshapes]
    return input, output


# ==========================================================
# Test without input and output decorator
@inputlayer
@outputconv('sigmoid')
def make_conv_w_topbottom(input, input_props, output_props, params):
    output = Conv2D(params[0], (6, 4))(input[0])
    return input, output


def test_dense_decorators():
    inp = {'testin': {'shape': (10,)}}
    oup = {'testout': {'shape': (3,)}}

    funclist = [make_dense_w_top_str, make_dense_w_top_dict,
                make_dense_w_top_func,
                make_dense_w_bottom, make_dense_w_topbottom]

    i, o = make_dense_wo_decorator(None, inp, oup, (30, 'relu'))
    ref_model = Model(i, o)
    for func in funclist:
        i, o = func(None, inp, oup, (30, 'relu'))
        model = Model(i, o)
        for i in range(len(model.layers)):
            np.testing.assert_equal(model.layers[i].input_shape,
                                    ref_model.layers[i].input_shape)
            np.testing.assert_equal(model.layers[i].output_shape,
                                    ref_model.layers[i].output_shape)


def test_conv_decorators():

    inp = {'testin': {'shape': (10, 4, 1)}}
    oup = {'testout': {'shape': (5, 1, 3)}}

    funclist = [make_conv_w_top_str,
                make_conv_w_top_dict,
                make_conv_w_top_func,
                make_conv_w_bottom, make_conv_w_topbottom]

    i, o = make_conv_wo_decorator(None, inp, oup, (30, 'relu'))
    ref_model = Model(i, o)
    ref_model.summary()

    for func in funclist:
        i, o = func(None, inp, oup, (30, 'relu'))
        model = Model(i, o)
        for i in range(len(model.layers)):
            np.testing.assert_equal(model.layers[i].input_shape,
                                    ref_model.layers[i].input_shape)
            np.testing.assert_equal(model.layers[i].output_shape,
                                    ref_model.layers[i].output_shape)
