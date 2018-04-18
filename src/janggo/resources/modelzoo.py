from keras import Input
from keras.layers import Dense
from keras.layers import Flatten

from janggo import inputlayer
from janggo import outputconv
from janggo import outputdense


@inputlayer
@outputdense
def _fnn_model1(inputs, inp, oup, params):
    layer = inputs[0]
    layer = Flatten()(layer)
    output = Dense(params[0])(layer)
    return inputs, output

@inputlayer
@outputconv
def _cnn_model2(inputs, inp, oup, params):
    with inputs.use('dna') as inlayer:
        layer = inlayer
    return inputs, layer

def _model3():
    inputs = Input((100,), name='x')
    output = Dense(1, activation='sigmoid')(inputs)

    return inputs, output
