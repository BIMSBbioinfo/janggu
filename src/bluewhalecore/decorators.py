"""Decorators to build neural networks.

This module contains decorators for automatically
appending an input and output layer of a neural network
with the correct dimensions.
There goal is to avoid having to define those layers.
Only the network body needs to be defined.
"""
from functools import wraps

from keras.layers import Dense
from keras.layers import Input


def outputlayer(func):
    """Output layer decorator

    This decorator appends an output layer to the
    network with the correct shape, activation and name.
    """
    @wraps(func)
    def add(inputs, inshapes, outshapes, params):
        inputs, outputs = func(inputs, inshapes, outshapes, params)
        outputs = [Dense(outshapes[name]['shape'][0],
                         activation=outshapes[name]['activation'],
                         name=name)(outputs) for name in outshapes]
        return inputs, outputs
    return add


def inputlayer(func):
    """Input layer decorator

    This decorator appends an input layer to the
    network with the correct shape and name.
    """
    @wraps(func)
    def add(inputs, inshapes, outshapes, params):
        inputs = [Input(inshapes[name]['shape'], name=name)
                  for name in inshapes]
        inputs, outputs = func(inputs, inshapes, outshapes, params)
        return inputs, outputs
    return add
