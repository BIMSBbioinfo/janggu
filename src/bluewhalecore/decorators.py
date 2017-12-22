from functools import wraps

from keras.layers import Dense
from keras.layers import Input


def outputlayer(func):
    @wraps(func)
    def add(input, inshapes, outshapes, params):
        input, output = func(input, inshapes, outshapes, params)
        print('outputlayer')
        output = [Dense(outshapes[name]['shape'][0],
                  activation=outshapes[name]['activation'],
                  name=name)(output) for name in outshapes]
        return input, output
    return add


def inputlayer(func):
    @wraps(func)
    def add(input, inshapes, outshapes, params):
        input = [Input(inshapes[name]['shape'], name=name)
                 for name in inshapes]
        print('inputlayer')
        input, output = func(input, inshapes, outshapes, params)
        return input, output
    return add
