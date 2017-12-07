from functools import wraps

from keras.layers import Dense
from keras.layers import Input


def toplayer(func):
    @wraps(func)
    def addOutput(inshapes, outshapes, params):
        input, output = func(inshapes, outshapes, params)
        print('toplayer')
        output = [Dense(outshapes[name]['shape'][0],
                  activation=outshapes[name]['activation'],
                  name=name)(output) for name in outshapes]
        return input, output
    return addOutput


def bottomlayer(func):
    @wraps(func)
    def addInput(inshapes, outshapes, params):
        input = [Input(inshapes[name]['shape'], name=name)
                 for name in inshapes]
        print('bottomlayer')
        input, output = func(input, outshapes, params)
        return input, output
    return addInput
