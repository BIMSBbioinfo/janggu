import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from bluewhalecore.bluewhale import BlueWhale
from bluewhalecore.data.data import inputShape
from bluewhalecore.data.data import outputShape
from bluewhalecore.data.nparr import NumpyBwDataset
from bluewhalecore.decorators import bottomlayer
from bluewhalecore.decorators import toplayer

np.random.seed(1234)

# Original MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Wrap it as BwDataset
bw_x_train = NumpyBwDataset('x', x_train)
bw_x_test = NumpyBwDataset('x', x_test)
bw_y_train = NumpyBwDataset('y', y_train)
bw_y_test = NumpyBwDataset('y', y_test)


# Define the body of the network
@bottomlayer
@toplayer
def ffn(input, inparams, outparams, otherparams):
    layer = Flatten()(input[0])
    output = Dense(otherparams[0], activation=otherparams[1])(layer)
    return input, output


# Option 1:
# Instantiate model from input and output shape
K.clear_session()
np.random.seed(1234)
bw = BlueWhale.fromShape('mnist_ffn', inputShape(bw_x_train),
                         outputShape(bw_y_train, 'categorical_crossentropy'),
                         modeldef=(ffn, (10, 'relu',)))
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000)
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)


# Option 2:
# Instantiate the model manually
def fmodel():
    input = Input(shape=(28, 28), name='x')
    layer = Flatten()(input)
    layer = Dense(10, activation='relu')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# then use it in bluewhale
K.clear_session()
np.random.seed(1234)
bw = BlueWhale('bw', fmodel())
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000)

print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)


# For comparison, here is how the model would train without BlueWhale
K.clear_session()
np.random.seed(1234)
m = fmodel()
h = m.fit(x_train, y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)
