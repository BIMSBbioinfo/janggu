import pprint

import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import roc_auc_score

from bluewhalecore import Evaluator
from bluewhalecore import bluewhale_fit_generator
from bluewhalecore import bluewhale_predict_generator
from bluewhalecore.bluewhale import BlueWhale
from bluewhalecore.data import NumpyBwDataset
from bluewhalecore.data import inputShape
from bluewhalecore.data import outputShape
from bluewhalecore.decorators import inputlayer
from bluewhalecore.decorators import outputlayer
from bluewhalecore.evaluate import bw_auprc
from bluewhalecore.evaluate import bw_auroc
from bluewhalecore.evaluate import bw_av_auprc
from bluewhalecore.evaluate import bw_av_auroc

np.random.seed(1234)

# Original MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Wrap it as BwDataset
bw_x_train = NumpyBwDataset('x', x_train)
bw_x_test = NumpyBwDataset('x', x_test)
bw_y_train = NumpyBwDataset('y', y_train,
                            samplenames=[str(i) for i in range(10)])
bw_y_test = NumpyBwDataset('y', y_test,
                           samplenames=[str(i) for i in range(10)])


# Option 2:
# Instantiate the model manually
def kerasmodel():
    input = Input(shape=(28, 28), name='x')
    layer = Flatten()(input)
    layer = Dense(10, activation='tanh')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Option 2:
# Instantiate the model manually
def bluewhalemodel():
    input = Input(shape=(28, 28), name='x')
    layer = Flatten()(input)
    layer = Dense(10, activation='tanh')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = BlueWhale(inputs=input, outputs=output, name='mnist')
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Option 3:
# Define the body of the network using the decorators.
# This option is used with BlueWhale.fromShape
@inputlayer
@outputlayer
def ffn(input, inparams, outparams, otherparams):
    layer = Flatten()(input[0])
    output = Dense(otherparams[0], activation=otherparams[1])(layer)
    return input, output


def auc3(ytrue, ypred):
    yt = np.zeros((len(ytrue),))
    yt[ytrue[:, 2] == 1] = 1
    pt = ypred[:, 2]
    return roc_auc_score(yt, pt)


# The datastructures provided by bluewhalecore can be immediately
# supplied to keras, because they mimic numpy arrays
K.clear_session()
np.random.seed(1234)
m = kerasmodel()
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=100, verbose=0)
ypred = m.predict(bw_x_test)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


# then use it in bluewhale
K.clear_session()
np.random.seed(1234)
bw = bluewhalemodel()
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=100, verbose=0)
ypred = bw.predict(bw_x_test)

print('Option 2a')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


# then use it in bluewhale
K.clear_session()
np.random.seed(1234)
bw = bluewhalemodel()
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=100,
           generator=bluewhale_fit_generator,
           workers=3, verbose=0)
ypred = bw.predict(bw_x_test)

print('Option 2b')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


# then use it in bluewhale
K.clear_session()
np.random.seed(1234)
bw = bluewhalemodel()
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=100,
           generator=bluewhale_fit_generator,
           workers=3, verbose=0)
ypred = bw.predict(bw_x_test, generator=bluewhale_predict_generator)

print('Option 2c')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


# For comparison, here is how the model would train without BlueWhale
K.clear_session()
np.random.seed(1234)
m = kerasmodel()
h = m.fit(x_train, y_train, epochs=30, batch_size=100,
          shuffle=False, verbose=0)
ypred = m.predict(bw_x_test)
print('Option 3')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


# Instantiate model from input and output shape
K.clear_session()
np.random.seed(1234)
bw = BlueWhale.fromShape(inputShape(bw_x_train),
                         outputShape(bw_y_train, 'categorical_crossentropy'),
                         'mnist_ffn',
                         modeldef=(ffn, (10, 'tanh',)))
h = bw.fit(bw_x_train, bw_y_train, epochs=30, batch_size=100, verbose=0)
ypred = bw.predict(bw_x_test)
print('Option 4')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(bw_y_test[:], ypred)))
print('#' * 40)


evaluator = Evaluator()

# Evaluate the results
evaluator.dump(bw, bw_x_test, bw_y_test,
               elementwise_score={'auROC': bw_auroc, 'auPRC': bw_auprc},
               combined_score={'av-auROC': bw_av_auroc,
                               'av-auPRC': bw_av_auprc}, batch_size=100,
               use_multiprocessing=True)

for row in evaluator.db.results.find():
    pprint.pprint(row)
