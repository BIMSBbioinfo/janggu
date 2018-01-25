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

from janggo import MongoDbEvaluator
from janggo import janggo_fit_generator
from janggo import janggo_predict_generator
from janggo.data import NumpyDataset
from janggo.data import input_props
from janggo.data import output_props
from janggo.decorators import inputlayer
from janggo.decorators import outputlayer
from janggo.evaluate import auprc
from janggo.evaluate import auroc
from janggo.evaluate import av_auprc
from janggo.evaluate import av_auroc
from janggo.model import Janggo

np.random.seed(1234)

# Original MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Wrap it as Dataset
blg_x_train = NumpyDataset('x', x_train)
blg_x_test = NumpyDataset('x', x_test)
blg_y_train = NumpyDataset('y', y_train,
                              samplenames=[str(i) for i in range(10)])
blg_y_test = NumpyDataset('y', y_test,
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
def janggomodel():
    input = Input(shape=(28, 28), name='x')
    layer = Flatten()(input)
    layer = Dense(10, activation='tanh')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = Janggo(inputs=input, outputs=output, name='mnist')
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Option 3:
# Define the body of the network using the decorators.
# This option is used with Janggo.create_by_shape
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


# The datastructures provided by janggo can be immediately
# supplied to keras, because they mimic numpy arrays
K.clear_session()
np.random.seed(1234)
m = kerasmodel()
h = m.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100, verbose=0)
ypred = m.predict(blg_x_test)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


# then use it in janggo
K.clear_session()
np.random.seed(1234)
bw = janggomodel()
h = bw.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100, verbose=0)
ypred = bw.predict(blg_x_test)

print('Option 2a')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


# then use it in janggo
K.clear_session()
np.random.seed(1234)
bw = janggomodel()
h = bw.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100,
           generator=janggo_fit_generator,
           workers=3, verbose=0)
ypred = bw.predict(blg_x_test)

print('Option 2b')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


# then use it in janggo
K.clear_session()
np.random.seed(1234)
bw = janggomodel()
h = bw.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100,
           generator=janggo_fit_generator,
           workers=3, verbose=0)
ypred = bw.predict(blg_x_test, generator=janggo_predict_generator)

print('Option 2c')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


# For comparison, here is how the model would train without Janggo
K.clear_session()
np.random.seed(1234)
m = kerasmodel()
h = m.fit(x_train, y_train, epochs=30, batch_size=100,
          shuffle=False, verbose=0)
ypred = m.predict(blg_x_test)
print('Option 3')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


# Instantiate model from input and output shape
K.clear_session()
np.random.seed(1234)
bw = Janggo.create_by_shape(input_props(blg_x_train),
                            output_props(blg_y_train,
                                         'categorical_crossentropy'),
                            'mnist_ffn',
                            modeldef=(ffn, (10, 'tanh',)),
                            metrics=['acc'])
h = bw.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100, verbose=0)
ypred = bw.predict(blg_x_test)
print('Option 4')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)


evaluator = MongoDbEvaluator()

# Evaluate the results
evaluator.dump(bw, blg_x_test, blg_y_test,
               elementwise_score={'auROC': auroc, 'auPRC': auprc},
               combined_score={'av-auROC': av_auroc,
                               'av-auPRC': av_auprc}, batch_size=100,
               use_multiprocessing=True)

for row in evaluator.db.results.find():
    pprint.pprint(row)
