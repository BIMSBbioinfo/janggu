import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import roc_auc_score

from janggo import janggo_fit_generator
from janggo import janggo_predict_generator
from janggo.data import NumpyDataset
from janggo.data import input_props
from janggo.data import output_props
from janggo.decorators import inputlayer
from janggo.decorators import outputlayer
from janggo.evaluation import EvaluatorList
from janggo.evaluation import ScoreEvaluator
from janggo.evaluation import auprc
from janggo.evaluation import auroc
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
    input_ = Input(shape=(28, 28), name='x')
    layer = Flatten()(input_)
    layer = Dense(10, activation='tanh')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = Model(inputs=input_, outputs=output)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Option 2:
# Instantiate the model manually
def janggomodel():
    input_ = Input(shape=(28, 28), name='x')
    layer = Flatten()(input_)
    layer = Dense(10, activation='tanh')(layer)
    output = Dense(10, activation='sigmoid', name='y')(layer)
    model = Janggo(inputs=input_, outputs=output, name='mnist')
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Option 3:
# Define the body of the network using the decorators.
# This option is used with Janggo.create_by_shape
@inputlayer
@outputlayer
def ffn(input_, inparams, outparams, otherparams):
    layer = Flatten()(input_[0])
    output = Dense(otherparams[0], activation=otherparams[1])(layer)
    return input_, output


def auc3(ytrue_, ypred_):
    yt = np.zeros((len(ytrue_),))
    yt[ytrue_[:, 2] == 1] = 1
    pt = ypred_[:, 2]
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
                            metrics=['acc'], outputdir='mnist_result')
h = bw.fit(blg_x_train, blg_y_train, epochs=30, batch_size=100, verbose=0)
ypred = bw.predict(blg_x_test)
print('Option 4')
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('AUC: {}'.format(auc3(blg_y_test[:], ypred)))
print('#' * 40)

auc_eval = ScoreEvaluator('mnist_result', 'auROC', auroc)
prc_eval = ScoreEvaluator('mnist_result', 'auPRC', auprc)
evaluators = EvaluatorList('mnist_result', [auc_eval, prc_eval])

# Evaluate the results
evaluators.evaluate(blg_x_test, blg_y_test, datatags=['test_set'],
                    batch_size=100, use_multiprocessing=True)
evaluators.dump()
