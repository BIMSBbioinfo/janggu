import os

import numpy as np
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model

from janggo import Janggo
from janggo import inputlayer
from janggo import janggo_fit_generator
from janggo import outputlayer
from janggo.data import Dna
from janggo.data import NumpyWrapper
from janggo.data import input_props
from janggo.data import output_props
from janggo.evaluation import EvaluatorList
from janggo.evaluation import ScoreEvaluator
from janggo.evaluation import auprc
from janggo.evaluation import auroc

np.random.seed(1234)

DATA_PATH = pkg_resources.resource_filename('janggo', 'resources/')
OCT4_FILE = os.path.join(DATA_PATH, 'stemcells.fa')
X1 = Dna.create_from_fasta('dna', fastafile=OCT4_FILE, order=1)
MAFK_FILE = os.path.join(DATA_PATH, 'mafk.fa')
X2 = Dna.create_from_fasta('dna', fastafile=MAFK_FILE, order=1)

DNA = Dna.create_from_fasta('dna', fastafile=[OCT4_FILE, MAFK_FILE],
                                   order=1)
DNA_ONEHOT = NumpyWrapper('dna', np.concatenate((X1[:], X2[:])))

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = NumpyWrapper('y', Y, conditions='Oct4-binding')

auc_eval = ScoreEvaluator('oct4_result', 'auROC', auroc)
prc_eval = ScoreEvaluator('oct4_result', 'auPRC', auprc)
evaluators = EvaluatorList('oct4_result', [auc_eval, prc_eval])


# Option 1:
# One can use a keras model directly with all Datasets, because
# the datasets satisfy an interface that mimics ordinary numpy arrays.
def kerasmodel():
    input_ = Input(shape=(4, 200, 1), name='dna')
    layer = Conv2D(30, (4, 21), activation='relu')(input_)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid', name='y')(layer)
    model_ = Model(inputs=input_, outputs=output)
    model_.compile(optimizer='adadelta', loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model_


# For comparison, here is how the model would train without Janggo
K.clear_session()
np.random.seed(1234)
model = kerasmodel()
hist = model.fit({'dna': DNA_ONEHOT}, {'y': LABELS}, epochs=10, batch_size=100)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)


# Option 2:
# Instantiate an ordinary keras model
def janggomodel(name):
    input_ = Input(shape=(4, 200, 1), name='dna')
    layer = Conv2D(30, (4, 21), activation='relu')(input_)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid', name='y')(layer)
    model_ = Janggo(inputs=input_, outputs=output,
                    name=name, outputdir='oct4_result')
    model_.compile(optimizer='adadelta', loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model_


# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputlayer
def janggobody(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        layer = Conv2D(params[0], (inp['dna']['shape'][0], 21),
                       activation=params[1])(layer)
    output = GlobalAveragePooling2D()(layer)
    return inputs, output


K.clear_session()
model = Janggo.create_by_shape(input_props(DNA),
                               output_props(LABELS, 'binary_crossentropy'),
                               'oct4_cnn_create_shape_fit_1',
                               modeldef=(janggobody, (30, 'relu',)),
                               metrics=['acc'], outputdir='oct4_result')
hist = model.fit(DNA_ONEHOT, LABELS, epochs=10, batch_size=100)
model.kerasmodel.fit(DNA_ONEHOT, LABELS, epochs=10, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = Janggo.create_by_shape(input_props(DNA),
                               output_props(LABELS, 'binary_crossentropy'),
                               'oct4_cnn_create_shape_fit_2',
                               modeldef=(janggobody, (30, 'relu',)),
                               metrics=['acc'], outputdir='oct4_result')
hist = model.fit(DNA, LABELS, epochs=10, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = Janggo.create_by_shape(input_props(DNA),
                               output_props(LABELS, 'binary_crossentropy'),
                               'oct4_cnn_create_shape_fitgen_3',
                               modeldef=(janggobody, (30, 'relu',)),
                               metrics=['acc'], outputdir='oct4_result')
hist = model.fit(DNA_ONEHOT, LABELS, epochs=10, batch_size=100,
                 generator=janggo_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = Janggo.create_by_shape(input_props(DNA),
                               output_props(LABELS, 'binary_crossentropy'),
                               'oct4_cnn_create_shape_fitgen_4',
                               modeldef=(janggobody, (30, 'relu',)),
                               metrics=['acc'], outputdir='oct4_result')
hist = model.fit(DNA, LABELS, epochs=10, batch_size=100,
                 generator=janggo_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = janggomodel('oct4_cnn_create_keras_fit_5')
hist = model.fit(DNA_ONEHOT, LABELS, epochs=10, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = janggomodel('oct4_cnn_create_keras_fit_6')
hist = model.fit(DNA, LABELS, epochs=10, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = janggomodel('oct4_cnn_create_keras_fitgen_7')
hist = model.fit(DNA_ONEHOT, LABELS, epochs=10, batch_size=100,
                 generator=janggo_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

K.clear_session()
model = janggomodel('oct4_cnn_create_keras_fitgen_8')
hist = model.fit(DNA, LABELS, epochs=10, batch_size=100,
                 generator=janggo_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)


evaluators.evaluate(DNA, LABELS,
                    datatags=['test_set', 'dnaindex'],
                    batch_size=100, use_multiprocessing=True)
evaluators.evaluate(DNA_ONEHOT, LABELS,
                    datatags=['test_set', 'onehot'],
                    batch_size=100, use_multiprocessing=True)

# finally, dump everything into the file
evaluators.dump()
