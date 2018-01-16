import os

import numpy as np
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model

from bluewhalecore import BlueWhale
from bluewhalecore import MongoDbEvaluator
from bluewhalecore import inputlayer
from bluewhalecore import outputlayer
from bluewhalecore import bluewhale_fit_generator
from bluewhalecore.evaluate import bw_av_auroc
from bluewhalecore.evaluate import bw_av_auprc
from bluewhalecore.data import DnaBwDataset
from bluewhalecore.data import NumpyBwDataset
from bluewhalecore.data import input_shape
from bluewhalecore.data import output_shape

np.random.seed(1234)

DATA_PATH = pkg_resources.resource_filename('bluewhalecore', 'resources/')
OCT4_FILE = os.path.join(DATA_PATH, 'stemcells.fa')
X1 = DnaBwDataset.create_from_fasta('dna', fastafile=OCT4_FILE, order=1)
MAFK_FILE = os.path.join(DATA_PATH, 'mafk.fa')
X2 = DnaBwDataset.create_from_fasta('dna', fastafile=MAFK_FILE, order=1)

DNA = DnaBwDataset.create_from_fasta('dna', fastafile=[OCT4_FILE, MAFK_FILE],
                                     order=1)
DNA_ONEHOT = NumpyBwDataset('dna', np.concatenate((X1[:], X2[:])))

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = NumpyBwDataset('y', Y)

evaluator = MongoDbEvaluator()


# Option 1:
# One can use a keras model directly with all BwDatasets, because
# the datasets satisfy an interface that mimics ordinary numpy arrays.
def kerasmodel():
    input_ = Input(shape=(4, 200, 1))
    layer = Conv2D(30, (4, 21), activation='relu')(input_)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model_ = Model(inputs=input_, outputs=output)
    model_.compile(optimizer='adadelta', loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model_


# For comparison, here is how the model would train without BlueWhale
K.clear_session()
np.random.seed(1234)
model = kerasmodel()
hist = model.fit(DNA_ONEHOT, LABELS, epochs=300, batch_size=100)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)


# Option 2:
# Instantiate an ordinary keras model
def bluewhalemodel():
    input_ = Input(shape=(4, 200, 1), name='dna')
    layer = Conv2D(30, (4, 21), activation='relu')(input_)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid', name='y')(layer)
    model_ = BlueWhale(inputs=input_, outputs=output,
                       name='oct4_cnn')
    model_.compile(optimizer='adadelta', loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model_

# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputlayer
def bluewhalebody(inputs, inp, oup, params):
    layer = Conv2D(inp[0], (inp['dna']['shape'][2], 21),
                   activation=inp[1])(inputs[0])
    output = GlobalAveragePooling2D()(layer)
    return inputs, output



K.clear_session()
model = bluewhalemodel()
hist = model.fit(DNA_ONEHOT, LABELS, epochs=300, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA_ONEHOT, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['onehot'],
               modeltags=['fit'])


K.clear_session()
model = bluewhalemodel()
hist = model.fit(DNA, LABELS, epochs=300, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['dnaindex'],
               modeltags=['fit'])

K.clear_session()
model = bluewhalemodel()
hist = model.fit(DNA_ONEHOT, LABELS, epochs=300, batch_size=100,
                 generator=bluewhale_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA_ONEHOT, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['onehot'],
               modeltags=['fit_generator'])


K.clear_session()
model = bluewhalemodel()
hist = model.fit(DNA, LABELS, epochs=300, batch_size=100,
                 generator=bluewhale_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['dnaindex'],
               modeltags=['fit_generator'])


K.clear_session()
model = BlueWhale.create_by_shape(input_shape(DNA),
                              output_shape(LABELS, 'binary_crossentropy'),
                              'oct4_cnn',
                              modeldef=(bluewhalebody, (30, 'relu',)))
hist = model.fit(DNA_ONEHOT, LABELS, epochs=300, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA_ONEHOT, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['onehot'],
               modeltags=['fit'])


K.clear_session()
model = BlueWhale.create_by_shape(input_shape(DNA),
                              output_shape(LABELS, 'binary_crossentropy'),
                              'oct4_cnn_from_shape',
                              modeldef=(bluewhalebody, (30, 'relu',)))
hist = model.fit(DNA, LABELS, epochs=300, batch_size=100)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['dnaindex'],
               modeltags=['fit'])


K.clear_session()
model = BlueWhale.create_by_shape(input_shape(DNA),
                              output_shape(LABELS, 'binary_crossentropy'),
                              'oct4_cnn_from_shape',
                              modeldef=(bluewhalebody, (30, 'relu',)))
hist = model.fit(DNA_ONEHOT, LABELS, epochs=300, batch_size=100,
                 generator=bluewhale_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA_ONEHOT, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['onehot'],
               modeltags=['fit_generator'])

K.clear_session()
model = BlueWhale.create_by_shape(input_shape(DNA),
                              output_shape(LABELS, 'binary_crossentropy'),
                              'oct4_cnn_from_shape',
                              modeldef=(bluewhalebody, (30, 'relu',)))
hist = model.fit(DNA, LABELS, epochs=300, batch_size=100,
                 generator=bluewhale_fit_generator,
                 use_multiprocessing=True,
                 workers=3)
print('Option 2')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)
evaluator.dump(model, DNA, LABELS,
               combined_score={'AUC': bw_av_auroc, 'PRC': bw_av_auprc},
               datatags=['dnaindex'],
               modeltags=['fit_generator'])
