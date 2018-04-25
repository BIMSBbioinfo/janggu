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
from janggo.data import DnaDataset
from janggo.data import NumpyDataset
from janggo.data import input_props
from janggo.data import output_props
from janggo.evaluation import EvaluatorList
from janggo.evaluation import ScoreEvaluator
from janggo.evaluation import auprc
from janggo.evaluation import auroc


np.random.seed(1234)

DATA_PATH = pkg_resources.resource_filename('janggo', 'resources/')
OCT4_FILE = os.path.join(DATA_PATH, 'stemcells.fa')
X1 = DnaDataset.create_from_fasta('dna', fastafile=OCT4_FILE, order=1)
MAFK_FILE = os.path.join(DATA_PATH, 'mafk.fa')
X2 = DnaDataset.create_from_fasta('dna', fastafile=MAFK_FILE, order=1)

DNA = DnaDataset.create_from_fasta('dna', fastafile=[OCT4_FILE, MAFK_FILE],
                                   order=1)
DNA_ONEHOT = NumpyDataset('dna', np.concatenate((X1[:], X2[:])))

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = NumpyDataset('y', Y, samplenames='Oct4-binding')

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
model.summary()
hist = model.fit({'dna': DNA_ONEHOT}, {'y': LABELS}, epochs=10, batch_size=100)
print('Option 1')
print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)



# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputlayer
def janggobody(inputs, inp, oup, params):
    layer = inputs[0]
    # with inputs.use('dna') as layer:
    layer = Conv2D(params[0], (inp['dna']['shape'][2], 21),
                   activation=params[1])(layer)
    output = GlobalAveragePooling2D()(layer)
    return inputs, output


K.clear_session()
inp=input_props(DNA)
oup=output_props(LABELS, 'binary_crossentropy')
inputs, outputs = janggobody(None, inp, oup, (30, 'relu'))
model=Model(inputs, outputs)
model.summary()
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit({'dna': DNA_ONEHOT}, {'y': LABELS}, epochs=10, batch_size=100)
