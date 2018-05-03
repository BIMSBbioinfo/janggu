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
from janggo import outputdense
from janggo import InOutScorer
from janggo.data import Dna
from janggo.data import NumpyWrapper
from janggo.utils import dump_tsv
from janggo.utils import plot_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

np.random.seed(1234)

DATA_PATH = pkg_resources.resource_filename('janggo', 'resources/')
SAMPLE_1 = os.path.join(DATA_PATH, 'sample.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2.fa')
X1 = Dna.create_from_fasta('dna', fastafile=SAMPLE_1, order=1)

DNA = Dna.create_from_fasta('dna', fastafile=[SAMPLE_1, SAMPLE_2], order=1)

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = NumpyWrapper('y', Y, conditions='TF-binding')

auc_eval = InOutScorer('auROC', roc_auc_score, dumper=dump_tsv)
prc_eval = InOutScorer('PRC', precision_recall_curve, dumper=plot_score)
auprc_eval = InOutScorer('auPRC', average_precision_score, dumper=dump_tsv)


# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputdense('sigmoid')
def janggobody(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        layer = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                       activation=params[2])(layer)
    output = GlobalAveragePooling2D()(layer)
    return inputs, output


K.clear_session()
model = Janggo.create(template=janggobody,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=LABELS,
                      outputdir='oct4_result')

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(DNA, LABELS, epochs=100)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

model.evaluate(DNA, LABELS, datatags=['training_set'],
               callbacks=[auc_eval, prc_eval, auprc_eval])

auc_eval.dump(model.outputdir)
prc_eval.dump(model.outputdir)
auprc_eval.dump(model.outputdir)
