import os

import numpy as np
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Maximum
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggo import InOutScorer
from janggo import Janggo
from janggo import inputlayer
from janggo import outputdense
from janggo.data import Array
from janggo.data import Dna
from janggo.layers import Complement
from janggo.layers import Reverse
from janggo.utils import export_tsv
from janggo.utils import plot_score

np.random.seed(1234)

DATA_PATH = pkg_resources.resource_filename('janggo', 'resources/')
SAMPLE_1 = os.path.join(DATA_PATH, 'sample.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2.fa')
X1 = Dna.create_from_fasta('dna', fastafile=SAMPLE_1, order=1)

DNA = Dna.create_from_fasta('dna', fastafile=[SAMPLE_1, SAMPLE_2], order=1)

Y = np.zeros((len(DNA), 1))
Y[:len(X1)] = 1
LABELS = Array('y', Y, conditions='TF-binding')


def wrap_roc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(roc_auc_score(y_true, y_pred)))
    print('roc', aux)
    return fpr, tpr, aux


def wrap_prc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aux = str('({:.2%})'.format(average_precision_score(y_true, y_pred)))
    print('prc', aux)
    return recall, precision, aux


auc_eval = InOutScorer('auROC', roc_auc_score, exporter=export_tsv)
prc_eval = InOutScorer('PRC', wrap_prc, exporter=plot_score)
roc_eval = InOutScorer('ROC', wrap_roc, exporter=plot_score)
auprc_eval = InOutScorer('auPRC', average_precision_score, exporter=export_tsv)


# Option 3:
# Instantiate an ordinary keras model
@inputlayer
@outputdense('sigmoid')
def janggobody(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        forward = layer
    convlayer = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                       activation=params[2])
    revcomp = Reverse()(forward)
    revcomp = Complement()(revcomp)

    forward = convlayer(forward)
    revcomp = convlayer(revcomp)
    revcomp = Reverse()(revcomp)
    layer = Maximum()([forward, revcomp])
    output = GlobalAveragePooling2D()(layer)
    return inputs, output


K.clear_session()
model = Janggo.create(template=janggobody,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=LABELS,
                      outputdir='tf_predict')

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(DNA, LABELS, epochs=100)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

model.evaluate(DNA, LABELS, datatags=['training_set'],
               callbacks=[auc_eval, prc_eval, roc_eval, auprc_eval])
