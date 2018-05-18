import argparse
import os

import numpy as np
import pandas as pd
import pkg_resources
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Maximum
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggu import Scorer
from janggu import Janggu
from janggu import inputlayer
from janggu import outputdense
from janggu.data import Array
from janggu.data import Dna
from janggu.layers import Complement
from janggu.layers import Reverse
from janggu.utils import export_clustermap
from janggu.utils import export_json
from janggu.utils import export_plotly
from janggu.utils import export_score_plot
from janggu.utils import export_tsne
from janggu.utils import export_tsv

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('model', choices=['single', 'double'],
                    help="Single or double stranded model.")
PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")
PARSER.add_argument('-order', dest='order', type=int,
                    default=1,
                    help="One-hot order.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT']=args.path
# helper function
def nseqs(filename):
    """Extract the number of rows in the file.

    Note however, that this is a simplification
    that might not always work. In general, one would
    need to parse for '>' occurrences.
    """
    return sum((1 for line in open(filename) if line[0] == '>'))


# load the dataset
DATA_PATH = pkg_resources.resource_filename('janggu', 'resources/')
SAMPLE_1 = os.path.join(DATA_PATH, 'sample.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2.fa')

DNA = Dna.create_from_fasta('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                            order=args.order, datatags=['train'])

Y = np.asarray([1 for line in range(nseqs(SAMPLE_1))] +
               [0 for line in range(nseqs(SAMPLE_2))])
LABELS = Array('y', Y, conditions=['TF-binding'])
annot = pd.DataFrame(Y[:], columns=LABELS.conditions).applymap(
    lambda x: 'Oct4' if x == 1 else 'Mafk').to_dict(orient='list')


# evaluation metrics from sklearn.metrics
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


# instantiate various evaluation callback objects
# score metrics
auc_eval = Scorer('auROC', roc_auc_score, exporter=export_tsv)
prc_eval = Scorer('PRC', wrap_prc, exporter=export_score_plot)
roc_eval = Scorer('ROC', wrap_roc, exporter=export_score_plot)
auprc_eval = Scorer('auPRC', average_precision_score, exporter=export_tsv)

# clustering plots based on hidden features
heatmap_eval = Scorer('heatmap', exporter=export_clustermap)
tsne_eval = Scorer('tsne', exporter=export_tsne)

# output the predictions as tables or json files
pred_tsv = Scorer('pred', exporter=export_tsv)
pred_json = Scorer('pred', exporter=export_json)

# plotly will export a special table that is used for interactive browsing
# of the results
pred_plotly = Scorer('pred', exporter=export_plotly)

# Define the model templates
@inputlayer
@outputdense('sigmoid')
def single_stranded_model(inputs, inp, oup, params):
    with inputs.use('dna') as layer:
        layer = Conv2D(params[0], (params[1], layer.shape.as_list()[2]),
                       activation=params[2])(layer)
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output


@inputlayer
@outputdense('sigmoid')
def double_stranded_model(inputs, inp, oup, params):
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
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output


modeltemplate = single_stranded_model if args.model == 'single' \
                else double_stranded_model
K.clear_session()

# create a new model object
model = Janggu.create(template=modeltemplate,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=LABELS)

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(DNA, LABELS, epochs=100)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

SAMPLE_1 = os.path.join(DATA_PATH, 'sample_test.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2_test.fa')

DNA_TEST = Dna.create_from_fasta('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                                 order=args.order, datatags=['test'])


Y = np.asarray([1 for _ in range(nseqs(SAMPLE_1))] +
               [0 for _ in range(nseqs(SAMPLE_2))])
LABELS_TEST = Array('y', Y, conditions=['TF-binding'])
annot_test = pd.DataFrame(Y[:], columns=LABELS_TEST.conditions).applymap(
    lambda x: 'Oct4' if x == 1 else 'Mafk').to_dict(orient='list')

# do the evaluation on the training data
model.evaluate(DNA, LABELS, datatags=['train'],
               callbacks=[auc_eval, prc_eval, roc_eval, auprc_eval])

model.predict(DNA, datatags=['train'],
              callbacks=[pred_tsv, pred_json, pred_plotly],
              layername='motif',
              exporter_kwargs={'annot': annot,
                             'row_names': DNA.gindexer.chrs})
model.predict(DNA, datatags=['train'],
              callbacks=[heatmap_eval],
              layername='motif',
              exporter_kwargs={'annot': annot,
                             'z_score': 1})
model.predict(DNA, datatags=['train'],
              callbacks=[tsne_eval],
              layername='motif',
              exporter_kwargs={'annot': annot,
                             'alpha': .1})

# do the evaluation on the independent test data
model.evaluate(DNA_TEST, LABELS_TEST, datatags=['test'],
               callbacks=[auc_eval, prc_eval, roc_eval, auprc_eval])

model.predict(DNA_TEST, datatags=['test'],
              callbacks=[pred_tsv, pred_json, pred_plotly],
              layername='motif',
              exporter_kwargs={'annot': annot_test,
                             'row_names': DNA_TEST.gindexer.chrs})
model.predict(DNA_TEST, datatags=['test'],
              callbacks=[heatmap_eval],
              layername='motif',
              exporter_kwargs={'annot': annot_test,
                             'z_score': 1})
model.predict(DNA_TEST, datatags=['test'],
              callbacks=[tsne_eval],
              layername='motif',
              exporter_kwargs={'annot': annot_test,
                             'alpha': .1})
