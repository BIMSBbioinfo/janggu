import argparse
import os

import numpy as np
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Maximum
from pkg_resources import resource_filename
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggu import Janggu
from janggu import Scorer
from janggu import inputlayer
from janggu import outputconv
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse
from janggu.utils import ExportClustermap
from janggu.utils import ExportJson
from janggu.utils import ExportScorePlot
from janggu.utils import ExportTsne
from janggu.utils import ExportTsv

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('model', choices=['single', 'double', 'dnaconv'],
                    help="Single or double stranded model.")
PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")
PARSER.add_argument('-order', dest='order', type=int,
                    default=1,
                    help="One-hot order.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

# load the dataset
# The pseudo genome represents just a concatenation of all sequences
# in sample.fa and sample2.fa. Therefore, the results should be almost
# identically to the models obtained from classify_fasta.py.
REFGENOME = resource_filename('janggu', 'resources/pseudo_genome.fa')
# ROI contains regions spanning positive and negative examples
ROI_FILE = resource_filename('janggu', 'resources/roi_train.bed')
# PEAK_FILE only contains positive examples
PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')


# Training input and labels are purely defined genomic coordinates
DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                   roi=ROI_FILE,
                                   binsize=200,
                                   order=args.order,
                                   datatags=['ref'])

LABELS = Cover.create_from_bed('peaks', roi=ROI_FILE,
                               bedfiles=PEAK_FILE,
                               binsize=200,
                               resolution=200,
                               datatags=['train'])


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


# Define the model templates
@inputlayer
@outputconv('sigmoid')
def single_stranded_model(inputs, inp, oup, params):
    """ keras model that scans a DNA sequence using
    a number of motifs.

    This model only scans one strand for sequence patterns.
    """
    with inputs.use('dna') as layer:
        # the name in inputs.use() should be the same as the dataset name.
        layer = Conv2D(params[0], (params[1], 1),
                       activation=params[2])(layer)
    output = LocalAveragePooling2D(window_size=layer.shape.as_list()[1],
                                   name='motif')(layer)
    return inputs, output


@inputlayer
@outputconv('sigmoid')
def double_stranded_model(inputs, inp, oup, params):
    """ keras model for scanning both DNA strands.

    Sequence patterns may be present on either strand.
    By scanning both DNA strands with the same motifs (kernels)
    the performance of the model will generally improve.

    In the model below, this is achieved by reverse complementing
    the input tensor and keeping the convolution filters fixed.
    """
    with inputs.use('dna') as layer:
        # the name in inputs.use() should be the same as the dataset name.
        forward = layer
    convlayer = Conv2D(params[0], (params[1], 1),
                       activation=params[2])
    revcomp = Reverse()(forward)
    revcomp = Complement()(revcomp)

    forward = convlayer(forward)
    revcomp = convlayer(revcomp)
    revcomp = Reverse()(revcomp)
    layer = Maximum()([forward, revcomp])
    output = LocalAveragePooling2D(window_size=layer.shape.as_list()[1],
                                   name='motif')(layer)
    return inputs, output


@inputlayer
@outputconv('sigmoid')
def double_stranded_model_dnaconv(inputs, inp, oup, params):
    """ keras model for scanning both DNA strands.

    A more elegant way of scanning both strands for motif occurrences
    is achieved by the DnaConv2D layer wrapper, which internally
    performs the convolution operation with the normal kernel weights
    and the reverse complemented weights.
    """
    with inputs.use('dna') as layer:
        # the name in inputs.use() should be the same as the dataset name.
        layer = DnaConv2D(Conv2D(params[0], (params[1], 1),
                                 activation=params[2]))(layer)
    output = LocalAveragePooling2D(window_size=layer.shape.as_list()[1],
                                   name='motif')(layer)
    return inputs, output


if args.model == 'single':
    modeltemplate = single_stranded_model
elif args.model == 'double':
    modeltemplate = double_stranded_model
else:
    modeltemplate = double_stranded_model_dnaconv

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

ROI_FILE = resource_filename('janggu', 'resources/roi_test.bed')
# PEAK_FILE only contains positive examples
PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')


DNA_TEST = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                        roi=ROI_FILE,
                                        binsize=200,
                                        order=args.order,
                                        datatags=['ref'])

LABELS_TEST = Cover.create_from_bed('peaks',
                                    bedfiles=PEAK_FILE,
                                    roi=ROI_FILE,
                                    binsize=200,
                                    resolution=200,
                                    datatags=['test'])


# instantiate various evaluation callback objects
# score metrics
auc_eval = Scorer('auROC', roc_auc_score, exporter=ExportTsv())
prc_eval = Scorer('PRC', wrap_prc, exporter=ExportScorePlot())
roc_eval = Scorer('ROC', wrap_roc, exporter=ExportScorePlot())
auprc_eval = Scorer('auPRC', average_precision_score, exporter=ExportTsv())

# clustering plots based on hidden features
heatmap_eval = Scorer('heatmap', exporter=ExportClustermap(z_score=1.))
tsne_eval = Scorer('tsne', exporter=ExportTsne())

# output the predictions as tables or json files
pred_tsv = Scorer('pred', exporter=ExportTsv(row_names=DNA_TEST.gindexer.chrs))
pred_json = Scorer('pred', exporter=ExportJson(row_names=DNA_TEST.gindexer.chrs))

# plotly will export a special table that is used for interactive browsing
# of the results
pred_plotly = Scorer('pred', exporter=ExportTsv(row_names=DNA_TEST.gindexer.chrs,
                                                filesuffix='ply'))

# do the evaluation on the independent test data
model.evaluate(DNA_TEST, LABELS_TEST, datatags=['test'],
               callbacks=[auc_eval, prc_eval, roc_eval, auprc_eval])

pred = model.predict(DNA_TEST)
cov_pred = Cover.create_from_array('BindingProba', pred, LABELS_TEST.gindexer)


model.predict(DNA_TEST, datatags=['test'],
              callbacks=[pred_tsv, pred_json, pred_plotly],
              layername='motif')
model.predict(DNA_TEST, datatags=['test'],
              callbacks=[heatmap_eval, tsne_eval],
              layername='motif')
