import argparse
import os
import numpy as np
import pandas as pd
import pkg_resources

from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Maximum

from janggu import Janggu
from janggu import Scorer
from janggu import inputlayer
from janggu import outputdense
from janggu.data import Array
from janggu.data import Bioseq
from janggu.layers import Complement
from janggu.layers import DnaConv2D
from janggu.layers import Reverse
from janggu.utils import ExportClustermap
from janggu.utils import ExportTsv

import matplotlib
matplotlib.use('Agg')

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

# DNA sequences in one-hot encoding will be used as input
DNA = Bioseq.create_from_seq('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                             order=args.order, cache=True)

# An array of 1/0 will be used as labels for training
Y = np.asarray([[1] for line in range(nseqs(SAMPLE_1))] +
               [[0] for line in range(nseqs(SAMPLE_2))])
LABELS = Array('y', Y, conditions=['TF-binding'])
annot = pd.DataFrame(Y[:], columns=LABELS.conditions).applymap(
    lambda x: 'Oct4' if x == 1 else 'Mafk').to_dict(orient='list')

# Define the model templates

@inputlayer
@outputdense('sigmoid')
def single_stranded_model(inputs, inp, oup, params):
    """ keras model that scans a DNA sequence using
    a number of motifs.

    This model only scans one strand for sequence patterns.
    """
    with inputs.use('dna') as layer:
        # the name in inputs.use() should be the same as the dataset name.
        layer = Conv2D(params[0], (params[1], 1), activation=params[2])(layer)
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output


@inputlayer
@outputdense('sigmoid')
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
    convlayer = Conv2D(params[0], (params[1], 1), activation=params[2])
    revcomp = Reverse()(forward)
    revcomp = Complement()(revcomp)

    forward = convlayer(forward)
    revcomp = convlayer(revcomp)
    revcomp = Reverse()(revcomp)
    layer = Maximum()([forward, revcomp])
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output


@inputlayer
@outputdense('sigmoid')
def double_stranded_model_dnaconv(inputs, inp, oup, params):
    """ keras model for scanning both DNA strands.

    A more elegant way of scanning both strands for motif occurrences
    is achieved by the DnaConv2D layer wrapper, which internally
    performs the convolution operation with the normal kernel weights
    and the reverse complemented weights.
    """
    with inputs.use('dna') as layer:
        # the name in inputs.use() should be the same as the dataset name.
        conv = DnaConv2D(Conv2D(params[0],
                                (params[1], 1),
                                activation=params[2]), name='conv1')(layer)

    output = GlobalAveragePooling2D(name='motif')(conv)
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
                      outputs=LABELS,
                      name='fasta_seqs_m{}_o{}'.format(args.model, args.order))

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

# fit the model
hist = model.fit(DNA, LABELS, epochs=100)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

# load test data
SAMPLE_1 = os.path.join(DATA_PATH, 'sample_test.fa')
SAMPLE_2 = os.path.join(DATA_PATH, 'sample2_test.fa')

DNA_TEST = Bioseq.create_from_seq('dna', fastafile=[SAMPLE_1, SAMPLE_2],
                                  order=args.order, cache=True)

Y = np.asarray([[1] for _ in range(nseqs(SAMPLE_1))] +
               [[0] for _ in range(nseqs(SAMPLE_2))])
LABELS_TEST = Array('y', Y, conditions=['TF-binding'])
annot_test = pd.DataFrame(Y[:], columns=LABELS_TEST.conditions).applymap(
    lambda x: 'Oct4' if x == 1 else 'Mafk').to_dict(orient='list')

# clustering plots based on hidden features
heatmap_eval = Scorer('heatmap', exporter=ExportClustermap(annot=annot_test,
                                                           z_score=1.))

# output the predictions as tables or json files
pred_tsv = Scorer('pred', exporter=ExportTsv(annot=annot_test,
                                             row_names=DNA_TEST.gindexer.chrs))

# do the evaluation on the independent test data
# after the evaluation and prediction has been performed,
# the callbacks further process the results allowing
# to automatically generate summary statistics or figures
# into the JANGGU_OUTPUT directory.
model.evaluate(DNA_TEST, LABELS_TEST, datatags=['test'],
               callbacks=['auc', 'auprc', 'roc', 'auroc'])

pred = model.predict(DNA_TEST, datatags=['test'],
              callbacks=[pred_tsv, heatmap_eval],
              layername='motif')

pred = model.predict(DNA_TEST)
print('Oct4 predictions scores should be greater than Mafk scores:')
print('Prediction score examples for Oct4')
for i in range(4):
    print('{}.: {}'.format(i, pred[i]))
print('Prediction score examples for Mafk')
for i in range(1, 5):
    print('{}.: {}'.format(i, pred[-i]))

