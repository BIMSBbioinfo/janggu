import argparse
import os

import numpy as np
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from pkg_resources import resource_filename

from janggu import Janggu
from janggu import inputlayer
from janggu import outputdense
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import ReduceDim
from janggu.data import view
from janggu.layers import DnaConv2D

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')
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
ROI_FILE = resource_filename('janggu', 'resources/roi.bed')
ROI_TRAIN_FILE = resource_filename('janggu', 'resources/roi_train.bed')
ROI_TEST_FILE = resource_filename('janggu', 'resources/roi_test.bed')
# PEAK_FILE only contains positive examples
PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')

DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                   roi=ROI_FILE,
                                   order=args.order,
                                   binsize=200,
                                   store_whole_genome=True)

LABELS = Cover.create_from_bed('peaks', roi=ROI_FILE,
                               bedfiles=PEAK_FILE,
                               binsize=200,
                               resolution=200,
                               storage='sparse',
                               store_whole_genome=True)

# in case the dataset has been loaded with store_whole_genome=True,
# it is possible to reuse the same dataset by subsetting on different
# regions of the genome.
DNA_TRAIN = view(DNA, ROI_TRAIN_FILE)
LABELS_TRAIN = view(LABELS, ROI_TRAIN_FILE)
DNA_TEST = view(DNA, ROI_TEST_FILE)
LABELS_TEST = view(LABELS, ROI_TEST_FILE)

# Define the model templates

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
        layer = DnaConv2D(Conv2D(params[0], (params[1], 1),
                                 activation=params[2]))(layer)
    output = GlobalAveragePooling2D(name='motif')(layer)
    return inputs, output

K.clear_session()

# create a new model object
model = Janggu.create(template=double_stranded_model_dnaconv,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=ReduceDim(LABELS))

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

model.fit(DNA, ReduceDim(LABELS), epochs=100, validation_data=['pseudo2'])

# do the evaluation on the independent test data
model.evaluate(DNA_TEST, ReduceDim(LABELS_TEST), datatags=['test'],
               callbacks=['auc', 'auprc', 'roc', 'prc'])

pred = model.predict(DNA_TEST)
cov_pred = Cover.create_from_array('BindingProba', pred, LABELS_TEST.gindexer)

print('Oct4 predictions scores should be greater than Mafk scores:')
print('Prediction score examples for Oct4')
for i in range(4):
    print('{}.: {}'.format(i, cov_pred[i]))
print('Prediction score examples for Mafk')
for i in range(1, 5):
    print('{}.: {}'.format(i, cov_pred[-i]))

