import argparse
import os

import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Maximum
from keras.layers import Reshape
from pkg_resources import resource_filename
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from janggu.data import Bioseq
from janggu.data import Cover
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D
from janggu.layers import Reverse

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')

PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

# load the dataset
# The pseudo genome represents just a concatenation of all sequences
# in sample.fa and sample2.fa. Therefore, the results should be almost
# identically to the models obtained from classify_fasta.py.
REFGENOME = resource_filename('janggu', 'resources/pseudo_genome.fa')
# ROI contains regions spanning positive and negative examples
ROI_TRAIN = resource_filename('janggu', 'resources/roi_train.bed')
ROI_TEST = resource_filename('janggu', 'resources/roi_test.bed')
# PEAK_FILE only contains positive examples
PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')


DNA_TEST = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                        roi=ROI_TEST,
                                        binsize=200,
                                        order=1,
                                        datatags=['ref'])

LABELS_TEST = Cover.create_from_bed('peaks',
                                    bedfiles=PEAK_FILE,
                                    roi=ROI_TEST,
                                    binsize=200,
                                    resolution=None,
                                    datatags=['test'])

# Training input and labels are purely defined genomic coordinates
DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                   roi=ROI_TRAIN,
                                   binsize=200,
                                   datatags=['ref'])

LABELS = Cover.create_from_bed('peaks', roi=ROI_TRAIN,
                               bedfiles=PEAK_FILE,
                               binsize=200,
                               resolution=None,
                               datatags=['train'])


# define a keras model here

xin = Input((200, 1, 4))
layer = DnaConv2D(Conv2D(30, (21, 1),
                         activation='relu'))(xin)
layer = GlobalAveragePooling2D()(layer)
layer = Dense(1, activation='sigmoid')(layer)

# the last one is used to make the dimensionality compatible with
# the coverage dataset dimension
output = Reshape((1, 1, 1))(layer)

model = Model(xin, output)

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

hist = model.fit(DNA, LABELS, epochs=100,
                 validation_data=(DNA_TEST, LABELS_TEST))

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)


# convert the prediction to a cover object
pred = model.predict(DNA_TEST)
cov_pred = Cover.create_from_array('BindingProba', pred, LABELS_TEST.gindexer)

# predicted labels for chromosome pseudo1 should be higher than for pseudo2,
# because in the example all of chromosome pseudo1 is bound and pseudo2 is unbound
for chrom in ['pseudo1', 'pseudo2']:
    print('prediction score for {}: {}'.format(chrom, cov_pred[chrom, 0, 1]))


# predictions (or feature activities) can finally be exported to bigwig
cov_pred.export_to_bigwig(output_dir=args.path)
