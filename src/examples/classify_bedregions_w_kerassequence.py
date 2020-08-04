import argparse
import os

import numpy as np
from keras import Model

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Maximum
from keras.layers import Reshape
from pkg_resources import resource_filename

from janggu.data import Bioseq
from janggu.data import Cover
from janggu.layers import Complement, Reverse
from janggu.layers import DnaConv2D
from janggu.data.data import JangguSequence

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')

PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")

PARSER.add_argument('-model', dest='model',
                    default='single', choices=['single', 'double'],
                    help="Single or double stranded.")

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
                                        binsize=200)

LABELS_TEST = Cover.create_from_bed('peaks',
                                    bedfiles=PEAK_FILE,
                                    roi=ROI_TEST,
                                    binsize=200,
                                    resolution=None)

# Training input and labels are purely defined genomic coordinates
DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                   roi=ROI_TRAIN,
                                   binsize=200)

LABELS = Cover.create_from_bed('peaks', roi=ROI_TRAIN,
                               bedfiles=PEAK_FILE,
                               binsize=200,
                               resolution=None)


# define a keras model here

xin = Input((200, 1, 4), name="dna")
convl = Conv2D(30, (21, 1),
               activation='relu')

if args.model == 'double':
   layer = DnaConv2D(convl)(xin)
else:
   layer = convl(xin)

layer = GlobalAveragePooling2D()(layer)
layer = Dense(1, activation='sigmoid')(layer)

# the last one is used to make the dimensionality compatible with
# the coverage dataset dimensions.
# Alternatively, the ReduceDim dataset wrapper may be used to transform
# the output to a 2D dataset object.
output = Reshape((1, 1, 1), name="peaks")(layer)

model = Model(xin, output)

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

trainseq = JangguSequence(DNA, LABELS, batch_size=32)
valseq = JangguSequence(DNA_TEST, LABELS_TEST)

hist = model.fit(trainseq, epochs=500,
                 validation_data=valseq)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)


# convert the prediction to a cover object
pred = model.predict(valseq)
cov_pred = Cover.create_from_array('BindingProba', pred, LABELS_TEST.gindexer)

print('Prediction score examples for Oct4')
for i in range(4):
    print('{}.: {}'.format(i, cov_pred[i]))
print('Prediction score examples for Mafk')
for i in range(1, 5):
    print('{}.: {}'.format(i, cov_pred[-i]))

# predictions (or feature activities) can finally be exported to bigwig
cov_pred.export_to_bigwig(output_dir=args.path)
