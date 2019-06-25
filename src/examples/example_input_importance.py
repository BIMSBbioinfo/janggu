import argparse
import os

import numpy as np
import h5py
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from pkg_resources import resource_filename

from janggu import Janggu
from janggu import Scorer
from janggu import inputlayer
from janggu import outputdense
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import GenomicIndexer
from janggu.data import ReduceDim
from janggu.data import plotGenomeTrack
from janggu.data import LineTrack
from janggu.data import SeqTrack
from janggu.layers import DnaConv2D
from janggu import input_attribution

np.random.seed(1234)


# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')
PARSER.add_argument('-path', dest='path',
                    default='tf_results',
                    help="Output directory for the examples.")
PARSER.add_argument('-order', dest='order', type=int,
                    default=1,
                    help="One-hot order.")
PARSER.add_argument('-epochs', dest='epochs', type=int,
                    default=100,
                    help="Number of epochs to train the model.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

# load the dataset
# The pseudo genome represents just a concatenation of all sequences
# in sample.fa and sample2.fa. Therefore, the results should be almost
# identically to the models obtained from classify_fasta.py.
REFGENOME = resource_filename('janggu', 'resources/pseudo_genome.fa')
VCFFILE = resource_filename('janggu', 'resources/pseudo_snps.vcf')
# ROI contains regions spanning positive and negative examples
ROI_TRAIN_FILE = resource_filename('janggu', 'resources/roi_train.bed')
ROI_TEST_FILE = resource_filename('janggu', 'resources/roi_test.bed')

# PEAK_FILE only contains positive examples
PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')

# Training input and labels are purely defined genomic coordinates
DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                   roi=ROI_TRAIN_FILE,
                                   binsize=200,
                                   store_whole_genome=True,
                                   order=args.order)

LABELS = Cover.create_from_bed('peaks', roi=ROI_TRAIN_FILE,
                               bedfiles=PEAK_FILE,
                               binsize=200,
                               resolution=200)


DNA_TEST = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                        roi=ROI_TEST_FILE,
                                        binsize=200,
                                        store_whole_genome=True,
                                        order=args.order)

LABELS_TEST = Cover.create_from_bed('peaks',
                                    roi=ROI_TEST_FILE,
                                    bedfiles=PEAK_FILE,
                                    binsize=200,
                                    resolution=200)
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

modeltemplate = double_stranded_model_dnaconv

K.clear_session()

# create a new model object
model = Janggu.create(template=modeltemplate,
                      modelparams=(30, 21, 'relu'),
                      inputs=DNA,
                      outputs=ReduceDim(LABELS))

model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['acc'])

hist = model.fit(DNA, ReduceDim(LABELS), epochs=args.epochs)

print('#' * 40)
print('loss: {}, acc: {}'.format(hist.history['loss'][-1],
                                 hist.history['acc'][-1]))
print('#' * 40)

pred = model.predict(DNA_TEST)
cov_pred = Cover.create_from_array('BindingProba', pred, LABELS_TEST.gindexer)

print('Oct4 predictions scores should be greater than Mafk scores:')
print('Prediction score examples for Oct4')
for i in range(4):
    print('{}.: {}'.format(i, cov_pred[i]))
print('Prediction score examples for Mafk')
for i in range(1, 5):
    print('{}.: {}'.format(i, cov_pred[-i]))

# Extract the 4th interval to perform input feature importance attribution
# which represents an Oct4 bound region
gi = DNA.gindexer[3]
chrom = gi.chrom
start = gi.start
end = gi.end
attr_oct = input_attribution(model, DNA, chrom=chrom, start=start, end=end)

# visualize the important sequence features
plotGenomeTrack(SeqTrack(attr_oct[0]),
                chrom, start, end).savefig(os.path.join(
                    args.path, 'influence_oct4_example_order{}.png'.format(args.order)))

# For the comparison, extract an interval
# representing a Mafk bound region and visualize the
# important features.
gi = DNA.gindexer[7796]
chrom = gi.chrom
start = gi.start
end = gi.end
attr_mafk = input_attribution(model, DNA, chrom=chrom, start=start, end=end)

plotGenomeTrack(SeqTrack(attr_mafk[0]),
                chrom, start, end).savefig(os.path.join(
                    args.path,
                    'influence_mafk_example_order{}.png'.format(args.order)))
# output directory for the variant effect prediction
vcfoutput = os.path.join(os.environ['JANGGU_OUTPUT'], 'vcfoutput')
os.makedirs(vcfoutput, exist_ok=True)

# perform variant effect prediction using Bioseq object and
# a VCF file
scoresfile, variantsfile = model.predict_variant_effect(DNA,
                                                        VCFFILE,
                                                        conditions=['feature'],
                                                        output_folder=vcfoutput)

scoresfile = os.path.join(vcfoutput, 'scores.hdf5')
variantsfile = os.path.join(vcfoutput, 'snps.bed.gz')

# parse the variant effect predictions (difference between
# reference and alternative variant) into a Cover object
# for the purpose of visualization
f = h5py.File(scoresfile, 'r')

gindexer = GenomicIndexer.create_from_file(variantsfile, None, None)

snpcov = Cover.create_from_array('snps', f['diffscore'],
                                 gindexer,
                                 store_whole_genome=True,
                                 padding_value=np.nan)
snpcov = Cover.create_from_array('snps', f['diffscore'],
                                 gindexer,
                                 store_whole_genome=False,
                                 padding_value=np.nan)


gi = DNA.gindexer[3]
chrom = gi.chrom
start = gi.start
end = gi.end

plotGenomeTrack([LineTrack(snpcov,
                           linestyle="None"), SeqTrack(attr_oct[0])],
                chrom, start, end).savefig(os.path.join(
                    args.path,
                    'predicted_variant_effects_oct4_order{}.png'.format(args.order)))

