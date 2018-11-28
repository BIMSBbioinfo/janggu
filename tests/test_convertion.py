import os

import matplotlib
matplotlib.use('AGG')

import numpy as np
from keras import backend as K
from keras.layers import Conv2D
from pkg_resources import resource_filename

from janggu import Janggu
from janggu import inputlayer
from janggu import outputconv
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.layers import DnaConv2D
from janggu.layers import LocalAveragePooling2D

def test_create_from_array(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # load the dataset
    # The pseudo genome represents just a concatenation of all sequences
    # in sample.fa and sample2.fa. Therefore, the results should be almost
    # identically to the models obtained from classify_fasta.py.
    REFGENOME = resource_filename('janggu', 'resources/pseudo_genome.fa')
    # ROI contains regions spanning positive and negative examples
    ROI_FILE = resource_filename('janggu', 'resources/roi_train.bed')
    # PEAK_FILE only contains positive examples
    PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')

    DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                       roi=ROI_FILE,
                                       binsize=200, stepsize=200,
                                       order=1,
                                       store_whole_genome=True,
                                       datatags=['ref'])

    LABELS = Cover.create_from_bed('peaks', roi=ROI_FILE,
                                   bedfiles=PEAK_FILE,
                                   binsize=200, stepsize=200,
                                   resolution=200,
                                   store_whole_genome=True,
                                   datatags=['train'])

    @inputlayer
    @outputconv('sigmoid')
    def double_stranded_model_dnaconv(inputs, inp, oup, params):
        with inputs.use('dna') as layer:
            layer = DnaConv2D(Conv2D(params[0], (params[1], 1),
                                     activation=params[2]))(layer)
        output = LocalAveragePooling2D(window_size=layer.shape.as_list()[1],
                                       name='motif')(layer)
        return inputs, output

    modeltemplate = double_stranded_model_dnaconv

    K.clear_session()

    # create a new model object
    model = Janggu.create(template=modeltemplate,
                          modelparams=(30, 21, 'relu'),
                          inputs=DNA,
                          outputs=LABELS)

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['acc'])

    pred = model.predict(DNA)

    cov_out = Cover.create_from_array('BindingProba', pred, LABELS.gindexer,
                                      store_whole_genome=True)

    assert pred.shape == cov_out.shape

    np.testing.assert_equal(pred, cov_out[:])

    assert len(cov_out.gindexer) == len(pred)
    assert len(cov_out.garray.handle) == 1


def test_create_from_array_whole_genome_false(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # load the dataset
    # The pseudo genome represents just a concatenation of all sequences
    # in sample.fa and sample2.fa. Therefore, the results should be almost
    # identically to the models obtained from classify_fasta.py.
    REFGENOME = resource_filename('janggu', 'resources/pseudo_genome.fa')
    # ROI contains regions spanning positive and negative examples
    ROI_FILE = resource_filename('janggu', 'resources/roi_train.bed')
    # PEAK_FILE only contains positive examples
    PEAK_FILE = resource_filename('janggu', 'resources/scores.bed')

    DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                       roi=ROI_FILE,
                                       binsize=200, stepsize=200,
                                       order=1,
                                       store_whole_genome=False,
                                       datatags=['ref'])

    LABELS = Cover.create_from_bed('peaks', roi=ROI_FILE,
                                   bedfiles=PEAK_FILE,
                                   binsize=200, stepsize=200,
                                   resolution=200,
                                   store_whole_genome=False,
                                   datatags=['train'])

    @inputlayer
    @outputconv('sigmoid')
    def double_stranded_model_dnaconv(inputs, inp, oup, params):
        with inputs.use('dna') as layer:
            layer = DnaConv2D(Conv2D(params[0], (params[1], 1),
                                     activation=params[2]))(layer)
        output = LocalAveragePooling2D(window_size=layer.shape.as_list()[1],
                                       name='motif')(layer)
        return inputs, output

    modeltemplate = double_stranded_model_dnaconv

    K.clear_session()

    # create a new model object
    model = Janggu.create(template=modeltemplate,
                          modelparams=(30, 21, 'relu'),
                          inputs=DNA,
                          outputs=LABELS)

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['acc'])

    pred = model.predict(DNA)

    cov_out = Cover.create_from_array('BindingProba', pred, LABELS.gindexer,
                                      store_whole_genome=False)

    assert pred.shape == cov_out.shape

    np.testing.assert_equal(pred, cov_out[:])

    assert len(cov_out.gindexer) == len(pred)
    assert len(cov_out.garray.handle) == len(pred)
