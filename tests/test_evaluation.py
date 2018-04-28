import os

import HTSeq
import numpy
import pkg_resources
import pyBigWig
from keras import Input
from keras import Model
from keras.layers import Dense
from keras.layers import Flatten

from janggo.data import GenomicIndexer
from janggo.data import NumpyWrapper
from janggo.evaluation import _export_to_bed
from janggo.evaluation import _export_to_bigwig
from janggo.evaluation import _input_dimension_match
from janggo.evaluation import _output_dimension_match


def test_input_dims():
    data = NumpyWrapper('testa', numpy.zeros((10, 10, 1)))
    xin = Input((10, 1), name='testy')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _input_dimension_match(m, data)

    xin = Input((20, 10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _input_dimension_match(m, data)
    # more input datasets supplied than inputs to models
    assert not _input_dimension_match(m, [data, data])

    xin = Input((10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _input_dimension_match(m, data)


def test_output_dims():
    data = NumpyWrapper('testa', numpy.zeros((10, 10, 1)))
    label = NumpyWrapper('testy', numpy.zeros((10, 1)))
    xin = Input(data.shape, name='asdf')
    out = Flatten()(xin)
    out = Dense(1)(out)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _output_dimension_match(m, label)

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(2, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _output_dimension_match(m, label)

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(1, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _output_dimension_match(m, label)

    assert _output_dimension_match(m, None)


def test_export_bigwig_predict(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    values = numpy.ones((10, 4, 1, 2)) * .5
    values[:, :, :, 1] += 1

    _export_to_bigwig('test', 'denselayer',
                      tmpdir.strpath, gi, values,
                      ['sample1', 'sample2'], 'out')

    files = [os.path.join(tmpdir.strpath, 'export',
                          'out.test.denselayer.sample1.bigwig'),
             os.path.join(tmpdir.strpath, 'export',
                          'out.test.denselayer.sample2.bigwig')]
    for file_, value in zip(files, [0.5, 1.5]):
        # if file exists and output is correct
        bw = pyBigWig.open(file_)

        for idx, region in enumerate(gi):
            co = bw.values(region.chrom,
                           region.start*gi.resolution +
                           gi.binsize//2 - gi.resolution//2,
                           (region.end)*gi.resolution -
                           gi.binsize//2 + gi.resolution//2 - 1)
            print(co)
            numpy.testing.assert_equal(numpy.mean(co), value)


def test_export_bed_predict(tmpdir):
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    values = numpy.ones((10, 4, 1, 2)) * .5
    values[:, :, :, 1] += 1

    _export_to_bed('test', 'denselayer',
                   tmpdir.strpath, gi, values,
                   ['sample1', 'sample2'], 'out')

    files = [os.path.join(tmpdir.strpath, 'export',
                          'out.test.denselayer.sample1.bed'),
             os.path.join(tmpdir.strpath, 'export',
                          'out.test.denselayer.sample2.bed')]
    for file_, value in zip(files, [0.5, 1.5]):
        # if file exists and output is correct
        bed = iter(HTSeq.BED_Reader(file_))

        for idx, region in enumerate(gi):
            breg = next(bed)

            numpy.testing.assert_equal(breg.score, value)
