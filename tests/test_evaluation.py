import json
import os

import HTSeq
import numpy
import pandas
import pkg_resources
import pyBigWig
from keras import Input
from keras import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input

from janggo import Janggo
from janggo import inputlayer
from janggo import janggo_fit_generator
from janggo import janggo_predict_generator
from janggo import outputconv
from janggo import outputdense
from janggo.data import GenomicIndexer
from janggo.data import NumpyWrapper
from janggo.evaluation import EvaluatorList
from janggo.evaluation import ScoreEvaluator
from janggo.evaluation import _export_to_bed
from janggo.evaluation import _export_to_bigwig
from janggo.evaluation import _input_dimension_match
from janggo.evaluation import _output_dimension_match
from janggo.evaluation import dump_tsv
from janggo.evaluation import plot_score


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


def test_various_outputs(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    generators: YES
    """

    inputs = NumpyWrapper("x", numpy.random.random((100, 10)))
    outputs = NumpyWrapper('y', numpy.random.randint(2, size=(100, 1)),
                           conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggo.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest',
                        outputdir=tmpdir.strpath)

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32,
            generator=janggo_fit_generator,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs, generator=janggo_predict_generator,
                       use_multiprocessing=False)
    numpy.testing.assert_equal(len(pred[:, numpy.newaxis]), len(inputs))
    numpy.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs, generator=janggo_fit_generator,
                 use_multiprocessing=False)

    dummy_eval_json = ScoreEvaluator('score', lambda y_true, y_pred: 0.15)
    dummy_eval_tsv = ScoreEvaluator('score', lambda y_true, y_pred: 0.15,
                                    dumper=dump_tsv)
    dummy_eval_plot = ScoreEvaluator('score',
                                     lambda y_true, y_pred:
                                     ([0., 0.5, 0.5, 1.],
                                      [0.5, 0.5, 1., 1.],
                                      [0.8, 0.4, 0.35, 0.1]), dumper=plot_score)

    evaluators = EvaluatorList([dummy_eval_json,
                                dummy_eval_tsv,
                                dummy_eval_plot],
                               path=tmpdir.strpath,
                               model_filter='ptest')

    evaluators.evaluate(inputs, outputs, datatags=['validation_set'])

    # check correctness of json
    with open(os.path.join(tmpdir.strpath, "evaluation",
                           "score.json"), 'r') as f:
        content = json.load(f)
        # now nptest was evaluated
        assert 'nptest,y,random' in content

    # check correctness of table
    assert pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation",
                                        "score.tsv"), sep='\t').value[0] == 0.15

    # check if plot was produced
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", "score.png"))
