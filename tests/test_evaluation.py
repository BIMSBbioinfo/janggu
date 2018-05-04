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

from janggo import Janggo
from janggo import inputlayer
from janggo import outputconv
from janggo import outputdense
from janggo.data import Array
from janggo.data import GenomicIndexer
from janggo.evaluation import InOutScorer
from janggo.evaluation import InScorer
from janggo.evaluation import _dimension_match
from janggo.utils import export_bed
from janggo.utils import export_bigwig
from janggo.utils import export_tsv
from janggo.utils import plot_score


def test_input_dims():
    data = Array('testa', numpy.zeros((10, 10, 1)))
    xin = Input((10, 1), name='testy')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _dimension_match(m, data, 'input_layers')

    xin = Input((20, 10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _dimension_match(m, data, 'input_layers')
    # more input datasets supplied than inputs to models
    assert not _dimension_match(m, [data, data], 'input_layers')

    xin = Input((10, 1), name='testa')
    out = Dense(1)(xin)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _dimension_match(m, data, 'input_layers')


def test_output_dims():
    data = Array('testa', numpy.zeros((10, 10, 1)))
    label = Array('testy', numpy.zeros((10, 1)))
    xin = Input(data.shape, name='asdf')
    out = Flatten()(xin)
    out = Dense(1)(out)
    m = Model(xin, out)

    # False due to mismatch of names
    assert not _dimension_match(m, label, 'output_layers')

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(2, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert not _dimension_match(m, label, 'output_layers')

    xin = Input(data.shape, name='testa')
    out = Flatten()(xin)
    out = Dense(1, name='testy')(out)
    m = Model(xin, out)

    # False due to mismatch of dims
    assert _dimension_match(m, label, 'output_layers')

    assert _dimension_match(m, None, 'output_layers')


def get_janggo(inputs, outputs, tmpdir):
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
    return bwm


def get_janggo_conv(inputs, outputs, tmpdir):
    @inputlayer
    @outputconv('sigmoid')
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
    return bwm


def test_output_json_score(tmpdir):

    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggo(inputs, outputs, tmpdir)

    dummy_eval = InOutScorer('score', lambda y_true, y_pred: 0.15)

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    # check correctness of json
    with open(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                           "score.json"), 'r') as f:
        content = json.load(f)
        # now nptest was evaluated
        assert 'nptest,y,random' in content


def test_output_tsv_score(tmpdir):
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggo(inputs, outputs, tmpdir)

    dummy_eval = InOutScorer('score', lambda y_true, y_pred: 0.15, exporter=export_tsv)

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    assert pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                                        "score.tsv"),
                           sep='\t').value[0] == 0.15


def test_output_plot_score(tmpdir):
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggo(inputs, outputs, tmpdir)

    dummy_eval = InOutScorer('score',
                             lambda y_true, y_pred:
                             ([0., 0.5, 0.5, 1.],
                              [0.5, 0.5, 1., 1.],
                              [0.8, 0.4, 0.35, 0.1]), exporter=plot_score)

    dummy_eval_par = InOutScorer('score',
                                 lambda y_true, y_pred:
                                 ([0., 0.5, 0.5, 1.],
                                  [0.5, 0.5, 1., 1.],
                                  [0.8, 0.4, 0.35, 0.1]), exporter=plot_score,
                                 export_args={'figsize': (10, 12),
                                              'xlabel': 'FPR',
                                              'ylabel': 'TPR',
                                              'fform': 'eps'})

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval, dummy_eval_par])

    # check if plot was produced
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, "score.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, "score.eps"))


def test_output_bed_loss_resolution_equal_stepsize(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 1, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 1, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo_conv(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=200)

    dummy_eval = InOutScorer('loss', lambda t, p: [0.1] * len(t),
                             exporter=export_bed, export_args={'gindexer': gi})

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.nptest.y.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = iter(HTSeq.BED_Reader(file_.format('c1')))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(reg.score, 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_loss_resolution_unequal_stepsize(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    # dummy_eval = InOutScorer('loss', lambda t, p: -t * numpy.log(p),
    #                    exporter=export_bed, export_args={'gindexer': gi})
    dummy_eval = InOutScorer('loss', lambda t, p: [0.1] * len(t),
                             exporter=export_bed, export_args={'gindexer': gi})

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.nptest.y.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = iter(HTSeq.BED_Reader(file_.format('c1')))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(reg.score, 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 28, 'There should be 28 regions in the bed file.'


def test_output_bed_predict_resolution_equal_stepsize(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 1, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 1, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo_conv(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=200)

    dummy_eval = InScorer('pred', lambda p: [0.1] * len(p),
                          exporter=export_bed, export_args={'gindexer': gi},
                          conditions=['c1', 'c2', 'c3', 'c4'])

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'prediction', bwm.name,
                         'pred.nptest.y.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = iter(HTSeq.BED_Reader(file_.format('c1')))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(reg.score, 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_predict_denseout(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 10)))
    outputs = Array('y', numpy.random.random((7, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=200)

    dummy_eval = InScorer('pred', lambda p: [0.1] * len(p),
                          exporter=export_bed, export_args={'gindexer': gi},
                          conditions=['c1', 'c2', 'c3', 'c4'])

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'prediction', bwm.name,
                         'pred.nptest.y.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = iter(HTSeq.BED_Reader(file_.format('c1')))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(reg.score, 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_predict_resolution_unequal_stepsize(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    dummy_eval = InScorer('pred', lambda p: [0.1] * len(p),
                          exporter=export_bed, export_args={'gindexer': gi},
                          conditions=['c1', 'c2', 'c3', 'c4'])

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'prediction', bwm.name,
                         'pred.nptest.y.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = iter(HTSeq.BED_Reader(file_.format('c1')))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(reg.score, 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 28, 'There should be 28 regions in the bed file.'


def test_output_bigwig_predict_denseout(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 10)))
    outputs = Array('y', numpy.random.random((7, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=200)

    dummy_eval = InScorer('pred', lambda p: [0.1] * len(p),
                          exporter=export_bigwig, export_args={'gindexer': gi},
                          conditions=['c1', 'c2', 'c3', 'c4'])

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'prediction', bwm.name,
                         'pred.nptest.y.{}.bigwig')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('c1'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.1, rtol=1e-5)


def test_output_bigwig_predict_convout(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo_conv(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    dummy_eval = InScorer('pred', lambda p: [0.2] * len(p),
                          exporter=export_bigwig, export_args={'gindexer': gi},
                          conditions=['c1', 'c2', 'c3', 'c4'])

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'prediction', bwm.name,
                         'pred.nptest.y.{}.bigwig')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('c1'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.2, rtol=1e-5)


def test_output_bigwig_loss_resolution_unequal_stepsize(tmpdir):
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggo(inputs, outputs, tmpdir)
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200,
                                         resolution=50)

    # dummy_eval = InOutScorer('loss', lambda t, p: -t * numpy.log(p),
    #                    exporter=export_bed, export_args={'gindexer': gi})
    dummy_eval = InOutScorer('loss', lambda t, p: [0.2] * len(t),
                             exporter=export_bigwig, export_args={'gindexer': gi})

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.nptest.y.{}.bigwig')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('c1'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.2, rtol=1e-5)
