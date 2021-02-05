import json
import os

import numpy
import pandas
import pkg_resources
import pyBigWig
import pytest
from keras import Input
from keras import Model
from keras.layers import Dense
from keras.layers import Flatten
from pybedtools import BedTool

from janggu import Janggu
from janggu import inputlayer
from janggu import outputconv
from janggu import outputdense
from janggu.data import Array
from janggu.data import GenomicIndexer
from janggu.evaluation import Scorer
#from janggu.evaluation import _dimension_match
from janggu.utils import ExportBed
from janggu.utils import ExportBigwig
from janggu.utils import ExportClustermap
from janggu.utils import ExportScorePlot
from janggu.utils import ExportTsne
from janggu.utils import ExportTsv


def get_janggu(inputs, outputs):
    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]
    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')
    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=bwm.outputdir)
    assert not os.path.exists(storage)
    return bwm


def get_janggu_conv(inputs, outputs):
    @inputlayer
    @outputconv('sigmoid')
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=bwm.outputdir)
    assert not os.path.exists(storage)
    return bwm


def test_output_score_by_name(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggu(inputs, outputs)

    dummy_eval = Scorer('score', lambda y_true, y_pred: 0.15, immediate_export=False)

    bwm.evaluate(inputs, outputs, callbacks=['auc', 'roc', 'prc',
                                             'auprc', 'auroc',
                                             'cor', 'mae', 'mse',
                                             'var_explained', dummy_eval])

    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "auc.tsv"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "prc.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "roc.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "cor.tsv"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "mae.tsv"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "mse.tsv"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "var_explained.tsv"))
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "auprc.tsv"))
    assert not os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "score.json"))

    dummy_eval.export(os.path.join(tmpdir.strpath, dummy_eval.subdir), bwm.name)
    assert os.path.exists(os.path.join(tmpdir.strpath, "evaluation", bwm.name, "score.json"))

    with pytest.raises(ValueError):
        bwm.evaluate(inputs, outputs, callbacks=['adsf'])


def test_output_json_score(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggu(inputs, outputs)

    # check exception if no scoring function is provided
    dummy_eval = Scorer('score')

    with pytest.raises(ValueError):
        bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    dummy_eval = Scorer('score', lambda y_true, y_pred: 0.15)

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    # check correctness of json
    with open(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                           "score.json"), 'r') as f:
        content = json.load(f)
        # now nptest was evaluated
        print(content)
        assert 'random' in content


def test_output_tsv_score(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggu(inputs, outputs)

    dummy_eval = Scorer('score', lambda y_true, y_pred: 0.15, exporter=ExportTsv())

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    print(pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                                        "score.tsv"),
                           sep='\t', header=[0]))
    assert pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                                        "score.tsv"),
                           sep='\t', header=[0]).iloc[0, 0] == 0.15


def test_output_export_score_plot(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    bwm = get_janggu(inputs, outputs)

    dummy_eval = Scorer('score',
                        lambda y_true, y_pred:
                        ([0., 0.5, 0.5, 1.],
                         [0.5, 0.5, 1., 1.],
                         [0.8, 0.4, 0.35, 0.1]),
                        exporter=ExportScorePlot())

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    dummy_eval = Scorer('score',
                        lambda y_true, y_pred:
                        ([0., 0.5, 0.5, 1.],
                         [0.5, 0.5, 1., 1.],
                         [0.8, 0.4, 0.35, 0.1]),
                        exporter=ExportScorePlot(figsize=(10,12),
                                                 xlabel='FPR',
                                                 ylabel='TPR',
                                                 fform='eps'))

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    # check if plot was produced
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, "score.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, "score.eps"))


def test_output_export_clustermap(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        with inputs.use('x') as layer:
            outputs = Dense(3, name='hidden')(layer)
        return inputs, outputs

    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    dummy_eval = Scorer('cluster', exporter=ExportClustermap())

    bwm.predict(inputs, layername='hidden',
                callbacks=[dummy_eval])

    dummy_eval = Scorer('cluster', exporter=ExportClustermap(fform='eps',
                                                             annot={'annot':[1]*50 + [0]*50}))
    bwm.predict(inputs, layername='hidden',
                callbacks=[dummy_eval])

    # check if plot was produced
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, 'hidden',
                                       "cluster.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, 'hidden',
                                       "cluster.eps"))


@pytest.mark.filterwarnings("ignore:the matrix")
def test_output_export_tsne(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def _model(inputs, inp, oup, params):
        with inputs.use('x') as layer:
            outputs = Dense(3, name='hidden')(layer)
        return inputs, outputs

    bwm = Janggu.create(_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest')

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    dummy_eval = Scorer('tsne', exporter=ExportTsne())

    bwm.predict(inputs, layername='hidden',
                callbacks=[dummy_eval])

    dummy_eval = Scorer('tsne', exporter=ExportTsne(fform='eps',
                                                    annot={'annot':[1]*50 + [0]*50},
                                                    figsize=(10, 10)))
    bwm.predict(inputs, layername='hidden',
                callbacks=[dummy_eval])
    # check if plot was produced
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, 'hidden',
                                       "tsne.png"))
    assert os.path.exists(os.path.join(tmpdir.strpath,
                                       "evaluation", bwm.name, 'hidden',
                                       "tsne.eps"))


def test_output_bed_loss_resolution_equal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 1, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 1, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu_conv(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('loss', lambda t, p: [0.1] * len(t),
                        exporter=ExportBed(gindexer=gi, resolution=200))

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.{}.bed')
    print(os.listdir(os.path.join(tmpdir.strpath, 'evaluation', bwm.name)))
    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = BedTool(file_.format('c1'))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(float(reg.score), 0.1)
        nreg += 1
#        numpy.testing.assert_equal(breg.score, value)

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_loss_resolution_unequal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    # dummy_eval = Scorer('loss', lambda t, p: -t * numpy.log(p),
    #                    exporter=export_bed, export_args={'gindexer': gi})
    dummy_eval = Scorer('loss', lambda t, p: [0.1] * len(t),
                        exporter=ExportBed(gindexer=gi, resolution=50))

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.{}.bed')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bed = BedTool(file_.format('c1'))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(float(reg.score), 0.1)
        nreg += 1

    assert nreg == 28, 'There should be 28 regions in the bed file.'


def test_output_bed_predict_resolution_equal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 1, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 1, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu_conv(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('pred', lambda p: [0.1] * len(p),
                        exporter=ExportBed(gindexer=gi, resolution=200))

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'pred.{}.bed')

    print(os.listdir(os.path.join(tmpdir.strpath, 'evaluation', bwm.name)))
    for cond in ['0', '1', '2', '3']:
        assert os.path.exists(file_.format(cond))

    bed = BedTool(file_.format('0'))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(float(reg.score), 0.1)
        nreg += 1

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_predict_denseout(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 10)))
    outputs = Array('y', numpy.random.random((7, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('pred', lambda p: [0.1] * len(p),
                        exporter=ExportBed(gindexer=gi, resolution=200))

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'pred.{}.bed')

    for cond in ['0', '1', '2', '3']:
        assert os.path.exists(file_.format(cond))

    bed = BedTool(file_.format('0'))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(float(reg.score), 0.1)
        nreg += 1

    assert nreg == 7, 'There should be 7 regions in the bed file.'


def test_output_bed_predict_resolution_unequal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('pred', lambda p: [0.1] * len(p),
                        exporter=ExportBed(gindexer=gi, resolution=50))

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'pred.{}.bed')

    for cond in ['0', '1', '2', '3']:
        assert os.path.exists(file_.format(cond))

    bed = BedTool(file_.format('0'))

    nreg = 0
    for reg in bed:
        numpy.testing.assert_equal(float(reg.score), 0.1)
        nreg += 1

    assert nreg == 28, 'There should be 28 regions in the bed file.'


def test_output_bigwig_predict_denseout(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 10)))
    outputs = Array('y', numpy.random.random((7, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('pred', lambda p: [0.1] * len(p),
                        exporter=ExportBigwig(gindexer=gi))

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'pred.{}.bigwig')

    for cond in ['0', '1', '2', '3']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('0'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.1, rtol=1e-5)


def test_output_bigwig_predict_convout(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu_conv(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('pred', lambda p: [0.2] * len(p),
                        exporter=ExportBigwig(gindexer=gi))

    bwm.predict(inputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'pred.{}.bigwig')

    for cond in ['0', '1', '2', '3']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('0'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.2, rtol=1e-5)


def test_output_bigwig_loss_resolution_equal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=200)

    dummy_eval = Scorer('loss', lambda t, p: [0.2] * len(t),
                        exporter=ExportBigwig(gindexer=gi))

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.{}.bigwig')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('c1'))

    co = bw.values('chr1', 600, 2000)

    numpy.testing.assert_allclose(numpy.mean(co), 0.2, rtol=1e-5)


def test_output_bigwig_loss_resolution_unequal_stepsize(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    # generate loss
    #
    # resolution < stepsize
    inputs = Array("x", numpy.random.random((7, 4, 1, 10)))
    outputs = Array('y', numpy.random.random((7, 4, 1, 4)),
                    conditions=['c1', 'c2', 'c3', 'c4'])

    bwm = get_janggu(inputs, outputs)
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/10regions.bed')

    gi = GenomicIndexer.create_from_file(data_path,
                                         binsize=200,
                                         stepsize=50)

    dummy_eval = Scorer('loss', lambda t, p: [0.2] * len(t),
                        exporter=ExportBigwig(gindexer=gi))

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval])

    file_ = os.path.join(tmpdir.strpath, 'evaluation', bwm.name,
                         'loss.{}.bigwig')

    for cond in ['c1', 'c2', 'c3', 'c4']:
        assert os.path.exists(file_.format(cond))

    bw = pyBigWig.open(file_.format('c1'))

    co = bw.values('chr1', 600, 2000-150)

    numpy.testing.assert_allclose(numpy.mean(co), 0.2, rtol=1e-5)


def test_output_tsv_score_across_conditions(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    inputs = Array("x", numpy.random.random((100, 10)))
    outputs = Array('y', numpy.random.randint(2, size=(100, 2)),
                    conditions=['c1', 'c2'])

    bwm = get_janggu(inputs, outputs)

    dummy_eval = Scorer('score', lambda y_true, y_pred: 0.15,
                        exporter=ExportTsv())
    dummy_evalacross = Scorer('scoreacross',
                              lambda y_true, y_pred: 0.15,
                              exporter=ExportTsv(),
                              percondition=False)

    bwm.evaluate(inputs, outputs, callbacks=[dummy_eval, dummy_evalacross])

    # percondition=True
    assert pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                                        "score.tsv"),
                           sep='\t', header=[0]).shape == (1, 2)
    # percondition=False
    val = pandas.read_csv(os.path.join(tmpdir.strpath, "evaluation", bwm.name,
                                       "scoreacross.tsv"),
                          sep='\t', header=[0])
    assert val['across'][0] == .15
    assert val.shape == (1, 1)
