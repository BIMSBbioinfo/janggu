import os

import numpy as np
import pkg_resources
import pytest
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input

from bluewhalecore import BlueWhale
from bluewhalecore import MongoDbEvaluator
from bluewhalecore import bluewhale_fit_generator
from bluewhalecore import bluewhale_predict_generator
from bluewhalecore import inputlayer
from bluewhalecore import outputlayer
from bluewhalecore.cli import main
from bluewhalecore.data import DnaBwDataset
from bluewhalecore.data import NumpyBwDataset
from bluewhalecore.data import TabBwDataset
from bluewhalecore.data import input_shape
from bluewhalecore.data import output_shape
from bluewhalecore.data import read_bed
from bluewhalecore.evaluate import bw_accuracy
from bluewhalecore.evaluate import bw_auprc
from bluewhalecore.evaluate import bw_auroc
from bluewhalecore.evaluate import bw_av_auprc
from bluewhalecore.evaluate import bw_av_auroc
from bluewhalecore.evaluate import bw_f1


def test_main():
    """Basic main test"""
    main([])


def test_bluewhale_instance(tmpdir):
    """Test BlueWhale creation by shape and name."""
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = read_bed(os.path.join(data_path, 'regions.bed'))
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBwDataset.create_from_refgenome('dna', refgenome=refgenome,
                                             storage='ndarray',
                                             regions=regions, order=1)

    ctcf = TabBwDataset('ctcf', filename=csvfile)

    @inputlayer
    @outputlayer
    def _cnn_model(inputs, inp, oup, params):
        layer = Flatten()(inputs)
        output = Dense(params[0])(layer)
        return inputs, output

    bwm = BlueWhale.create_by_shape(input_shape(dna),
                                    output_shape(ctcf, 'binary_crossentropy'),
                                    'dna_ctcf_HepG2.cnn',
                                    (_cnn_model, (2,)),
                                    outputdir=tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()

    assert os.path.exists(storage)

    BlueWhale.create_by_name('dna_ctcf_HepG2.cnn', outputdir=tmpdir.strpath)


def test_bluewhale_train_predict_option1(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: by_shape
    Input args: BwDataset
    generators: NO
    """

    inputs = NumpyBwDataset("X", np.random.random((1000, 100)))
    outputs = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    @inputlayer
    @outputlayer
    def test_model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = BlueWhale.create_by_shape(input_shape(inputs),
                                    output_shape(outputs,
                                                 'binary_crossentropy'),
                                    'nptest',
                                    (test_model, None),
                                    outputdir=tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)


def test_bluewhale_train_predict_option2(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(BwDataset)
    generators: NO
    """

    inputs = NumpyBwDataset("x", np.random.random((1000, 100)))
    outputs = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    def _model(path):
        inputs = Input((100,), name='x')
        output = Dense(1, activation='sigmoid', name='y')(inputs)
        model = BlueWhale(inputs=inputs, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict([inputs])
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs])


def test_bluewhale_train_predict_option3(tmpdir, mongodb):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: list(np.array)
    generators: NO
    """

    inputs = np.random.random((1000, 100))
    outputs = np.random.randint(2, size=(1000, 1))

    def _model(path):
        inputs = Input((100,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = BlueWhale(inputs=inputs, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)
    with pytest.raises(TypeError):
        bwm.fit([inputs], [outputs], epochs=2, batch_size=32,
                generator=bluewhale_fit_generator)
    assert os.path.exists(storage)

    pred = bwm.predict([inputs])
    with pytest.raises(TypeError):
        bwm.predict([inputs], batch_size=32,
                    generator=bluewhale_predict_generator)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs])
    with pytest.raises(TypeError):
        bwm.evaluate([inputs], [outputs], batch_size=32,
                     generator=bluewhale_fit_generator)


def test_bluewhale_train_predict_option4(tmpdir, mongodb):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: np.array
    generators: YES
    """

    inputs = np.random.random((1000, 100))
    outputs = np.random.randint(2, size=(1000, 1))

    def _model(path):
        inputs = Input((100,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = BlueWhale(inputs=inputs, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)
    with pytest.raises(TypeError):
        bwm.fit(inputs, outputs, epochs=2, batch_size=32,
                generator=bluewhale_fit_generator)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)
    with pytest.raises(TypeError):
        bwm.predict(inputs, batch_size=32,
                    generator=bluewhale_predict_generator)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)
    with pytest.raises(TypeError):
        bwm.evaluate(inputs, outputs, batch_size=32,
                     generator=bluewhale_fit_generator)


def test_bluewhale_train_predict_option5(tmpdir, mongodb):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(BwDataset)
    generators: YES
    MongoDb: YES
    """

    inputs = NumpyBwDataset("x", np.random.random((1000, 100)))
    outputs = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    def _model(path):
        inputs = Input((100,), name='x')
        output = Dense(1, name='y', activation='sigmoid')(inputs)
        model = BlueWhale(inputs=inputs, outputs=output, name='test_model',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32,
            generator=bluewhale_fit_generator,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict([inputs], generator=bluewhale_predict_generator,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs], generator=bluewhale_fit_generator,
                 use_multiprocessing=False)

    evaluator = MongoDbEvaluator()
    evaluator.database = mongodb
    evaluator.dump(
        bwm, [inputs], outputs,
        elementwise_score={'auROC': bw_auroc, 'auPRC': bw_auprc,
                           'Accuracy': bw_accuracy, "F1": bw_f1},
        combined_score={'av-auROC': bw_av_auroc,
                        'av-auPRC': bw_av_auprc},
        datatags=['testing'],
        modeltags=['modeltesting'],
        batch_size=100,
        use_multiprocessing=False)


def test_bluewhale_train_predict_option6(tmpdir, mongodb):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: BwDataset
    generators: YES
    MongoDb: YES
    """

    inputs = NumpyBwDataset("x", np.random.random((1000, 100)))
    outputs = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    @inputlayer
    @outputlayer
    def _model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = BlueWhale.create_by_shape(input_shape(inputs),
                                    output_shape(outputs,
                                                 'binary_crossentropy'),
                                    'nptest',
                                    (_model, None),
                                    outputdir=tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32,
            generator=bluewhale_fit_generator,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs, generator=bluewhale_predict_generator,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs, generator=bluewhale_fit_generator,
                 use_multiprocessing=False)

    evaluator = MongoDbEvaluator()
    evaluator.database = mongodb
    evaluator.dump(
        bwm, inputs, outputs,
        elementwise_score={'auROC': bw_auroc, 'auPRC': bw_auprc,
                           'Accuracy': bw_accuracy, "F1": bw_f1},
        combined_score={'av-auROC': bw_av_auroc,
                        'av-auPRC': bw_av_auprc},
        datatags=['testing'],
        modeltags=['modeltesting'],
        batch_size=100,
        use_multiprocessing=False)
