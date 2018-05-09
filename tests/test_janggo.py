import os

import matplotlib
import numpy as np
import pkg_resources
import pytest
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input

from janggo import Janggo
from janggo import inputlayer
from janggo import outputconv
from janggo import outputdense
from janggo.cli import main
from janggo.data import Array
from janggo.data import Cover
from janggo.data import Dna
from janggo.data import Table
from janggo.layers import Complement
from janggo.layers import LocalAveragePooling2D
from janggo.layers import Reverse

matplotlib.use('AGG')


def test_main():
    """Basic main test"""
    main([])


def test_localaveragepooling2D(tmpdir):
    # some test data
    testin = np.ones((1, 10, 1, 3))
    testin[:, :, :, 1] += 1
    testin[:, :, :, 2] += 2

    # test local average pooling
    lin = Input((10, 1, 3))
    out = LocalAveragePooling2D(3)(lin)
    m = Janggo(lin, out, outputdir=tmpdir.strpath)

    testout = m.predict(testin)
    np.testing.assert_equal(testout, testin[:, :8, :, :])

    # more tests
    testin = np.ones((1, 3, 1, 2))
    testin[:, 0, :, :] = 0
    testin[:, 2, :, :] = 2
    testin[:, :, :, 1] += 1

    # test local average pooling
    lin = Input((3, 1, 2))
    out = LocalAveragePooling2D(3)(lin)
    m = Janggo(lin, out, outputdir=tmpdir.strpath)

    testout = m.predict(testin)
    np.testing.assert_equal(testout.shape, (1, 1, 1, 2))
    np.testing.assert_equal(testout[0, 0, 0, 0], 1)
    np.testing.assert_equal(testout[0, 0, 0, 1], 2)


def test_janggo_generate_name(tmpdir):

    def _cnn_model(inputs, inp, oup, params):
        inputs = Input((10, 1))
        layer = Flatten()(inputs)
        output = Dense(params[0])(layer)
        return inputs, output

    bwm = Janggo.create(_cnn_model, modelparams=(2,), outputdir=tmpdir.strpath)
    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggo.create_by_name(bwm.name, outputdir=tmpdir.strpath)


def test_janggo_instance_dense(tmpdir):
    """Test Janggo creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    csvfile = os.path.join(data_path, 'sample.csv')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Dna.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    regions=bed_file, order=1)

    ctcf = Table('ctcf', filename=csvfile)

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs['.']
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # due to No input name . defined
        bwm = Janggo.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name='dna_ctcf_HepG2-cnn',
                            outputdir=tmpdir.strpath)

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs[list()]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # due to Wrong type for indexing
        bwm = Janggo.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name='dna_ctcf_HepG2-cnn',
                            outputdir=tmpdir.strpath)

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs()[0]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output

    with pytest.raises(Exception):
        # name with dot not allowed. could be mistaken for a file-ending
        bwm = Janggo.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name='dna_ctcf_HepG2.cnn',
                            outputdir=tmpdir.strpath)
    with pytest.raises(Exception):
        # name with must be string
        bwm = Janggo.create(_cnn_model, modelparams=(2,),
                            inputs=dna,
                            outputs=ctcf,
                            name=12342134,
                            outputdir=tmpdir.strpath)

    # test with given model name
    bwm = Janggo.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn',
                        outputdir=tmpdir.strpath)
    # test with auto. generated modelname.
    bwm = Janggo.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn',
                        outputdir=tmpdir.strpath)

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs[0]
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output
    bwm = Janggo.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn',
                        outputdir=tmpdir.strpath)

    @inputlayer
    @outputdense('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        layer = inputs['dna']
        layer = Complement()(layer)
        layer = Reverse()(layer)
        layer = Flatten()(layer)
        output = Dense(params[0])(layer)
        return inputs, output
    bwm = Janggo.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn',
                        outputdir=tmpdir.strpath)
    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggo.create_by_name('dna_ctcf_HepG2-cnn', outputdir=tmpdir.strpath)


def test_janggo_instance_conv(tmpdir):
    """Test Janggo creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    posfile = os.path.join(data_path, 'positive.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Dna.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    regions=bed_file, order=1)

    ctcf = Cover.create_from_bed(
        "positives",
        bedfiles=posfile,
        regions=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        dimmode='all',
        storage='ndarray')

    @inputlayer
    @outputconv('sigmoid')
    def _cnn_model(inputs, inp, oup, params):
        with inputs.use('dna') as inlayer:
            layer = inlayer
        layer = Complement()(layer)
        layer = Reverse()(layer)
        return inputs, layer

    bwm = Janggo.create(_cnn_model, modelparams=(2,),
                        inputs=dna,
                        outputs=ctcf,
                        name='dna_ctcf_HepG2-cnn',
                        outputdir=tmpdir.strpath)

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')
    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()
    bwm.summary()

    assert os.path.exists(storage)

    Janggo.create_by_name('dna_ctcf_HepG2-cnn', outputdir=tmpdir.strpath)


def test_janggo_train_predict_option1(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: by_shape
    Input args: Dataset
    """

    inputs = Array("X", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    @inputlayer
    @outputdense('sigmoid')
    def test_model(inputs, inp, oup, params):
        return inputs, inputs[0]

    bwm = Janggo.create(test_model,
                        inputs=inputs,
                        outputs=outputs,
                        name='nptest',
                        outputdir=tmpdir.strpath)

    bwm.compile(optimizer='adadelta', loss='binary_crossentropy')

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)


def test_janggo_train_predict_option2(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(Dataset)
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    def _model(path):
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid', name='y')(inputs)
        model = Janggo(inputs=inputs, outputs=output, name='test',
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


def test_janggo_train_predict_option3(tmpdir):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: list(np.array)
    """

    inputs = np.random.random((100, 10))
    outputs = np.random.randint(2, size=(100, 1))

    def _model(path):
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = Janggo(inputs=inputs, outputs=output, name='test',
                       outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32)
    assert os.path.exists(storage)

    pred = bwm.predict([inputs])

    bwm.predict([inputs], batch_size=32)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs])

    bwm.evaluate([inputs], [outputs], batch_size=32)


def test_janggo_train_predict_option4(tmpdir):
    """Train, predict and evaluate on dummy data.

    Only works without generators and without evaluators.

    create: NO
    Input args: np.array
    """

    inputs = np.random.random((100, 10))
    outputs = np.random.randint(2, size=(100, 1))

    def _model(path):
        inputs = Input((10,), name='x')
        output = Dense(1, activation='sigmoid')(inputs)
        model = Janggo(inputs=inputs, outputs=output, name='test',
                       outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    # This used to not work with normal numpy arrays,
    # but now the numpy arrays are matched automatically
    # with the layer names.
    bwm.fit(inputs, outputs, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs)

    bwm.predict(inputs, batch_size=32)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs)

    bwm.evaluate(inputs, outputs, batch_size=32)


def test_janggo_train_predict_option5(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: NO
    Input args: list(Dataset)
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
                    conditions=['random'])

    def _model(path):
        inputs = Input((10,), name='x')
        output = Dense(1, name='y', activation='sigmoid')(inputs)
        model = Janggo(inputs=inputs, outputs=output, name='test_model',
                       outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = _model(tmpdir.strpath)

    storage = bwm._storage_path(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([inputs], [outputs], epochs=2, batch_size=32,
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict([inputs],
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate([inputs], [outputs],
                 use_multiprocessing=False)


def test_janggo_train_predict_option6(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
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
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs,
                 use_multiprocessing=False)


def test_janggo_train_predict_option7(tmpdir):
    """Train, predict and evaluate on dummy data.

    create: YES
    Input args: Dataset
    validation_set: YES
    batch_size: None
    """

    inputs = Array("x", np.random.random((100, 10)))
    outputs = Array('y', np.random.randint(2, size=(100, 1)),
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

    bwm.fit(inputs, outputs, epochs=2,
            validation_data=(inputs, outputs),
            use_multiprocessing=False)

    assert os.path.exists(storage)

    pred = bwm.predict(inputs,
                       use_multiprocessing=False)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(inputs))
    np.testing.assert_equal(pred.shape, outputs.shape)
    bwm.evaluate(inputs, outputs,
                 use_multiprocessing=False)
