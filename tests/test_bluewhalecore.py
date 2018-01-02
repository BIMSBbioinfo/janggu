import os

import numpy as np
import pkg_resources
from genomeutils.regions import readBed
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input

from bluewhalecore.bluewhale import BlueWhale
from bluewhalecore.cli import main
from bluewhalecore.data.data import inputShape
from bluewhalecore.data.data import outputShape
from bluewhalecore.data.dna import DnaBwDataset
from bluewhalecore.data.nparr import NumpyBwDataset
from bluewhalecore.data.tab import TabBwDataset
from bluewhalecore.decorators import inputlayer
from bluewhalecore.decorators import outputlayer


def test_main():
    main([])


def test_bluewhale_instance(tmpdir):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = readBed(os.path.join(data_path, 'regions.bed'))
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBwDataset.extractRegionsFromRefGenome('dna', refgenome=refgenome,
                                                   regions=regions, order=1)

    ctcf = TabBwDataset('ctcf', filename=csvfile)

    @inputlayer
    @outputlayer
    def cnn_model(input, inp, oup, params):
        layer = Flatten()(input)
        output = Dense(params[0])(layer)
        return input, output

    bwm = BlueWhale.fromShape(inputShape(dna),
                              outputShape(ctcf, 'binary_crossentropy'),
                              'dna_ctcf_HepG2.cnn',
                              (cnn_model, (2,)),
                              outputdir=tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)

    bwm.save()

    assert os.path.exists(storage)

    BlueWhale.fromName('dna_ctcf_HepG2.cnn', outputdir=tmpdir.strpath)


def test_bluewhale_train_predict_option1(tmpdir):
    X = NumpyBwDataset("X", np.random.random((1000, 100)))
    y = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    @inputlayer
    @outputlayer
    def test_model(input, inp, oup, params):
        return input, input[0]

    bwm = BlueWhale.fromShape(inputShape(X),
                              outputShape(y, 'binary_crossentropy'),
                              'nptest',
                              (test_model, None),
                              outputdir=tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(X, y, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(X)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(X))
    np.testing.assert_equal(pred.shape, y.shape)
    bwm.evaluate(X, y)


def test_bluewhale_train_predict_option2(tmpdir):
    X = NumpyBwDataset("x", np.random.random((1000, 100)))
    y = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    def test_model(path):
        input = Input((100,), name='x')
        output = Dense(1, name='y')(input)
        model = BlueWhale(inputs=input, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = test_model(tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([X], [y], epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict([X])
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(X))
    np.testing.assert_equal(pred.shape, y.shape)
    bwm.evaluate([X], [y])


def test_bluewhale_train_predict_option3(tmpdir):
    X = np.random.random((1000, 100))
    y = np.random.randint(2, size=(1000, 1))

    def test_model(path):
        input = Input((100,), name='x')
        output = Dense(1)(input)
        model = BlueWhale(inputs=input, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = test_model(tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit([X], [y], epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict([X])
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(X))
    np.testing.assert_equal(pred.shape, y.shape)
    bwm.evaluate([X], [y])

def test_bluewhale_train_predict_option4(tmpdir):
    X = np.random.random((1000, 100))
    y = np.random.randint(2, size=(1000, 1))

    def test_model(path):
        input = Input((100,), name='x')
        output = Dense(1)(input)
        model = BlueWhale(inputs=input, outputs=output, name='test',
                          outputdir=path)
        model.compile(optimizer='adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    bwm = test_model(tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    bwm.fit(X, y, epochs=2, batch_size=32)

    assert os.path.exists(storage)

    pred = bwm.predict(X)
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(X))
    np.testing.assert_equal(pred.shape, y.shape)
    bwm.evaluate(X, y)
