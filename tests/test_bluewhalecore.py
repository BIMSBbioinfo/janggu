import os

import numpy as np
import pkg_resources
import pytest
from genomeutils.regions import readBed
from keras.layers import Dense
from keras.layers import Flatten

from bluewhalecore.bluewhale import BlueWhale
from bluewhalecore.cli import main
from bluewhalecore.data.data import inputShape
from bluewhalecore.data.data import outputShape
from bluewhalecore.data.dna import DnaBwDataset
from bluewhalecore.data.nparr import NumpyBwDataset
from bluewhalecore.data.tab import TabBwDataset
from bluewhalecore.decorators import bottomlayer
from bluewhalecore.decorators import toplayer


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

    @bottomlayer
    @toplayer
    def cnn_model(input, inp, oup, params):
        layer = Flatten()(input)
        output = Dense(params[0])(layer)
        return input, output

    bwm = BlueWhale.fromShape('dna_train_ctcf_HepG2.cnn',
                              inputShape(dna),
                              outputShape(ctcf, 'binary_crossentropy'),
                              (cnn_model, (2,)),
                              outputdir=tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)

    bwm.saveKerasModel()

    assert os.path.exists(storage)

    BlueWhale.fromName('dna_train_ctcf_HepG2.cnn', outputdir=tmpdir.strpath)


def test_bluewhale_train_predict(tmpdir):
    X = NumpyBwDataset("X", np.random.random((1000, 100)))
    y = NumpyBwDataset('y', np.random.randint(2, size=(1000, 1)))

    @bottomlayer
    @toplayer
    def test_model(input, inp, oup, params):
        return input, input[0]

    bwm = BlueWhale.fromShape('nptest',
                              inputShape(X),
                              outputShape(y, 'binary_crossentropy'),
                              (test_model, None),
                              outputdir=tmpdir.strpath)

    storage = bwm.storagePath(bwm.name, outputdir=tmpdir.strpath)
    assert not os.path.exists(storage)

    with pytest.warns(UserWarning):
        bwm.fit(X, y, epochs=1)

    assert os.path.exists(storage)

    pred = bwm.predict(X)
    print(pred.shape)
    print(pred)
    print(pred[0])
    np.testing.assert_equal(len(pred[:, np.newaxis]), len(X))
    np.testing.assert_equal(pred.shape[1:], y.shape)
