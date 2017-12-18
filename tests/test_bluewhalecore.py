import os
import pkg_resources
from bluewhalecore.cli import main

from keras.layers import Flatten
from keras.layers import Dense
from bluewhalecore.bluewhale import BlueWhale
from bluewhalecore.data.data import inputShape
from bluewhalecore.data.data import outputShape
from bluewhalecore.data.dna import DnaBwDataset
from bluewhalecore.data.tab import TabBwDataset
from genomeutils.regions import readBed
from bluewhalecore.decorators import toplayer
from bluewhalecore.decorators import bottomlayer


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

    print(bwm)
