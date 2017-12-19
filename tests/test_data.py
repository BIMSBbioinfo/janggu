import os

import pkg_resources
import pytest
from genomeutils.regions import readBed

from bluewhalecore.data.data import inputShape
from bluewhalecore.data.data import outputShape
from bluewhalecore.data.dna import DnaBwDataset
from bluewhalecore.data.dna import RevCompDnaBwDataset
from bluewhalecore.data.tab import TabBwDataset


def test_inshape():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = readBed(os.path.join(data_path, 'regions.bed'))

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBwDataset.extractRegionsFromRefGenome('dna', refgenome=refgenome,
                                                   regions=regions, order=1)
    rcdna = RevCompDnaBwDataset('rcdna', dna)

    sh = inputShape([dna, rcdna])
    assert 'dna' in sh
    assert 'rcdna' in sh

    with pytest.raises(Exception):
        inputShape((0,))


def test_outshape():
    data_path = pkg_resources.resource_filename('bluewhalecore',
                                                'resources/')
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    ctcf1 = TabBwDataset('ctcf1', filename=csvfile)
    ctcf2 = TabBwDataset('ctcf2', filename=csvfile)

    sh = outputShape([ctcf1, ctcf2], 'binary_crossentropy')

    assert 'ctcf1' in sh
    assert 'ctcf2' in sh

    with pytest.raises(Exception):
        outputShape((0,))
