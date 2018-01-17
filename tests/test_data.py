import os

import pkg_resources
import pytest

from bluewhalecore.data import DnaBwDataset
from bluewhalecore.data import RevCompDnaBwDataset
from bluewhalecore.data import TabBwDataset
from bluewhalecore.data import input_shape
from bluewhalecore.data import output_shape


def test_inshape():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBwDataset.create_from_refgenome('dna', refgenome=refgenome,
                                             storage='ndarray',
                                             regions=bed_file, order=1)
    rcdna = RevCompDnaBwDataset('rcdna', dna)

    sh = input_shape([dna, rcdna])
    assert 'dna' in sh
    assert 'rcdna' in sh

    with pytest.raises(Exception):
        input_shape((0,))


def test_outshape():
    data_path = pkg_resources.resource_filename('bluewhalecore',
                                                'resources/')
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    ctcf1 = TabBwDataset('ctcf1', filename=csvfile)
    ctcf2 = TabBwDataset('ctcf2', filename=csvfile)

    sh = output_shape([ctcf1, ctcf2], 'binary_crossentropy')

    assert 'ctcf1' in sh
    assert 'ctcf2' in sh

    with pytest.raises(Exception):
        output_shape((0,))
