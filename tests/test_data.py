import os

import pkg_resources
import pytest

from beluga.data import DnaBlgDataset
from beluga.data import RevCompDnaBlgDataset
from beluga.data import TabBlgDataset
from beluga.data import input_shape
from beluga.data import output_shape


def test_inshape():
    data_path = pkg_resources.resource_filename('beluga', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBlgDataset.create_from_refgenome('dna', refgenome=refgenome,
                                              storage='ndarray',
                                              regions=bed_file, order=1)
    rcdna = RevCompDnaBlgDataset('rcdna', dna)

    sh = input_shape([dna, rcdna])
    assert 'dna' in sh
    assert 'rcdna' in sh

    with pytest.raises(Exception):
        input_shape((0,))


def test_outshape():
    data_path = pkg_resources.resource_filename('beluga',
                                                'resources/')
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    ctcf1 = TabBlgDataset('ctcf1', filename=csvfile)
    ctcf2 = TabBlgDataset('ctcf2', filename=csvfile)

    sh = output_shape([ctcf1, ctcf2], 'binary_crossentropy')

    assert 'ctcf1' in sh
    assert 'ctcf2' in sh

    with pytest.raises(Exception):
        output_shape((0,))
