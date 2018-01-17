import os

import numpy as np
import pkg_resources
import pytest

from beluga.data import DnaBlgDataset
from beluga.data import RevCompDnaBlgDataset
from beluga.data import TabBlgDataset
from beluga.data import dna2ind
from beluga.data import input_shape
from beluga.data import output_shape
from beluga.data import sequences_from_fasta


def test_inshape():
    data_path = pkg_resources.resource_filename('beluga', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaBlgDataset.create_from_refgenome('dna', refgenome=refgenome,
                                              storage='ndarray',
                                              regions=bed_file, order=1)
    rcdna = RevCompDnaBlgDataset('rcdna', dna)

    props = input_shape([dna, rcdna])
    assert 'dna' in props
    assert 'rcdna' in props

    with pytest.raises(Exception):
        input_shape((0,))


def test_outshape():
    data_path = pkg_resources.resource_filename('beluga',
                                                'resources/')
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    ctcf1 = TabBlgDataset('ctcf1', filename=csvfile)
    ctcf2 = TabBlgDataset('ctcf2', filename=csvfile)

    props = output_shape([ctcf1, ctcf2], 'binary_crossentropy')

    assert 'ctcf1' in props
    assert 'ctcf2' in props

    with pytest.raises(Exception):
        output_shape((0,), loss='binary_crossentropy')


def test_dna2ind():
    data_path = pkg_resources.resource_filename('beluga', 'resources/')
    filename = os.path.join(data_path, 'oct4.fa')
    seqs = sequences_from_fasta(filename)

    np.testing.assert_equal(dna2ind(seqs[0]), dna2ind(str(seqs[0].seq)))
    np.testing.assert_equal(dna2ind(seqs[0]), dna2ind(seqs[0].seq))

    with pytest.raises(TypeError):
        # wrong type: int
        dna2ind(0)
