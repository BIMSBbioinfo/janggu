import os
import matplotlib
matplotlib.use('AGG')

import numpy as np
import pkg_resources
import pytest

from janggo.data import DnaDataset
from janggo.data import TabDataset
from janggo.data.data import data_props
from janggo.utils import dna2ind
from janggo.utils import sequences_from_fasta


def test_dna_props_extraction():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    dna = DnaDataset.create_from_refgenome('dna', refgenome=refgenome,
                                           storage='ndarray',
                                           regions=bed_file, order=1)

    props = data_props(dna)
    assert 'dna' in props
    assert props['dna']['shape'] == (500, 4, 1)

    with pytest.raises(Exception):
        data_props((0,))


def test_tab_props_extraction():
    data_path = pkg_resources.resource_filename('janggo',
                                                'resources/')
    csvfile = os.path.join(data_path, 'ctcf_sample.csv')

    ctcf1 = TabDataset('ctcf1', filename=csvfile)
    ctcf2 = TabDataset('ctcf2', filename=csvfile)

    props = data_props([ctcf1, ctcf2])

    assert 'ctcf1' in props
    assert 'ctcf2' in props
    assert props['ctcf1']['shape'] == (1,)
    assert props['ctcf2']['shape'] == (1,)

    with pytest.raises(Exception):
        data_props((0,))


def test_dna2ind():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    filename = os.path.join(data_path, 'oct4.fa')
    seqs = sequences_from_fasta(filename)

    np.testing.assert_equal(dna2ind(seqs[0]), dna2ind(str(seqs[0].seq)))
    np.testing.assert_equal(dna2ind(seqs[0]), dna2ind(seqs[0].seq))

    with pytest.raises(TypeError):
        # wrong type: int
        dna2ind(0)
