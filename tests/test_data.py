import os

import matplotlib
import pkg_resources
import pytest

from janggu.data import Dna
from janggu.data import Table
from janggu.data.data import _data_props

matplotlib.use('AGG')


def test_dna_props_extraction():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Dna.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    regions=bed_file, order=1)

    props = _data_props(dna)
    assert 'dna' in props
    assert props['dna']['shape'] == (200, 4, 1)

    with pytest.raises(Exception):
        _data_props((0,))


def test_tab_props_extraction():
    data_path = pkg_resources.resource_filename('janggu',
                                                'resources/')
    csvfile = os.path.join(data_path, 'sample.csv')

    sam1 = Table('sample1', filename=csvfile)
    sam2 = Table('sample2', filename=csvfile)

    props = _data_props([sam1, sam2])

    assert 'sample1' in props
    assert 'sample2' in props
    assert props['sample1']['shape'] == (1,)
    assert props['sample2']['shape'] == (1,)

    with pytest.raises(Exception):
        _data_props((0,))
