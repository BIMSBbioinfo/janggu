import os

import matplotlib
import pkg_resources
import pytest

from janggu.data import Bioseq
from janggu.data import split_train_test
from janggu.data import Table
from janggu.data.data import _data_props

matplotlib.use('AGG')


def test_dna_props_extraction(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    regions=bed_file,
                                    binsize=200, stepsize=200,
                                    order=1)

    props = _data_props(dna)
    assert 'dna' in props
    assert props['dna']['shape'] == (200, 1, 4)

    with pytest.raises(Exception):
        _data_props((0,))


def test_split_train_test():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       regions=bed_file,
                                       binsize=200, stepsize=200,
                                       order=1, store_whole_genome=True)

    traindna, testdna = split_train_test(dna, holdout_chroms='chr2')

    assert len(traindna) == 50
    assert len(testdna) == 50
    assert len(dna) == len(traindna) + len(testdna)


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
