import os

import matplotlib
import pkg_resources
import pytest

from janggu.data import Bioseq
from janggu.data import split_train_test
from janggu.data import subset
from janggu.data import view
from janggu.data.data import _data_props

matplotlib.use('AGG')


def test_dna_props_extraction(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                    storage='ndarray',
                                    roi=bed_file,
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
                                       roi=bed_file,
                                       binsize=200, stepsize=200,
                                       order=1, store_whole_genome=True)

    traindna, testdna = split_train_test(dna, holdout_chroms='chr2')

    assert len(traindna) == 50
    assert len(testdna) == 50
    assert len(dna) == len(traindna) + len(testdna)

    traindna, testdna = split_train_test([dna, dna], holdout_chroms='chr2')

    assert len(traindna[0]) == 50
    assert len(testdna[0]) == 50
    assert len(dna) == len(traindna[0]) + len(testdna[0])


def test_subset_include_chrname_test():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       roi=bed_file,
                                       binsize=200, stepsize=200,
                                       order=1, store_whole_genome=True)

    subdna = subset(dna, include_regions='chr2')

    assert len(subdna) == 50


def test_subset_exclude_chrname_test():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       roi=bed_file,
                                       binsize=200, stepsize=200,
                                       order=1, store_whole_genome=True)

    subdna = subset(dna, exclude_regions='chr2')

    assert len(subdna) == 50


def test_view_bed_test():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')
    bedsub_file = os.path.join(data_path, 'scored_sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       roi=bed_file,
                                       binsize=200, stepsize=200,
                                       order=1, store_whole_genome=True)

    subdna = view(dna, use_regions=bedsub_file)

    assert len(subdna) == 4
