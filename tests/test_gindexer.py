import os

import matplotlib
matplotlib.use('AGG')

import numpy as np
import pkg_resources
import pytest

from janggo.data import GenomicIndexer


def test_gindexer_errors():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'regions.bed'),
                                        binsize=0, stepsize=50)

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'regions.bed'),
                                        binsize=10, stepsize=0)
    with pytest.raises(ValueError):
        # due to resolution > stepsize
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'regions.bed'),
                                        binsize=200, stepsize=50,
                                        resolution=100)


def test_gindexer_merged():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'regions.bed'), binsize=200, stepsize=50)
    np.testing.assert_equal(len(gi), 14344)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr1')), 14344)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr10')), 0)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr10')), 14344)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr1')), 0)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 800, '.'))

    iv = gi[7]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 950, 1150, '.'))


def test_gindexer_indiv():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'indiv_regions.bed'), binsize=200, stepsize=50)
    np.testing.assert_equal(len(gi), 14344)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr1')), 14344)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr10')), 0)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr10')), 14344)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr1')), 0)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 800, '.'))

    iv = gi[7]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 950, 1150, '.'))


def test_gindexer_merged_variable_length_ranges():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    # with fixed size
    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'regions.bed'), binsize=100000,
        stepsize=100000)
    np.testing.assert_equal(len(gi), 5+1)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 100600, '.'))
    iv = gi[-1]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 570400, 670400, '.'))

    # with variable size regions
    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'regions.bed'), binsize=100000,
        stepsize=100000, fixed_size_batches=False)
    np.testing.assert_equal(len(gi), 5+1+2)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 100600, '.'))
    iv = gi[-1]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 670400, 724100, '.'))
