import os

import matplotlib
import numpy as np
import pkg_resources
import pytest

from janggo.data import GenomicIndexer

matplotlib.use('AGG')


def test_gindexer_errors():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'sample.bed'),
                                        binsize=0, stepsize=50)

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'sample.bed'),
                                        binsize=10, stepsize=0)
    with pytest.raises(ValueError):
        # due to resolution > stepsize
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'sample.bed'),
                                        binsize=200, stepsize=50,
                                        resolution=100)


def test_gindexer_merged():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'sample.bed'), binsize=200, stepsize=200)
    np.testing.assert_equal(len(gi), 100)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr1')), 50)

    np.testing.assert_equal(len(gi.idx_by_chrom(include='chr10')), 0)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr2')), 50)
    np.testing.assert_equal(len(gi.idx_by_chrom(exclude='chr10')), 100)


def test_gindexer_merged_variable_length_ranges():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    # with fixed size
    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'sample.bed'), binsize=3000, stepsize=3000)
    np.testing.assert_equal(len(gi), 6)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 15000, 18000, '+'))
    iv = gi[-1]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr2', 21000, 24000, '-'))

    # with variable size regions
    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'sample.bed'), binsize=3000,
        stepsize=3000, fixed_size_batches=False)
    np.testing.assert_equal(len(gi), 8)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 15000, 18000, '+'))
    iv = gi[-1]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr2', 24000, 25000, '-'))
