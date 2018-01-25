import os

import numpy as np
import pkg_resources
import pytest

from janggo.data import BlgGenomicIndexer


def test_gindexer_errors():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    with pytest.raises(ValueError):
        BlgGenomicIndexer.create_from_file(os.path.join(data_path,
                                                        'regions.bed'),
                                           resolution=0, stride=50)

    with pytest.raises(ValueError):
        BlgGenomicIndexer.create_from_file(os.path.join(data_path,
                                                        'regions.bed'),
                                           resolution=10, stride=0)


def test_gindexer_merged():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    gi = BlgGenomicIndexer.create_from_file(
        os.path.join(data_path, 'regions.bed'), resolution=200, stride=50)
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

    gi = BlgGenomicIndexer.create_from_file(
        os.path.join(data_path, 'indiv_regions.bed'), resolution=200, stride=50)
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
