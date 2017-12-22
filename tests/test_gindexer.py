import os

import numpy as np
import pkg_resources
from genomeutils.regions import readBed

from bluewhalecore.data import BwGenomicIndexer


def test_gindexer_merged():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = readBed(os.path.join(data_path, 'regions.bed'))

    gi = BwGenomicIndexer(os.path.join(data_path, 'regions.bed'),
                          resolution=200, stride=50)
    np.testing.assert_equal(len(gi), 14344)

    gi = BwGenomicIndexer(regions, resolution=200, stride=50)
    np.testing.assert_equal(len(gi), 14344)

    np.testing.assert_equal(len(gi.idxByChrom(include='chr1')), 14344)

    np.testing.assert_equal(len(gi.idxByChrom(include='chr10')), 0)
    np.testing.assert_equal(len(gi.idxByChrom(exclude='chr10')), 14344)
    np.testing.assert_equal(len(gi.idxByChrom(exclude='chr1')), 0)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 800, '.'))

    iv = gi[7]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 950, 1150, '.'))


def test_gindexer_indiv():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = readBed(os.path.join(data_path, 'indiv_regions.bed'))

    gi = BwGenomicIndexer(os.path.join(data_path, 'indiv_regions.bed'),
                          resolution=200, stride=50)
    np.testing.assert_equal(len(gi), 14344)

    gi = BwGenomicIndexer(regions, resolution=200, stride=50)
    np.testing.assert_equal(len(gi), 14344)

    np.testing.assert_equal(len(gi.idxByChrom(include='chr1')), 14344)

    np.testing.assert_equal(len(gi.idxByChrom(include='chr10')), 0)
    np.testing.assert_equal(len(gi.idxByChrom(exclude='chr10')), 14344)
    np.testing.assert_equal(len(gi.idxByChrom(exclude='chr1')), 0)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 600, 800, '.'))

    iv = gi[7]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 950, 1150, '.'))
