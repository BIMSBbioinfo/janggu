import os

import matplotlib
import numpy as np
import pkg_resources
import pytest
import pandas as pd

from janggu.data import GenomicIndexer

matplotlib.use('AGG')

def test_gindexer_short_interval():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')


    gi = GenomicIndexer.create_from_file(os.path.join(data_path,
                                                 'sample_equalsize.bed'),
                                    binsize=200, stepsize=200)
    assert len(gi) == 4
    gi = GenomicIndexer.create_from_file(os.path.join(data_path,
                                                 'sample_equalsize.bed'),
                                    binsize=180, stepsize=20)
    assert len(gi) == 8
    gi = GenomicIndexer.create_from_file(os.path.join(data_path,
                                                 'sample_equalsize.bed'),
                                    binsize=210, stepsize=20, zero_padding=False)
    assert len(gi) == 0

    gi = GenomicIndexer.create_from_file(os.path.join(data_path,
                                                 'sample_equalsize.bed'),
                                    binsize=210, stepsize=20, zero_padding=True)
    assert len(gi) == 4


def test_gindexer_short_interval_with_dataframe():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    df = pd.read_csv(os.path.join(data_path, 'sample_equalsize.bed'),
                     sep='\t', header=None, names=['chrom', 'start', 'end'])

    gi = GenomicIndexer.create_from_file(df,
                                         binsize=200, stepsize=200)
    assert len(gi) == 4
    gi = GenomicIndexer.create_from_file(df,
                                         binsize=180, stepsize=20)
    assert len(gi) == 8
    gi = GenomicIndexer.create_from_file(df,
                                         binsize=210, stepsize=20,
                                         zero_padding=False)
    assert len(gi) == 0

    gi = GenomicIndexer.create_from_file(df,
                                         binsize=210, stepsize=20,
                                         zero_padding=True)
    assert len(gi) == 4


def test_gindexer_errors():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'sample.bed'),
                                        binsize=0, stepsize=50)

    with pytest.raises(ValueError):
        GenomicIndexer.create_from_file(os.path.join(data_path,
                                                     'sample.bed'),
                                        binsize=10, stepsize=0)
    with pytest.raises(ValueError):
        # due to flank < 0
        GenomicIndexer.create_from_file(os.path.join(data_path, 'sample.bed'),
                                        binsize=200, stepsize=50, flank=-1)
    # due to unequal intervals
    gi=GenomicIndexer.create_from_file(os.path.join(data_path, 'scores.bed'),
                                    binsize=None, stepsize=None, flank=0)
    #print(len(gi))
    #for reg in gi:
    #    print(reg)
    GenomicIndexer.create_from_file(os.path.join(data_path, 'scores.bed'),
                                    binsize=200, stepsize=200, flank=0)

def test_gindexer_merged():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'sample.bed'), binsize=200, stepsize=200)
    np.testing.assert_equal(len(gi), 100)
    gi2 = gi.filter_by_region(include='chr1')
    gi3 = gi.filter_by_region(include='chr10')
    gi4 = gi.filter_by_region(exclude='chr2')
    gi5 = gi.filter_by_region(exclude='chr10')


    np.testing.assert_equal(len(gi2), 50)

    np.testing.assert_equal(len(gi3), 0)
    np.testing.assert_equal(len(gi4), 50)
    np.testing.assert_equal(len(gi5), 100)


def test_gindexer_merged_variable_length_ranges():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    # with fixed size
    gi = GenomicIndexer.create_from_file(
        os.path.join(data_path, 'sample.bed'), binsize=3000, stepsize=3000,
        zero_padding=False)
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
        stepsize=3000, zero_padding=True)
    np.testing.assert_equal(len(gi), 8)

    iv = gi[0]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr1', 15000, 18000, '+'))
    iv = gi[-1]
    np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                            ('chr2', 24000, 25000, '-'))
