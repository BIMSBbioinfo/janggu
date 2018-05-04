import os

import matplotlib
matplotlib.use('AGG')

import numpy as np
import pandas
import pkg_resources
import pytest
from HTSeq import GenomicInterval

from janggo.data import Cover


def test_coverage_from_bam_sanity():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, "yeast.bed")

    bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
    Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        regions=bed_file,
        binsize=1, stepsize=1,
        flank=0,
        storage='ndarray')

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bam(
            1.2,
            bamfiles=bamfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            regions=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            regions=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')


def test_coverage_from_bigwig_sanity():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, "yeast.bed")

    bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
    Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        regions=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bigwig(
            1.2,
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        # resolution must be greater than stepsize
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=200, stepsize=50,
            resolution=300,
            flank=0,
            storage='ndarray')


def test_coverage_from_bed_sanity():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    bwfile_ = os.path.join(data_path, "indiv_regions.bed")
    Cover.create_from_bed(
        'test',
        bedfiles=bwfile_,
        regions=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bed(
            1.2,
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        # resolution must be greater than stepsize
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=200, stepsize=50,
            resolution=300,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        csvfile = os.path.join(data_path, 'ctcf_sample.csv')
        # must be a bed file
        Cover.create_from_bed(
            'test',
            bedfiles=csvfile,
            regions=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')


def test_load_coveragedataset_bam_stranded(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    interval = GenomicInterval("chrIII", 217330, 217350, "+")

    flank = 4

    for store in ['ndarray', 'hdf5']:
        # base pair binsize
        # print(store)
        cvdata = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            binsize=1, stepsize=1,
            flank=flank,
            storage=store,
            cachedir=tmpdir.strpath if store == 'hdf5' else None)

        cvdata_bam_unstranded_bed = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            binsize=1, stepsize=1,
            flank=flank,
            storage=store,
            cachedir=tmpdir.strpath if store == 'hdf5' else None)

        np.testing.assert_equal(len(cvdata), 40)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2*flank + 1, 2, 1))
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.start+1),
            (cinterval.chrom, cinterval.start, cinterval.end))

        np.testing.assert_equal(cvdata.covers[interval].sum(), 2)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][:20, :, :, 0],
                                cvdata[:][20:, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][:20, :, :, :],
                                cvdata[:][:20, :, :, :])

    for store in ['ndarray', 'hdf5']:
        # 20 bp binsize
        # print(store)
        cvdata = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            binsize=20, stepsize=20,
            flank=flank,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata_bam_unstranded_bed = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            binsize=20, stepsize=20,
            flank=flank,
            storage=store,
            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2*flank + 20, 2, 1))

        np.testing.assert_equal(cvdata.covers[interval].sum(), 2)

        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.end),
            (cinterval.chrom, cinterval.start, cinterval.end))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 2*flank + 20, 2, 1))
        np.testing.assert_equal(x[0, 11, :, 0], [1.0, 0.0])
        np.testing.assert_equal(x[0, 23, :, 0], [1.0, 0.0])
        np.testing.assert_equal(x[0, 27, :, 0], [1.0, 0.0])

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[0][:, :, :, :],
                                cvdata[1][:, ::-1, ::-1, :])

        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][0, :, :, :],
                                cvdata[:][0, :, :, :])


def test_load_coveragedataset_bigwig_unstranded_resolution1_bin1(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    gsize = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                            index_col='chr').to_dict()['length']

    # gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    flank = 4
    resolution = 1
    interval = GenomicInterval("chrIII", 217330, 217350, "+")
    cachedir = tmpdir.strpath

    for store in ['ndarray', 'hdf5']:
        # base pair binsize
        print(store)
        cvdata_bigwig = Cover.create_from_bigwig(
            "yeast_I_II_III.bw_res1_str",
            bigwigfiles=bwfile_,
            regions=bed_file,
            genomesize=gsize,
            binsize=1, stepsize=1,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir if store == 'hdf5' else None)
        cvdata_bigwig_us = Cover.create_from_bigwig(
            "yeast_I_II_III.bw_res1_unstr",
            bigwigfiles=bwfile_,
            regions=bed_file_unstranded,
            binsize=1, stepsize=1,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir if store == 'hdf5' else None)
        cvdata = cvdata_bigwig
        np.testing.assert_equal(len(cvdata), 40)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2*flank + 1, 1, 1))
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start//resolution,
             (interval.start + resolution)//resolution, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))
        # We observe twice the coverage here.
        # For an actual analysis, this is not desired.
        # There one would have to make sure, that regions
        # are non-self-overlapping to avoid double counting evidence.
        # But for testing purposes this is ok.
        # UPDATE:
        # With the new GenomicArray, we obtain half the coverage,
        # because we only assign once per region. If a region gets assignments
        # multiple times, it overwrites the previous value.
        piv = interval.copy()
        piv.start //= resolution
        piv.end = piv.start + 1
        print(piv)
        print(cvdata.covers[piv])
        np.testing.assert_equal(cvdata.covers[piv].mean(), 1.)
        x = cvdata[3]
        np.testing.assert_equal(x.shape, (1, 2*flank + 1, 1, 1))
        np.testing.assert_equal(x[0, 2*flank, 0, 0], 2.0)
        x = cvdata[7]
        np.testing.assert_equal(x.shape, (1, 2*flank + 1, 1, 1))
        np.testing.assert_equal(x[0, flank, 0, 0], 2.0)
        x = cvdata[11]
        np.testing.assert_equal(x.shape, (1, 2*flank + 1, 1, 1))
        np.testing.assert_equal(x[0, 0, 0, 0], 2.0)
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)
        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][:20, :, :, 0],
                                cvdata[:][20:, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        print(cvdata_bigwig_us[:].dtype)
        print(cvdata[:].dtype)
        print((cvdata_bigwig_us[:] - cvdata[:20]).sum())
        np.testing.assert_equal(cvdata_bigwig_us[:], cvdata[:20])


def test_load_coveragedataset_bigwig_unstranded_resolution1_bin20(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    gsize = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                            index_col='chr').to_dict()['length']

    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    flank = 4
    resolution = 1
    interval = GenomicInterval("chrIII", 217330, 217350, "+")
    cachedir = tmpdir.strpath

    for store in ['ndarray', 'hdf5']:
        # 20 bp binsize
        print(store)
        cvdata_bigwig = Cover.create_from_bigwig(
            "yeast_I_II_III.bw_res20_str",
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=20, stepsize=20,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir)
        cvdata_bigwig_us = Cover.create_from_bigwig(
            "yeast_I_II_III.bw_res20_unstr",
            bigwigfiles=bwfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            binsize=20, stepsize=20,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir)
        cvdata = cvdata_bigwig
        piv = interval.copy()
        piv.start //= resolution
        piv.end = piv.start + 1
        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2*flank + 20, 1, 1))
        print(piv)
        print(cvdata.covers[piv])
        np.testing.assert_allclose(cvdata.covers[interval].sum(), 1.7)
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start//resolution, interval.end//resolution, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))
        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 2*flank + 20, 1, 1))
        np.testing.assert_allclose(x[0, flank, 0, 0], 1.7)
        # check if slicing works
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)
        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][0, :, :, 0],
                                cvdata[:][1, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bigwig_us[:][0, :, :, :],
                                cvdata[:][0, :, :, :])


def test_load_coveragedataset_bigwig_unstranded_resolution10_bin20(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    gsize = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                            index_col='chr').to_dict()['length']

    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    flank = 4
    resolution = 10
    interval = GenomicInterval("chrIII", 217330, 217350, "+")
    cachedir = tmpdir.strpath

    for store in ['ndarray', 'hdf5']:
        # 20 bp binsize
        print(store)
        cvdata_bigwig = Cover.create_from_bigwig(
            "yeast_I_II_III.bed_res20_str",
            bigwigfiles=bwfile_,
            regions=bed_file,
            binsize=20, stepsize=20,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir)
        cvdata_bigwig_us = Cover.create_from_bigwig(
            "yeast_I_II_III.bw_res20_unstr",
            bigwigfiles=bwfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            binsize=20, stepsize=20,
            resolution=resolution,
            flank=flank,
            storage=store,
            cachedir=cachedir)
        cvdata = cvdata_bigwig
        piv = interval.copy()
        piv.start //= resolution
        piv.end = piv.start + 1
        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2*flank + 2, 1, 1))
        print(piv)
        print(cvdata.covers[piv])
        np.testing.assert_allclose(cvdata.covers[piv].sum(), 1.7)
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start//resolution, interval.end//resolution, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))
        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 2*flank + 2, 1, 1))
        np.testing.assert_allclose(x[0, flank, 0, 0], 1.7)
        # check if slicing works
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)
        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][0, :, :, 0],
                                cvdata[:][1, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bigwig_us[:][0, :, :, :],
                                cvdata[:][0, :, :, :])


def test_load_coveragedataset_bed_unstranded_resolution50_bin200(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "indiv_regions.bed")

    gsize = {'chr1': 20000}

    bed_file = os.path.join(data_path, "regions.bed")

    flank = 4
    resolution = 50

    cachedir = tmpdir.strpath

    for store in ['ndarray', 'hdf5']:
        # 20 bp binsize
        print(store)
        cvdata_bigwig = Cover.create_from_bed(
            "yeast_I_II_III.bed_res20_str",
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=200, stepsize=50,
            resolution=resolution,
            flank=flank,
            dimmode='all',
            storage=store,
            cachedir=cachedir if store == 'hdf5' else None)

        cvdata = cvdata_bigwig

        np.testing.assert_equal(len(cvdata), 14344)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 4 + 2*4, 1, 1))
        np.testing.assert_equal(cvdata[99].shape, (1, 12, 1, 1))
        np.testing.assert_equal(cvdata[99].sum(), 1 + 11)

        cinterval = cvdata.gindexer[99]
        np.testing.assert_equal(('chr1', 111, 115),
                                (cinterval.chrom, cinterval.start,
                                 cinterval.end))

        # with flank the label should be shifted. therefore, we do not
        # find the score=10 at index 99, because the score value from the
        # upstream position is returned.
        cvdata_bigwig = Cover.create_from_bed(
            "yeast_I_II_III.bed_res20_str",
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=200, stepsize=50,
            genomesize=gsize,
            resolution=resolution,
            flank=1,
            dimmode='first',
            storage=store,
            cachedir=cachedir if store == 'hdf5' else None)

        cvdata = cvdata_bigwig

        np.testing.assert_equal(len(cvdata), 14344)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 1, 1))
        np.testing.assert_equal(cvdata[99].shape, (1, 1, 1, 1))
        np.testing.assert_equal(cvdata[99].sum(), 1)

        cinterval = cvdata.gindexer[99]
        np.testing.assert_equal(('chr1', 111, 115),
                                (cinterval.chrom, cinterval.start,
                                 cinterval.end))

        # now use without flank, otherwise this would introduce a shift.
        cvdata_bigwig = Cover.create_from_bed(
            "yeast_I_II_III.bed_res20_str",
            bedfiles=bwfile_,
            regions=bed_file,
            binsize=200, stepsize=50,
            resolution=resolution,
            flank=0,
            dimmode='first',
            storage=store,
            cachedir=cachedir if store == 'hdf5' else None)

        cvdata = cvdata_bigwig

        np.testing.assert_equal(len(cvdata), 14344)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 1, 1))
        np.testing.assert_equal(cvdata[99].shape, (1, 1, 1, 1))
        np.testing.assert_equal(cvdata[99].sum(), 1)

        cinterval = cvdata.gindexer[99]
        np.testing.assert_equal(('chr1', 111, 115),
                                (cinterval.chrom, cinterval.start,
                                 cinterval.end))


def test_load_coveragedataset_bed_binary():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "scored_region.bed")

    bed_file = os.path.join(data_path, "regions.bed")

    resolution = 50

    store = 'ndarray'

    cvdata_bigwig = Cover.create_from_bed(
        "yeast_I_II_III.bed_res20_str",
        bedfiles=bwfile_,
        regions=bed_file,
        binsize=50, stepsize=50,
        resolution=resolution,
        dimmode='all',
        storage=store,
        mode='binary')

    cvdata = cvdata_bigwig

    np.testing.assert_equal(len(cvdata), 14350)
    np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 1, 1))
    np.testing.assert_equal(cvdata[0].shape, (1, 1, 1, 1))
    np.testing.assert_equal(cvdata[0].sum(), 1)
    np.testing.assert_equal(cvdata[1].sum(), 1)
    np.testing.assert_equal(cvdata[99].sum(), 0)


def test_load_coveragedataset_bed_scored():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "scored_region.bed")

    bed_file = os.path.join(data_path, "regions.bed")

    resolution = 50

    store = 'ndarray'

    cvdata_bigwig = Cover.create_from_bed(
        "yeast_I_II_III.bed_res20_str",
        bedfiles=bwfile_,
        regions=bed_file,
        binsize=50, stepsize=50,
        resolution=resolution,
        dimmode='all',
        storage=store,
        mode='score')

    cvdata = cvdata_bigwig

    np.testing.assert_equal(len(cvdata), 14350)
    np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 1, 1))
    np.testing.assert_equal(cvdata[0].shape, (1, 1, 1, 1))
    np.testing.assert_equal(cvdata[0].sum(), 1)
    np.testing.assert_equal(cvdata[1].sum(), 2)
    np.testing.assert_equal(cvdata[99].sum(), 0)


def test_load_coveragedataset_bed_categorical():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "scored_region.bed")

    bed_file = os.path.join(data_path, "regions.bed")

    resolution = 50

    store = 'ndarray'

    cvdata_bigwig = Cover.create_from_bed(
        "yeast_I_II_III.bed_res20_str",
        bedfiles=bwfile_,
        regions=bed_file,
        binsize=50, stepsize=50,
        resolution=resolution,
        dimmode='all',
        storage=store,
        mode='categorical')

    cvdata = cvdata_bigwig

    np.testing.assert_equal(len(cvdata), 14350)
    np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 1, 9))
    np.testing.assert_equal(cvdata[0].shape, (1, 1, 1, 9))
    np.testing.assert_equal(cvdata[0][0, 0, 0, 1], 1)
    np.testing.assert_equal(cvdata[1][0, 0, 0, 2], 1)
    np.testing.assert_equal(cvdata[2][0, 0, 0, 3], 1)
    np.testing.assert_equal(cvdata[99].sum(), 0)
