import os

import numpy as np
import pandas
import pkg_resources
from HTSeq import GenomicInterval
import matplotlib
matplotlib.use('AGG')

from janggo.data import CoverageDataset


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

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # base pair resolution
        # print(store)
        cvdata = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank, stranded=True,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata_bam_unstranded_bed = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank, stranded=True,
            storage=store,
            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 40)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2, 2*flank + 1, 1))
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.start + 1),
            (cinterval.chrom, cinterval.start, cinterval.end))

        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 2)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][:20, :, :, 0],
                                cvdata[:][20:, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][:20, :, :, :],
                                cvdata[:][:20, :, :, :])

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # 20 bp resolution
        # print(store)
        cvdata = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank, stranded=True,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata_bam_unstranded_bed = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank, stranded=True,
            storage=store,
            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2, 2*flank + 1, 1))

        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 2)

        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.end),
            (cinterval.chrom, cinterval.start, cinterval.end))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 2, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 2.0)
        np.testing.assert_equal(x[0, 1, flank, 0], 0.0)

        # should be the opposite of above
        x = cvdata[1]
        np.testing.assert_equal(x.shape, (1, 2, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 0.0)
        np.testing.assert_equal(x[0, 1, flank, 0], 2.0)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][0, :, :, 0],
                                cvdata[:][1, ::-1, ::-1, 0])

        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][0, :, :, :],
                                cvdata[:][0, :, :, :])


def test_load_coveragedataset_bam_unstranded(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']
    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    flank = 4
    interval = GenomicInterval("chrIII", 217330, 217350, "+")

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # base pair resolution
        print(store)
        cvdata_bam = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank, stranded=False,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata_bam_unstranded_bed = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank, stranded=False,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata = cvdata_bam
        np.testing.assert_equal(len(cvdata), 40)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))
        print(list(cvdata.covers[0][interval]))
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start,
             interval.start + 1, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))

        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 2)

        x = cvdata[3]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, 2*flank, 0], 1.0)

        x = cvdata[7]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 1.0)

        x = cvdata[11]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, 0, 0], 1.0)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][:20, :, :, 0],
                                cvdata[:][20:, ::-1, ::-1, 0])

        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][:20, :, :, :],
                                cvdata[:][:20, :, :, :])

    for store in ['step', 'ndarray', 'memmap', 'hdf5']:
        # 20 bp resolution
        print(store)
        cvdata_bam = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank, stranded=False,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata_bam_unstranded_bed = CoverageDataset.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank, stranded=False,
            storage=store,
            cachedir=tmpdir.strpath)

        cvdata = cvdata_bam
        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))

        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 2)

        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.end, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 2.0)

        # check if slicing works
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][0, :, :, 0],
                                cvdata[:][1, ::-1, ::-1, 0])

        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bam_unstranded_bed[:][0, :, :, :],
                                cvdata[:][0, :, :, :])


def test_load_coveragedataset_bigwig_unstranded(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bwfile_ = os.path.join(data_path, "yeast_I_II_III.bw")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    gsize = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                            index_col='chr').to_dict()['length']

    # gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "yeast.bed")
    bed_file_unstranded = os.path.join(data_path, "yeast_unstranded.bed")

    flank = 4
    interval = GenomicInterval("chrIII", 217330, 217350, "+")
    cachedir = tmpdir.strpath

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # base pair resolution
        print(store)

        cvdata_bigwig = CoverageDataset.create_from_bigwig(
            "yeast_I_II_III.bw_res1_str",
            bigwigfiles=bwfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank,
            storage=store,
            cachedir=cachedir)

        cvdata_bigwig_us = CoverageDataset.create_from_bigwig(
            "yeast_I_II_III.bw_res1_unstr",
            bigwigfiles=bwfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=1, stride=1,
            flank=flank,
            storage=store,
            cachedir=cachedir)

        cvdata = cvdata_bigwig
        np.testing.assert_equal(len(cvdata), 40)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))
        print(list(cvdata.covers[0][interval]))
        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start,
             interval.start + 1, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))

        # We observe twice the coverage here.
        # For an actual analysis, this is not desired.
        # There one would have to make sure, that regions
        # are non-self-overlapping to avoid double counting evidence.
        # But for testing purposes this is ok.
        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 34*2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 34*2)

        x = cvdata[3]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, 2*flank, 0], 4.0)

        x = cvdata[7]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 4.0)

        x = cvdata[11]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, 0, 0], 4.0)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][:20, :, :, 0],
                                cvdata[:][20:, ::-1, ::-1, 0])
        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bigwig_us[:][:20, :, :, :]*2,
                                cvdata[:][:20, :, :, :])

    for store in ['step', 'ndarray', 'memmap', 'hdf5']:
        # 20 bp resolution
        print(store)

        cvdata_bigwig = CoverageDataset.create_from_bigwig(
            "yeast_I_II_III.bw_res20_str",
            bigwigfiles=bwfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank,
            storage=store,
            cachedir=cachedir)

        cvdata_bigwig_us = CoverageDataset.create_from_bigwig(
            "yeast_I_II_III.bw_res20_unstr",
            bigwigfiles=bwfile_,
            regions=bed_file_unstranded,
            genomesize=gsize,
            resolution=20, stride=20,
            flank=flank,
            storage=store,
            cachedir=cachedir)

        cvdata = cvdata_bigwig
        np.testing.assert_equal(len(cvdata), 2)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))

        np.testing.assert_equal(sum(list(cvdata.covers[0][interval])), 34*2)
        np.testing.assert_equal(cvdata.covers[0][interval].sum(), 34*2)

        cinterval = cvdata.gindexer[0]
        np.testing.assert_equal(
            (interval.chrom, interval.start, interval.end, interval.strand),
            (cinterval.chrom, cinterval.start,
             cinterval.end, cinterval.strand))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 34.0*2)

        # check if slicing works
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

        # testing forward and reverse complement
        np.testing.assert_equal(cvdata[:][0, :, :, 0],
                                cvdata[:][1, ::-1, ::-1, 0])

        # Also check unstranded bed variant
        np.testing.assert_equal(cvdata_bigwig_us[:][0, :, :, :]*2,
                                cvdata[:][0, :, :, :])
