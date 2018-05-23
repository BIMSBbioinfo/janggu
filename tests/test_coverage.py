import os

import matplotlib
matplotlib.use('AGG')

import numpy as np
import pandas
import pkg_resources
import pytest

from janggu.data import Cover


def test_cover_from_bam_sanity(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    print(os.environ['JANGGU_OUTPUT'])
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bamfile_ = os.path.join(data_path, "sample.bam")
    Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        regions=bed_file,
        binsize=1, stepsize=1,
        flank=0,
        storage='ndarray')
    Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
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


def test_cover_from_bigwig_sanity(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bwfile_ = os.path.join(data_path, "sample.bw")
    Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        regions=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')
    Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        resolution=50,
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


def test_cover_from_bed_sanity(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    bwfile_ = os.path.join(data_path, "scored_sample.bed")
    Cover.create_from_bed(
        'test',
        bedfiles=bwfile_,
        regions=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')
    Cover.create_from_bed(
        'test',
        bedfiles=bwfile_,
        regions=bed_file,
        resolution=50,
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


def test_cover_bam(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        # print(store)
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            storage=store)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 200, 2, 1))

        # the region is read relative to the forward strand
        # read on the reverse strand
        val = np.where(cover[4] == 1)
        np.testing.assert_equal(cover[4].sum(), 1.)
        np.testing.assert_equal(val[1][0], 179)  # pos
        np.testing.assert_equal(val[2][0], 1)  # strand

        # two reads on the forward strand
        val = np.where(cover[13] == 1)
        np.testing.assert_equal(cover[13].sum(), 2.)
        np.testing.assert_equal(val[1], np.asarray([162, 178]))  # pos
        np.testing.assert_equal(val[2], np.asarray([0, 0]))  # strand

        # the region is read relative to the reverse strand
        # for index 50
        # read on the reverse strand
        val = np.where(cover[52] == 1)
        np.testing.assert_equal(cover[52].sum(), 2.)
        np.testing.assert_equal(val[1], np.asarray([9, 89]))  # pos
        np.testing.assert_equal(val[2], np.asarray([0, 0]))  # strand

        # two reads on the forward strand
        val = np.where(cover[96] == 1)
        np.testing.assert_equal(cover[96].sum(), 1.)
        np.testing.assert_equal(val[1], np.asarray([25]))  # pos
        np.testing.assert_equal(val[2], np.asarray([1]))  # strand


def test_load_bam_resolution10(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        # print(store)
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            regions=bed_file,
            genomesize=gsize,
            resolution=10,
            storage=store)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 20, 2, 1))

        # the region is read relative to the forward strand
        # read on the reverse strand
        val = np.where(cover[4] == 1)
        np.testing.assert_equal(cover[4].sum(), 1.)
        np.testing.assert_equal(val[1][0], 17)  # pos
        np.testing.assert_equal(val[2][0], 1)  # strand

        # two reads on the forward strand
        val = np.where(cover[13] == 1)
        np.testing.assert_equal(cover[13].sum(), 2.)
        np.testing.assert_equal(val[1], np.asarray([16, 17]))  # pos
        np.testing.assert_equal(val[2], np.asarray([0, 0]))  # strand

        # the region is read relative to the reverse strand
        # for index 50
        # read on the reverse strand
        val = np.where(cover[52] == 1)
        np.testing.assert_equal(cover[52].sum(), 2.)
        np.testing.assert_equal(val[1], np.asarray([0, 8]))  # pos
        np.testing.assert_equal(val[2], np.asarray([0, 0]))  # strand

        # two reads on the forward strand
        val = np.where(cover[96] == 1)
        np.testing.assert_equal(cover[96].sum(), 1.)
        np.testing.assert_equal(val[1], np.asarray([2]))  # pos
        np.testing.assert_equal(val[2], np.asarray([1]))  # strand


def test_load_cover_bigwig_default(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bwfile_ = os.path.join(data_path, "sample.bw")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    gsize = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                            index_col='chr').to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        print(store)
        cover = Cover.create_from_bigwig(
            "cov",
            bigwigfiles=bwfile_,
            regions=bed_file,
            genomesize=gsize,
            storage=store)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))

        # there is one read in the region
        np.testing.assert_allclose(cover[4].sum(), 36./200)
        np.testing.assert_allclose(cover[52].sum(), 2*36./200)


def test_load_cover_bigwig_resolution1(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bwfile_ = os.path.join(data_path, "sample.bw")

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        print(store)
        cover = Cover.create_from_bigwig(
            "cov",
            bigwigfiles=bwfile_,
            regions=bed_file,
            resolution=1,
            storage=store)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 200, 1, 1))

        # there is one read in the region 4
        np.testing.assert_allclose(cover[4].sum(), 36)
        np.testing.assert_equal(cover[4][0, :, 0, 0],
         np.asarray(
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        # and two reads in region 52
        np.testing.assert_allclose(cover[52].sum(), 2*36)
        np.testing.assert_equal(cover[52][0, :, 0, 0],
        np.asarray(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))


def test_load_cover_bed_binary(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    for store in ['ndarray', 'sparse']:
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            mode='binary')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            resolution=50,
            mode='binary')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            resolution=50,
            dimmode='first',
            mode='binary')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)


def test_load_cover_bed_scored(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    for store in ['ndarray', 'sparse']:
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            mode='score')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 5)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            resolution=50,
            mode='score')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*5)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            resolution=50,
            dimmode='first',
            mode='score')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 5)


def test_load_cover_bed_categorical(tmpdir):
    os.environ['JANGGU_OUTPUT']=tmpdir.strpath
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    for store in ['ndarray', 'sparse']:
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            regions=bed_file,
            storage=store,
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 6))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            resolution=50,
            storage=store,
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 6))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            regions=bed_file,
            resolution=50,
            storage=store,
            dimmode='first',
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 6))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
