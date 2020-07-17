import os
from itertools import product

import matplotlib
matplotlib.use('AGG')  # pylint: disable=

import numpy as np
import pandas
import pkg_resources
import pytest
from pybedtools import BedTool

from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import Transpose
from janggu.data import GenomicIndexer
from janggu.data import plotGenomeTrack
from janggu.data import LineTrack
from janggu.data import SeqTrack
from janggu.data import HeatTrack


def test_channel_last_first():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bwfile_ = os.path.join(data_path, "sample.bw")

    cover = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        resolution=1,
        binsize=200,
        roi=bed_file,
        store_whole_genome=True,
        storage='ndarray')
    assert cover.shape == (100, 200, 1, 1)
    assert cover[0].shape == (1, 200, 1, 1)
    cover1 = cover

    cover = Transpose(Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        resolution=1,
        binsize=200,
        roi=bed_file,
        store_whole_genome=True,
        storage='ndarray'), axis=(0, 3, 2, 1))
    assert cover.shape == (100, 1, 1, 200)
    assert cover[0].shape == (1, 1, 1, 200)

    np.testing.assert_equal(cover1[0], np.transpose(cover[0], (0, 3, 2, 1)))


def test_cover_roi_binsize_padding(tmpdir):

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample_equalsize.bed')
    print(pandas.read_csv(bed_file,
                          sep='\t', header=None,
                          names=['chrom', 'start', 'end',
                                 'name', 'score', 'strand']))

    roi_file = os.path.join(data_path, "sample.bed")
    roi = pandas.read_csv(roi_file,
                          sep='\t', header=None,
                          names=['chrom', 'start', 'end',
                                 'name', 'score', 'strand'])

    roi.end.iloc[0] += 12
    roi.end.iloc[1] += 111
    print(roi)

    with pytest.raises(ValueError):
        # error due to binsize not being a multiple of resolution
        Cover.create_from_bed('test',
                              bedfiles=bed_file,
                              roi=roi, binsize=30,
                              stepsize=30,
                              store_whole_genome=True,
                              cache=False, resolution=7)

    with pytest.raises(ValueError):
        # interval starts must align with resolution intervals
        rroi = roi.copy()
        rroi.start += 1
        Cover.create_from_bed('test',
                              bedfiles=bed_file,
                              roi=rroi, binsize=30,
                              stepsize=30,
                              store_whole_genome=True,
                              cache=False, resolution=30)

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    storage=store,
                                    cache=False, resolution=10)
        assert len(cov) == 68
        assert cov.shape == (68, 30, 1, 1)
        [c for c in cov]

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    cache=False, resolution=3)
        assert len(cov) == 68
        assert cov.shape == (68, 100, 1, 1)
        [c for c in cov]

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    storage=store,
                                    cache=False, resolution=3)
        assert len(cov) == 68
        assert cov.shape == (68, 100, 1, 1)
        [c for c in cov]

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    storage=store,
                                    cache=False, resolution=3)
        assert len(cov) == 68
        assert cov.shape == (68, 100, 1, 1)
        [c for c in cov]

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    storage=store,
                                    cache=False, resolution=100)
        assert len(cov) == 68
        assert cov.shape == (68, 3, 1, 1)
        [c for c in cov]
    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cov = Cover.create_from_bed('test',
                                    bedfiles=bed_file,
                                    roi=roi, binsize=300,
                                    stepsize=300,
                                    store_whole_genome=swg,
                                    cache=False, resolution=100,
                                    storage=store,
                                    zero_padding=False)
        assert len(cov) == 66
        assert cov.shape == (66, 3, 1, 1)
        [c for c in cov]

    bwfile_ = os.path.join(data_path, "sample.bw")

    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cover = Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            resolution=100,
            binsize=300,
            roi=roi,
            storage=store,
            store_whole_genome=swg)
        assert len(cover) == 68
        assert cover.shape == (68, 3, 1, 1)
        [c for c in cover]
    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cover = Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            resolution=100,
            binsize=300,
            roi=roi, zero_padding=False,
            storage=store,
            store_whole_genome=swg)
        assert len(cover) == 66
        assert cover.shape == (66, 3, 1, 1)
        [c for c in cover]

    bamfile_ = os.path.join(data_path, "sample.bam")
    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cover = Cover.create_from_bam(
            'test',
            bamfile_,
            resolution=100,
            binsize=300,
            roi=roi,
            stranded=False,
            storage=store,
            store_whole_genome=swg)
        assert len(cover) == 68
        assert cover.shape == (68, 3, 1, 1)
        [c for c in cover]
    for swg, store in product([True, False], ['ndarray', 'sparse']):
        cover = Cover.create_from_bam(
            'test',
            bamfile_,
            resolution=100,
            binsize=300,
            roi=roi, zero_padding=False,
            stranded=False,
            storage=store,
            store_whole_genome=swg)
        assert len(cover) == 66
        assert cover.shape == (66, 3, 1, 1)
        [c for c in cover]

def test_cover_export_bigwig(tmpdir):
    path = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bwfile_ = os.path.join(data_path, "sample.bw")

    for resolution in [1, 50]:
        for storage in [True, False]:
            print('resolution=', resolution)
            print('store_whole_genome', storage)
            cover = Cover.create_from_bigwig(
                'test',
                bigwigfiles=bwfile_,
                resolution=resolution,
                binsize=200,
                roi=bed_file,
                store_whole_genome=storage,
                storage='ndarray')

            cover.export_to_bigwig(output_dir=path)

            cov2 = Cover.create_from_bigwig('test',
                bigwigfiles='{path}/{name}.{sample}.bigwig'.format(
                path=path, name=cover.name,
                sample=cover.conditions[0]),
                resolution=resolution,
                binsize=200,
                roi=bed_file,
                store_whole_genome=storage,
                storage='ndarray')

            assert cover.shape == (100, 200 // resolution, 1, 1)
            assert cover.shape == cov2.shape
            np.testing.assert_allclose(cover[:].sum(), 1044.0 / resolution)
            np.testing.assert_allclose(cov2[:].sum(), 1044.0 / resolution)


def test_bam_genomic_interval_access():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bamfile_ = os.path.join(data_path, "sample.bam")

    for reso, shift, storage in product([1, 50], [0, 1], [True, False]):
            cover = Cover.create_from_bam(
                'test',
                bamfiles=bamfile_,
                roi=bed_file,
                flank=0,
                storage='ndarray',
                store_whole_genome=storage,
                resolution=reso)

            for i in range(len(cover)):
                print('storage :',storage,'/ resolution :',reso,'/ shift :',shift)

                np.testing.assert_equal(np.repeat(cover[i],
                                    cover.garray.resolution,
                                    axis=1), cover[cover.gindexer[i]])

                chrom, start, end, strand = cover.gindexer[i].chrom, \
                    cover.gindexer[i].start, \
                    cover.gindexer[i].end, \
                    cover.gindexer[i].strand

                np.testing.assert_equal(np.repeat(cover[i],
                                        cover.garray.resolution, axis=1),
                                        cover[chrom, start, end, strand])

                np.testing.assert_equal(cover[chrom, start, end, strand],
                                        cover[chrom, start-1, end+1, strand][:, 1:-1, :, :])
                if shift != 0:
                    start += shift * reso
                    end += shift * reso

                    if strand != '-':
                        gicov = cover[chrom, start, end, strand][:, :(-shift*reso),:,:]
                        np.testing.assert_equal(cover[i][:, shift:,:, :],
                            gicov.reshape((1, gicov.shape[1]//reso, reso, 2, 1))[:, :, 0, :, :])
                    else:
                        gicov = cover[chrom, start, end, strand][:, (shift*reso):,:,:]
                        np.testing.assert_equal(cover[i][:, :-shift,:, :],
                        gicov.reshape((1, gicov.shape[1]//reso, reso, 2, 1))[:, :, 0, :, :])


def test_bigwig_genomic_interval_access():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bamfile_ = os.path.join(data_path, "sample.bw")

    for reso, shift, storage in product([1, 50], [0, 1], [True, False]):
            cover = Cover.create_from_bigwig(
                'test',
                bigwigfiles=bamfile_,
                roi=bed_file,
                flank=0,
                storage='ndarray',
                store_whole_genome=storage,
                resolution=reso)

            for i in range(len(cover)):
                print('storage :',storage,'/ resolution :',reso,'/ shift :',shift)

                np.testing.assert_equal(np.repeat(cover[i],
                                    cover.garray.resolution,
                                    axis=1), cover[cover.gindexer[i]])

                chrom, start, end, strand = cover.gindexer[i].chrom, \
                    cover.gindexer[i].start, \
                    cover.gindexer[i].end, \
                    cover.gindexer[i].strand

                np.testing.assert_equal(np.repeat(cover[i],
                                    cover.garray.resolution, axis=1),
                                    cover[chrom, start, end, strand])

                if shift != 0:
                    start += shift * reso
                    end += shift * reso

                    if strand != '-':
                        gicov = cover[chrom, start, end, strand][:, :(-shift*reso),:,:]
                        np.testing.assert_equal(cover[i][:, shift:,:, :],
                            gicov.reshape((1, gicov.shape[1]//reso, reso, 1, 1))[:, :, 0, :, :])
                    else:
                        gicov = cover[chrom, start, end, strand][:, (shift*reso):,:,:]
                        np.testing.assert_equal(cover[i][:, :-shift,:, :],
                        gicov.reshape((1, gicov.shape[1]//reso, reso, 1, 1))[:, :, 0, :, :])


def test_bed_genomic_interval_access():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bamfile_ = os.path.join(data_path, "sample.bed")


    for reso, shift, storage in product([1, 50], [0, 1], [True, False]):
        cover = Cover.create_from_bed(
            'test',
            bedfiles=bamfile_,
            roi=bed_file,
            flank=0,
            storage='ndarray',
            store_whole_genome=storage,
            resolution=reso)

        for i in range(len(cover)):
            print('storage :',storage,'/ resolution :',reso,'/ shift :',shift)

            np.testing.assert_equal(np.repeat(cover[i],
                                cover.garray.resolution,
                                axis=1), cover[cover.gindexer[i]])

            chrom, start, end, strand = cover.gindexer[i].chrom, \
                cover.gindexer[i].start, \
                cover.gindexer[i].end, \
                cover.gindexer[i].strand

            np.testing.assert_equal(np.repeat(cover[i],
                                cover.garray.resolution, axis=1),
                                cover[chrom, start, end, strand])

            if shift != 0:
                start += shift * reso
                end += shift * reso

                if strand != '-':
                    gicov = cover[chrom, start, end, strand][:, :(-shift*reso),:,:]
                    np.testing.assert_equal(cover[i][:, shift:,:, :],
                        gicov.reshape((1, gicov.shape[1]//reso, reso, 1, 1))[:, :, 0, :, :])
                else:
                    gicov = cover[chrom, start, end, strand][:, (shift*reso):,:,:]
                    np.testing.assert_equal(cover[i][:, :-shift,:, :],
                    gicov.reshape((1, gicov.shape[1]//reso, reso, 1, 1))[:, :, 0, :, :])


def test_bam_inferred_binsize():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive.bed")

    bamfile_ = os.path.join(data_path, "sample.bam")

    cover = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=bed_file,
        flank=0,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 200, 2, 1)


def test_bigwig_inferred_binsize():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive.bed")

    bwfile_ = os.path.join(data_path, "sample.bw")

    cover = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        resolution=1,
        roi=bed_file,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)


def test_bed_unsync_roi_targets():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive.bed")
    bed_shift_file = os.path.join(data_path, "positive_shift.bed")

    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=None,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 1, 1, 1)
    assert cover[:].sum() == 1

    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=50,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 4, 1, 1)
    assert cover[:].sum() == 1


    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=50,
        store_whole_genome=True,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 4, 1, 1)
    assert cover[:].sum() == 1

    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=1,
        store_whole_genome=False,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)
    assert cover[0].sum() == 49

    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=1,
        store_whole_genome=True,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)
    assert cover[:].sum() == 49

    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=bed_file,
        resolution=1,
        store_whole_genome=True,
        storage='ndarray', minoverlap=.5)
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)
    assert cover[:].sum() == 0

    # check bed file loading without roi
    cover_ = Cover.create_from_bed(
        'test',
        bedfiles=bed_shift_file,
        roi=None,
        resolution=1,
        store_whole_genome=True,
        storage='ndarray', minoverlap=.5)

    cover_.gindexer = cover.gindexer
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)
    assert cover[:].sum() == 0

def test_bed_inferred_binsize():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive.bed")


    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=bed_file,
        resolution=1,
        storage='ndarray')
    assert len(cover) == 25
    assert cover.shape == (25, 200, 1, 1)

    bed_file = os.path.join(data_path, "positive_gap.bed")
    cover = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=bed_file,
        resolution=1,
        store_whole_genome=True,
        storage='ndarray')
    assert len(cover) == 2
    assert cover.shape == (2, 50, 1, 1)

def test_bed_overreaching_ends_whole_genome():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "bed_test.bed")

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            'test',
            bedfiles=bed_file,
            roi=bed_file,
            binsize=2,
            flank=20,
            resolution=1,
            store_whole_genome=True,
            storage=store)
        assert len(cover) == 9
        assert cover.shape == (9, 2+2*20, 1, 1)
        np.testing.assert_equal(cover[0].sum(), 18)
        np.testing.assert_equal(cover[:].sum(), 9*18)


def test_bed_overreaching_ends_part_genome():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "bed_test.bed")

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            'test',
            bedfiles=bed_file,
            roi=bed_file,
            binsize=2,
            flank=2,
            resolution=1,
            store_whole_genome=False,
            storage=store)
        assert len(cover) == 9
        assert cover.shape == (9, 2+2*2, 1, 1)
        np.testing.assert_equal(cover[0].sum(), 4)
        np.testing.assert_equal(cover[:].sum(), 6*7 + 8)


def test_bed_store_whole_genome_option():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive_shift.bed")

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=bed_file,
        store_whole_genome=True,
        storage='ndarray')
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bed_file,
        roi=bed_file,
        store_whole_genome=False,
        storage='ndarray')

    assert len(cover1) == 1
    assert len(cover2) == len(cover1)
    assert cover1.shape == (1, 49, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], np.ones(cover1.shape))
    np.testing.assert_equal(cover2[:], np.ones(cover1.shape))


def test_bed_store_whole_genome_option_dataframe(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    # as pd.dataframe
    roi = pandas.read_csv(bed_file,
                          sep='\t', header=None,
                          names=['chrom', 'start', 'end',
                                 'name', 'score', 'strand'])

    print(roi.head())
    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=True,
        cache=False,
        storage='ndarray')
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=False,
        cache=True,
        storage='ndarray')

    print(cover1.gindexer[0])
    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    np.testing.assert_equal(cover1[:], np.ones(cover1.shape))

    # as bedtool
    roi = BedTool(bed_file)
    print(roi)

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=True,
        storage='ndarray')
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=False,
        cache=True,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    np.testing.assert_equal(cover1[:], np.ones(cover1.shape))

    # as interval list
    roi = [iv for iv in BedTool(bed_file)]
    print(roi)

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=True,
        storage='ndarray')
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bed_file,
        roi=roi,
        binsize=200, stepsize=200,
        store_whole_genome=False,
        cache=True,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    np.testing.assert_equal(cover1[:], np.ones(cover1.shape))


def test_bigwig_store_whole_genome_option():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")
    bwfile_ = os.path.join(data_path, "sample.bw")

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=bed_file,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile_,
        roi=bed_file,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover3 = Cover.create_from_bigwig(
        'test3',
        bigwigfiles=bwfile_,
        roi=bed_file,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        nan_to_num=False,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 1044.0
    assert cover3[:].sum() == 1044.0


def test_bigwig_store_whole_genome_option_dataframe(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")
    bwfile_ = os.path.join(data_path, "sample.bw")

    # as dataframe
    roi = pandas.read_csv(bed_file,
                          sep='\t', header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand'])

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')
    cover3 = Cover.create_from_bigwig(
        'test3',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        nan_to_num=False,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 1044.0
    assert cover3[:].sum() == 1044.0

    # as bedtool
    roi = BedTool(bed_file)

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')
    cover3 = Cover.create_from_bigwig(
        'test3',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        nan_to_num=False,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 1044.0
    assert cover3[:].sum() == 1044.0

    # as list of intervals
    roi = [iv for iv in roi]

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')
    cover3 = Cover.create_from_bigwig(
        'test3',
        bigwigfiles=bwfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        nan_to_num=False,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 1044.0
    assert cover3[:].sum() == 1044.0



def test_bam_store_whole_genome_option_dataframe(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")
    bamfile_ = os.path.join(data_path, "sample.bam")

    # as dataframe
    roi = pandas.read_csv(bed_file,
                          sep='\t', header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand'])

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 2, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 29.

    # as bedtool
    roi = BedTool(bed_file)

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 2, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 29.

    # as list of intervals
    roi = [iv for iv in roi]

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bamfile_,
        roi=roi,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        cache=True,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 2, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 29.


def test_bam_store_whole_genome_option():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")
    bamfile_ = os.path.join(data_path, "sample.bam")

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=bed_file,
        store_whole_genome=True,
        binsize=200, stepsize=200,
        storage='ndarray')
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bamfile_,
        roi=bed_file,
        store_whole_genome=False,
        binsize=200, stepsize=200,
        storage='ndarray')

    assert len(cover1) == 100
    assert len(cover2) == len(cover1)
    assert cover1.shape == (100, 200, 2, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])
    assert cover1[:].sum() == 29.


def test_cover_from_bam_sanity():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bamfile_ = os.path.join(data_path, "sample.bam")
    cover = Cover.create_from_bam(
        'test',
        bamfiles=bamfile_,
        roi=bed_file,
        binsize=200, stepsize=200,
        flank=0,
        storage='ndarray')
    cover[0]

    with pytest.raises(IndexError):
        # not interable
        cover[1.2]

    cov2 = Cover.create_from_bam(
           'test',
           bamfiles=bamfile_,
           storage='ndarray',
           store_whole_genome=True)

    assert len(cover.gindexer) == len(cover.garray.handle['data'])
    assert len(cov2.garray.handle) != len(cover.garray.handle['data'])

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bam(
            1.2,
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        # bamfile does not exist
        Cover.create_from_bam(
            'test',
            bamfiles="",
            roi=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        # bamfile does not exist
        Cover.create_from_bam(
            'test',
            bamfiles=[],
            roi=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')

    with pytest.warns(FutureWarning):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage='ndarray',
            overwrite=True)

    with pytest.warns(FutureWarning):
        Cover.create_from_bam(
            'test',
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage='ndarray',
            datatags=['asdf'])


def test_cover_from_bigwig_sanity():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "sample.bed")

    bwfile_ = os.path.join(data_path, "sample.bw")
    cover = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')
    cover[0]
    assert len(cover.gindexer) == 394
    assert len(cover.garray.handle['data']) == 394

    cover = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray',
        store_whole_genome=True)
    cover[0]
    assert len(cover.gindexer) == 394
    assert len(cover.garray.handle) == 2
    cov2 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile_,
        resolution=7,
        storage='ndarray',
        store_whole_genome=True)

    assert len(cov2.garray.handle) == 2
    assert cov2['chr1', 100, 200].shape == (1, 100, 1, 1)

    with pytest.raises(Exception):
        cov2.shape
    with pytest.raises(Exception):
        cov2[0]

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bigwig(
            1.2,
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')

    with pytest.warns(FutureWarning):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            flank=0,
            storage='ndarray',
            overwrite=True)
    with pytest.warns(FutureWarning):
        Cover.create_from_bigwig(
            'test',
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage='ndarray',
            datatags=['asdf'])


def test_cover_from_bed_sanity():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    bwfile_ = os.path.join(data_path, "scored_sample.bed")
    cover = Cover.create_from_bed(
        'test',
        bedfiles=bwfile_,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        flank=0,
        storage='ndarray')
    cover[0]
    Cover.create_from_bed(
        'test',
        bedfiles=bwfile_,
        roi=bed_file,
        binsize=200, stepsize=50,
        resolution=50,
        storage='ndarray')

    with pytest.raises(Exception):
        # name must be a string
        Cover.create_from_bed(
            1.2,
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')

    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=1,
            flank=-1,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=1, stepsize=-1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=-1, stepsize=1,
            flank=0,
            storage='ndarray')
    with pytest.raises(Exception):
        csvfile = os.path.join(data_path, 'ctcf_sample.csv')
        # must be a bed file
        Cover.create_from_bed(
            'test',
            bedfiles=csvfile,
            roi=bed_file,
            binsize=1, stepsize=1,
            storage='ndarray')
    with pytest.warns(FutureWarning):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            flank=0,
            storage='ndarray',
            overwrite=True)
    with pytest.warns(FutureWarning):
        Cover.create_from_bed(
            'test',
            bedfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage='ndarray',
            datatags=['asdf'])


def test_cover_bam_unstranded():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")


    cover = Cover.create_from_bam(
        "yeast_I_II_III.bam",
        bamfiles=bamfile_,
        roi=bed_file,
        binsize=200, stepsize=200,
        genomesize=gsize,
        stranded=False)

    np.testing.assert_equal(len(cover), 100)
    np.testing.assert_equal(cover.shape, (100, 200, 1, 1))

    # the region is read relative to the forward strand
    # read on the reverse strand
    val = np.where(cover[4] == 1)
    np.testing.assert_equal(cover[4].sum(), 1.)
    np.testing.assert_equal(val[1][0], 179)  # pos

    # two reads on the forward strand
    val = np.where(cover[13] == 1)
    np.testing.assert_equal(cover[13].sum(), 2.)
    np.testing.assert_equal(val[1], np.asarray([162, 178]))  # pos

    # the region is read relative to the reverse strand
    # for index 50
    # read on the reverse strand
    val = np.where(cover[52] == 1)
    np.testing.assert_equal(cover[52].sum(), 2.)
    np.testing.assert_equal(val[1], np.asarray([9, 89]))  # pos

    # two reads on the forward strand
    val = np.where(cover[96] == 1)
    np.testing.assert_equal(cover[96].sum(), 1.)
    np.testing.assert_equal(val[1], np.asarray([25]))  # pos


def test_cover_bam_paired_5pend():
    # sample2.bam contains paired end examples,
    # unmapped examples, unmapped mate and low quality example
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample2.bam")

    cover = Cover.create_from_bam(
        "yeast_I_II_III.bam",
        bamfiles=bamfile_,
        stranded=False,
        pairedend='5pend',
        min_mapq=30,
        store_whole_genome=True)

    assert cover.garray.handle['ref'].sum() == 4, cover.garray.handle['ref']

    # the read starts at index 6 and tlen is 39
    assert cover.garray.handle['ref'][6, 0, 0] == 1
    # another read maps to index 24
    assert cover.garray.handle['ref'][24, 0, 0] == 1


def test_cover_bam_paired_midpoint():
    # sample2.bam contains paired end examples,
    # unmapped examples, unmapped mate and low quality example
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample2.bam")


    cover = Cover.create_from_bam(
        "yeast_I_II_III.bam",
        bamfiles=bamfile_,
        stranded=False,
        pairedend='midpoint',
        min_mapq=30,
        store_whole_genome=True)

    assert cover.garray.handle['ref'].sum() == 2, cover.garray.handle['ref']
    print(cover.garray.handle['ref'])
    # the read starts at index 6 and tlen is 39
    assert cover.garray.handle['ref'][6 + 39//2, 0, 0] == 1
    # another read maps to index 34
    assert cover.garray.handle['ref'][34, 0, 0] == 1


def test_cover_bam_list(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=[bamfile_],
            roi=bed_file,
            conditions=['condition2'],
            normalizer='tpm',
            binsize=200, stepsize=200)

def test_cover_bam(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            genomesize=gsize,
            storage=store, cache=True)

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
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store, store_genome in product(['ndarray', 'hdf5', 'sparse'], [True, False]):
        # base pair binsize
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            genomesize=gsize,
            resolution=10,
            store_whole_genome=store_genome,
            storage=store, cache=True)

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


def test_load_bam_resolutionNone(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bamfile_ = os.path.join(data_path, "sample.bam")
    gsfile_ = os.path.join(data_path, 'sample.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        cover1 = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            genomesize=gsize,
            resolution=1,
            storage=store, cache=True)
        cover = Cover.create_from_bam(
            "yeast_I_II_III.bam",
            bamfiles=bamfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            genomesize=gsize,
            resolution=None,
            storage=store, cache=True)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 2, 1))

        np.testing.assert_equal(cover1[:].sum(axis=1), cover[:].sum(axis=1))


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
            roi=bed_file,
            binsize=200, stepsize=200,
            genomesize=gsize,
            storage=store,
            store_whole_genome=True,
            cache=True)

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 200, 1, 1))

        # there is one read in the region
        np.testing.assert_allclose(cover[4].sum(), 36.)
        np.testing.assert_allclose(cover[52].sum(), 2*36.)

    cover = Cover.create_from_bigwig(
        "cov",
        bigwigfiles=bwfile_,
        roi=bed_file,
        binsize=200, stepsize=200,
        genomesize=gsize,
        store_whole_genome=False, cache=True)

    np.testing.assert_equal(len(cover), 100)
    np.testing.assert_equal(cover.shape, (100, 200, 1, 1))

    # there is one read in the region
    np.testing.assert_allclose(cover[4].sum(), 36.)
    np.testing.assert_allclose(cover[52].sum(), 2*36.)

def test_load_cover_bigwig_resolution1(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bwfile_ = os.path.join(data_path, "sample.bw")

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        print(store)
        cover = Cover.create_from_bigwig(
            "cov",
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=1,
            storage=store, cache=True)

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



def test_load_cover_bigwig_resolutionNone(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bwfile_ = os.path.join(data_path, "sample.bw")

    bed_file = os.path.join(data_path, "sample.bed")

    for store in ['ndarray', 'hdf5', 'sparse']:
        # base pair binsize
        print(store)
        cover1 = Cover.create_from_bigwig(
            "cov",
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=1,
            storage=store, cache=True)

        cover = Cover.create_from_bigwig(
            "cov",
            bigwigfiles=bwfile_,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=None,
            storage=store, cache=True,
            collapser='sum')
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))

        np.testing.assert_equal(cover1[:].sum(axis=1), cover[:].sum(axis=1))


def test_load_cover_bed_binary(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    for store in ['ndarray', 'hdf5', 'sparse']:
        print('store', store)
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            mode='binary', cache=True)
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage=store,
            resolution=50,
            collapser='max',
            mode='binary', cache=True)
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50_firstdim",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage=store,
            resolution=None,
            collapser='max',
            mode='binary', cache=True)
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)

        cover = Cover.create_from_bed(
            "cov50_firstdim",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            storage=store,
            store_whole_genome=True,
            resolution=200,
            collapser='max',
            mode='binary', cache=True)
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)


def test_load_cover_bed_scored():
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    for store in ['ndarray', 'sparse']:
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            store_whole_genome=True,
            mode='score')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 5)
        np.testing.assert_equal(cover[50].sum(), 0)
        np.testing.assert_equal(cover[54].sum(), 4)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
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
            roi=bed_file,
            storage=store,
            resolution=None,
            binsize=200, stepsize=200,
            collapser='max',
            mode='score')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 5)


def test_load_cover_bed_categorical():
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    with pytest.raises(ValueError):
        # Only one bed file allowed.
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=[score_file] * 2,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            mode='categorical')

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[4], [[[[0., 0., 0., 1.]]]])

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=50,
            storage=store,
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            resolution=None,
            binsize=200, stepsize=200,
            storage=store,
            collapser='max',
            mode='categorical')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[4], [[[[0., 0., 0., 1.]]]])


def test_load_cover_bed_score_category():
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    with pytest.raises(ValueError):
        # Only one bed file allowed.
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=[score_file] * 2,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            mode='score_category')

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            mode='score_category')

        assert cover.conditions == ['1', '2', '4', '5']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[4], [[[[0., 0., 0., 1.]]]])

        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            conditions=['1', '2', '4', '5'],
            storage=store,
            mode='score_category')

        assert cover.conditions == ['1', '2', '4', '5']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[4], [[[[0., 0., 0., 1.]]]])

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=50,
            storage=store,
            mode='score_category')

        assert cover.conditions == ['1', '2', '4', '5']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            resolution=None,
            binsize=200, stepsize=200,
            storage=store,
            collapser='max',
            mode='score_category')

        assert cover.conditions == ['1', '2', '4', '5']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 4))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[4], [[[[0., 0., 0., 1.]]]])


def test_load_cover_bedgraph():
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/sample.bedgraph')

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            mode='bedgraph')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), .5)
        np.testing.assert_equal(cover[4], [[[[.5]]]])

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=50,
            storage=store,
            mode='bedgraph')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*.5)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            resolution=None,
            binsize=200, stepsize=200,
            storage=store,
            collapser='max',
            mode='bedgraph')

        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 1))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), .5)
        np.testing.assert_equal(cover[4], [[[[.5]]]])


def test_load_cover_bed_name_category():
    bed_file = pkg_resources.resource_filename('janggu', 'resources/sample.bed')
    score_file = pkg_resources.resource_filename('janggu',
                                                 'resources/scored_sample.bed')

    with pytest.raises(ValueError):
        # Only one bed file allowed.
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=[score_file] * 2,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            mode='name_category')

    for store in ['ndarray', 'sparse']:
        print(store)
        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            storage=store,
            mode='name_category')

        assert cover.conditions == ['state1', 'state2']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 2))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[3], [[[[1., 0.]]]])
        np.testing.assert_equal(cover[4], [[[[0., 1.]]]])

        cover = Cover.create_from_bed(
            "cov",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=200,
            conditions=['state1', 'state2'],
            storage=store,
            mode='name_category')

        assert cover.conditions == ['state1', 'state2']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 2))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[3], [[[[1., 0.]]]])
        np.testing.assert_equal(cover[4], [[[[0., 1.]]]])

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            binsize=200, stepsize=200,
            resolution=50,
            storage=store,
            mode='name_category')

        assert cover.conditions == ['state1', 'state2']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 4, 1, 2))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 4*1)

        cover = Cover.create_from_bed(
            "cov50",
            bedfiles=score_file,
            roi=bed_file,
            resolution=None,
            binsize=200, stepsize=200,
            storage=store,
            collapser='max',
            mode='name_category')

        assert cover.conditions == ['state1', 'state2']
        np.testing.assert_equal(len(cover), 100)
        np.testing.assert_equal(cover.shape, (100, 1, 1, 2))
        np.testing.assert_equal(cover[0].sum(), 0)
        np.testing.assert_equal(cover[4].sum(), 1)
        np.testing.assert_equal(cover[3], [[[[1., 0.]]]])
        np.testing.assert_equal(cover[4], [[[[0., 1.]]]])


def test_filter_by_region():

    roi_file = pkg_resources.resource_filename('janggu',
                                 'resources/bed_test.bed')

    roi = GenomicIndexer.create_from_file(regions=roi_file, binsize=2, stepsize=2)
    np.testing.assert_equal(len(roi), 9)

    np.testing.assert_equal((roi[0].chrom, roi[0].start, roi[0].end), ('chr1', 0, 2))
    np.testing.assert_equal((roi[-1].chrom, roi[-1].start, roi[-1].end), ('chr1', 16, 18))

    test1 = roi.filter_by_region(include='chr1', start=0, end=18)

    for i in range(len(test1)):
        np.testing.assert_equal(test1[i], roi[i])

    test2 = roi.filter_by_region(include='chr1', start=5, end=10)
    np.testing.assert_equal(len(test2), 3)
    np.testing.assert_equal((test2[0].chrom, test2[0].start, test2[0].end), ('chr1', 4, 6))
    np.testing.assert_equal((test2[1].chrom, test2[1].start, test2[1].end), ('chr1', 6, 8))
    np.testing.assert_equal((test2[2].chrom, test2[2].start, test2[2].end), ('chr1', 8, 10))

    test3 = roi.filter_by_region(include='chr1', start=5, end=11)
    np.testing.assert_equal(len(test3), 4)
    np.testing.assert_equal((test3[0].chrom, test3[0].start, test3[0].end), ('chr1', 4, 6))
    np.testing.assert_equal((test3[1].chrom, test3[1].start, test3[1].end), ('chr1', 6, 8))
    np.testing.assert_equal((test3[2].chrom, test3[2].start, test3[2].end), ('chr1', 8, 10))
    np.testing.assert_equal((test3[3].chrom, test3[3].start, test3[3].end), ('chr1', 10, 12))

    test4 = roi.filter_by_region(include='chr1', start=6, end=10)
    np.testing.assert_equal(len(test4), 2)
    np.testing.assert_equal((test4[0].chrom, test4[0].start, test4[0].end), ('chr1', 6, 8))
    np.testing.assert_equal((test4[1].chrom, test4[1].start, test4[1].end), ('chr1', 8, 10))

    test5 = roi.filter_by_region(include='chr1', start=6, end=11)
    np.testing.assert_equal(len(test5), 3)
    np.testing.assert_equal((test5[0].chrom, test5[0].start, test5[0].end), ('chr1', 6, 8))
    np.testing.assert_equal((test5[1].chrom, test5[1].start, test5[1].end), ('chr1', 8, 10))
    np.testing.assert_equal((test5[2].chrom, test5[2].start, test5[2].end), ('chr1', 10, 12))

    test6 = roi.filter_by_region(include='chr1', start=20, end=30)
    np.testing.assert_equal(len(test6), 0)


def test_plotgenometracks_bigwigs():

    roi = pkg_resources.resource_filename('janggu', 'resources/sample.bed')

    bw_file = pkg_resources.resource_filename('janggu', 'resources/sample.bw')

    cover = Cover.create_from_bigwig('coverage2',
                                     bigwigfiles=bw_file,
                                     roi=roi,
                                     binsize=200,
                                     stepsize=200,
                                     resolution=50)

    cover2 = Cover.create_from_bigwig('morecoverage',
                                      bigwigfiles=[bw_file] * 4,
                                      roi=roi,
                                      binsize=200,
                                      stepsize=200,
                                      resolution=50)

    # line plots
    a = plotGenomeTrack([cover,cover2],'chr1',16000,18000)
    a = plotGenomeTrack(cover,'chr1',16000,18000)

    a = plotGenomeTrack(LineTrack(cover),'chr1',16000,18000)

    a = plotGenomeTrack([cover,cover2],'chr1',16000,18000, plottypes=['heatmap'] * 2)
    with pytest.raises(AssertionError):
        # differing number of plottypes and coverage objects raises an error
        a = plotGenomeTrack(cover,'chr1',16000,18000, plottypes=['heatmap'] * 2)
    with pytest.raises(ValueError):
        # coverage not a sequence
        a = plotGenomeTrack(cover,'chr1',16000,18000, plottypes=['seqplot'])
    with pytest.raises(ValueError):
        # coverage not a sequence
        a = plotGenomeTrack(cover2,'chr1',16000,18000, plottypes=['seqplot'])


def test_plotgenometracks_bams():

    roi = pkg_resources.resource_filename('janggu', 'resources/sample.bed')

    bw_file = pkg_resources.resource_filename('janggu', 'resources/sample.bam')

    cover = Cover.create_from_bam('coverage',
                                  bamfiles=bw_file,
                                  roi=roi,
                                  binsize=200,
                                  stepsize=200,
                                  resolution=50)

    # line plots
    a = plotGenomeTrack(cover,'chr1',16000,18000)

    a = plotGenomeTrack([cover,cover],'chr1',16000,18000, plottypes=['heatmap'] * 2)

    a = plotGenomeTrack([HeatTrack(cover), HeatTrack(cover)],'chr1',16000,18000)
    a = plotGenomeTrack([LineTrack(cover)],'chr1',16000,18000)


def test_plotgenometracks_seqplot():

    roi = pkg_resources.resource_filename('janggu', 'resources/sample.bed')

    refgenome = pkg_resources.resource_filename('janggu',
                                               'resources/sample_genome.fa')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       roi=roi, order=1,
                                       store_whole_genome=True)

    a = plotGenomeTrack(dna,'chr1',16000,18000, plottypes=['seqplot'])

    a = plotGenomeTrack(SeqTrack(dna), 'chr1', 16000, 18000)

def test_padding_value_nan():
    variantsfile = pkg_resources.resource_filename('janggu', 'resources/pseudo_snps.vcf')
    gindexer = GenomicIndexer.create_from_file(variantsfile, None, None)
    array = np.zeros((len(gindexer), 3))

    snpcov = Cover.create_from_array('snps', array,
                                     gindexer,
                                     store_whole_genome=True,
                                     padding_value=np.nan)

    assert snpcov.shape == (6, 1, 1, 3)

    np.testing.assert_equal(snpcov['pseudo1', 650, 670][0,:,0,0],
                            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                      0., np.nan,  0., np.nan,  0.,  0.,  0.,
                                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))

    snpcov = Cover.create_from_array('snps', array,
                                     gindexer,
                                     store_whole_genome=False,
                                     padding_value=np.nan)

    assert snpcov.shape == (6, 1, 1, 3)

    np.testing.assert_equal(snpcov['pseudo1', 650, 670][0,:,0,0],
                            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                      0., np.nan,  0., np.nan,  0.,  0.,  0.,
                                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))


def test_bedgraph():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "positive.bed")
    bgfile_ = os.path.join(data_path, "positive.bedgraph")

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bgfile_,
        roi=bed_file,
        mode='bedgraph',
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bgfile_,
        roi=bed_file,
        mode='bedgraph',
        store_whole_genome=False)

    assert len(cover1) == 25
    assert len(cover2) == len(cover1)
    assert cover1.shape == (25, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

def test_fulltilebigwig():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile.bed")
    bwfile = os.path.join(data_path, "sample.bw")

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi,
        store_whole_genome=False)

    assert len(cover1) == 2
    assert len(cover2) == len(cover1)
    assert cover1.shape == (2, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

def test_fulltilebigwig2():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile2.bed")
    bwfile = os.path.join(data_path, "sample.bw")

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi,
        store_whole_genome=False)

    assert len(cover1) == 3
    assert len(cover2) == len(cover1)
    assert cover1.shape == (3, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bigwig(
        'test',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=True)
    cover2 = Cover.create_from_bigwig(
        'test2',
        bigwigfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])


def test_fulltilebam():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile.bed")
    bwfile = os.path.join(data_path, "sample.bam")

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        stranded=False,
        roi=roi,
        store_whole_genome=False)

    assert len(cover1) == 2
    assert len(cover2) == len(cover1)
    assert cover1.shape == (2, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        stranded=False,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        stranded=False,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

def test_fulltilebam2():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile2.bed")
    bwfile = os.path.join(data_path, "sample.bam")

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        roi=roi,
        stranded=False,
        store_whole_genome=False)

    assert len(cover1) == 3
    assert len(cover2) == len(cover1)
    assert cover1.shape == (3, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        stranded=False,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bam(
        'test',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        stranded=False,
        store_whole_genome=True)
    cover2 = Cover.create_from_bam(
        'test2',
        bamfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        stranded=False,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])


def test_fulltilebed():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile.bed")
    bwfile = os.path.join(data_path, "sample.bed")

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi,
        store_whole_genome=False)

    assert len(cover1) == 2
    assert len(cover2) == len(cover1)
    assert cover1.shape == (2, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=False)

    assert len(cover1) == 300
    assert len(cover2) == len(cover1)
    assert cover1.shape == (300, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

def test_fulltilebed2():

    import pkg_resources
    import os
    from janggu.data import Cover

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    roi = os.path.join(data_path, "sample_fulltile2.bed")
    bwfile = os.path.join(data_path, "sample.bed")

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi,
        store_whole_genome=False)

    assert len(cover1) == 3
    assert len(cover2) == len(cover1)
    assert cover1.shape == (3, 30000, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 200, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

    cover1 = Cover.create_from_bed(
        'test',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=True)
    cover2 = Cover.create_from_bed(
        'test2',
        bedfiles=bwfile,
        roi=roi, binsize=200,
        flank=150,
        store_whole_genome=False)

    assert len(cover1) == 450
    assert len(cover2) == len(cover1)
    assert cover1.shape == (450, 500, 1, 1)
    assert cover1.shape == cover2.shape
    np.testing.assert_equal(cover1[:], cover2[:])

