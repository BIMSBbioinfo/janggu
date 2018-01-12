import os

import numpy as np
import pandas
import pkg_resources
# from genomeutils.refgenome import getGenomeSize
from HTSeq import GenomicInterval

from bluewhalecore.data import CoverageBwDataset


def test_load_coveragedataset_bam_stranded(tmpdir):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    regions = pandas.DataFrame({'chr': ['chrIII'],
                                'start': [217330],
                                'end': [217350]})
    iv = GenomicInterval("chrIII", 217330, 217350, "+")

    flank = 4

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # base pair resolution
        # print(store)
        cvdata = CoverageBwDataset.from_bam("yeast_I_II_III.bam",
                                            bam=bamfile_,
                                            regions=regions,
                                            genomesize=gsize,
                                            resolution=1, stride=1,
                                            flank=flank, stranded=True,
                                            storage=store,
                                            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 20)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2, 2*flank + 1, 1))
        civ = cvdata.gindexer[0]
        np.testing.assert_equal((iv.chrom, iv.start, iv.start + 1),
                                (civ.chrom, civ.start, civ.end))

        np.testing.assert_equal(sum(list(cvdata.covers[0][iv])), 2)
        np.testing.assert_equal(cvdata.covers[0][iv].sum(), 2)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # 20 bp resolution
        # print(store)
        cvdata = CoverageBwDataset.from_bam("yeast_I_II_III.bam",
                                            bam=bamfile_,
                                            regions=regions,
                                            genomesize=gsize,
                                            resolution=20, stride=20,
                                            flank=flank, stranded=True,
                                            storage=store,
                                            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 1)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 2, 2*flank + 1, 1))

        np.testing.assert_equal(sum(list(cvdata.covers[0][iv])), 2)
        np.testing.assert_equal(cvdata.covers[0][iv].sum(), 2)

        civ = cvdata.gindexer[0]
        np.testing.assert_equal((iv.chrom, iv.start, iv.end),
                                (civ.chrom, civ.start, civ.end))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 2, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 2.0)
        np.testing.assert_equal(x[0, 1, flank, 0], 0.0)

        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)


def test_load_coveragedataset_bam_unstranded(tmpdir):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    bamfile_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsfile_ = os.path.join(data_path, 'sacCer3.chrom.sizes')

    content = pandas.read_csv(gsfile_, sep='\t', names=['chr', 'length'],
                              index_col='chr')

    gsize = content.to_dict()['length']

    regions = pandas.DataFrame({'chr': ['chrIII'],
                                'start': [217330],
                                'end': [217350]})

    flank = 4
    iv = GenomicInterval("chrIII", 217330, 217350, ".")

    for store in ['step', 'memmap', 'ndarray', 'hdf5']:
        # base pair resolution
        print(store)
        cvdata = CoverageBwDataset.from_bam("yeast_I_II_III.bam",
                                            bam=bamfile_,
                                            regions=regions,
                                            genomesize=gsize,
                                            resolution=1, stride=1,
                                            flank=flank, stranded=False,
                                            storage=store,
                                            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 20)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))
        print(list(cvdata.covers[0][iv]))
        civ = cvdata.gindexer[0]
        np.testing.assert_equal((iv.chrom, iv.start, iv.start + 1, iv.strand),
                                (civ.chrom, civ.start, civ.end, civ.strand))

        np.testing.assert_equal(sum(list(cvdata.covers[0][iv])), 2)
        np.testing.assert_equal(cvdata.covers[0][iv].sum(), 2)

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

    for store in ['step', 'ndarray', 'memmap', 'hdf5']:
        # 20 bp resolution
        print(store)
        cvdata = CoverageBwDataset.from_bam("yeast_I_II_III.bam",
                                            bam=bamfile_,
                                            regions=regions,
                                            genomesize=gsize,
                                            resolution=20, stride=20,
                                            flank=flank, stranded=False,
                                            storage=store,
                                            cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 1)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))

        np.testing.assert_equal(sum(list(cvdata.covers[0][iv])), 2)
        np.testing.assert_equal(cvdata.covers[0][iv].sum(), 2)

        civ = cvdata.gindexer[0]
        np.testing.assert_equal((iv.chrom, iv.start, iv.end, iv.strand),
                                (civ.chrom, civ.start, civ.end, civ.strand))

        x = cvdata[0]
        np.testing.assert_equal(x.shape, (1, 1, 2*flank + 1, 1))
        np.testing.assert_equal(x[0, 0, flank, 0], 2.0)

        # check if slicing works
        np.testing.assert_equal(cvdata[:].shape, cvdata.shape)
