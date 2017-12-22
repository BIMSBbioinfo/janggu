import os

import numpy as np
import pandas
import pkg_resources
from genomeutils.refgenome import getGenomeSize
from HTSeq import GenomicInterval

from bluewhalecore.data import CoverageBwDataset


def test_load_coveragedataset_bam_stranded(tmpdir):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    file_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsize = getGenomeSize('sacCer3', outputdir=tmpdir.strpath)

    regions = pandas.DataFrame({'chr': ['chrIII'],
                                'start': [217330],
                                'end': [217350]})
    iv = GenomicInterval("chrIII", 217330, 217350, "+")

    flank = 4

    for store in ['step', 'memmap', 'ndarray']:
        # base pair resolution
        cvdata = CoverageBwDataset.fromBam("yeast_I_II_III.bam",
                                           bam=file_,
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

    for store in ['step', 'memmap', 'ndarray']:
        cvdata = CoverageBwDataset.fromBam("yeast_I_II_III.bam",
                                           bam=file_,
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


def test_load_coveragedataset_bam_unstranded(tmpdir):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    file_ = os.path.join(data_path, "yeast_I_II_III.bam")
    gsize = getGenomeSize('sacCer3', outputdir=tmpdir.strpath)

    regions = pandas.DataFrame({'chr': ['chrIII'],
                                'start': [217330],
                                'end': [217350]})

    flank = 4
    iv = GenomicInterval("chrIII", 217330, 217350, ".")

    for store in ['step', 'memmap', 'ndarray']:
        # base pair resolution
        cvdata = CoverageBwDataset.fromBam("yeast_I_II_III.bam",
                                           bam=file_,
                                           regions=regions,
                                           genomesize=gsize,
                                           resolution=1, stride=1,
                                           flank=flank, stranded=False,
                                           storage=store,
                                           cachedir=tmpdir.strpath)

        np.testing.assert_equal(len(cvdata), 20)
        np.testing.assert_equal(cvdata.shape, (len(cvdata), 1, 2*flank + 1, 1))
        civ = cvdata.gindexer[0]
        np.testing.assert_equal((iv.chrom, iv.start, iv.start + 1, iv.strand),
                                (civ.chrom, civ.start, civ.end, civ.strand))

        np.testing.assert_equal(sum(list(cvdata.covers[0][iv])), 2)
        np.testing.assert_equal(cvdata.covers[0][iv].sum(), 2)

    for store in ['step', 'memmap', 'ndarray']:
        cvdata = CoverageBwDataset.fromBam("yeast_I_II_III.bam",
                                           bam=file_,
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
