import numpy as np
import pytest
from genomeutils.regions import readBed

from bluewhalecore.data.dna import DnaBwDataset

reglen = 200
flank = 150
stride = 50


def datalen(regions):
    reglens = ((regions.end - regions.start - reglen + stride)//stride).sum()
    return reglens


def dna_templ(order):
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions,
                        order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data.getData(indices)
    np.testing.assert_equal(dna.shape, (len(indices), 1, pow(4, order),
                                        reglen + 2*flank - order + 1))

    np.testing.assert_equal(data.shape, (1, pow(4, order),
                                         reglen + 2*flank - order + 1))

    # Check length
    np.testing.assert_equal(len(data), datalen(regions))


def test_dna_dims_order_1():
    order = 1
    dna_templ(order)


def test_dna_dims_order_2():
    order = 2
    dna_templ(order)


def revcomp(order):
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions,
                        order=order)

    indices = [600, 500, 400]

    # actual shape of DNA
    dna = data.getData(indices)
    rcrcdna = data.as_revcomp(data.as_revcomp(dna))

    np.testing.assert_equal(dna, rcrcdna)


def test_revcomp_order_1():
    revcomp(1)


def test_revcomp_order_2():
    revcomp(2)


def test_revcomp_rcmatrix():
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions, order=1)
    np.testing.assert_equal(data.rcmatrix,
                            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0]]))

    data = DnaBwDataset('train', refgenome=refgenome, regions=regions, order=2)
    np.testing.assert_equal(data.rcmatrix[0],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1]))
    np.testing.assert_equal(data.rcmatrix[4],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 1, 0]))
    np.testing.assert_equal(data.rcmatrix[8],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 1, 0, 0]))
    np.testing.assert_equal(data.rcmatrix[12],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      1, 0, 0, 0]))

    np.testing.assert_equal(data.rcmatrix[1],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(data.rcmatrix[5],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(data.rcmatrix[9],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(data.rcmatrix[13],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 0, 0]))


def test_rcmatrix_identity():

    for order in range(1, 4):
        regions = readBed('resources/regions.bed')
        refgenome = 'resources/genome.fa'

        data = DnaBwDataset('train', refgenome=refgenome, regions=regions,
                            order=order)
        np.testing.assert_equal(np.eye(pow(4, order)),
                                np.matmul(data.rcmatrix, data.rcmatrix))


def test_dna_dataset_sanity():
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome='', regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome='test', regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome=refgenome, regions=None, order=1)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome=refgenome, regions=regions, order=0)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome=refgenome, regions=regions, flank=-1)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome=refgenome, regions=regions, reglen=0)
    with pytest.raises(Exception):
        DnaBwDataset('train', refgenome=refgenome, regions=regions, stride=0)
