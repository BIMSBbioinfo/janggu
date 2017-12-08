import numpy as np
import pytest
from genomeutils.regions import readBed

from bluewhalecore.data.dna import DnaBwDataset
from bluewhalecore.data.dna import RevCompDnaBwDataset

reglen = 200
flank = 150
stride = 50


def datalen(regions):
    reglens = ((regions.end - regions.start - reglen + stride)//stride).sum()
    return reglens


def dna_templ(order):
    regions = readBed('resources/regions.bed')
    indiv_reg = readBed('resources/indiv_regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                    refgenome=refgenome,
                                                    regions=regions,
                                                    order=order)
    idata = DnaBwDataset.extractRegionsFromRefGenome('itrain',
                                                     refgenome=refgenome,
                                                     regions=indiv_reg,
                                                     order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data.getData(indices)
    # this is the actual numpy array
    np.testing.assert_equal(dna.shape, (len(indices), pow(4, order),
                                        reglen + 2*flank - order + 1, 1))

    # this is the bwdataset
    np.testing.assert_equal(data.shape, (pow(4, order),
                                         reglen + 2*flank - order + 1, 1))

    # Check length
    np.testing.assert_equal(len(data), datalen(regions))

    # test if the two arrays (one is read from a merged bed and one
    # from individual bed regions) are the same
    np.testing.assert_equal(data.getData(indices), idata.getData(indices))
    np.testing.assert_equal(len(data), len(idata))


def test_read_ranges_from_file():

    reglen = 200
    flank = 150
    order = 1

    refgenome = 'resources/genome.fa'
    regions = 'resources/regions.bed'

    data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                    refgenome=refgenome,
                                                    regions=regions,
                                                    order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data.getData(indices)
    np.testing.assert_equal(dna.shape, (len(indices), pow(4, order),
                                        reglen + 2*flank - order + 1, 1))

    np.testing.assert_equal(data.shape, (pow(4, order),
                                         reglen + 2*flank - order + 1, 1))


def test_dna_dims_order_1():
    order = 1
    dna_templ(order)


def test_dna_dims_order_2():
    order = 2
    dna_templ(order)


def revcomp(order):
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                    refgenome=refgenome,
                                                    regions=regions,
                                                    order=order)
    rcdata = RevCompDnaBwDataset('rctrain', data)
    rcrcdata = RevCompDnaBwDataset('rcrctrain', rcdata)

    indices = [600, 500, 400]

    # actual shape of DNA
    dna = data.getData(indices)
    rcrcdna = rcrcdata.getData(indices)

    np.testing.assert_equal(dna, rcrcdna)


def test_revcomp_order_1():
    revcomp(1)


def test_revcomp_order_2():
    revcomp(2)


def test_revcomp_rcmatrix():
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                    refgenome=refgenome,
                                                    regions=regions, order=1)
    rcdata = RevCompDnaBwDataset('rctrain', data)

    np.testing.assert_equal(rcdata.rcmatrix,
                            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0]]))

    data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                    refgenome=refgenome,
                                                    regions=regions, order=2)
    rcdata = RevCompDnaBwDataset('rctrain', data)

    np.testing.assert_equal(rcdata.rcmatrix[0],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1]))
    np.testing.assert_equal(rcdata.rcmatrix[4],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 1, 0]))
    np.testing.assert_equal(rcdata.rcmatrix[8],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 1, 0, 0]))
    np.testing.assert_equal(rcdata.rcmatrix[12],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      1, 0, 0, 0]))

    np.testing.assert_equal(rcdata.rcmatrix[1],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcdata.rcmatrix[5],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcdata.rcmatrix[9],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcdata.rcmatrix[13],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 0, 0]))


def test_rcmatrix_identity():

    for order in range(1, 4):
        regions = readBed('resources/regions.bed')
        refgenome = 'resources/genome.fa'

        data = DnaBwDataset.extractRegionsFromRefGenome('train',
                                                        refgenome=refgenome,
                                                        regions=regions,
                                                        order=order)
        rcdata = RevCompDnaBwDataset('rctrain', data)

        np.testing.assert_equal(np.eye(pow(4, order)),
                                np.matmul(rcdata.rcmatrix, rcdata.rcmatrix))


def test_dna_dataset_sanity():
    regions = readBed('resources/regions.bed')

    refgenome = 'resources/genome.fa'

    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome='',
                                                 regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome='test',
                                                 regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome=refgenome,
                                                 regions=None, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome=refgenome,
                                                 regions=regions, order=0)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome=refgenome,
                                                 regions=regions, flank=-1)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome=refgenome,
                                                 regions=regions, reglen=0)
    with pytest.raises(Exception):
        DnaBwDataset.extractRegionsFromRefGenome('train', refgenome=refgenome,
                                                 regions=regions, stride=0)


def test_read_dna_from_fasta():

    order = 1
    filename = 'resources/oct4.fa'
    data = DnaBwDataset.fromFasta('train', fastafile=filename, order=1)

    np.testing.assert_equal(len(data), 3997)
    np.testing.assert_equal(data.shape, (pow(4, order), 200, 1))
