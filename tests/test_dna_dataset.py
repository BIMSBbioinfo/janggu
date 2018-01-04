import os

import numpy as np
import pkg_resources
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
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    regions = readBed(os.path.join(data_path, 'regions.bed'))
    indiv_reg = readBed(os.path.join(data_path, 'indiv_regions.bed'))

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                      regions=regions,
                                      order=order)
    idata = DnaBwDataset.fromRefGenome('itrain', refgenome=refgenome,
                                       regions=indiv_reg,
                                       order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data[indices]
    # this is the actual numpy array
    np.testing.assert_equal(dna.shape, (len(indices), pow(4, order),
                                        reglen + 2*flank - order + 1, 1))

    # this is the bwdataset
    np.testing.assert_equal(data.shape, (len(data), pow(4, order),
                                         reglen + 2*flank - order + 1, 1))

    # Check length
    np.testing.assert_equal(len(data), datalen(regions))

    # test if the two arrays (one is read from a merged bed and one
    # from individual bed regions) are the same
    np.testing.assert_equal(data[indices], idata[indices])
    np.testing.assert_equal(len(data), len(idata))


def test_read_ranges_from_file():

    reglen = 200
    flank = 150
    order = 1

    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    refgenome = os.path.join(data_path, 'genome.fa')
    regions = os.path.join(data_path, 'regions.bed')

    data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                      regions=regions,
                                      order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data[indices]
    np.testing.assert_equal(dna.shape, (len(indices), pow(4, order),
                                        reglen + 2*flank - order + 1, 1))

    np.testing.assert_equal(data.shape, (len(data), pow(4, order),
                                         reglen + 2*flank - order + 1, 1))


def test_dna_dims_order_1():
    order = 1
    dna_templ(order)


def test_dna_dims_order_2():
    order = 2
    dna_templ(order)


def revcomp(order):
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    regions = readBed(os.path.join(data_path, 'regions.bed'))

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                      regions=regions,
                                      order=order)
    rcdata = RevCompDnaBwDataset('rctrain', data)
    rcrcdata = RevCompDnaBwDataset('rcrctrain', rcdata)

    indices = [600, 500, 400]

    # actual shape of DNA
    dna = data[indices]
    rcrcdna = rcrcdata[indices]

    np.testing.assert_equal(dna, rcrcdna)


def test_revcomp_order_1():
    revcomp(1)


def test_revcomp_order_2():
    revcomp(2)


def test_revcomp_rcmatrix():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    regions = readBed(os.path.join(data_path, 'regions.bed'))

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                      regions=regions, order=1)
    rcdata = RevCompDnaBwDataset('rctrain', data)

    np.testing.assert_equal(rcdata.rcmatrix,
                            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0]]))

    data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
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
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    for order in range(1, 4):
        regions = readBed(os.path.join(data_path, 'regions.bed'))
        refgenome = os.path.join(data_path, 'genome.fa')

        data = DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                          regions=regions, order=order)
        rcdata = RevCompDnaBwDataset('rctrain', data)

        np.testing.assert_equal(np.eye(pow(4, order)),
                                np.matmul(rcdata.rcmatrix, rcdata.rcmatrix))


def test_dna_dataset_sanity():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
    regions = readBed(os.path.join(data_path, 'regions.bed'))

    refgenome = os.path.join(data_path, 'genome.fa')

    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome='',
                                   regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome='test',
                                   regions=regions, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                   regions=None, order=1)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                   regions=regions, order=0)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                   regions=regions, flank=-1)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                   regions=regions, reglen=0)
    with pytest.raises(Exception):
        DnaBwDataset.fromRefGenome('train', refgenome=refgenome,
                                   regions=regions, stride=0)


def test_read_dna_from_fasta_order_1():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'oct4.fa')
    data = DnaBwDataset.fromFasta('train', fastafile=filename, order=order)

    np.testing.assert_equal(len(data), 4)
    np.testing.assert_equal(data.shape, (len(data), pow(4, order), 200, 1))
    np.testing.assert_equal(data[0].shape, (1, 4, 200, 1))

    # correctness of the first sequence - uppercase
    # cacagcagag
    np.testing.assert_equal(data[0][0, 0, :5, 0], np.asarray([0, 1, 0, 1, 0]))
    np.testing.assert_equal(data[0][0, 1, :5, 0], np.asarray([1, 0, 1, 0, 0]))
    np.testing.assert_equal(data[0][0, 3, :5, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[0][0, 2, :5, 0], np.asarray([0, 0, 0, 0, 1]))

    # correctness of the second sequence - uppercase
    # cncact
    np.testing.assert_equal(data[1][0, 0, :5, 0], np.asarray([0, 0, 0, 1, 0]))
    np.testing.assert_equal(data[1][0, 1, :5, 0], np.asarray([1, 0, 1, 0, 1]))
    np.testing.assert_equal(data[1][0, 2, :5, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[1][0, 3, :5, 0], np.asarray([0, 0, 0, 0, 0]))

    # correctness of the third sequence - lowercase
    # aagtta
    np.testing.assert_equal(data[2][0, 0, :5, 0], np.asarray([1, 1, 0, 0, 0]))
    np.testing.assert_equal(data[2][0, 1, :5, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[2][0, 2, :5, 0], np.asarray([0, 0, 1, 0, 0]))
    np.testing.assert_equal(data[2][0, 3, :5, 0], np.asarray([0, 0, 0, 1, 1]))

    # correctness of the third sequence - lowercase
    # cnaagt
    np.testing.assert_equal(data[3][0, 0, :5, 0], np.asarray([0, 0, 1, 1, 0]))
    np.testing.assert_equal(data[3][0, 1, :5, 0], np.asarray([1, 0, 0, 0, 0]))
    np.testing.assert_equal(data[3][0, 2, :5, 0], np.asarray([0, 0, 0, 0, 1]))
    np.testing.assert_equal(data[3][0, 3, :5, 0], np.asarray([0, 0, 0, 0, 0]))


def test_read_dna_from_fasta_order_2():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    order = 2
    filename = os.path.join(data_path, 'oct4.fa')
    data = DnaBwDataset.fromFasta('train', fastafile=filename, order=order)

    np.testing.assert_equal(len(data), 4)
    np.testing.assert_equal(data.shape, (len(data), 16, 199, 1))

    # correctness of the first sequence - uppercase
    # cacagc
    print(data[0][0, :, :5, 0])
    np.testing.assert_equal(data[0][0, 4, 0, 0], 1)
    np.testing.assert_equal(data[0][0, 1, 1, 0], 1)
    np.testing.assert_equal(data[0][0, 4, 2, 0], 1)
    np.testing.assert_equal(data[0][0, 2, 3, 0], 1)
    np.testing.assert_equal(data[0][0, 9, 4, 0], 1)
    np.testing.assert_equal(data[0][:, :, :5, :].sum(), 5)

    # correctness of the second sequence - uppercase
    # cncact
    # np.testing.assert_equal(data[0][5, 0, 0], 1)
    # np.testing.assert_equal(data[0][2, 1, 0], 1)
    np.testing.assert_equal(data[1][0, 4, 2, 0], 1)
    np.testing.assert_equal(data[1][0, 1, 3, 0], 1)
    np.testing.assert_equal(data[1][0, 7, 4, 0], 1)
    np.testing.assert_equal(data[1][:, :, :5, :].sum(), 3)

    # correctness of the third sequence - lowercase
    # aagtta
    np.testing.assert_equal(data[2][0, 0, 0, 0], 1)
    np.testing.assert_equal(data[2][0, 2, 1, 0], 1)
    np.testing.assert_equal(data[2][0, 11, 2, 0], 1)
    np.testing.assert_equal(data[2][0, 15, 3, 0], 1)
    np.testing.assert_equal(data[2][0, 12, 4, 0], 1)
    np.testing.assert_equal(data[2][0, :, :5, :].sum(), 5)

    # correctness of the third sequence - lowercase
    # cnaagt
    np.testing.assert_equal(data[3][0, 0, 2, 0], 1)
    np.testing.assert_equal(data[3][0, 2, 3, 0], 1)
    np.testing.assert_equal(data[3][0, 11, 4, 0], 1)
    np.testing.assert_equal(data[3][0, :, :5, :].sum(), 3)
