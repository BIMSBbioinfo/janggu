import numpy as np
from genomeutils.regions import readBed
from bluewhalecore.data.dna_dataset import DnaBwDataset

reglen = 200
flank = 150
stride = 50


def datalen(regions):
    # check the lengths, indices, etc.
    # order = 1
    # flank = 150
    #reglen = 200
    #stride = 50
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


def test_dna_order_1():

    order = 1
    dna_templ(order)


def test_dna_order_2():

    order = 2
    dna_templ(order)
