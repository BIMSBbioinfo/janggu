import os

import matplotlib
import numpy as np
import pkg_resources
import pytest
from HTSeq import BED_Reader
from keras.layers import Input
from keras.models import Model

from janggo.data import Dna
from janggo.layers import Complement
from janggo.layers import Reverse
from janggo.utils import NMAP
from janggo.utils import complement_permmatrix
from janggo.utils import sequences_from_fasta

matplotlib.use('AGG')

reglen = 200
flank = 150
stepsize = 50


def _seqToOneHot(seqs):
    onehots = []
    for seq in seqs:
        onehots.append(_getOneHotSeq(seq.seq))
    return np.concatenate(onehots, axis=0)


def _getOneHotSeq(seq):
    m = len(seq.alphabet.letters)
    n = len(seq)
    result = np.zeros((1, n, m, 1), dtype="float32")
    for i in range(len(seq)):
        result[0, i, NMAP[seq[i]], 0] = 1
    return result


def datalen(bed_file):
    reglens = 0
    reader = BED_Reader(bed_file)
    for reg in reader:
        reglens += (reg.iv.end - reg.iv.start - reglen + stepsize)//stepsize
    return reglens


def dna_templ(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_merged = os.path.join(data_path, 'regions.bed')
    bed_indiv = os.path.join(data_path, 'indiv_regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = Dna.create_from_refgenome('train', refgenome=refgenome,
                                     regions=bed_merged,
                                     storage='ndarray',
                                     reglen=reglen,
                                     flank=flank,
                                     order=order)
    idata = Dna.create_from_refgenome('itrain', refgenome=refgenome,
                                      regions=bed_indiv,
                                      reglen=reglen,
                                      flank=flank,
                                      storage='ndarray',
                                      order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data[indices]
    # this is the actual numpy array
    np.testing.assert_equal(dna.shape, (len(indices),
                                        reglen + 2*flank - order + 1,
                                        pow(4, order), 1))

    # this is the bwdataset
    np.testing.assert_equal(data.shape, (len(data),
                                         reglen + 2*flank - order + 1,
                                         pow(4, order), 1))

    # Check length
    np.testing.assert_equal(len(data), datalen(bed_merged))

    # test if the two arrays (one is read from a merged bed and one
    # from individual bed regions) are the same
    np.testing.assert_equal(data[indices], idata[indices])
    np.testing.assert_equal(len(data), len(idata))


def test_read_ranges_from_file():

    order = 1

    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    refgenome = os.path.join(data_path, 'genome.fa')
    regions = os.path.join(data_path, 'regions.bed')

    data = Dna.create_from_refgenome('train', refgenome=refgenome,
                                     regions=regions,
                                     storage='ndarray',
                                     reglen=reglen,
                                     flank=flank,
                                     order=order)

    indices = [1, 600, 1000]

    # Check correctness of idna4idx
    np.testing.assert_equal(data.idna4idx(indices).shape, (len(indices),
                            reglen + 2*flank - order + 1))

    # actual shape of DNA
    dna = data[indices]
    np.testing.assert_equal(dna.shape, (len(indices),
                                        reglen + 2*flank - order + 1,
                                        pow(4, order), 1))

    np.testing.assert_equal(data.shape, (len(data),
                                         reglen + 2*flank - order + 1,
                                         pow(4, order), 1))


def test_dna_dims_order_1():
    order = 1
    dna_templ(order)


def test_dna_dims_order_2():
    order = 2
    dna_templ(order)


def reverse_layer(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = Dna.create_from_refgenome('train', refgenome=refgenome,
                                     regions=bed_file,
                                     storage='ndarray',
                                     reglen=reglen,
                                     flank=flank,
                                     order=order)

    dna_in = Input(shape=data.shape[1:], name='dna')
    rdna_layer = Reverse()(dna_in)

    rmod = Model(dna_in, rdna_layer)

    indices = [600, 500, 400]

    # actual shape of DNA
    dna = data[indices]
    np.testing.assert_equal(dna[:, ::-1, :, :], rmod.predict(dna))


def complement_layer(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = Dna.create_from_refgenome('train', refgenome=refgenome,
                                     regions=bed_file,
                                     storage='ndarray',
                                     reglen=reglen,
                                     flank=flank,
                                     order=order)

    dna_in = Input(shape=data.shape[1:], name='dna')
    cdna_layer = Complement()(dna_in)
    cmod = Model(dna_in, cdna_layer)

    indices = [600, 500, 400]

    # actual shape of DNA
    dna = data[indices]

    cdna = cmod.predict(dna)
    ccdna = cmod.predict(cdna)

    with pytest.raises(Exception):
        np.testing.assert_equal(dna, cdna)
    np.testing.assert_equal(dna, ccdna)


def test_reverse_order_1():
    reverse_layer(1)


def test_reverse_order_2():
    reverse_layer(2)


def test_complement_order_1():
    complement_layer(1)


def test_complement_order_2():
    complement_layer(2)


def test_revcomp_rcmatrix():

    rcmatrix = complement_permmatrix(1)

    np.testing.assert_equal(rcmatrix,
                            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0]]))

    rcmatrix = complement_permmatrix(2)

    np.testing.assert_equal(rcmatrix[0],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1]))
    np.testing.assert_equal(rcmatrix[4],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 1, 0]))
    np.testing.assert_equal(rcmatrix[8],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 1, 0, 0]))
    np.testing.assert_equal(rcmatrix[12],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      1, 0, 0, 0]))

    np.testing.assert_equal(rcmatrix[1],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcmatrix[5],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcmatrix[9],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0]))
    np.testing.assert_equal(rcmatrix[13],
                            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 0, 0]))


def test_rcmatrix_identity():

    for order in range(1, 4):

        rcmatrix = complement_permmatrix(order)

        np.testing.assert_equal(np.eye(pow(4, order)),
                                np.matmul(rcmatrix, rcmatrix))


def test_dna_dataset_sanity(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    with pytest.raises(Exception):
        # name must be a string
        Dna.create_from_refgenome(1.23, refgenome='',
                                  storage='ndarray',
                                  regions=bed_file, order=1)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome='',
                                  storage='ndarray',
                                  regions=bed_file, order=1)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome='test',
                                  storage='ndarray',
                                  regions=bed_file, order=1)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  regions=None, order=1)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  regions=bed_file, order=0)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  regions=bed_file, flank=-1)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  regions=bed_file, reglen=0)
    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  regions=bed_file, stepsize=0)

    with pytest.raises(Exception):
        Dna.create_from_refgenome('train', refgenome=refgenome,
                                  storage='step',
                                  regions=bed_file, order=1,
                                  cachedir=tmpdir.strpath)

    assert not os.path.exists(os.path.join(tmpdir.strpath, 'train',
                                           'genome.fa', 'chr1..nmm'))

    Dna.create_from_refgenome('train', refgenome=refgenome,
                              storage='hdf5',
                              regions=bed_file, order=1,
                              cachedir=tmpdir.strpath)

    assert os.path.exists(os.path.join(tmpdir.strpath, 'train',
                                       'storage.unstranded.h5'))


def test_read_dna_from_fasta_order_1(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'oct4.fa')
    data = Dna.create_from_fasta('train', fastafile=filename,
                                 order=order)

    np.testing.assert_equal(len(data), 4)
    np.testing.assert_equal(data.shape, (len(data), 200, pow(4, order), 1))
    np.testing.assert_equal(data[0].shape, (1, 200, 4, 1))

    # correctness of the first sequence - uppercase
    # cacagcagag
    np.testing.assert_equal(data[0][0, :5, 0, 0], np.asarray([0, 1, 0, 1, 0]))
    np.testing.assert_equal(data[0][0, :5, 1, 0], np.asarray([1, 0, 1, 0, 0]))
    np.testing.assert_equal(data[0][0, :5, 3, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[0][0, :5, 2, 0], np.asarray([0, 0, 0, 0, 1]))

    # correctness of the second sequence - uppercase
    # cncact
    np.testing.assert_equal(data[1][0, :5, 0, 0], np.asarray([0, 0, 0, 1, 0]))
    np.testing.assert_equal(data[1][0, :5, 1, 0], np.asarray([1, 0, 1, 0, 1]))
    np.testing.assert_equal(data[1][0, :5, 2, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[1][0, :5, 3, 0], np.asarray([0, 0, 0, 0, 0]))

    # correctness of the third sequence - lowercase
    # aagtta
    np.testing.assert_equal(data[2][0, :5, 0, 0], np.asarray([1, 1, 0, 0, 0]))
    np.testing.assert_equal(data[2][0, :5, 1, 0], np.asarray([0, 0, 0, 0, 0]))
    np.testing.assert_equal(data[2][0, :5, 2, 0], np.asarray([0, 0, 1, 0, 0]))
    np.testing.assert_equal(data[2][0, :5, 3, 0], np.asarray([0, 0, 0, 1, 1]))

    # correctness of the third sequence - lowercase
    # cnaagt
    np.testing.assert_equal(data[3][0, :5, 0, 0], np.asarray([0, 0, 1, 1, 0]))
    np.testing.assert_equal(data[3][0, :5, 1, 0], np.asarray([1, 0, 0, 0, 0]))
    np.testing.assert_equal(data[3][0, :5, 2, 0], np.asarray([0, 0, 0, 0, 1]))
    np.testing.assert_equal(data[3][0, :5, 3, 0], np.asarray([0, 0, 0, 0, 0]))


def test_read_dna_from_fasta_order_2(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    order = 2
    filename = os.path.join(data_path, 'oct4.fa')
    data = Dna.create_from_fasta('train', fastafile=filename,
                                 order=order,
                                 cachedir=tmpdir.strpath)

    np.testing.assert_equal(len(data), 4)
    np.testing.assert_equal(data.shape, (len(data),  199, 16, 1))

    # correctness of the first sequence - uppercase
    # cacagc
    np.testing.assert_equal(data[0][0, 0, 4, 0], 1)
    np.testing.assert_equal(data[0][0, 1, 1, 0], 1)
    np.testing.assert_equal(data[0][0, 2, 4, 0], 1)
    np.testing.assert_equal(data[0][0, 3, 2, 0], 1)
    np.testing.assert_equal(data[0][0, 4, 9, 0], 1)
    np.testing.assert_equal(data[0][:, :5, :, :].sum(), 5)

    # correctness of the second sequence - uppercase
    # cncact
    # np.testing.assert_equal(data[0][5, 0, 0], 1)
    # np.testing.assert_equal(data[0][2, 1, 0], 1)
    np.testing.assert_equal(data[1][0, 2, 4, 0], 1)
    np.testing.assert_equal(data[1][0, 3, 1, 0], 1)
    np.testing.assert_equal(data[1][0, 4, 7, 0], 1)
    np.testing.assert_equal(data[1][:, :5, :, :].sum(), 3)

    # correctness of the third sequence - lowercase
    # aagtta
    np.testing.assert_equal(data[2][0, 0, 0, 0], 1)
    np.testing.assert_equal(data[2][0, 1, 2, 0], 1)
    np.testing.assert_equal(data[2][0, 2, 11, 0], 1)
    np.testing.assert_equal(data[2][0, 3, 15, 0], 1)
    np.testing.assert_equal(data[2][0, 4, 12, 0], 1)
    np.testing.assert_equal(data[2][0, :5, :, :].sum(), 5)

    # correctness of the third sequence - lowercase
    # cnaagt
    np.testing.assert_equal(data[3][0, 2, 0, 0], 1)
    np.testing.assert_equal(data[3][0, 3, 2, 0], 1)
    np.testing.assert_equal(data[3][0, 4, 11, 0], 1)
    np.testing.assert_equal(data[3][0, :5, :, :].sum(), 3)


def test_stemcell_onehot_identity():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    filename = os.path.join(data_path, 'stemcells.fa')
    data = Dna.create_from_fasta('dna', fastafile=filename)

    oh = _seqToOneHot(sequences_from_fasta(filename))

    np.testing.assert_equal(data[:], oh)


def _dna_with_region_strandedness(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed = os.path.join(data_path, 'region_w_strand.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = Dna.create_from_refgenome('train', refgenome=refgenome,
                                     regions=bed,
                                     storage='ndarray',
                                     reglen=reglen,
                                     flank=flank,
                                     order=order)

    dna_in = Input(shape=data.shape[1:], name='dna')
    rdna_layer = Reverse()(dna_in)
    rcdna_layer = Complement()(rdna_layer)
    mod = Model(dna_in, rcdna_layer)

    dna = data[[0, 1]]
    rcdna = mod.predict(data)

    np.testing.assert_equal(data.shape, (2,
                                         reglen + 2*flank - order + 1,
                                         pow(4, order), 1))
    np.testing.assert_equal(dna.shape, rcdna.shape)

    np.testing.assert_equal(data[0], rcdna[1:2])
    np.testing.assert_equal(data[1], rcdna[:1])
    np.testing.assert_equal(data[:], rcdna[::-1])

    with pytest.raises(Exception):
        np.testing.assert_equal(data[0], data[1])

    with pytest.raises(Exception):
        np.testing.assert_equal(rcdna[0], rcdna[1])


def test_dna_with_region_strandedness():
    _dna_with_region_strandedness(1)
    _dna_with_region_strandedness(2)
