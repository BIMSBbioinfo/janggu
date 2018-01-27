import os

import numpy as np
import pkg_resources
import pytest
from HTSeq import BED_Reader

from janggo.data import DnaDataset
from janggo.data import RevCompDnaDataset
from janggo.data import sequences_from_fasta
from janggo.data.utils import NMAP

reglen = 200
flank = 150
stride = 50


# adopted from secomo
def _seqToOneHot(seqs):
    onehots = []
    for seq in seqs:
        onehots.append(_getOneHotSeq(seq.seq))
    return np.concatenate(onehots, axis=0)


def _getOneHotSeq(seq):
    m = len(seq.alphabet.letters)
    n = len(seq)
    result = np.zeros((1, m, n, 1), dtype="float32")
    for i in range(len(seq)):
        result[0, NMAP[seq[i]], i, 0] = 1
    return result


def datalen(bed_file):
    reglens = 0
    reader = BED_Reader(bed_file)
    for reg in reader:
        reglens += (reg.iv.end - reg.iv.start - reglen + stride)//stride
    return reglens


def dna_templ(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_merged = os.path.join(data_path, 'regions.bed')
    bed_indiv = os.path.join(data_path, 'indiv_regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            regions=bed_merged,
                                            storage='ndarray',
                                            order=order)
    idata = DnaDataset.create_from_refgenome('itrain', refgenome=refgenome,
                                             regions=bed_indiv,
                                             storage='ndarray',
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

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            regions=regions,
                                            storage='ndarray',
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
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            regions=bed_file,
                                            storage='ndarray',
                                            order=order)
    rcdata = RevCompDnaDataset('rctrain', data)
    rcrcdata = RevCompDnaDataset('rcrctrain', rcdata)

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
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            storage='ndarray',
                                            regions=bed_file, order=1)
    rcdata = RevCompDnaDataset('rctrain', data)

    np.testing.assert_equal(rcdata.rcmatrix,
                            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0]]))

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            storage='ndarray',
                                            regions=bed_file, order=2)
    rcdata = RevCompDnaDataset('rctrain', data)

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
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    for order in range(1, 4):
        bed_file = os.path.join(data_path, 'regions.bed')
        refgenome = os.path.join(data_path, 'genome.fa')

        data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                                storage='ndarray',
                                                regions=bed_file, order=order)
        rcdata = RevCompDnaDataset('rctrain', data)

        np.testing.assert_equal(np.eye(pow(4, order)),
                                np.matmul(rcdata.rcmatrix, rcdata.rcmatrix))


def test_dna_dataset_sanity(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')
    bed_file = os.path.join(data_path, 'regions.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome='',
                                         storage='ndarray',
                                         regions=bed_file, order=1)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome='test',
                                         storage='ndarray',
                                         regions=bed_file, order=1)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='ndarray',
                                         regions=None, order=1)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='ndarray',
                                         regions=bed_file, order=0)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='ndarray',
                                         regions=bed_file, flank=-1)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='ndarray',
                                         regions=bed_file, reglen=0)
    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='ndarray',
                                         regions=bed_file, stride=0)

    with pytest.raises(Exception):
        DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                         storage='step',
                                         regions=bed_file, order=1,
                                         cachedir=tmpdir.strpath)

    assert not os.path.exists(os.path.join(tmpdir.strpath, 'train',
                                           'genome.fa', 'chr1..nmm'))

    DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                     storage='memmap',
                                     regions=bed_file, order=1,
                                     cachedir=tmpdir.strpath)

    assert os.path.exists(os.path.join(tmpdir.strpath, 'train', 'genome.fa',
                                       'chr1..nmm'))

    DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                     storage='hdf5',
                                     regions=bed_file, order=1,
                                     cachedir=tmpdir.strpath)

    assert os.path.exists(os.path.join(tmpdir.strpath, 'train', 'genome.fa',
                                       'chr1..h5'))


def test_read_dna_from_fasta_order_1(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'oct4.fa')
    data = DnaDataset.create_from_fasta('train', fastafile=filename,
                                        order=order)

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


def test_read_dna_from_fasta_order_2(tmpdir):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    order = 2
    filename = os.path.join(data_path, 'oct4.fa')
    data = DnaDataset.create_from_fasta('train', fastafile=filename,
                                        order=order,
                                        cachedir=tmpdir.strpath)

    np.testing.assert_equal(len(data), 4)
    np.testing.assert_equal(data.shape, (len(data), 16, 199, 1))

    # correctness of the first sequence - uppercase
    # cacagc
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


def test_stemcell_onehot_identity():
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    filename = os.path.join(data_path, 'stemcells.fa')
    data = DnaDataset.create_from_fasta('dna', fastafile=filename)

    oh = _seqToOneHot(sequences_from_fasta(filename))

    np.testing.assert_equal(data[:], oh)


def _dna_with_region_strandedness(order):
    data_path = pkg_resources.resource_filename('janggo', 'resources/')

    bed = os.path.join(data_path, 'region_w_strand.bed')

    refgenome = os.path.join(data_path, 'genome.fa')

    data = DnaDataset.create_from_refgenome('train', refgenome=refgenome,
                                            regions=bed,
                                            storage='ndarray',
                                            order=order)
    rcdata = RevCompDnaDataset('rctrain', data)

    np.testing.assert_equal(data.shape, (2, pow(4, order),
                                         reglen + 2*flank - order + 1, 1))

    np.testing.assert_equal(data[0], rcdata[1])
    np.testing.assert_equal(data[1], rcdata[0])

    with pytest.raises(Exception):
        np.testing.assert_equal(data[0], data[1])

    with pytest.raises(Exception):
        np.testing.assert_equal(rcdata[0], rcdata[1])


def test_dna_with_region_strandedness():
    _dna_with_region_strandedness(1)
    _dna_with_region_strandedness(2)
