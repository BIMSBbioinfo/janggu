import os

import matplotlib
import numpy as np
import pkg_resources
import pytest
from HTSeq import BED_Reader
from keras.layers import Input
from keras.models import Model

from janggu.data import Bioseq
from janggu.data import GenomicIndexer
from janggu.layers import Complement
from janggu.layers import Reverse
from janggu.utils import complement_permmatrix
from janggu.utils import sequences_from_fasta

matplotlib.use('AGG')

binsize = 200
flank = 150
stepsize = 50


def datalen(bed_file):
    binsizes = 0
    reader = BED_Reader(bed_file)
    for reg in reader:
        binsizes += (reg.iv.end - reg.iv.start - binsize + stepsize)//stepsize
    return binsizes

def test_dna_first_last_channel():

    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data1 = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     channel_last=True)
    assert data1.shape == (2, 10000, 1, 4)
    assert data1[0].shape == (1, 10000, 1, 4)

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     channel_last=False)
    assert data.shape == (2, 4, 10000, 1)
    assert data[0].shape == (1, 4, 10000, 1)

    np.testing.assert_equal(data1[0], np.transpose(data[0], (0, 2, 3, 1)))


def test_dna_loading_from_seqrecord(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 2
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')
    seqs = sequences_from_fasta(refgenome)

    data = Bioseq.create_from_refgenome('train', refgenome=seqs,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     order=order)


def test_dna_genomic_interval_access(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 2
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     order=order)

    with pytest.raises(Exception):
        # due to store_whole_genome = False
        data[data.gindexer[0]]

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     order=order,
                                     store_whole_genome=True)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    chrom = data.gindexer[0].chrom
    start = data.gindexer[0].start
    end = data.gindexer[0].end
    np.testing.assert_equal(data[0], data[(chrom, start, end)])
    np.testing.assert_equal(data[0], data[chrom, start, end])

def test_dna_dims_order_1_from_subset(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 1
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     storage='ndarray',
                                     order=order)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    assert len(data.garray.handle) == 2

    # for order 1
    assert len(data) == 2
    assert data.shape == (2, 10000, 1, 4)


def test_dna_dims_order_1_from_subset(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 1
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     binsize=200, stepsize=200,
                                     storage='ndarray',
                                     order=order)

    assert len(data.garray.handle) == 100

    # for order 1
    assert len(data) == 100
    assert data.shape == (100, 200, 1, 4)
    # the correctness of the sequence extraction was also
    # validated using:
    # bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr1:15000-25000
    # ATTGTGGTGA...
    # this sequence is read from the forward strand
    np.testing.assert_equal(data[0][0, :10, 0, :],
                            np.asarray([[1, 0, 0, 0],  # A
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # C
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [0, 0, 1, 0],  # G
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [1, 0, 0, 0]],  # A
                            dtype='int8'))

    # bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr2:15000-25000
    # ggggaagcaa...
    # this sequence is read from the reverse strand
    # so we have ...ttgcttcccc
    np.testing.assert_equal(data[50][0, -10:, 0, :],
                            np.asarray([[0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [0, 1, 0, 0],  # C
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0]],  # C
                            dtype='int8'))


def test_dna_dims_order_1_from_reference(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 1
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    gindexer = GenomicIndexer.create_from_file(bed_merged, 200, 200)

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                        storage='ndarray',
                                        order=order,
                                        store_whole_genome=True)
    data.gindexer = gindexer
    assert len(data.garray.handle) == 2
    assert 'chr1' in data.garray.handle
    assert 'chr2' in data.garray.handle

    # for order 1
    assert len(data) == 100
    assert data.shape == (100, 200, 1, 4)
    # the correctness of the sequence extraction was also
    # validated using:
    # bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr1:15000-25000
    # ATTGTGGTGA...
    # this sequence is read from the forward strand
    np.testing.assert_equal(data[0][0, :10, 0, :],
                            np.asarray([[1, 0, 0, 0],  # A
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # C
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [0, 0, 1, 0],  # G
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [1, 0, 0, 0]],  # A
                            dtype='int8'))

    # bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr2:15000-25000
    # ggggaagcaa...
    # this sequence is read from the reverse strand
    # so we have ...ttgcttcccc
    np.testing.assert_equal(data[50][0, -10:, 0, :],
                            np.asarray([[0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 1, 0],  # G
                                        [0, 1, 0, 0],  # C
                                        [0, 0, 0, 1],  # T
                                        [0, 0, 0, 1],  # T
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0],  # C
                                        [0, 1, 0, 0]],  # C
                            dtype='int8'))


def test_dna_dims_order_2(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 2
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.bed')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_merged,
                                     binsize=200,
                                     storage='ndarray',
                                     order=order)

    # for order 1
    assert len(data) == 100
    assert data.shape == (100, 199, 1, 16)
    # the correctness of the sequence extraction was also
    # validated using:
    # >bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr1:15000-25000
    # ATTGTGGTGAC...
    np.testing.assert_equal(
        data[0][0, :10, 0, :],
        np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # AT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # TT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # TG
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # GT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # TG
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # GG
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # GT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # TG
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # GA
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # AC
                   dtype='int8'))

    # bedtools getfasta -fi sample_genome.fa -bed sample.bed
    # >chr2:15000-25000
    # ggggaagcaag...
    # this sequence is read from the reverse strand
    # so we have ...cttgcttcccc
    np.testing.assert_equal(
        data[50][0, -10:, 0, :],
        np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # CT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # TT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # TG
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # GC
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # CT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # TT
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # TC
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CC
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CC
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # CC
                   dtype='int8'))


def reverse_layer(order):
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_file,
                                     storage='ndarray',
                                     binsize=binsize,
                                     flank=flank,
                                     order=order)

    dna_in = Input(shape=data.shape[1:], name='dna')
    rdna_layer = Reverse()(dna_in)

    rmod = Model(dna_in, rdna_layer)

    # actual shape of DNA
    dna = data[0]
    np.testing.assert_equal(dna[:, ::-1, :, :], rmod.predict(dna))


def complement_layer(order):
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=bed_file,
                                     storage='ndarray',
                                     binsize=binsize,
                                     flank=flank,
                                     order=order)

    dna_in = Input(shape=data.shape[1:], name='dna')
    cdna_layer = Complement()(dna_in)
    cmod = Model(dna_in, cdna_layer)

    # actual shape of DNA
    dna = data[0]

    cdna = cmod.predict(dna)
    ccdna = cmod.predict(cdna)

    with pytest.raises(Exception):
        np.testing.assert_equal(dna, cdna)
    np.testing.assert_equal(dna, ccdna)


def test_reverse_order_1(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    reverse_layer(1)


def test_reverse_order_2(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    reverse_layer(2)


def test_complement_order_1(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    complement_layer(1)


def test_complement_order_2(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    complement_layer(2)


def test_revcomp_rcmatrix(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath

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
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, 'sample.bed')

    refgenome = os.path.join(data_path, 'sample_genome.fa')

    with pytest.raises(Exception):
        # name must be a string
        Bioseq.create_from_refgenome(1.23, refgenome='',
                                  storage='ndarray',
                                  roi=bed_file, order=1)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome='',
                                  storage='ndarray',
                                  roi=bed_file, order=1)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome='test',
                                  storage='ndarray',
                                  roi=bed_file, order=1)

    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file, order=0)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file, flank=-1)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file, binsize=0)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file, stepsize=0)

    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='step',
                                  roi=bed_file, order=1)

    assert not os.path.exists(os.path.join(tmpdir.strpath, 'train',
                                           'storage.h5'))

    Bioseq.create_from_refgenome('train', refgenome=refgenome,
                              storage='ndarray',
                              roi=None, order=1,
                              store_whole_genome=True)
    Bioseq.create_from_refgenome('train', refgenome=refgenome,
                              storage='hdf5',
                              roi=bed_file, order=1, cache=True)

    assert os.path.exists(os.path.join(tmpdir.strpath, 'datasets', 'train',
                                       'order1', 'storage.h5'))


def test_read_dna_from_biostring_order_1():

    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'sample.fa')
    seqs = sequences_from_fasta(filename)
    data = Bioseq.create_from_seq('train', fastafile=seqs,
                                 order=order, cache=False)

    np.testing.assert_equal(len(data), 3897)
    np.testing.assert_equal(data.shape, (3897, 200, 1, 4))
    np.testing.assert_equal(
        data[0][0, :10, 0, :],
        np.asarray([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0]], dtype='int8'))


def test_read_dna_from_fasta_order_1():

    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'sample.fa')
    data = Bioseq.create_from_seq('train', fastafile=filename,
                                 order=order, cache=False)

    np.testing.assert_equal(len(data), 3897)
    np.testing.assert_equal(data.shape, (3897, 200, 1, 4))
    np.testing.assert_equal(
        data[0][0, :10, 0, :],
        np.asarray([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0]], dtype='int8'))


def test_read_dna_from_fasta_order_2():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 2
    filename = os.path.join(data_path, 'sample.fa')
    data = Bioseq.create_from_seq('train', fastafile=filename,
                                 order=order, cache=False)

    np.testing.assert_equal(len(data), 3897)
    np.testing.assert_equal(data.shape, (3897, 199, 1, 16))
    np.testing.assert_equal(
        data[0][0, :10, 0, :],
        np.asarray([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype='int8'))

def test_read_protein_sequences():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    order = 1
    filename = os.path.join(data_path, 'sample_protein.fa')
    data = Bioseq.create_from_seq('train', fastafile=filename,
                                 order=order, seqtype='protein', fixedlen=1000)
    np.testing.assert_equal(len(data), 3)
    np.testing.assert_equal(data.shape, (3, 1000, 1, 20))
    np.testing.assert_equal(
        data[0][0, :4, 0, :],
        np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='int8'))
    np.testing.assert_equal(
        data[0][0, -2:, 0, :], np.zeros((2, 20), dtype='int8'))

    data = Bioseq.create_from_seq('train', fastafile=filename,
                                 order=order, seqtype='protein', fixedlen=5)
    np.testing.assert_equal(len(data), 3)
    np.testing.assert_equal(data.shape, (3, 5, 1, 20))
    np.testing.assert_equal(
        data[0][0, :4, 0, :],
        np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='int8'))
