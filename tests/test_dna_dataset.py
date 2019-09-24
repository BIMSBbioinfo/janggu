import glob
import os

import matplotlib
import numpy as np
import pandas
import pkg_resources
import pytest
from keras.layers import Input
from keras.models import Model
from pybedtools import BedTool
from pybedtools import Interval

from janggu.data import Bioseq
from janggu.data import GenomicIndexer
from janggu.data import VariantStreamer
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
    reader = BedTool(bed_file)
    for reg in reader:
        binsizes += (reg.iv.end - reg.iv.start - binsize + stepsize)//stepsize
    return binsizes


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
                                     store_whole_genome=True,
                                     order=order)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    chrom = data.gindexer[0].chrom
    start = data.gindexer[0].start
    end = data.gindexer[0].end
    np.testing.assert_equal(data[0], data[(chrom, start, end)])
    np.testing.assert_equal(data[0], data[chrom, start, end])


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

def test_dna_dims_order_1_from_subset_dataframe(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    order = 1
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_merged = os.path.join(data_path, 'sample.gtf')
    refgenome = os.path.join(data_path, 'sample_genome.fa')

    roi = pandas.read_csv(bed_merged,
                          sep='\t', header=None, usecols=[0, 2, 3, 4, 5,  6], skiprows=2,
                          names=['chrom', 'name', 'start', 'end', 'score', 'strand'])
    roi.start -= 1
    print(roi)

    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=roi,
                                     storage='ndarray',
                                     store_whole_genome=True,
                                     order=order)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    assert len(data.garray.handle) == 2

    # for order 1
    assert len(data) == 2
    assert data.shape == (2, 10000, 1, 4)
    assert data[:].sum() == 20000

    roi = BedTool(bed_merged)
    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=roi,
                                     storage='ndarray',
                                     store_whole_genome=True,
                                     order=order)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    assert len(data.garray.handle) == 2

    # for order 1
    assert len(data) == 2
    assert data.shape == (2, 10000, 1, 4)
    assert data[:].sum() == 20000

    roi = [iv for iv in BedTool(bed_merged)]
    data = Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=roi,
                                     storage='ndarray',
                                     store_whole_genome=True,
                                     order=order)

    np.testing.assert_equal(data[0], data[data.gindexer[0]])
    assert len(data.garray.handle) == 2

    # for order 1
    assert len(data) == 2
    assert data.shape == (2, 10000, 1, 4)
    assert data[:].sum() == 20000


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
    assert data[:].sum() == 20000


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

    assert len(data.garray.handle['data']) == 100

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

    with pytest.warns(FutureWarning):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file,
                                  datatags=['help'])

    with pytest.warns(FutureWarning):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='ndarray',
                                  roi=bed_file,
                                  overwrite=True)
    with pytest.raises(Exception):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='step',
                                  roi=bed_file, order=1)

    assert not os.path.exists(os.path.join(tmpdir.strpath, 'train',
                                           'storage.h5'))
    with pytest.raises(ValueError):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  storage='sparse',
                                  roi=None, order=1,
                                  store_whole_genome=True)
    with pytest.raises(ValueError):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                  roi=bed_file, order=0,
                                  store_whole_genome=True)
    with pytest.raises(ValueError):
        Bioseq.create_from_refgenome('train', refgenome=refgenome,
                                     roi=None, store_whole_genome=False)

    Bioseq.create_from_refgenome('train', refgenome=refgenome,
                              storage='ndarray',
                              roi=None, order=1,
                              store_whole_genome=True)
    file_ = glob.glob(os.path.join(tmpdir.strpath, 'datasets', 'train', '*.h5'))
    assert len(file_) == 0
    print(refgenome)
    print(bed_file)
    Bioseq.create_from_refgenome('train', refgenome=refgenome,
        storage='ndarray',
        roi=bed_file, order=1, cache=True)
    Bioseq.create_from_refgenome('train', refgenome=refgenome,
                              storage='hdf5',
                              roi=bed_file, order=1, cache=True)
    # a cache file must exist now
    file_ = glob.glob(os.path.join(tmpdir.strpath, 'datasets', 'train', '*.h5'))
    assert len(file_) == 1

    # reload the cached file
    Bioseq.create_from_refgenome('train', refgenome=refgenome,
                              storage='hdf5',
                              roi=bed_file, order=1, cache=True)



def test_read_dna_from_biostring_order_1():

    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 1
    filename = os.path.join(data_path, 'sample.fa')
    seqs = sequences_from_fasta(filename)
    with pytest.raises(ValueError):
        data = Bioseq.create_from_seq('train', fastafile=seqs, storage='sparse',
                                     order=order, cache=False)

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
    for store_genome in [True, False]:
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


def test_dnabed_overreaching_ends_whole_genome():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "bed_test.bed")
    filename = os.path.join(data_path, 'sample_genome.fa')

    bioseq = Bioseq.create_from_refgenome(
        'test',
        refgenome=filename,
        roi=bed_file,
        binsize=2,
        flank=20,
        store_whole_genome=True,
        storage='ndarray', cache=False)
    assert len(bioseq) == 9
    assert bioseq.shape == (9, 2+2*20, 1, 4)
    # test if beginning is correctly padded
    np.testing.assert_equal(bioseq[0].sum(), 22)
    # test if end is correctly padded
    np.testing.assert_equal(bioseq['chr1', 29990, 30010].sum(), 10)


def test_dnabed_overreaching_ends_partial_genome():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')
    bed_file = os.path.join(data_path, "bed_test.bed")
    filename = os.path.join(data_path, 'sample_genome.fa')

    bioseq = Bioseq.create_from_refgenome(
        'test',
        refgenome=filename,
        roi=bed_file,
        binsize=2,
        flank=20,
        store_whole_genome=False,
        storage='ndarray')
    assert len(bioseq) == 9
    assert bioseq.shape == (9, 2+2*20, 1, 4)
    np.testing.assert_equal(bioseq[0].sum(), 22)
    np.testing.assert_equal(bioseq[-1].sum(), 42 - 4)


@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_variant_streamer_order_1(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 1

    refgenome = os.path.join(data_path, 'sample_genome.fa')
    vcffile = os.path.join(data_path, 'sample.vcf')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       binsize=50,
                                       store_whole_genome=True,
                                       order=order)

    # even binsize
    vcf = VariantStreamer(dna, vcffile, binsize=10, batch_size=1)
    it_vcf = iter(vcf.flow())
    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # C to T
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    assert names[0] == 'refmismatch'
    np.testing.assert_equal(reference, alternative)
    np.testing.assert_equal(alternative[0,4,0,:], np.array([0,1,0,0]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # C to T
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,4,0,:], np.array([0,1,0,0]))
    np.testing.assert_equal(alternative[0,4,0,:], np.array([0,0,0,1]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # T to C
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,4,0,:], np.array([0,0,0,1]))
    np.testing.assert_equal(alternative[0,4,0,:], np.array([0,1,0,0]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # A to G
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,4,0,:], np.array([1,0,0,0]))
    np.testing.assert_equal(alternative[0,4,0,:], np.array([0,0,1,0]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # G to A
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,4,0,:], np.array([0,0,1,0]))
    np.testing.assert_equal(alternative[0,4,0,:], np.array([1,0,0,0]))

    # odd binsize
    vcf = VariantStreamer(dna, vcffile, binsize=3, batch_size=1)
    it_vcf = iter(vcf.flow())

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # C to T
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    assert names[0] == 'refmismatch'
    np.testing.assert_equal(reference, alternative)
    np.testing.assert_equal(alternative[0,1,0,:], np.array([0,1,0,0]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # C to T
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,1,0,:], np.array([0,1,0,0]))
    np.testing.assert_equal(alternative[0,1,0,:], np.array([0,0,0,1]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # T to C
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,1,0,:], np.array([0,0,0,1]))
    np.testing.assert_equal(alternative[0,1,0,:], np.array([0,1,0,0]))


@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_variant_streamer_order_12_ignore_ref_match(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    refgenome = os.path.join(data_path, 'sample_genome.fa')
    vcffile = os.path.join(data_path, 'sample.vcf')

    for order in [1, 2]:

        dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                           storage='ndarray',
                                           binsize=50,
                                           store_whole_genome=True,
                                           order=order)
    
        # even binsize
        vcf = VariantStreamer(dna, vcffile, binsize=10, batch_size=1,
                              ignore_reference_match=True)
        it_vcf = iter(vcf.flow())
        names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
        # C to T
        print(names, chroms, poss, ra, aa)
        print(reference)
        print(alternative)

        assert names[0] == 'refmismatch'
        #np.testing.assert_equal(reference, alternative)
        np.testing.assert_equal(np.abs(reference-alternative).sum(), 2*order)
        #np.testing.assert_equal(alternative[0,4,0,:], np.array([0,1,0,0]))
    
        # odd binsize
        vcf = VariantStreamer(dna, vcffile, binsize=3, batch_size=1,
                              ignore_reference_match=True)
        it_vcf = iter(vcf.flow())
    
        names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
        # C to T
        print(names, chroms, poss, ra, aa)
        print(reference)
        print(alternative)
        assert names[0] == 'refmismatch'
        np.testing.assert_equal(np.abs(reference-alternative).sum(), 2*order)
        #np.testing.assert_equal(alternative[0,1,0,:], np.array([0,1,0,0]))

@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_variant_streamer_order_1_revcomp(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 1

    refgenome = os.path.join(data_path, 'sample_genome.fa')
    vcffile = os.path.join(data_path, 'sample.vcf')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       binsize=50,
                                       store_whole_genome=True,
                                       order=order)

    annot = BedTool([Interval('chr2', 110, 130, '-')])

    # even binsize
    vcf = VariantStreamer(dna, vcffile, binsize=10, batch_size=1)
    it_vcf = iter(vcf.flow())
    next(it_vcf)
    # C to T
    #print(names, chroms, poss, ra, aa)
    #print(reference)
    #print(alternative)
    #assert names[0] == 'refmismatch'
    #np.testing.assert_equal(reference, alternative)
    #np.testing.assert_equal(alternative[0,4,0,:], np.array([0,1,0,0]))

    next(it_vcf)
    # C to T
    #print(names, chroms, poss, ra, aa)
    #print(reference)
    #print(alternative)
    #np.testing.assert_equal(reference[0,4,0,:], np.array([0,1,0,0]))
    #np.testing.assert_equal(alternative[0,4,0,:], np.array([0,0,0,1]))

    names, chroms, poss, ra, aa, reference, alternative = next(it_vcf)
    # T to C
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
#    np.testing.assert_equal(reference[0,4,0,:], np.array([0,0,0,1]))
#    np.testing.assert_equal(alternative[0,4,0,:], np.array([0,1,0,0]))

    # even binsize
    vcf = VariantStreamer(dna, vcffile, binsize=10, batch_size=1,
                          annotation=annot)
    it_vcf = iter(vcf.flow())
    next(it_vcf)
    # C to T


    next(it_vcf)
    # C to T

    names, chroms, poss, ra, aa, reference2, alternative2 = next(it_vcf)
    # T to C
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference, reference2[:,::-1, :, ::-1])
    np.testing.assert_equal(alternative, alternative2[:,::-1, :, ::-1])


@pytest.mark.filterwarnings("ignore:The truth value")
def test_janggu_variant_streamer_order_2(tmpdir):
    os.environ['JANGGU_OUTPUT'] = tmpdir.strpath
    """Test Janggu creation by shape and name. """
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    order = 2

    refgenome = os.path.join(data_path, 'sample_genome.fa')
    vcffile = os.path.join(data_path, 'sample.vcf')

    dna = Bioseq.create_from_refgenome('dna', refgenome=refgenome,
                                       storage='ndarray',
                                       binsize=50,
                                       store_whole_genome=True,
                                       order=order)

    vcf = VariantStreamer(dna, vcffile, binsize=10, batch_size=1)
    it_vcf = iter(vcf.flow())

    names, chroms, poss,  ra, aa, reference, alternative = next(it_vcf)
    # ACT -> ATT
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    assert names[0] == 'refmismatch'
    np.testing.assert_equal(reference, alternative)

    names, chroms, poss,  ra, aa, reference, alternative = next(it_vcf)
    # ACT -> ATT
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,3,0,1], 1)
    np.testing.assert_equal(reference[0,4,0,7], 1)
    np.testing.assert_equal(alternative[0,3,0,3], 1)
    np.testing.assert_equal(alternative[0,4,0,15], 1)


    names, chroms, poss, ra, aa,  reference, alternative = next(it_vcf)
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    # CTC -> CCC
    np.testing.assert_equal(reference[0,3,0,7], 1)
    np.testing.assert_equal(reference[0,4,0,13], 1)
    np.testing.assert_equal(alternative[0,3,0,5], 1)
    np.testing.assert_equal(alternative[0,4,0,5], 1)

    names, chroms, poss, ra, aa,  reference, alternative = next(it_vcf)
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    # GAC -> GGC
    np.testing.assert_equal(reference[0,3,0,8], 1)
    np.testing.assert_equal(reference[0,4,0,1], 1)
    np.testing.assert_equal(alternative[0,3,0,10], 1)
    np.testing.assert_equal(alternative[0,4,0,9], 1)

    names, chroms, poss, ra, aa,  reference, alternative = next(it_vcf)
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    # CGG -> CAG
    np.testing.assert_equal(reference[0,3,0,6], 1)
    np.testing.assert_equal(reference[0,4,0,10], 1)
    np.testing.assert_equal(alternative[0,3,0,4], 1)
    np.testing.assert_equal(alternative[0,4,0,2], 1)

    vcf = VariantStreamer(dna, vcffile, binsize=5, batch_size=1)
    it_vcf = iter(vcf.flow())

    names, chroms, poss,  ra, aa, reference, alternative = next(it_vcf)
    # ACT -> ATT
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    assert names[0] == 'refmismatch'
    np.testing.assert_equal(reference, alternative)

    names, chroms, poss,  ra, aa, reference, alternative = next(it_vcf)
    # ACT -> ATT
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    np.testing.assert_equal(reference[0,1,0,1], 1)
    np.testing.assert_equal(reference[0,2,0,7], 1)
    np.testing.assert_equal(alternative[0,1,0,3], 1)
    np.testing.assert_equal(alternative[0,2,0,15], 1)


    names, chroms, poss, ra, aa,  reference, alternative = next(it_vcf)
    print(names, chroms, poss, ra, aa)
    print(reference)
    print(alternative)
    # CTC -> CCC
    np.testing.assert_equal(reference[0,1,0,7], 1)
    np.testing.assert_equal(reference[0,2,0,13], 1)
    np.testing.assert_equal(alternative[0,1,0,5], 1)
    np.testing.assert_equal(alternative[0,2,0,5], 1)
