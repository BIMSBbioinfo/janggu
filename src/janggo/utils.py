"""Utilities for janggo.data """

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from HTSeq import BED_Reader
from HTSeq import GFF_Reader

if sys.version_info[0] < 3:  # pragma: no cover
    from urllib import urlcleanup, urlretrieve
else:
    from urllib.request import urlcleanup, urlretrieve


def sequences_from_fasta(fasta):
    """Obtains nucleotide sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Fasta-filename

    Returns
        List of Biostring sequences
    """

    file_ = open(fasta)
    gen = SeqIO.parse(file_, "fasta", IUPAC.unambiguous_dna)
    seqs = [item for item in gen]

    return seqs


LETTERMAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

NMAP = defaultdict(lambda: -1024)
NMAP.update(LETTERMAP)


def dna2ind(seq):
    """Transforms a nucleotide sequence into an int array.

    In this array, we use the mapping
    {'A':0, 'C':1, 'G':2, 'T':3}
    Any other characters (e.g. 'N') are represented by a large negative value
    to avoid confusion with valid nucleotides.

    Parameters
    ----------
    seq : str, Bio.SeqRecord or Bio.Seq.Seq
        Nucleotide sequence represented as string, SeqRecord or Seq.

    Returns
    -------
    list(int)
        Integer array representation of the nucleotide sequence.
    """

    if isinstance(seq, SeqRecord):
        seq = seq.seq
    if isinstance(seq, (str, Seq)):
        return [NMAP[x] for x in seq]
    else:
        raise TypeError('dna2ind: Format is not supported')


def as_onehot(idna, order):
    """Converts a index sequence into one-hot representation.

    This method is used to transform a nucleotide sequence
    for a given batch, represented by integer indices,
    into a one-hot representation.

    Parameters
    ----------
    idna: numpy.array
        Array that holds the indices for a given batch.
        The dimensions of the array correspond to
        `(batch_size, sequence_length + 2*flank - order + 1)`.
    order: int
        Order of the sequence representation. Used for higher-order
        motif modelling.

    Returns
    -------
    numpy.array
        One-hot representation of the batch. The dimension
        of the array is given by
        `(batch_size, sequence length, pow(4, order), 1)`
    """

    onehot = np.zeros((len(idna),
                       idna.shape[1], pow(4, order), 1), dtype='int8')
    for nuc in np.arange(pow(4, order)):
        onehot[:, :, nuc, 0][idna == nuc] = 1

    return onehot


def complement_index(idx, order):
    rev_idx = np.arange(4)[::-1]
    irc = 0
    for iord in range(order):
        nuc = idx % 4
        idx = idx // 4
        irc += rev_idx[nuc] * pow(4, order - iord - 1)

    return irc


def complement_permmatrix(order):
    """This function returns a permutation matrix for computing
    the complementary DNA strand one-hot representation for a given order.

    Parameters
    ----------
    order : int
        Order of the one-hot representation

    Returns
    -------
    np.array
        Permutation matrix
    """
    perm = np.zeros((pow(4, order), pow(4, order)))
    for idx in range(pow(4, order)):
        jdx = complement_index(idx, order)
        perm[jdx, idx] = 1
    return perm


def get_genome_size(refgenome='hg19', outputdir='./', skip_random=True):
    """Get genome size.

    This function loads the genome size for a specified reference genome
    into a dict. The reference genome sizes are obtained from
    UCSC genome browser.

    Parameters
    ----------
    refgenome : str
        Reference genome name. Default: 'hg19'.
    outputdir : str
        Directory in which the downloaded *.chrom.sizes file will be stored.
    skipRandom : True

    Returns
    -------
    dict()
        Dictionary with chromosome names as keys and their respective lengths
        as values.
    """

    outputfile = os.path.join(outputdir, '{}.chrom.sizes'.format(refgenome))
    if not os.path.exists(outputfile):  # pragma: no cover
        # not part of unit tests, because this requires internet connection
        urlpath = 'http://hgdownload.cse.ucsc.edu/goldenPath/{}/bigZips/{}.chrom.sizes'.format(refgenome, refgenome)

        # From the md5sum.txt we extract the
        print("Downloading {}".format(urlpath))
        urlcleanup()
        urlretrieve(urlpath.format(refgenome), outputfile)

    content = pd.read_csv(outputfile, sep='\t', names=['chr', 'length'],
                          index_col='chr')
    if skip_random:
        fltr = [True if len(name.split('_')) <= 1 else False for name in content.index]
        content = content[fltr]
    return content.to_dict()['length']


def get_genome_size_from_bed(bedfile, flank):
    """Get genome size.

    This function loads the genome size for a specified reference genome
    into a dict. The reference genome sizes are obtained from
    UCSC genome browser.

    Parameters
    ----------
    bedfile : str
        Bed or GFF file containing the regions of interest
    flank : int
        Flank to increase the window sizes.

    Returns
    -------
    dict()
        Dictionary with chromosome names as keys and their respective lengths
        as values.
    """

    if isinstance(bedfile, str) and bedfile.endswith('.bed'):
        regions_ = BED_Reader(bedfile)
    elif isinstance(bedfile, str) and (bedfile.endswith('.gff') or
                                       bedfile.endswith('.gtf')):
        regions_ = GFF_Reader(bedfile)
    else:
        raise Exception('Regions must be a bed, gff or gtf-file.')

    gsize = {}
    for region in regions_:
        if region.iv.chrom not in gsize:
            gsize[region.iv.chrom] = region.iv.end + flank
        elif gsize[region.iv.chrom] < region.iv.end + flank:
            gsize[region.iv.chrom] = region.iv.end + flank
    return gsize
