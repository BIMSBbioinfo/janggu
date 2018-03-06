"""Utilities for janggo.data """

from collections import defaultdict

import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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
        `(batch_size, pow(4, order), sequence length, 1)`
    """

    onehot = np.zeros((len(idna), pow(4, order),
                       idna.shape[1], 1), dtype='int8')
    for nuc in np.arange(pow(4, order)):
        onehot[:, nuc, :, 0][idna == nuc] = 1

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
