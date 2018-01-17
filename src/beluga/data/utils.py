"""Utilities for beluga.data """

from collections import defaultdict

import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from beluga.data.data import BlgDataset


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


def input_shape(bwdata):
    """Extracts the shape of a provided Input-BlgDataset.

    Parameters
    ---------
    bwdata : :class:`BlgDataset` or list(:class:`BlgDataset`)
        BlgDataset or list(BlgDataset).

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corrsponding
        shape as value.
    """
    if isinstance(bwdata, BlgDataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        data = {}
        for bwdatum in bwdata:
            shape = bwdatum.shape[1:]
            if shape == ():
                shape = (1,)
            data[bwdatum.name] = {'shape': shape}
        return data
    else:
        raise Exception('inputSpace wrong argument: {}'.format(bwdata))


def output_shape(bwdata, loss, activation='sigmoid',
                 loss_weight=1.):
    """Extracts the shape of a provided Output-BlgDataset.

    Parameters
    ---------
    bwdata : :class:`BlgDataset` or list(:class:`BlgDataset`)
        BlgDataset or list(BlgDataset).
    loss : str or objective function.
        Keras compatible loss function. See https://keras.io/losses.
    activation : str
        Output activation function. Default: 'sigmoid'.
    loss_weights : float
        Loss weight used for fitting the model. Default: 1.

    Returns
    -------
    dict
        Dictionary description of the network output.
    """

    if isinstance(bwdata, BlgDataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        data = {}
        for bwdatum in bwdata:
            shape = bwdatum.shape[1:]
            if shape == ():
                shape = (1,)
            data[bwdatum.name] = {'shape': shape,
                                  'loss': loss,
                                  'loss_weight': loss_weight,
                                  'activation': activation}
        return data
    else:
        raise Exception('outputSpace wrong argument: {}'.format(bwdata))
