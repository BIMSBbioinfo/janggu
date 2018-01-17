"""Utilities for beluga.data """

from collections import defaultdict

from Bio import SeqIO
from Bio.Alphabet import IUPAC
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
    seq : str or Bio.SeqRecord
        Nucleotide sequence as a string or SeqRecord object.

    Returns
    -------
    list(int)
        Integer array representation of the nucleotide sequence.
    """

    if isinstance(seq, str):
        return [NMAP[x] for x in seq]
    elif isinstance(seq, SeqRecord):
        return [NMAP[x] for x in seq.seq]
    else:
        raise Exception('dna2ind: Format is not supported')


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
