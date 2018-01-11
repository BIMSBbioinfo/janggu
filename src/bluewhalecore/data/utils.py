from collections import defaultdict

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from bluewhalecore.data.data import BwDataset


def sequencescreate_from_fasta(fasta):
    """Obtains nucleotide sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Fasta-filename

    Returns
        List of Biostring sequences
    """

    file_ = open(fasta)
    gen = SeqIO.parse(file_, "fasta")
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


def read_bed(bedfile, trunc=None, usecols=None, names=None, sort_by=None):
    """Read content of a bedfile as pandas.DataFrame.

    Parameters
    ----------
    bedfile : str
        bed-filename.
    trunc : int or None
        Truncate the regions to be of equal length. Default: None
        means no truncation is applied.
    usecols : list(int)
        Specifies which columns of the file to use. Default: [0, 1, 2].
    names : list(str)
        Specifies the column-names. Default: ['chr', 'start', 'end'].
    sortBy : str or None
        Sorts the DataFrame by the specified column. Default: None means
        no sorting is applied.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the bed-file content.
    """

    if not usecols:
        usecols = [0, 1, 2]

    if not names:
        names = ["chr", "start", "end"]

    # currently I am only interested in using cols 1-3
    bed = pd.read_csv(bedfile, sep="\t", header=None, usecols=usecols,
                      names=names)

    if isinstance(sort_by, str):
        bed.sort_values(sort_by, ascending=False, inplace=True)

    if isinstance(trunc, int):
        if trunc < 0:
            raise Exception('read_bed: trunc must be greater than zero.')

        bed.start = (bed.end - bed.start)//2 + bed.start - trunc
        bed.end = bed.start + 2*trunc

    return bed


def input_shape(bwdata):
    """Extracts the shape of a provided Input-BwDataset.

    Parameters
    ---------
    bwdata : :class:`BwDataset` or list(:class:`BwDataset`)
        BwDataset or list(BwDataset).

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corrsponding
        shape as value.
    """
    if isinstance(bwdata, BwDataset):
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
    """Extracts the shape of a provided Output-BwDataset.

    Parameters
    ---------
    bwdata : :class:`BwDataset` or list(:class:`BwDataset`)
        BwDataset or list(BwDataset).
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

    if isinstance(bwdata, BwDataset):
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
