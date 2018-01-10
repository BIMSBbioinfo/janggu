from collections import defaultdict

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def sequencesFromFasta(fasta):
    """Obtains nucleotide sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Fasta-filename

    Returns
        List of Biostring sequences
    """

    h = open(fasta)
    gen = SeqIO.parse(h, "fasta")
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
        return map(lambda x: NMAP[x], seq)
    elif isinstance(seq, SeqRecord):
        return map(lambda x: NMAP[x], str(seq.seq))
    else:
        raise Exception('dna2ind: Format is not supported')


def readBed(bedfile, trunc=None, usecols=[0, 1, 2],
            names=["chr", "start", "end"], sortBy=None):
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

    # currently I am only interested in using cols 1-3
    bed = pd.read_csv(bedfile, sep="\t", header=None, usecols=usecols,
                      names=names)

    if isinstance(sortBy, str):
        bed.sort_values(sortBy, ascending=False, inplace=True)

    if isinstance(trunc, int):
        if trunc < 0:
            raise Exception('readBed: trunc must be greater than zero.')

        bed.start = (bed.end - bed.start)//2 + bed.start - trunc
        bed.end = bed.start + 2*trunc

    return bed
