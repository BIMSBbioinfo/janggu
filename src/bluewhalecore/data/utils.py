from collections import defaultdict

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
