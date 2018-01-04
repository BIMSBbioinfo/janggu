import sys

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict


def sequencesFromFasta(fasta):
    """ Obtain fasta-formated sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Filename of the fastafile

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
    """ Transforms a nucleotide sequence into an int8 array.

    In this array, we use the mapping
    {'A':0, 'C':1, 'G':2, 'T':3}
    """

    if isinstance(seq, str):
        return map(lambda x: NMAP[x], seq)
    elif isinstance(seq, SeqRecord):
        return map(lambda x: NMAP[x], str(seq.seq))
    else:
        raise Exception('dna2ind: Format is not supported')
