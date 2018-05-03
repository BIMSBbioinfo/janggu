"""Utilities for janggo """

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
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


if pyBigWig.numpy == 0:
    raise Exception('pyBigWig must be installed with numpy support. '
                    'Please install numpy before pyBigWig to ensure that.')


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
        urlpath = 'http://hgdownload.cse.ucsc.edu/goldenPath/{}/bigZips/{}.chrom.sizes'.format(
            refgenome, refgenome)

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


def dump_json(output_dir, name, results, append=True):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, name + '.json')
    try:
        with open(filename, 'r') as jsonfile:
            content = json.load(jsonfile)
    except IOError:
        content = {}  # needed for py27
    with open(filename, 'w') as jsonfile:
        if not append:
            content = {}
        content.update({','.join(key): results[key] for key in results})
        json.dump(content, jsonfile)


def dump_tsv(output_dir, name, results):
    """Method that dumps the results as tsv file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, name + '.tsv')
    pd.DataFrame.from_dict(results, orient='index').to_csv(filename, sep='\t')


def plot_score(output_dir, name, results, figsize=None, xlabel=None,
               ylabel=None, fform=None):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax_ = fig.add_axes([0.1, 0.1, .55, .5])

    ax_.set_title(name)
    for key in results:
        # avg might be returned using a custom function
        x_score, y_score, avg = results[key]['value']
        ax_.plot(x_score, y_score,
                 label="{}".format('-'.join(key)))

    lgd = ax_.legend(bbox_to_anchor=(1.05, 1),
                     loc=2, prop={'size': 10}, ncol=1)
    if xlabel is not None:
        ax_.set_xlabel(xlabel, size=14)
    if ylabel is not None:
        ax_.set_ylabel(ylabel, size=14)
    if fform is not None:
        fform = fform
    else:
        fform = 'png'
    filename = os.path.join(output_dir, name + '.' + fform)
    fig.savefig(filename, format=fform,
                dpi=1000,
                bbox_extra_artists=(lgd,), bbox_inches="tight")


def export_bigwig(output_dir, name, results, gindexer=None):

    """Export predictions to bigwig."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if gindexer is None:
        raise ValueError('Please specify a GenomicIndexer for export_to_bigiwig')

    genomesize = {}

    # extract genome size from gindexer
    # check also if sorted and non-overlapping
    last_interval = {}
    for region in gindexer:
        if region.chrom in last_interval:
            if region.start < last_interval[region.chrom]:
                raise ValueError('The regions in the bed/gff-file must be sorted'
                                 ' and mutually disjoint. Please, sort and merge'
                                 ' the regions before exporting the bigwig format')
        if region.chrom not in genomesize:
            genomesize[region.chrom] = region.end
            last_interval[region.chrom] = region.end
        if genomesize[region.chrom] < region.end:
            genomesize[region.chrom] = region.end

    bw_header = [(chrom, genomesize[chrom]*gindexer.resolution)
                 for chrom in genomesize]

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file
    for modelname, layername, condition in results:
        bw_file = pyBigWig.open(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bigwig'.format(
                prefix=name, model=modelname,
                output=layername, condition=condition)), 'w')
        bw_file.addHeader(bw_header)
        pred = results[modelname, layername, condition]['value']
        nsplit = len(pred)//len(gindexer)
        for idx, region in enumerate(gindexer):

            val = [p for p in pred[idx:(idx+nsplit)]
                   for _ in range(gindexer.resolution)]

            bw_file.addEntries(region.chrom,
                               int(region.start*gindexer.resolution),
                               values=val,
                               span=1,
                               step=1)
        bw_file.close()


def export_bed(output_dir, name, results, gindexer=None):
    """Export predictions to bed."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if gindexer is None:
        raise ValueError('Please specify a GenomicIndexer for export_to_bed')

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file

    for modelname, layername, condition in results:
        bed_content = pd.DataFrame(columns=['chr', 'start',
                                            'end', 'name', 'score'])
        for ridx, region in enumerate(gindexer):
            pred = results[modelname, layername, condition]['value']

            print(pred)
            print(region)
            print(gindexer)
            print('len(gi)={}'.format(len(gindexer)))
            nsplit = len(pred)//len(gindexer)
            stepsize = (region.end-region.start)//nsplit
            starts = list(range(region.start,
                                region.end,
                                stepsize))
            ends = list(range(region.start + stepsize,
                              region.end + stepsize,
                              stepsize))
            cont = {'chr': [region.chrom] * nsplit,
                    'start': [s*gindexer.resolution for s in starts],
                    'end': [e*gindexer.resolution for e in ends],
                    'name': ['.'] * nsplit,
                    'score': pred[ridx*nsplit:((ridx+1)*nsplit)]}
            bed_entry = pd.DataFrame(cont)
            bed_content = bed_content.append(bed_entry, ignore_index=True)

        bed_content.to_csv(os.path.join(
            output_dir,
            '{prefix}.{model}.{output}.{condition}.bed'.format(
                prefix=name, model=modelname,
                output=layername, condition=condition)),
                           sep='\t', header=False, index=False,
                           columns=['chr', 'start', 'end', 'name', 'score'])
