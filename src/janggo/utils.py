"""Utilities for janggo """

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from HTSeq import BED_Reader
from HTSeq import GFF_Reader
from sklearn.manifold import TSNE

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


def _complement_index(idx, order):
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
        jdx = _complement_index(idx, order)
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


def _get_genomic_reader(filename):
    """regions from a BED_Reader or GFF_Reader.
    """
    if isinstance(filename, str) and filename.endswith('.bed'):
        regions_ = BED_Reader(filename)
    elif isinstance(filename, str) and (filename.endswith('.gff') or
                                        filename.endswith('.gtf')):
        regions_ = GFF_Reader(filename)
    else:
        raise Exception('Regions must be a bed, gff or gtf-file.')

    return regions_


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

    regions_ = _get_genomic_reader(bedfile)

    gsize = {}
    for region in regions_:
        if region.iv.chrom not in gsize:
            gsize[region.iv.chrom] = region.iv.end + flank
        elif gsize[region.iv.chrom] < region.iv.end + flank:
            gsize[region.iv.chrom] = region.iv.end + flank
    return gsize


def export_json(output_dir, name, results, filesuffix='json',
                annot=None, row_names=None):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    filesuffix : str
        Target file ending.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels.
    row_names : None or list
        Row names.
    """

    filename = os.path.join(output_dir, name + '.' + filesuffix)

    content = {}  # needed for py27
    with open(filename, 'w') as jsonfile:
        try:
            content.update({'-'.join(key): results[key]['value'].tolist() for key in results})
        except AttributeError:
            content.update({'-'.join(key): results[key]['value'] for key in results})
        if annot is not None:
            content.update({'annot': annot})

        if row_names is not None:
            content.update({'row_names': row_names})
        json.dump(content, jsonfile)


def export_tsv(output_dir, name, results, annot=None, row_names=None):
    """Method that dumps the results as tsv file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    name : str
        Output name.
    results : dict
        Dictionary containing the evaluation results which needs to be stored.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels.
    row_names : None, list
        List of row names
    """

    filename = os.path.join(output_dir, name + '.tsv')
    try:
        # check if the result is iterable
        iter(results[list(results.keys())[0]]['value'])
        _rs = {'-'.join(k): results[k]['value'] for k in results}
    except TypeError:
        # if the result is not iterable, wrap it up as list
        _rs = {'-'.join(k): [results[k]['value']] for k in results}
    df = pd.DataFrame.from_dict(_rs)
    for an in annot or []:
        df['annot.'+an] = annot[an]
    if row_names is not None:
        df['row_names'] = row_names
    df.to_csv(filename, sep='\t', index=False)


def export_plotly(output_dir, name, results, annot=None, row_names=None):
    # this produces a normal json file, but for the dedicated
    # purpose of visualization in the dash app.
    export_tsv(output_dir, name, results, 'ply', annot, row_names)


def export_score_plot(output_dir, name, results, figsize=None, xlabel=None,
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

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax_ = fig.add_axes([0.1, 0.1, .55, .5])

    ax_.set_title(name)
    for mname, lname, cname in results:
        # avg might be returned using a custom function
        print(mname, lname, cname)
        x_score, y_score, auxstr = results[mname, lname, cname]['value']
        label = "{}".format('-'.join([mname[:8], lname, cname]))
        if isinstance(auxstr, str):
            label += ' ' + auxstr
        ax_.plot(x_score, y_score, label=label)

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
    """Export predictions to bigwig.

    This function can be used as exporter with :class:`InOutScorer`.
    """

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
    """Export predictions to bed.

    This function can be used as exporter with :class:`InOutScorer`.
    """

    if gindexer is None:
        raise ValueError('Please specify a GenomicIndexer for export_to_bed')

    # the last dimension holds the conditions. Each condition
    # needs to be stored in a separate file

    for modelname, layername, condition in results:
        bed_content = pd.DataFrame(columns=['chr', 'start',
                                            'end', 'name', 'score'])
        for ridx, region in enumerate(gindexer):
            pred = results[modelname, layername, condition]['value']

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


def export_clustermap(output_dir, name, results, fform=None, figsize=None,
                      annot=None,
                      method='ward', metric='euclidean', z_score=None,
                      standard_scale=None, row_cluster=True, col_cluster=True,
                      row_linkage=None, col_linkage=None, row_colors=None,
                      col_colors=None, mask=None, **kwargs):
    """Create of clustermap of the feature activities.

    This method utilizes
    `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_
    to illustrate feature activities of the neural net.
    """

    if annot is not None:

        firstkey = list(annot.keys())[0]
        pal = sns.color_palette('hls', len(set(annot[firstkey])))
        lut = dict(zip(set(annot[firstkey]), pal))
        row_colors = [lut[k] for k in annot[firstkey]]

    _rs = {k: results[k]['value'] for k in results}
    data = pd.DataFrame.from_dict(_rs)
    if fform is not None:
        fform = fform
    else:
        fform = 'png'

    sns.clustermap(data,
                   method=method,
                   metric=metric,
                   z_score=z_score,
                   standard_scale=standard_scale,
                   row_cluster=row_cluster,
                   col_cluster=col_cluster,
                   row_linkage=row_linkage,
                   col_linkage=col_linkage,
                   col_colors=col_colors,
                   row_colors=row_colors,
                   mask=mask,
                   figsize=figsize,
                   **kwargs).savefig(os.path.join(output_dir,
                                                  name + '.' + fform),
                                     format=fform, dpi=700)


def export_tsne(output_dir, name, results, figsize=None,
                cmap=None, colors=None, norm=None, alpha=None, fform=None,
                annot=None):
    """Create a plot of the 2D t-SNE embedding of the feature activities."""

    _rs = {k: results[k]['value'] for k in results}
    data = pd.DataFrame.from_dict(_rs)

    tsne = TSNE()
    embedding = tsne.fit_transform(data.values)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    if annot is not None:
        firstkey = list(annot.keys())[0]
        pal = sns.color_palette('hls', len(set(annot[firstkey])))
        lut = dict(zip(set(annot[firstkey]), pal))
        row_colors = [lut[k] for k in annot[firstkey]]

        for label in lut:

            plt.scatter(x=embedding[np.asarray(annot[firstkey])==label, 0],
                        y=embedding[np.asarray(annot[firstkey])==label, 1],
                        c=lut[label],
                        label=label,
                        norm=norm, alpha=alpha)

        plt.legend()
    else:
        plt.scatter(x=embedding[annot[firstkey]==label, 0],
                    y=embedding[annot[firstkey]==label, 1],
                    c=colors, cmap=cmap,
                    norm=norm, alpha=alpha)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    if fform is not None:
        fform = fform
    else:
        fform = 'png'

    fig.savefig(os.path.join(output_dir, name + '.' + fform),
                format=fform, dpi=700)
