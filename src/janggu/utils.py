"""Utilities for janggu """

import json
import os
from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pybedtools import BedTool
from pybedtools import Interval

import matplotlib.pyplot as plt
import pyBigWig
import seaborn as sns

from sklearn.manifold import TSNE  # pylint: disable=import-error


try:
    from urllib import urlcleanup, urlretrieve
except ImportError:
    try:
        from urllib.request import urlcleanup, urlretrieve
    except ImportError as exception:  # pragma: no cover
        urlretrieve = None
        urlcleanup = None


def _get_output_root_directory():
    """Function returns the output root directory."""
    if "JANGGU_OUTPUT" not in os.environ:  # pragma: no cover

        os.environ['JANGGU_OUTPUT'] = os.path.join(os.path.expanduser("~"),
                                                   'janggu_results')
        print('environment variable JANGGU_OUTPUT not set.'
              ' Will use JANGGU_OUTPUT={}'.format(os.environ['JANGGU_OUTPUT']))
    return os.environ['JANGGU_OUTPUT']


def _get_output_data_location(datatags):
    """Function returns the output location for the dataset.

    Parameters
    ------------
    datatags : list(str)
        Tags describing the dataset. E.g. ['testdna'].
    """
    outputdir = _get_output_root_directory()

    args = (outputdir, 'datasets')
    if datatags is not None:
        args += tuple(datatags)

    return os.path.join(*args)


def sequences_from_fasta(fasta, string='dna'):
    """Obtains nucleotide or peptide sequences from a fasta file.

    Parameters
    -----------
    fasta : str
        Fasta-filename
    string : str
        Either dna or protein.

    Returns
        List of Biostring sequences
    """

    file_ = open(fasta)
    if string == 'dna':
        alpha = IUPAC.unambiguous_dna
    else:
        alpha = IUPAC.protein
    gen = SeqIO.parse(file_, "fasta", alpha)
    seqs = [item for item in gen]

    return seqs


def _to_list(objs):
    """Makes a list of objs"""
    if objs is None:
        return []
    elif not isinstance(objs, list):
        return [objs]
    else:
        return objs

def _check_valid_files(list_of_files):
    if len(list_of_files) == 0:
        raise ValueError("Empty list fo files.")
    for f in list_of_files:
        if not os.path.exists(f):
            raise ValueError("File {} does not exist.".format(f))
    return list_of_files

NOLETTER = -100000000
LETTERMAP = {k: i for i, k in enumerate(sorted(IUPAC.unambiguous_dna.letters))}
NNUC = len(IUPAC.unambiguous_dna.letters)

# mapping of nucleotides to integers
NMAP = defaultdict(lambda: NOLETTER)
NMAP.update(LETTERMAP)

# mapping of amino acids to integers
LETTERMAP = {k: i for i, k in enumerate(sorted(IUPAC.protein.letters))}
PMAP = defaultdict(lambda: NOLETTER)
PMAP.update(LETTERMAP)


def seq2ind(seq):
    """Transforms a biological sequence into an int array.

    Each nucleotide or amino acid maps to an integer between
    zero and len(alphabet) - 1.

    Any other characters (e.g. 'N') are represented by a negative value
    to avoid confusion with valid nucleotides.

    Parameters
    ----------
    seq : str, Bio.SeqRecord or Bio.Seq.Seq
        Sequence represented as string, SeqRecord or Seq.

    Returns
    -------
    list(int)
        Integer array representation of the biological sequence.
    """

    if isinstance(seq, SeqRecord):
        seq = seq.seq
    if isinstance(seq, (str, Seq)):
        if type(seq.alphabet) is type(IUPAC.unambiguous_dna):
            return [NMAP[x.upper()] for x in seq]
        # else proteins should be used
        return [PMAP[x.upper()] for x in seq]
    raise TypeError('seq2ind: Format is not supported')


def sequence_padding(seqs, length):
    """This function truncates or pads the sequences
    to achieve fixed length sequences.

    Padding of the sequence is achieved by appending '.'

    Parameters
    ----------
    seqs : str, Bio.SeqRecord or Bio.Seq.Seq
        Sequence represented as string, SeqRecord or Seq.

    Returns
    -------
    list(Bio.SeqRecord)
        Padded sequence represented as string, SeqRecord or Seq.
    """
    seqs_ = deepcopy(seqs)
    for idx, seq in enumerate(seqs_):
        if len(seq) < length:
            seqs_[idx] += '.' * (length - len(seq))
        else:
            seqs_[idx] = seq[:length]
    return seqs_


def as_onehot(iseq, order, alphabetsize):
    """Converts a index sequence into one-hot representation.

    This method is used to transform a biological sequence
    for a given batch, represented by integer indices,
    into a one-hot representation.

    Parameters
    ----------
    iseq: numpy.array
        Array that holds the indices for a given batch.
        The dimensions of the array correspond to
        `(batch_size, sequence_length + 2*flank - order + 1)`.
    order: int
        Order of the sequence representation. Used for higher-order
        motif modelling.
    alphabetsize : int
        Alphabetsize.

    Returns
    -------
    numpy.array
        One-hot representation of the batch. The dimension
        of the array is given by
        `(batch_size, sequence length, 1, pow(alphabetsize, order))`
    """

    onehot = np.zeros((len(iseq),
                       iseq.shape[1], 1,
                       pow(alphabetsize, order)), dtype='int8')
    for nuc in np.arange(pow(alphabetsize, order)):
        onehot[:, :, 0, nuc][iseq == nuc] = 1

    return onehot


def _complement_index(idx, order):
    rev_idx = np.arange(NNUC)[::-1]
    irc = 0
    for iord in range(order):
        nuc = idx % NNUC
        idx = idx // NNUC
        irc += rev_idx[nuc] * pow(NNUC, order - iord - 1)

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
    perm = np.zeros((pow(NNUC, order), pow(NNUC, order)))
    for idx in range(pow(NNUC, order)):
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
    if urlcleanup is None:  # pragma: no cover
        raise Exception('\n\nurllib not available. Please install urllib3.\n\n')

    outputfile = os.path.join(outputdir, '{}.chrom.sizes'.format(refgenome))
    if not os.path.exists(outputfile):  # pragma: no cover
        # not part of unit tests, because this requires internet connection
        urlpath = 'http://hgdownload.cse.ucsc.edu/goldenPath/{ref1}/bigZips/{ref2}.chrom.sizes'\
            .format(ref1=refgenome, ref2=refgenome)

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


def _get_genomic_reader(inputregions):
    """regions from a BedTool
    """
    if isinstance(inputregions, BedTool):
        # already in the desired format
        return inputregions

    if isinstance(inputregions, pd.DataFrame):
        if np.any([c not in inputregions.columns \
                   for c in ['chrom', 'start', 'end']]):
            raise ValueError("The dataframe must contain the columns"
                             "'chrom', 'start' and 'end'")
        if 'strand' not in inputregions.columns:
            inputregions['strand'] = '.'

        # transform the pandas dataframe to a list of intervals
        inputregions = [Interval(str(row['chrom']),
                                 int(row['start']),
                                 int(row['end']),
                                 strand=str(row['strand'])) for \
                                 _, row in inputregions.iterrows()]

    # at this point the inputregions are either a filename pointing to a
    # bedfile or a list of intervals. Both of which are understood by BedTool.
    regions_ = BedTool(inputregions)

    return regions_


def _iv_to_str(chrom, start, end):
    """Converts a genomic interval into a string representation 'chr:start-end'."""
    return '{}:{}-{}'.format(chrom, start, end)


def _str_to_iv(givstr, template_extension=0):
    """Converts a string representation 'chr:start-end' into genomic coordinates."""
    sub = givstr.split(':')
    if len(sub) == 1:
        return (sub[0], )

    chr_ = sub[0]

    second_split = sub[1].split('-')
    # if the interval contains negative values, they will be zero-padded later
    if second_split[0] == '':
        # start is a negative value
        second_split = second_split[1:]
        second_split[0] = '-' + second_split[0]
    if second_split[1] == '':
        # if end is a negative value it does not make sense.
        raise ValueError('Invalid genomic location string: {}'.format(givstr))

    start = int(second_split[0])
    end = int(second_split[1])

    return (chr_, start - template_extension, end + template_extension)


def get_genome_size_from_regions(regions):
    """Get genome size.

    This function loads the genome size for a specified reference genome
    into a dict. The reference genome sizes are obtained from
    UCSC genome browser.

    Parameters
    ----------
    regions : str or GenomicIndexer
        Either a path pointing to a BED or GFF file containing genomic regions
        or a GenomicIndexer object.

    Returns
    -------
    dict()
        Dictionary with chromosome names as keys and their respective lengths
        as values.
    """

    regions_ = regions
    if isinstance(regions, str):
        regions_ = _get_genomic_reader(regions)

    gsize = OrderedDict()
    for interval in regions_:
        if interval.chrom not in gsize:
            gsize[interval.chrom] = interval.end
        elif gsize[interval.chrom] < interval.end:
            gsize[interval.chrom] = interval.end
    return gsize


class ExportJson(object):
    """Method that dumps the results in a json file.

    Parameters
    ----------
    filesuffix : str
        Target file ending. Default: 'json'.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels. Default: None.
    row_names : None or list
        List of row names. Default: None.
    """
    def __init__(self, filesuffix='json',
                 annot=None, row_names=None):

        self.filesuffix = filesuffix
        self.annot = annot
        self.row_names = row_names

    def __call__(self, output_dir, name, results):
        filesuffix = self.filesuffix
        annot = self.annot
        row_names = self.row_names

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


class ExportTsv(object):
    """Method that dumps the results as tsv file.

    This class can be used to export general table summaries.

    Parameters
    ----------
    filesuffix : str
        File ending. Default: 'tsv'.
    annot: None, dict
        Annotation data. If encoded as dict the key indicates the name,
        while the values holds a list of annotation labels. For example,
        this can be used to store the true output labels.
        Default: None.
    row_names : None, list
        List of row names. For example, chromosomal loci. Default: None.
    """
    def __init__(self, filesuffix='tsv', annot=None, row_names=None):

        self.filesuffix = filesuffix
        self.annot = annot
        self.row_names = row_names

    def __call__(self, output_dir, name, results):
        filesuffix = self.filesuffix
        annot = self.annot
        row_names = self.row_names

        filename = os.path.join(output_dir, name + '.' + filesuffix)
        try:
            # check if the result is iterable
            iter(results[list(results.keys())[0]]['value'])
            _rs = {'-'.join(k): results[k]['value'] for k in results}
        except TypeError:
            # if the result is not iterable, wrap it up as list
            _rs = {'-'.join(k): [results[k]['value']] for k in results}
        _df = pd.DataFrame.from_dict(_rs)
        for _an in annot or []:
            _df['annot.' + _an] = annot[_an]
        if row_names is not None:
            _df['row_names'] = row_names
        _df.to_csv(filename, sep='\t', index=False)


class ExportScorePlot(object):
    """Exporting score plot.

    This class can be used for producing an AUC or PRC plot.

    Parameters
    ----------
    figsize : tuple(int, int)
        Used to specify the figure size for matplotlib.
    xlabel : str or None
        xlabel used for the plot.
    ylabel : str or None
        ylabel used for the plot.
    fform : str or None
        Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    """
    def __init__(self, figsize=None, xlabel=None, ylabel=None, fform=None):
        self.figsize = figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fform = fform
        if self.fform is None:
            self.fform = 'png'

    def __call__(self, output_dir, name, results):

        if plt is None:  # pragma: no cover
            raise Exception('matplotlib not available. Please install matplotlib.')

        if self.figsize is not None:
            fig = plt.figure(figsize=self.figsize)
        else:
            fig = plt.figure()

        ax_ = fig.add_axes([0.1, 0.1, .55, .5])

        ax_.set_title(name)
        for keys in results:
            if isinstance(keys, str):
                k = (keys,)
            else:
                k = keys
            # avg might be returned using a custom function
            x_score, y_score, auxstr = results[keys]['value']
            label = "{}".format('-'.join(k))
            if isinstance(auxstr, str):
                label += ' ' + auxstr
            ax_.plot(x_score, y_score, label=label)

        lgd = ax_.legend(bbox_to_anchor=(1.05, 1),
                         loc=2, prop={'size': 10}, ncol=1)
        if self.xlabel is not None:
            ax_.set_xlabel(self.xlabel, size=14)
        if self.ylabel is not None:
            ax_.set_ylabel(self.ylabel, size=14)

        filename = os.path.join(output_dir, name + '.' + self.fform)
        fig.savefig(filename, format=self.fform,
                    dpi=1000,
                    bbox_extra_artists=(lgd,), bbox_inches="tight")


class ExportBigwig(object):
    """Export predictions to bigwig.

    This function exports the predictions to bigwig format which allows you to
    inspect the predictions in a genome browser.
    Importantly, gindexer must contain non-overlapping windows!

    Parameters
    ----------
    gindexer : GenomicIndexer
        GenomicIndexer that links the prediction for a certain region to
        its associated genomic coordinates.
    """
    def __init__(self, gindexer):
        self.gindexer = gindexer

    def __call__(self, output_dir, name, results):

        if pyBigWig is None:  # pragma: no cover
            raise Exception('pyBigWig not available. '
                            '`export_bigwig` requires pyBigWig to be installed.')

        genomesize = OrderedDict()

        # extract genome size from gindexer
        # check also if sorted and non-overlapping
        for region in self.gindexer:
            if region.chrom not in genomesize:
                genomesize[region.chrom] = region.end
            if genomesize[region.chrom] < region.end:
                genomesize[region.chrom] = region.end

        bw_header = [(str(chrom), genomesize[chrom])
                     for chrom in genomesize]

        # the last dimension holds the conditions. Each condition
        # needs to be stored in a separate file
        for keys in results:
            if isinstance(keys, str):
                k = (keys,)
            else:
                k = keys
            bw_file = pyBigWig.open(os.path.join(
                output_dir,
                '{prefix}.{key}.bigwig'.format(
                    prefix=name, key='.'.join(k))), 'w')
            bw_file.addHeader(bw_header)
            pred = results[keys]['value']
            # compute the ratio between binsize and stepsize
            bsss = max(np.ceil(float(self.gindexer.binsize) / float(self.gindexer.stepsize)), 1.)
            ppi = int(np.rint(len(pred)/(len(self.gindexer) - 1. + bsss)))

            # case 1) stepsize >= binsize
            # then bsss = 1; ppi = len(pred)/len(gindexer)
            #
            # case 2) stepsize < binsize
            # then bsss > 1; ppi = len(pred)/ (len(gindexer) -1 + bsss)
            resolution = int(region.length / bsss) // ppi
            for idx, region in enumerate(self.gindexer):

                val = [float(p) for p in pred[(idx*ppi):((idx+1)*ppi)]]

                bw_file.addEntries(str(region.chrom),
                                   int(region.start),
                                   values=val,
                                   span=int(resolution),
                                   step=int(resolution))
            bw_file.close()


class ExportBed(object):
    """Export predictions to bed.

    This function exports the predictions to bed format which allows you to
    inspect the predictions in a genome browser.

    Parameters
    ----------
    gindexer : GenomicIndexer
        GenomicIndexer that links the prediction for a certain region to
        its associated genomic coordinates.
    resolution : int
        Used to output the results.
    """
    def __init__(self, gindexer, resolution):
        self.gindexer = gindexer
        self.resolution = resolution

    def __call__(self, output_dir, name, results):

        gindexer = self.gindexer
        resolution = self.resolution

        # the last dimension holds the conditions. Each condition
        # needs to be stored in a separate file

        for keys in results:
            if isinstance(keys, str):
                k = (keys,)
            else:
                k = keys
            bed_content = pd.DataFrame(columns=['chr', 'start',
                                                'end', 'name', 'score'])
            for ridx, region in enumerate(gindexer):
                pred = results[keys]['value']

                nsplit = (region.end-region.start)//resolution

                starts = list(range(region.start,
                                    region.end,
                                    resolution))
                ends = list(range(region.start + resolution,
                                  region.end + resolution,
                                  resolution))
                cont = {'chr': [region.chrom] * nsplit,
                        'start': [s for s in starts],
                        'end': [e for e in ends],
                        'name': ['.'] * nsplit,
                        'score': pred[ridx*nsplit:((ridx+1)*nsplit)]}

                bed_entry = pd.DataFrame(cont)
                bed_content = bed_content.append(bed_entry, ignore_index=True, sort=False)

            bed_content.to_csv(os.path.join(
                output_dir,
                '{prefix}.{key}.bed'.format(
                    prefix=name, key='.'.join(k))),
                               sep='\t', header=False, index=False,
                               columns=['chr', 'start', 'end', 'name', 'score'])


class ExportClustermap(object):
    """Create of clustermap of the feature activities.

    This method utilizes
    `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_
    to illustrate feature activities of the neural net.

    Parameters
    ----------
    fform : str or None
        Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    figsize : tuple(int, int) or None
        Used to specify the figure size for matplotlib.
    annot : None
        Row annotation is used to create a row color annotation track
        for the figure.
    method : str
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
        Default: 'ward'
    metric : str
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
        Default: 'euclidean'
    z_score : int or None
        Whether to transform rows or columns to z-scores.
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    standard_scale :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_cluster : boolean
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_linkage :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    [row/col]_colors :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    mask :
        See `seaborn.clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_.
    """

    def __init__(self, fform=None, figsize=None,  # pylint: disable=too-many-locals
                 annot=None,
                 method='ward', metric='euclidean', z_score=None,
                 standard_scale=None, row_cluster=True, col_cluster=True,
                 row_linkage=None, col_linkage=None, row_colors=None,
                 col_colors=None, mask=None):

        self.fform = fform
        self.figsize = figsize
        self.annot = annot
        self.method = method
        self.metric = metric
        self.z_score = z_score
        self.standard_scale = standard_scale
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_linkage = row_linkage
        self.col_linkage = col_linkage
        self.row_colors = row_colors
        self.col_colors = col_colors
        self.mask = mask

    def __call__(self, output_dir, name, results):

        if sns is None:  # pragma: no cover
            raise Exception('seaborn not available. Please install seaborn.')

        annot = self.annot
        if annot is not None:

            firstkey = list(annot.keys())[0]
            pal = sns.color_palette('hls', len(set(annot[firstkey])))
            lut = dict(zip(set(annot[firstkey]), pal))
            self.row_colors = [lut[k] for k in annot[firstkey]]

        _rs = {k: results[k]['value'] for k in results}
        data = pd.DataFrame.from_dict(_rs)

        fform = self.fform

        if fform is not None:
            fform = fform
        else:
            fform = 'png'

        sns.clustermap(data,
                       method=self.method,
                       metric=self.metric,
                       z_score=self.z_score,
                       standard_scale=self.standard_scale,
                       row_cluster=self.row_cluster,
                       col_cluster=self.col_cluster,
                       row_linkage=self.row_linkage,
                       col_linkage=self.col_linkage,
                       col_colors=self.col_colors,
                       row_colors=self.row_colors,
                       mask=self.mask,
                       figsize=self.figsize).savefig(
                           os.path.join(output_dir, name + '.' + fform),
                           format=fform, dpi=700)


class ExportTsne(object):
    """Create a plot of the 2D t-SNE embedding of the feature activities.

    Parameters
    ----------
    fform : str or None
     Output file format. E.g. 'png', 'eps', etc. Default: 'png'.
    figsize : tuple(int, int) or None
     Used to specify the figure size for matplotlib.
    cmap : None or colormap
     Optional argument for matplotlib.pyplot.scatter. Only used if annot
     is abscent. Otherwise, marker colors are derived from the annotation.
    colors : None
     Optional argument for matplotlib.pyplot.scatter.
     Only used if annot
     is abscent. Otherwise, marker colors are derived from the annotation.
    norm :
     Optional argument for matplotlib.pyplot.scatter.
    alpha : None or float
     Opacity used for scatter plot markers.
    annot : None or dict.
     If annotation data is available, the color of the markers is automatically
     derived for the annotation.
    """

    def __init__(self, figsize=None, cmap=None, colors=None, norm=None,
                 alpha=None, fform=None, annot=None):

        self.figsize = figsize
        self.cmap = cmap
        self.colors = colors
        self.norm = norm
        self.alpha = alpha
        self.fform = fform
        self.annot = annot

    def __call__(self, output_dir, name, results):
        figsize = self.figsize
        annot = self.annot

        if TSNE is None:  # pragma: no cover
            raise Exception('scikit-learn not available. '
                            'Please install scikit-learn to be able to use export_tsne.')
        if plt is None:  # pragma: no cover
            raise Exception('matplotlib not available. '
                            'Please install matplotlib to be able to use export_tsne.')

        _rs = {k: results[k]['value'] for k in results}
        data = pd.DataFrame.from_dict(_rs)

        tsne = TSNE()
        embedding = tsne.fit_transform(data.values)

        if self.figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

        if self.annot is not None:
            firstkey = list(annot.keys())[0]
            pal = sns.color_palette('hls', len(set(annot[firstkey])))
            lut = dict(zip(set(annot[firstkey]), pal))

            for label in lut:
                plt.scatter(x=embedding[np.asarray(annot[firstkey]) == label, 0],
                            y=embedding[np.asarray(annot[firstkey]) == label, 1],
                            c=lut[label],
                            label=label,
                            norm=self.norm, alpha=self.alpha)

            plt.legend()
        else:
            plt.scatter(x=embedding[:, 0],
                        y=embedding[:, 1],
                        c=self.colors, cmap=self.cmap,
                        norm=self.norm, alpha=self.alpha)

        plt.axis('off')
        fform = self.fform
        if fform is not None:
            fform = fform
        else:
            fform = 'png'

        fig.savefig(os.path.join(output_dir, name + '.' + fform),
                    format=fform, dpi=700)


def trim_bed(inputbed, outputbed, divby):
    """Trims starts and ends of intervals.

    This function takes a BED file and produces a BED file
    with starts and ends being trimmed such that they
    are divisible by divby.

    Parameters
    ----------
    inputbed : str
        Input BED file.
    outputbed : str
        Trimmed output BED file.
    divby : int
        Factor to trim the starts and ends with.
    """
    with open(outputbed, 'w') as bed:
        regions = _get_genomic_reader(inputbed)
        for region in regions:
            start = int(np.ceil(region.start / divby)) * divby
            end = (region.end // divby) * divby
            bed.write('{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n'
                      .format(chrom=region.chrom,
                              start=start,
                              end=end,
                              name=region.name,
                              score=region.score if region.score is not None else 0,
                              strand=region.strand))
