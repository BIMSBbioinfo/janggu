import os

import numpy as np
from HTSeq import GenomicInterval

from beluga.data.data import BlgDataset
from beluga.data.genomic_indexer import BlgGenomicIndexer
from beluga.data.htseq_extension import BlgGenomicArray
from beluga.data.utils import as_onehot
from beluga.data.utils import dna2ind
from beluga.data.utils import sequences_from_fasta
from beluga.data.utils import REV_COMP_MAP as REV_COMP


class DnaBlgDataset(BlgDataset):
    """DnaBlgDataset class.

    This datastructure holds a DNA sequence for the purpose of a deep learning
    application.
    The sequence can conventiently fetched from a raw fasta-file.
    Upon indexing or slicing of the dataset, the one-hot representation
    for the respective locus will be returned.

    Note
    ----
    Caching is only used with storage mode 'memmap' or 'hdf5'.
    We recommend to use 'hdf5' for performance reasons.

    Parameters
    -----------
    name : str
        Name of the dataset
    garray : :class:`BlgGenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`BlgGenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Flanking regions in basepairs to be extended up and downstream.
        Default: 150.
    order : int
        Order for the one-hot representation. Default: 1.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.

    Attributes
    -----------
    name : str
        Name of the dataset
    garray : :class:`BlgGenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`BlgGenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Flanking regions in basepairs to be extended up and downstream.
        Default: 150.
    order : int
        Order for the one-hot representation. Default: 1.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.
    """

    _order = None
    _flank = None

    def __init__(self, name, garray, gindexer, flank=150, order=1):

        self.flank = flank
        self.order = order
        self.garray = garray
        self.gindexer = gindexer
        self._rcindex = [_rcindex(idx, order) for idx in range(pow(4, order))]

        BlgDataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _make_genomic_array(name, fastafile, order, storage, cachedir='',
                            overwrite=False):
        """Create a genomic array or reload an existing one."""

        # Load sequences from refgenome
        seqs = []
        if isinstance(fastafile, str):
            fastafile = [fastafile]

        for fasta in fastafile:
            seqs += sequences_from_fasta(fasta)

        chromlens = {}

        for seq in seqs:
            chromlens[seq.id] = len(seq) - order + 1

        if not (storage == 'memmap' or storage == 'ndarray' or
                storage == 'hdf5'):
            raise Exception('storage must be memmap, ndarray or hdf5')

        filename = '_'.join([os.path.basename(fasta) for fasta in fastafile])
        if storage == 'memmap':
            cachedir = os.path.join(cachedir, name,
                                    os.path.basename(filename))
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)

            paths = [os.path.join(cachedir, '{}..nmm'.format(x))
                     for x in iter(chromlens)]
            nmms = [os.path.exists(p) for p in paths]
        elif storage == 'hdf5':
            cachedir = os.path.join(cachedir, name,
                                    os.path.basename(filename))
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)

            paths = [os.path.join(cachedir, '{}..h5'.format(x))
                     for x in iter(chromlens)]
            nmms = [os.path.exists(p) for p in paths]
        else:
            nmms = [False]
            cachedir = ''

        garray = BlgGenomicArray(chromlens, stranded=False,
                                 typecode='int16',
                                 storage=storage, memmap_dir=cachedir,
                                 overwrite=overwrite)

        if all(nmms) and not overwrite:
            print('Reload BlgGenomicArray from {}'.format(cachedir))
        else:
            # Convert sequences to index array
            print('Convert sequences to index array')
            for seq in seqs:
                interval = GenomicInterval(seq.id, 0,
                                           len(seq) - order + 1, '.')

                dna = np.asarray(dna2ind(seq), dtype='int16')

                if order > 1:
                    # for higher order motifs, this part is used
                    filter_ = np.asarray([pow(4, i) for i in range(order)])
                    dna = np.convolve(dna, filter_, mode='valid')

                garray[interval] = dna

        return garray

    @classmethod
    def create_from_refgenome(cls, name, refgenome, regions,
                              stride=50, reglen=200,
                              flank=150, order=1, storage='hdf5',
                              cachedir='', overwrite=False):
        """Create a DnaBlgDataset class from a reference genome.

        This requires a reference genome in fasta format as well as a bed-file
        that holds the regions of interest.

        Parameters
        -----------
        name : str
            Name of the dataset
        refgenome : str
            Fasta file.
        regions : str
            BED- or GFF-filename.
        reglen : int
            Region length in basepairs to be considered. Default: 200.
        stride : int
            Stride in basepairs for traversing the genome. Default: 50.
        flank : int
            Flanking regions in basepairs to be extended up and downstream.
            Default: 150.
        order : int
            Order for the one-hot representation. Default: 1.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'memmap' or
            'hdf5'. Default: 'hdf5'.
        cachedir : str
            Directory in which the cachefiles are located. Default: ''.
        """
        # fill up int8 rep of DNA
        # load dna, region index, and within region index

        gindexer = BlgGenomicIndexer.create_from_file(regions, reglen, stride)

        garray = cls._make_genomic_array(name, refgenome, order, storage,
                                         cachedir=cachedir,
                                         overwrite=overwrite)

        return cls(name, garray, gindexer, flank, order)

    @classmethod
    def create_from_fasta(cls, name, fastafile, storage='ndarray',
                          order=1, cachedir='', overwrite=False):
        """Create a DnaBlgDataset class from a fastafile.

        This allows to load sequence of equal lengths to be loaded from
        a fastafile.

        Parameters
        -----------
        name : str
            Name of the dataset
        fastafile : str or list(str)
            Fasta file or list of fasta files.
        order : int
            Order for the one-hot representation. Default: 1.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'memmap' or
            'hdf5'. Default: 'ndarray'.
        cachedir : str
            Directory in which the cachefiles are located. Default: ''.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        """
        garray = cls._make_genomic_array(name, fastafile, order, storage,
                                         cachedir=cachedir,
                                         overwrite=overwrite)

        seqs = []
        if isinstance(fastafile, str):
            fastafile = [fastafile]

        for fasta in fastafile:
            seqs += sequences_from_fasta(fasta)

        # Check if sequences are equally long
        lens = [len(seq) for seq in seqs]
        assert lens == [len(seqs[0])] * len(seqs), "Input sequences must " + \
            "be of equal length."

        # Chromnames are required to be Unique
        chroms = [seq.id for seq in seqs]
        assert len(set(chroms)) == len(seqs), "Sequence IDs must be unique."
        # now mimic a dataframe representing a bed file

        reglen = lens[0]
        flank = 0
        stride = 1

        gindexer = BlgGenomicIndexer(reglen, stride)
        gindexer.chrs = chroms
        gindexer.offsets = [0]*len(lens)
        gindexer.inregionidx = [0]*len(lens)
        gindexer.strand = ['.']*len(lens)

        return cls(name, garray, gindexer, flank, order)

    def __repr__(self):  # pragma: no cover
        return 'DnaBlgDataset("{}", <garray>, <gindexer>, \
                flank={}, order={})'\
                .format(self.name, self.flank, self.order)

    def idna4idx(self, idxs):
        """Extracts the DNA sequence for set of indices.

        This method gets as input a list of indices (e.g.
        corresponding to genomic ranges for a given batch) and returns
        the respective sequences as an index array.

        Parameters
        ----------
        idxs : list(int)
            List of region indexes

        Returns
        -------
        numpy.array
            Nucleotide sequences associated with the regions
            with shape `(len(idxs), sequence_length + 2*flank - order + 1)`
        """

        # for each index read use the adaptor indices to retrieve the seq.
        idna = np.empty((len(idxs), self.gindexer.resolution +
                         2*self.flank - self.order + 1), dtype="int16")

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]
            interval.start -= self.flank
            interval.end += self.flank - self.order + 1

            # Computing the forward or reverse complement of the
            # sequence, depending on the strand flag.
            if interval.strand in ['.', '+']:
                idna[i] = np.fromiter(self.garray[interval], dtype='int16')
            else:
                idna[i] = np.asarray(
                    [self._rcindex[val] for val in self.garray[interval]])[::-1]

        return idna

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = [idxs]
        if isinstance(idxs, slice):
            idxs = range(idxs.start if idxs.start else 0,
                         idxs.stop if idxs.stop else len(self),
                         idxs.step if idxs.step else 1)
        try:
            iter(idxs)
        except TypeError:
            raise IndexError('DnaBlgDataset.__getitem__: '
                             + 'index must be iterable')

        data = as_onehot(self.idna4idx(idxs), self.order)

        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        return (len(self), pow(4, self.order), self.gindexer.resolution +
                2*self.flank - self.order + 1, 1)

    @property
    def order(self):
        """Order of the one-hot representation"""
        return self._order

    @order.setter
    def order(self, value):
        if not isinstance(value, int) or value < 1:
            raise Exception('order must be a positive integer')
        if value > 4:
            raise Exception('order support only up to order=4.')
        self._order = value

    @property
    def flank(self):
        """Flanking bins"""
        return self._flank

    @flank.setter
    def flank(self, value):
        if not isinstance(value, int) or value < 0:
            raise Exception('_flank must be a non-negative integer')
        self._flank = value


def _rcindex(idx, order):
    rev_idx = np.arange(4)[::-1]
    irc = 0
    for iord in range(order):
        nuc = idx % 4
        idx = idx // 4
        irc += rev_idx[nuc] * pow(4, order - iord - 1)

    return irc


def _rcpermmatrix(order):
    perm = np.zeros((pow(4, order), pow(4, order)))
    for idx in range(pow(4, order)):
        jdx = _rcindex(idx, order)
        perm[jdx, idx] = 1

    return perm


class RevCompDnaBlgDataset(BlgDataset):
    """RevCompDnaBlgDataset class.

    This datastructure for accessing the reverse complement of a given
    :class:`DnaBlgDataset`.

    Parameters
    -----------
    name : str
        Name of the dataset
    dnadata : :class:`DnaBlgDataset`
        Forward strand representation of the sequence sequence data.

    Attributes
    -----------
    name : str
        Name of the dataset
    dnadata : :class:`DnaBlgDataset`
        Forward strand representation of the sequence sequence data.
    """

    def __init__(self, name, dnadata):
        self.dna = dnadata
        self.order = self.dna.order

        self.rcmatrix = _rcpermmatrix(self.order)

        BlgDataset.__init__(self, '{}'.format(name))

    def __repr__(self):
        return 'RevDnaBlgDataset("{}", <DnaBlgDataset>)'.format(self.name)

    def _as_revcomp(self, data):
        """Compute the revere complement of a given sequence.

        This method computes the reverse complement for the
        entire batch of sequences by means of a permutation matrix
        that has been determined beforehand.

        Parameters
        ----------
        data : numpy.array
            DNA sequence in one hot representation

        Returns
        -------
        numpy.array
            Revere complement of the given DNA sequence
            in one-hot representation.
        """
        # compute the reverse complement of the original sequence
        # This is facilitated by, using rcmatrix (a permutation matrix),
        # which computes the complementary base for a given nucletide
        # Additionally, the sequences is reversed by ::-1
        rcdata = np.empty(data.shape)
        rcdata[:, :, :, 0] = np.matmul(self.rcmatrix, data[:, :, ::-1, 0])
        return rcdata

    def __getitem__(self, idxs):

        data = self.dna[idxs]
        data = self._as_revcomp(data)

        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.dna)

    @property
    def shape(self):
        """Shape of the Dataset"""
        return self.dna.shape
