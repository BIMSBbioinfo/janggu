import os

import numpy as np
from data import BwDataset
from genomic_indexer import BwGenomicIndexer
from HTSeq import GenomicInterval
from htseq_extension import BwGenomicArray
from pandas import DataFrame
from utils import dna2ind
from utils import sequencesFromFasta


class DnaBwDataset(BwDataset):
    """DnaBwDataset class.

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
    garray : :class:`BwGenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`BwGenomicIndexer`
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
    garray : :class:`BwGenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`BwGenomicIndexer`
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

    def __init__(self, name, garray, gindexer,
                 flank=150, order=1, cachedir=None):

        self.flank = flank
        self.order = order
        self.garray = garray
        self.gindexer = gindexer

        BwDataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _makeGenomicArray(name, fastafile, order, storage, cachedir='',
                          overwrite=False):
        """Create a genomic array or reload an existing one."""

        # Load sequences from refgenome
        seqs = sequencesFromFasta(fastafile)

        chromlens = {}

        for seq in seqs:
            chromlens[seq.id] = len(seq) - order + 1

        if not (storage == 'memmap' or storage == 'ndarray' or
                storage == 'hdf5'):
            raise Exception('storage must be memmap, ndarray or hdf5')

        if storage == 'memmap':
            memmap_dir = os.path.join(cachedir, name,
                                      os.path.basename(fastafile))
            if not os.path.exists(memmap_dir):
                os.makedirs(memmap_dir)

            ps = [os.path.join(memmap_dir, '{}..nmm'.format(x))
                  for x in chromlens.keys()]
            nmms = [os.path.exists(p) for p in ps]
        elif storage == 'hdf5':
            memmap_dir = os.path.join(cachedir, name,
                                      os.path.basename(fastafile))
            if not os.path.exists(memmap_dir):
                os.makedirs(memmap_dir)

            ps = [os.path.join(memmap_dir, '{}..h5'.format(x))
                  for x in chromlens.keys()]
            nmms = [os.path.exists(p) for p in ps]
        else:
            nmms = [False]
            memmap_dir = ''

        garray = BwGenomicArray(chromlens, stranded=False,
                                typecode='int16',
                                storage=storage, memmap_dir=memmap_dir,
                                overwrite=overwrite)

        if all(nmms) and not overwrite:
            print('Reload BwGenomicArray from {}'.format(memmap_dir))
        else:
            # Convert sequences to index array
            print('Convert sequences to index array')
            for seq in seqs:
                iv = GenomicInterval(seq.id, 0, len(seq) - order + 1, '.')

                dna = np.asarray(dna2ind(seq), dtype='int16')

                if order > 1:
                    # for higher order motifs, this part is used
                    filter = np.asarray([pow(4, i) for i in range(order)])
                    dna = np.convolve(dna, filter, mode='valid')

                garray[iv] = dna

        return garray

    @classmethod
    def fromRefGenome(cls, name, refgenome, regions,
                      stride=50, reglen=200,
                      flank=150, order=1, storage='hdf5',
                      cachedir='', overwrite=False):
        """Create a DnaBwDataset class from a reference genome.

        This requires a reference genome in fasta format as well as a bed-file
        that holds the regions of interest.

        Parameters
        -----------
        name : str
            Name of the dataset
        refgenome : str
            Fasta file.
        regions : :class:`pandas.DataFrame` or str
            bed-filename or content of a bed-file as :class:`pandas.DataFrame`.
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

        gindexer = BwGenomicIndexer(regions, reglen, stride)

        garray = cls._makeGenomicArray(name, refgenome, order, storage,
                                       cachedir=cachedir,
                                       overwrite=overwrite)

        return cls(name, garray, gindexer, flank, order)

    @classmethod
    def fromFasta(cls, name, fastafile, storage='ndarray',
                  order=1, cachedir='', overwrite=False):
        """Create a DnaBwDataset class from a fastafile.

        This allows to load sequence of equal lengths to be loaded from
        a fastafile.

        Parameters
        -----------
        name : str
            Name of the dataset
        fastafile : str
            Fasta file.
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
        garray = cls._makeGenomicArray(name, fastafile, order, storage,
                                       cachedir=cachedir,
                                       overwrite=overwrite)

        seqs = sequencesFromFasta(fastafile)

        # Check if sequences are equally long
        lens = [len(seq) for seq in seqs]
        assert lens == [len(seqs[0])] * len(seqs), "Input sequences must " + \
            "be of equal length."

        # Chromnames are required to be Unique
        chroms = [seq.id for seq in seqs]
        assert len(set(chroms)) == len(seqs), "Sequence IDs must be unique."
        # now mimic a dataframe representing a bed file

        regions = DataFrame({'chr': chroms, 'start': 0, 'end': lens})

        reglen = lens[0]
        flank = 0
        stride = 1

        gindexer = BwGenomicIndexer(regions, reglen, stride)

        return cls(name, garray, gindexer, flank, order)

    def __repr__(self):
        return 'DnaBwDataset("{}", <garray>, <gindexer>, \
                flank={}, order={})'\
                .format(self.name, self.flank, self.order)

    def idna4idx(self, idxs):
        """Extracts the DNA sequence for set of indices."""

        # for each index read use the adaptor indices to retrieve the seq.
        idna = np.empty((len(idxs), self.gindexer.resolution +
                         2*self.flank - self.order + 1), dtype="int16")

        for i, idx in enumerate(idxs):
            iv = self.gindexer[idx]
            iv.start -= self.flank
            iv.end += self.flank - self.order + 1

            idna[i] = np.asarray(list(self.garray[iv]))

        return idna

    def as_onehot(self, idna):
        """Converts the sequence into one-hot representation."""

        onehot = np.zeros((len(idna), pow(4, self.order),
                           idna.shape[1], 1), dtype='int8')
        for nuc in np.arange(pow(4, self.order)):
            Ilist = np.where(idna == nuc)
            onehot[Ilist[0], nuc, Ilist[1], 0] = 1

        return onehot

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = [idxs]
        if not isinstance(idxs, list):
            raise IndexError('DnaBwDataset.__getitem__ '
                             + 'requires "int" or "list"')

        data = self.as_onehot(self.idna4idx(idxs))

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the Dataset"""
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


class RevCompDnaBwDataset(BwDataset):
    """RevCompDnaBwDataset class.

    This datastructure for accessing the reverse complement of a given
    :class:`DnaBwDataset`.

    Parameters
    -----------
    name : str
        Name of the dataset
    dnadata : :class:`DnaBwDataset`
        Forward strand representation of the sequence sequence data.

    Attributes
    -----------
    name : str
        Name of the dataset
    dnadata : :class:`DnaBwDataset`
        Forward strand representation of the sequence sequence data.
    """

    def __init__(self, name, dnadata):
        self.dna = dnadata
        self.order = self.dna.order

        self.rcmatrix = self._rcpermmatrix(self.order)

        BwDataset.__init__(self, '{}'.format(name))

    def __repr__(self):
        return 'RevDnaBwDataset("{}", <DnaBwDataset>)'.format(self.name)

    def _rcindex(self, idx, order):
        x = np.arange(4)[::-1]
        irc = 0
        for o in range(order):
            nuc = idx % 4
            idx = idx // 4
            irc += x[nuc] * pow(4, order - o - 1)

        return irc

    def _rcpermmatrix(self, order):
        P = np.zeros((pow(4, order), pow(4, order)))
        for idx in range(pow(4, order)):
            jdx = self._rcindex(idx, order)
            P[jdx, idx] = 1

        return P

    def as_revcomp(self, data):
        # compute the reverse complement of the original sequence
        # This is facilitated by, using rcmatrix (a permutation matrix),
        # which computes the complementary base for a given nucletide
        # Additionally, the sequences is reversed by ::-1
        rcdata = np.empty(data.shape)
        rcdata[:, :, :, 0] = np.matmul(self.rcmatrix, data[:, :, ::-1, 0])
        return rcdata

    def __getitem__(self, idxs):

        data = self.dna[idxs]
        # self.as_onehot(self.idna4idx(idxs))
        data = self.as_revcomp(data)

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.dna)

    @property
    def shape(self):
        """Shape of the Dataset"""
        return self.dna.shape
