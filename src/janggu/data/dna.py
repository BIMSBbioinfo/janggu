"""Dna dataset"""

import numpy as np
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _complement_index
from janggu.utils import as_onehot
from janggu.utils import dna2ind
from janggu.utils import sequences_from_fasta


class Dna(Dataset):
    """Dna class.

    This datastructure holds a DNA sequence for the purpose of a deep learning
    application.
    The sequence can conventiently fetched from a raw fasta-file.
    Upon indexing or slicing of the dataset, the one-hot representation
    for the respective locus will be returned.

    Parameters
    -----------
    name : str
        Name of the dataset
    garray : :class:`GenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`GenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Flanking regions in basepairs to be extended up and downstream.
        Default: 150.
    order : int
        Order for the one-hot representation. Default: 1.
    """

    _order = None
    _flank = None
    _gindexer = None

    def __init__(self, name, garray, gindexer):

        self.garray = garray
        self.gindexer = gindexer
        self._rcindex = [_complement_index(idx, garray.order)
                         for idx in range(pow(4, garray.order))]

        Dataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _make_genomic_array(name, fastafile, order, storage, cache=True, datatags=None,
                            overwrite=False):
        """Create a genomic array or reload an existing one."""

        # always use int 16 to store dna indices
        # do not use int8 at the moment, because 'N' is encoded
        # as -1024, which causes an underflow with int8.
        dtype = 'int16'

        # Load sequences from refgenome
        seqs = []
        if isinstance(fastafile, str):
            fastafile = [fastafile]

        for fasta in fastafile:
            # += is necessary since sequences_from_fasta
            # returns a list
            seqs += sequences_from_fasta(fasta)

        # Extract chromosome lengths
        chromlens = {}

        for seq in seqs:
            chromlens[seq.id] = len(seq) - order + 1

        def _dna_loader(cover, seqs, order):
            print('Convert sequences to index array')
            for seq in seqs:
                interval = GenomicInterval(seq.id, 0,
                                           len(seq) - order + 1, '.')

                dna = np.asarray(dna2ind(seq), dtype=dtype)

                if order > 1:
                    # for higher order motifs, this part is used
                    filter_ = np.asarray([pow(4, i) for i in range(order)])
                    dna = np.convolve(dna, filter_, mode='valid')

                cover[interval, 0] = dna

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        datatags = [name] + datatags if datatags else [name]
        datatags += ['order{}'.format(order)]

        cover = create_genomic_array(chromlens, stranded=False,
                                     storage=storage,
                                     datatags=datatags,
                                     cache=cache,
                                     order=order,
                                     conditions=['idx'],
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     loader=_dna_loader,
                                     loader_args=(seqs, order))

        return cover

    @classmethod
    def create_from_refgenome(cls, name, refgenome, regions=None,
                              stepsize=200, binsize=200,
                              flank=0, order=1, storage='ndarray',
                              datatags=None,
                              cache=True, overwrite=False):
        """Create a Dna class from a reference genome.

        This requires a reference genome in fasta format as well as a bed-file
        that holds the regions of interest.

        Parameters
        -----------
        name : str
            Name of the dataset
        refgenome : str
            Fasta file.
        regions : str or None
            Bed-file defining the regions that comprise the dataset.
            If set to None, a genomic indexer must be attached later.
        binsize : int
            Binsize in basepairs to be read out. Default: 200.
        stepsize : int
            stepsize in basepairs for traversing the genome. Default: 200.
        flank : int
            Flanking regions in basepairs to be extended up and downstream.
            Default: 0.
        order : int
            Order for the one-hot representation. Default: 1.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'memmap' or
            'hdf5'. Default: 'hdf5'.
        datatags : list(str) or None
            List of datatags. Default: None.
        cache : boolean
            Whether to cache the dataset. Default: True.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        """
        # fill up int8 rep of DNA
        # load dna, region index, and within region index

        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        garray = cls._make_genomic_array(name, refgenome, order, storage,
                                         datatags=datatags,
                                         cache=cache,
                                         overwrite=overwrite)

        return cls(name, garray, gindexer)

    @classmethod
    def create_from_fasta(cls, name,  # pylint: disable=too-many-locals
                          fastafile,
                          storage='ndarray',
                          order=1,
                          datatags=None,
                          cache=True,
                          overwrite=False):
        """Create a Dna class from a fastafile.

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
        datatags : list(str) or None
            List of datatags. Default: None.
        cache : boolean
            Whether to cache the dataset. Default: True.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        """
        garray = cls._make_genomic_array(name, fastafile, order, storage,
                                         cache=cache, datatags=datatags,
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
        stepsize = 1

        # this is a special case. Here a GenomicIndexer will be created
        # with pseudo genomic coordinates
        gindexer = GenomicIndexer(reglen, stepsize, flank)
        gindexer.chrs = chroms
        gindexer.offsets = [0]*len(lens)
        gindexer.inregionidx = [0]*len(lens)
        gindexer.strand = ['.']*len(lens)
        gindexer.rel_end = [reglen + 2*flank]*len(lens)

        return cls(name, garray, gindexer)

    def __repr__(self):  # pragma: no cover
        return 'Dna("{}")'.format(self.name,)

    @property
    def gindexer(self):
        """GenomicIndexer property."""
        if self._gindexer is None:
            raise ValueError('GenomicIndexer has not been set yet. Please specify an indexer.')
        return self._gindexer

    @gindexer.setter
    def gindexer(self, gindexer):
        if gindexer is None:
            self._gindexer = None
            return

        if (gindexer.stepsize % self.garray.resolution) != 0:
            raise ValueError('gindexer.stepsize must be divisible by resolution')
        self._gindexer = gindexer

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
        idna = np.zeros((len(idxs), self.gindexer.binsize +
                         2*self.gindexer.flank - self.garray.order + 1), dtype="int16")

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            interval.end += - self.garray.order + 1

            # Computing the forward or reverse complement of the
            # sequence, depending on the strand flag.
            if interval.strand in ['.', '+']:
                idna[i] = np.asarray(self.garray[interval][:, 0, 0])
            else:
                idna[i] = np.asarray(
                    [self._rcindex[val] for val in self.garray[interval][:, 0, 0]])[::-1]

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
            raise IndexError('Dna.__getitem__: '
                             + 'index must be iterable')

        data = as_onehot(self.idna4idx(idxs), self.garray.order)

        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        return (len(self), self.gindexer.binsize +
                2*self.gindexer.flank - self.garray.order + 1,
                pow(4, self.garray.order), 1)
