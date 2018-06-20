"""Bioseq dataset"""

import Bio
import numpy as np
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _complement_index
from janggu.utils import _iv_to_str
from janggu.utils import as_onehot
from janggu.utils import seq2ind
from janggu.utils import sequence_padding
from janggu.utils import sequences_from_fasta


class Bioseq(Dataset):
    """Bioseq class.

    This datastructure holds a biological sequence
    and determines the one-hot encodeing
    for the purpose of a deep learning application.

    The sequence can represent nucleotide or peptide sequences,
    which can be conventiently fetched from a raw fasta-file.

    Parameters
    -----------
    name : str
        Name of the dataset
    garray : :class:`GenomicArray`
        A genomic array that holds the sequence data.
    gindxer : :class:`GenomicIndexer` or None
        A genomic index mapper that translates an integer index to a
        genomic coordinate. Can be None, if the Dataset is only loaded.
    """

    _order = None
    _alphabetsize = None
    _flank = None
    _gindexer = None

    def __init__(self, name, garray, gindexer):

        self.garray = garray
        self.gindexer = gindexer
        self._rcindex = [_complement_index(idx, garray.order)
                         for idx in range(pow(4, garray.order))]

        Dataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _make_genomic_array(name, fastafile, order, storage, seqtype, cache=True, datatags=None,
                            overwrite=False):
        """Create a genomic array or reload an existing one."""

        # always use int 16 to store bioseq indices
        # do not use int8 at the moment, because 'N' is encoded
        # as -1024, which causes an underflow with int8.
        dtype = 'int16'

        # Load sequences from refgenome
        seqs = []
        if isinstance(fastafile, str):
            fastafile = [fastafile]

        if not isinstance(fastafile[0], Bio.SeqRecord.SeqRecord):
            for fasta in fastafile:
                # += is necessary since sequences_from_fasta
                # returns a list
                seqs += sequences_from_fasta(fasta, seqtype)
        else:
            # This is already a list of SeqRecords
            seqs = fastafile

        # Extract chromosome lengths
        chromlens = {}

        for seq in seqs:
            chromlens[seq.id] = len(seq) - order + 1

        def _seq_loader(cover, seqs, order):
            print('Convert sequences to index array')
            for seq in seqs:
                interval = GenomicInterval(seq.id, 0,
                                           len(seq) - order + 1, '.')

                indarray = np.asarray(seq2ind(seq), dtype=dtype)

                if order > 1:
                    # for higher order motifs, this part is used
                    filter_ = np.asarray([pow(len(seq.seq.alphabet.letters),
                                              i) for i in range(order)])
                    indarray = np.convolve(indarray, filter_, mode='valid')

                cover[interval, 0] = indarray

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
                                     loader=_seq_loader,
                                     loader_args=(seqs, order))

        return cover

    @classmethod
    def create_from_refgenome(cls, name, refgenome, regions=None,
                              stepsize=200, binsize=200,
                              flank=0, order=1, storage='ndarray',
                              datatags=None,
                              cache=False, overwrite=False):
        """Create a Bioseq class from a reference genome.

        This requires a reference genome in fasta format to load the data from.
        If regions points to a bed file, the dataset will also be indexed
        accordingly. Otherwise, a GenomicIndexer must be attached later.

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
            Whether to cache the dataset. Default: False.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        """
        # fill up int8 rep of DNA
        # load bioseq, region index, and within region index

        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        if not isinstance(refgenome, Bio.SeqRecord.SeqRecord):
            seqs = sequences_from_fasta(refgenome, 'dna')
        else:
            # This is already a list of SeqRecords
            seqs = refgenome

        if gindexer is not None:
            # the genome is loaded with a bed file,
            # only the specific subset is loaded
            # to keep the memory overhead low.
            # Otherwise the entire reference genome is loaded.
            rgen = {seq.id: seq for seq in seqs}
            subseqs = []
            for giv in gindexer:
                subseq = rgen[giv.chrom][giv.start:(giv.end)]
                subseq.id = _iv_to_str(giv.chrom, giv.start, giv.end - order + 1)
                subseq.name = subseq.id
                subseq.description = subseq.id

                subseqs.append(subseq)
            seqs = subseqs

        garray = cls._make_genomic_array(name, seqs, order, storage, 'dna',
                                         datatags=datatags,
                                         cache=cache,
                                         overwrite=overwrite)

        garray._full_genome_stored = True if gindexer is None else False

        ob_ = cls(name, garray, gindexer)
        ob_._alphabetsize = len(seqs[0].seq.alphabet.letters)
        return ob_

    @classmethod
    def create_from_seq(cls, name,  # pylint: disable=too-many-locals
                        fastafile,
                        storage='ndarray',
                        seqtype='dna',
                        order=1,
                        fixedlen=None,
                        datatags=None,
                        cache=False,
                        overwrite=False):
        """Create a Bioseq class from a biological sequences.

        This allows to load sequence of equal lengths to be loaded.

        Parameters
        -----------
        name : str
            Name of the dataset
        fastafile : str or list(str) or list(Bio.SeqRecord)
            Fasta file or list of fasta files from which the sequences
            are loaded or a list of Bio.SeqRecord.SeqRecord.
        seqtype : str
            Indicates whether a nucleotide or peptide sequence is loaded
            using 'dna' or 'protein' respectively. Default: 'dna'.
        order : int
            Order for the one-hot representation. Default: 1.
        fixedlen : int or None
            Forces the sequences to be of equal length by truncation or
            padding. If set to None, it will be assumed that the sequences
            are already of equal length. An exception is raised if this is
            not the case.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'memmap' or
            'hdf5'. Default: 'ndarray'.
        datatags : list(str) or None
            List of datatags. Default: None.
        cache : boolean
            Whether to cache the dataset. Default: False.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        """
        seqs = []
        if isinstance(fastafile, str):
            fastafile = [fastafile]

        if not isinstance(fastafile[0], Bio.SeqRecord.SeqRecord):
            for fasta in fastafile:
                # += is necessary since sequences_from_fasta
                # returns a list
                seqs += sequences_from_fasta(fasta, seqtype)
        else:
            # This is already a list of SeqRecords
            seqs = fastafile

        if fixedlen is not None:
            seqs = sequence_padding(seqs, fixedlen)

        # Check if sequences are equally long
        lens = [len(seq) for seq in seqs]
        assert lens == [len(seqs[0])] * len(seqs), "Input sequences must " + \
            "be of equal length."

        # Chromnames are required to be Unique
        chroms = [seq.id for seq in seqs]
        assert len(set(chroms)) == len(seqs), "Sequence IDs must be unique."
        # now mimic a dataframe representing a bed file

        garray = cls._make_genomic_array(name, seqs, order, storage, seqtype,
                                         cache=cache, datatags=datatags,
                                         overwrite=overwrite)

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

        ob_ = cls(name, garray, gindexer)
        ob_._alphabetsize = len(seqs[0].seq.alphabet.letters)
        return ob_

    def __repr__(self):  # pragma: no cover
        return 'Bioseq("{}")'.format(self.name,)

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

    def iseq4idx(self, idxs):
        """Extracts the Bioseq sequence for set of indices.

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
            Nucleotide sequences associated with the region indices
            with shape `(len(idxs), sequence_length + 2*flank - order + 1)`
        """

        # for each index read use the adaptor indices to retrieve the seq.
        iseq = np.zeros((len(idxs), self.gindexer.binsize +
                         2*self.gindexer.flank - self.garray.order + 1), dtype="int16")

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            interval.end += - self.garray.order + 1

            # Computing the forward or reverse complement of the
            # sequence, depending on the strand flag.
            if interval.strand in ['.', '+']:
                iseq[i] = np.asarray(self.garray[interval][:, 0, 0])
            else:
                iseq[i] = np.asarray(
                    [self._rcindex[val] for val in self.garray[interval][:, 0, 0]])[::-1]

        return iseq

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
            raise IndexError('Bioseq.__getitem__: '
                             + 'index must be iterable')

        data = as_onehot(self.iseq4idx(idxs), self.garray.order,
                         self._alphabetsize)

        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        return (len(self), self.gindexer.binsize +
                2*self.gindexer.flank - self.garray.order + 1, 1,
                pow(self._alphabetsize, self.garray.order))
