"""Bioseq dataset"""

import Bio
import numpy as np
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _complement_index
from janggu.utils import _iv_to_str
from janggu.utils import _str_to_iv
from janggu.utils import as_onehot
from janggu.utils import seq2ind
from janggu.utils import sequence_padding
from janggu.utils import sequences_from_fasta


class SeqLoader:
    """SeqLoader class.

    This class loads a GenomicArray with sequences obtained
    from FASTA files.

    Parameters
    -----------
    seqs : list(Bio.SeqRecord)
        List of sequences contained in Biopython SeqRecords.
    order : int
        Order of the one-hot representation.
    """
    def __init__(self, seqs, order):
        self.seqs = seqs
        self.order = order

    def __call__(self, garray):
        seqs = self.seqs
        order = self.order
        dtype = garray.typecode

        print('Convert sequences to index array')
        for seq in seqs:
            if garray._full_genome_stored:
                interval = GenomicInterval(seq.id, 0,
                                           len(seq) - order + 1, '.')
            else:
                interval = GenomicInterval(*_str_to_iv(seq.id,
                                                       template_extension=0))

            indarray = np.asarray(seq2ind(seq), dtype=dtype)

            if order > 1:
                # for higher order motifs, this part is used
                filter_ = np.asarray([pow(len(seq.seq.alphabet.letters),
                                          i) for i in range(order)])
                indarray = np.convolve(indarray, filter_, mode='valid')

            garray[interval, 0] = indarray.reshape(-1, 1)

class Bioseq(Dataset):
    """Bioseq class.

    This class maintains a set of biological sequences,
    e.g. nucleotide or amino acid sequences,
    and determines its one-hot encoding.

    Parameters
    -----------
    name : str
        Name of the dataset
    garray : :class:`GenomicArray`
        A genomic array that holds the sequence data.
    gindexer : :class:`GenomicIndexer` or None
        A genomic index mapper that translates an integer index to a
        genomic coordinate. Can be None, if the Dataset is only loaded.
    alphabetsize : int
        Alphabetsize of the sequence.
    """

    _order = None
    _alphabetsize = None
    _flank = None
    _gindexer = None

    def __init__(self, name, garray, gindexer, alphabetsize, channel_last):

        self.garray = garray
        self.gindexer = gindexer
        self._alphabetsize = alphabetsize
        self._rcindex = [_complement_index(idx, garray.order)
                         for idx in range(pow(alphabetsize, garray.order))]
        self._channel_last = channel_last

        Dataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _make_genomic_array(name, fastafile, order, storage,
                            cache=True, datatags=None,
                            overwrite=False, store_whole_genome=True):
        """Create a genomic array or reload an existing one."""

        # always use int 16 to store bioseq indices
        # do not use int8 at the moment, because 'N' is encoded
        # as -1024, which causes an underflow with int8.
        dtype = 'int16'

        # Load sequences from refgenome
        seqs = fastafile

        # Extract chromosome lengths
        chromlens = {}

        for seq in seqs:
            chromlens[seq.id] = len(seq) - order + 1

        seqloader = SeqLoader(seqs, order)

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        datatags = [name] + datatags if datatags else [name]
        datatags += ['order{}'.format(order)]

        garray = create_genomic_array(chromlens, stranded=False,
                                      storage=storage,
                                      datatags=datatags,
                                      cache=cache,
                                      store_whole_genome=store_whole_genome,
                                      order=order,
                                      conditions=['idx'],
                                      overwrite=overwrite,
                                      typecode=dtype,
                                      loader=seqloader)

        return garray

    @classmethod
    def create_from_refgenome(cls, name, refgenome, roi=None,
                              binsize=None,
                              stepsize=None,
                              flank=0, order=1,
                              storage='ndarray',
                              datatags=None,
                              cache=False,
                              overwrite=False,
                              channel_last=True,
                              store_whole_genome=False):
        """Create a Bioseq class from a reference genome.

        This constructor loads nucleotide sequences from a reference genome.
        If regions of interest (ROI) is supplied, only the respective sequences
        are loaded, otherwise the entire genome is fetched.

        Parameters
        -----------
        name : str
            Name of the dataset
        refgenome : str
            Fasta file.
        roi : str or None
            Bed-file defining the region of interest.
            If set to None, the sequence will be
            fetched from the entire genome and a
            genomic indexer must be attached later.
            Otherwise, the coverage is only determined
            for the region of interest.
        binsize : int or None
            Binsize in basepairs. For binsize=None,
            the binsize will be determined from the bed-file directly
            which requires that all intervals in the bed-file are of equal
            length. Otherwise, the intervals in the bed-file will be
            split to subintervals of length binsize in conjunction with
            stepsize. Default: None.
        stepsize : int or None
            stepsize in basepairs for traversing the genome.
            If stepsize is None, it will be set equal to binsize.
            Default: None.
        flank : int
            Flanking region in basepairs to be extended up and downstream of each interval.
            Default: 0.
        order : int
            Order for the one-hot representation. Default: 1.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'hdf5' or
            'sparse'. Default: 'hdf5'.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        overwrite : boolean
            Overwrite the cachefiles. Default: False.
        store_whole_genome : boolean
            Indicates whether the whole genome or only ROI
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False.
        """
        # fill up int8 rep of DNA
        # load bioseq, region index, and within region index


        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        if not store_whole_genome and gindexer is None:
            raise ValueError('Either roi must be supplied or store_whole_genome must be True')

        if isinstance(refgenome, str):
            seqs = sequences_from_fasta(refgenome, 'dna')
        else:
            # This is already a list of SeqRecords
            seqs = refgenome

        if not store_whole_genome and gindexer is not None:
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

        garray = cls._make_genomic_array(name, seqs, order, storage,
                                         datatags=datatags,
                                         cache=cache,
                                         overwrite=overwrite,
                                         store_whole_genome=store_whole_genome)

        return cls(name, garray, gindexer,
                   alphabetsize=len(seqs[0].seq.alphabet.letters),
                   channel_last=channel_last)

    @classmethod
    def create_from_seq(cls, name,  # pylint: disable=too-many-locals
                        fastafile,
                        storage='ndarray',
                        seqtype='dna',
                        order=1,
                        fixedlen=None,
                        datatags=None,
                        cache=False,
                        channel_last=True,
                        overwrite=False):
        """Create a Bioseq class from a biological sequences.

        This constructor loads a set of nucleotide or amino acid sequences.
        By default, the sequence are assumed to be of equal length.
        Alternatively, sequences can be truncated and padded to a fixed length.


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
            zero-padding. If set to None, it will be assumed that the sequences
            are already of equal length. An exception is raised if this is
            not the case. Default: None.
        storage : str
            Storage mode for storing the sequence may be 'ndarray', 'hdf5' or
            'sparse'. Default: 'ndarray'.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
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

        garray = cls._make_genomic_array(name, seqs, order, storage,
                                         cache=cache, datatags=datatags,
                                         overwrite=overwrite,
                                         store_whole_genome=True)

        reglen = lens[0]
        flank = 0
        stepsize = 1

        # this is a special case. Here a GenomicIndexer will be created
        # with pseudo genomic coordinates
        gindexer = GenomicIndexer(reglen, stepsize, flank)
        gindexer.chrs = chroms
        gindexer.starts = [0]*len(lens)
        gindexer.strand = ['.']*len(lens)
        gindexer.ends = [reglen + 2*flank]*len(lens)

        return cls(name, garray, gindexer,
                   alphabetsize=len(seqs[0].seq.alphabet.letters),
                   channel_last=channel_last)

    def __repr__(self):  # pragma: no cover
        return 'Bioseq("{}")'.format(self.name,)

    @property
    def gindexer(self):
        """GenomicIndexer property."""
        if self._gindexer is None:
            raise ValueError('GenomicIndexer has not been set yet. '
                             'Please specify an indexer.')
        return self._gindexer

    @gindexer.setter
    def gindexer(self, gindexer):
        if gindexer is None:
            self._gindexer = None
            return

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
                         2*self.gindexer.flank - self.garray.order + 1),
                        dtype="int16")

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            dat = self._getsingleitem(interval)

            iseq[i, :len(dat)] = dat

        return iseq

    def _getsingleitem(self, interval):
        interval.end += - self.garray.order + 1

        # Computing the forward or reverse complement of the
        # sequence, depending on the strand flag.
        if interval.strand in ['.', '+']:
            return np.asarray(self.garray[interval][:, 0, 0])

        return np.asarray([self._rcindex[val] for val
                           in self.garray[interval][:, 0, 0]])[::-1]


    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            if len(idxs) == 3 or len(idxs) == 4:
                # interpret idxs as genomic interval
                idxs = GenomicInterval(*idxs)
            else:
                raise ValueError('idxs cannot be interpreted as genomic interval.'
                                 ' use (chr, start, end) or (chr, start, end, strand)')

        if isinstance(idxs, int):
            idxs = [idxs]
        elif isinstance(idxs, slice):
            idxs = range(idxs.start if idxs.start else 0,
                         idxs.stop if idxs.stop else len(self),
                         idxs.step if idxs.step else 1)
        elif isinstance(idxs, GenomicInterval):
            if not self.garray._full_genome_stored:
                raise ValueError('Indexing with GenomicInterval only possible '
                                 'when the whole genome (or chromosome) was loaded')

            data = np.zeros((1, idxs.length  - self.garray.order + 1))
            data[0] = self._getsingleitem(idxs)
            # accept a genomic interval directly
            data = as_onehot(data,
                             self.garray.order,
                             self._alphabetsize)
            for transform in self.transformations:
                data = transform(data)
            if not self._channel_last:
                data = np.transpose(data, (0, 3, 1, 2))
            return data

        try:
            iter(idxs)
        except TypeError:
            raise IndexError('Bioseq.__getitem__: '
                             + 'index must be iterable')

        data = as_onehot(self.iseq4idx(idxs), self.garray.order,
                         self._alphabetsize)

        for transform in self.transformations:
            data = transform(data)

        if not self._channel_last:
            data = np.transpose(data, (0, 3, 1, 2))

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        if self._channel_last:
            return (len(self), self.gindexer.binsize +
                    2*self.gindexer.flank - self.garray.order + 1, 1,
                    pow(self._alphabetsize, self.garray.order))

        return (len(self),
                pow(self._alphabetsize, self.garray.order),
                self.gindexer.binsize +
                2*self.gindexer.flank - self.garray.order + 1, 1)

    @property
    def ndim(self):
        return len(self.shape)
