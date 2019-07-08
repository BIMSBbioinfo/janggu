"""Bioseq dataset"""

import logging
import warnings
from collections import OrderedDict
from itertools import product

import Bio
import numpy as np
from progress.bar import Bar
from pybedtools import Interval
from pysam import VariantFile

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.data.genomicarray import create_sha256_cache
from janggu.utils import NMAP
from janggu.utils import NOLETTER
from janggu.utils import _complement_index
from janggu.utils import _iv_to_str
from janggu.utils import as_onehot
from janggu.utils import seq2ind
from janggu.utils import sequence_padding
from janggu.utils import sequences_from_fasta
from janggu.version import version


class GenomicSizeLazyLoader:
    """GenomicSizeLazyLoader class

    This class facilitates lazy loading of the DNA sequence.
    The DNA sequence is loaded to determine the length of the reference genome
    to allocate the required memory and to fetch the sequences.

    The call method is invoked for constructing a new genomic array.
    """
    def __init__(self, fastafile, seqtype, store_whole_genome, gindexer):
        self.fastafile = fastafile
        self.seqtype = seqtype
        self.store_whole_genome = store_whole_genome
        self.gindexer = gindexer
        self.seqs_ = None
        self.gsize_ = None

    def load_sequence(self):
        print('loading from lazy loader')
        store_whole_genome = self.store_whole_genome
        gindexer = self.gindexer

        if isinstance(self.fastafile, str):
            seqs = sequences_from_fasta(self.fastafile, self.seqtype)
        else:
            # This is already a list of SeqRecords
            seqs = self.fastafile

        if not store_whole_genome and gindexer is not None:
            # the genome is loaded with a bed file,
            # only the specific subset is loaded
            # to keep the memory overhead low.
            # Otherwise the entire reference genome is loaded.
            rgen = OrderedDict(((seq.id, seq) for seq in seqs))
            subseqs = []
            for giv in gindexer:
                subseq = rgen[giv.chrom][max(giv.start, 0):min(giv.end, len(rgen[giv.chrom]))]
                if giv.start < 0:
                    subseq = 'N' * (-giv.start) + subseq
                if len(subseq) < giv.length:
                    subseq = subseq + 'N' * (giv.length - len(subseq))
                subseq.id = _iv_to_str(giv.chrom, giv.start, giv.end)
                subseq.name = subseq.id
                subseq.description = subseq.id
                subseqs.append(subseq)
            seqs = subseqs
            gsize = gindexer

        if store_whole_genome:
            gsize = OrderedDict(((seq.id, len(seq)) for seq in seqs))
            gsize = GenomicIndexer.create_from_genomesize(gsize)

        self.gsize_ = gsize
        self.seqs_ = seqs

    @property
    def gsize(self):
        if self.gsize_ is None:
            self.load_sequence()
        return self.gsize_

    @property
    def seqs(self):  # pragma: no cover
        if self.seqs_ is None:
            self.load_sequence()
        return self.seqs_

    def __call__(self):
        return self.gsize

    def tostr(self):
        return "full_genome_lazy_loading"


class SeqLoader:
    """SeqLoader class.

    This class loads a GenomicArray with sequences obtained
    from FASTA files.

    Parameters
    -----------
    gsize : GenomicIndexer or callable
        GenomicIndexer indicating the genome size or callable
        that returns a genomic indexer for lazy loading.
    seqs : list(Bio.SeqRecord) or str
        List of sequences contained in Biopython SeqRecords or
        fasta file name
    order : int
        Order of the one-hot representation.
    """
    def __init__(self, gsize, seqs, order):
        self.seqs = seqs
        self.order = order
        self.gsize = gsize

    def __call__(self, garray):
        if callable(self.gsize):
            gsize = self.gsize()
            seqs = self.gsize.seqs
        else:
            gsize = self.gsize
            seqs = self.seqs
        order = self.order
        dtype = garray.typecode

        bar = Bar('Loading sequences', max=len(gsize))
        for region, seq in zip(gsize, seqs):

            indarray = np.asarray(seq2ind(seq), dtype=dtype)

            if order > 1:
                # for higher order motifs, this part is used
                filter_ = np.asarray([pow(len(seq.seq.alphabet.letters),
                                          i) for i in range(order)])
                indarray = np.convolve(indarray, filter_, mode='valid')

            garray[region, 0] = indarray.reshape(-1, 1)
            bar.next()
        bar.finish()


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
    alphabet : str
        String of sequence alphabet. For example, 'ACGT'.
    """

    _order = None
    _alphabet = None
    _alphabetsize = None
    _flank = None
    _gindexer = None

    def __init__(self, name, garray, gindexer, alphabet, channel_last):

        self.garray = garray
        self.gindexer = gindexer
        self._alphabet = alphabet
        self.conditions = [''.join(item) for item in
                           product(sorted(self._alphabet),
                                   repeat=self.garray.order)]
        self._alphabetsize = len(self._alphabet)
        self._rcindex = [_complement_index(idx, garray.order)
                         for idx in range(pow(self._alphabetsize, garray.order))]
        self._channel_last = channel_last

        Dataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _make_genomic_array(name, gsize, seqs, order, storage,
                            cache=None, datatags=None,
                            overwrite=False, store_whole_genome=True,
                            random_state=None):

        if overwrite:
            warnings.warn('overwrite=True is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)
        if datatags is not None:
            warnings.warn('datatags is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)

        """Create a genomic array or reload an existing one."""

        # always use int 16 to store bioseq indices
        # do not use int8 at the moment, because 'N' is encoded
        # as -1024, which causes an underflow with int8.
        dtype = 'int16'

        # Extract chromosome lengths
        seqloader = SeqLoader(gsize, seqs, order)

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        datatags = [name]

        if cache:
            files = seqs
            parameters = [gsize.tostr(),
                          storage, dtype, order,
                          store_whole_genome, version, random_state]
            cache_hash = create_sha256_cache(files, parameters)
        else:
            cache_hash = None

        garray = create_genomic_array(gsize, stranded=False,
                                      storage=storage,
                                      datatags=datatags,
                                      cache=cache_hash,
                                      store_whole_genome=store_whole_genome,
                                      order=order,
                                      conditions=['idx'],
                                      overwrite=overwrite,
                                      padding_value=NOLETTER,
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
                              random_state=None,
                              store_whole_genome=False):
        """Create a Bioseq class from a reference genome.

        This constructor loads nucleotide sequences from a reference genome.
        If regions of interest (ROI) is supplied, only the respective sequences
        are loaded, otherwise the entire genome is fetched.

        Parameters
        -----------
        name : str
            Name of the dataset
        refgenome : str or Bio.SeqRecord.SeqRecord
            Reference genome location pointing to a fasta file
            or a SeqRecord object from Biopython that contains the sequences.
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
            Storage mode for storing the sequence may be 'ndarray' or 'hdf5'.
            Default: 'ndarray'.
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
        random_state : None or int
            random_state used to internally randomize the dataset.
            This option is best used when consuming data for training
            from an HDF5 file. Since random data access from HDF5
            may be probibitively slow, this option allows to randomize
            the dataset during loading.
            In case an integer-valued random_state seed is supplied,
            make sure that all training datasets
            (e.g. input and output datasets) use the same random_state
            value so that the datasets are synchronized.
            Default: None means that no randomization is used.
        """
        # fill up int8 rep of DNA
        # load bioseq, region index, and within region index

        if storage not in ['ndarray', 'hdf5']:
            raise ValueError('Available storage options for Bioseq are: ndarray or hdf5')

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank,
                                                       random_state=random_state)
        else:
            gindexer = None

        if not store_whole_genome and gindexer is None:
            raise ValueError('Either roi must be supplied or store_whole_genome must be True')

        gsize = GenomicSizeLazyLoader(refgenome, 'dna', store_whole_genome, gindexer)

        garray = cls._make_genomic_array(name, gsize, [refgenome], order, storage,
                                         datatags=datatags,
                                         cache=cache,
                                         overwrite=overwrite,
                                         store_whole_genome=store_whole_genome,
                                         random_state=random_state)

        return cls(name, garray, gindexer,
                   alphabet='ACGT',
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
            Storage mode for storing the sequence may be 'ndarray' or 'hdf5'.
            Default: 'ndarray'.
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
        if storage not in ['ndarray', 'hdf5']:
            raise ValueError('Available storage options for Bioseq are: ndarray or hdf5')

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

        reglen = lens[0]
        flank = 0
        stepsize = 1

        gindexer = GenomicIndexer(reglen, stepsize, flank, zero_padding=False)
        for chrom in chroms:
            gindexer.add_interval(chrom, 0, reglen, '.')

        garray = cls._make_genomic_array(name, gindexer, seqs, order, storage,
                                         cache=cache, datatags=datatags,
                                         overwrite=overwrite,
                                         store_whole_genome=False)

        return cls(name, garray, gindexer,
                   alphabet=seqs[0].seq.alphabet.letters,
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
            if len(dat) < iseq.shape[1]:
                iseq[i, len(dat):] = NOLETTER

        return iseq

    def _getsingleitem(self, interval):

        # Computing the forward or reverse complement of the
        # sequence, depending on the strand flag.
        if interval.strand in ['.', '+']:
            return np.asarray(self.garray[interval][:, 0, 0])

        return np.asarray([self._rcindex[val] if val >= 0 else val for val
                           in self.garray[interval][:, 0, 0]])[::-1]


    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            if len(idxs) == 3 or len(idxs) == 4:
                # interpret idxs as genomic interval
                idxs = Interval(*idxs)
            else:
                raise ValueError('idxs cannot be interpreted as genomic interval.'
                                 ' use (chr, start, end) or (chr, start, end, strand)')

        if isinstance(idxs, int):
            idxs = [idxs]
        elif isinstance(idxs, slice):
            idxs = range(idxs.start if idxs.start else 0,
                         idxs.stop if idxs.stop else len(self),
                         idxs.step if idxs.step else 1)
        elif isinstance(idxs, Interval):
            if not self.garray._full_genome_stored:
                raise ValueError('Indexing with Interval only possible '
                                 'when the whole genome (or chromosome) was loaded')

            data = np.zeros((1, idxs.length  - self.garray.order + 1))
            data[0] = self._getsingleitem(idxs)
            # accept a genomic interval directly
            data = as_onehot(data,
                             self.garray.order,
                             self._alphabetsize)

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
    def ndim(self):  # pragma: no cover
        """ndim"""
        return len(self.shape)


class VariantStreamer:
    """VariantStreamer class.

    This class takes a :class:`Bioseq` object and variants from
    a VCF file.
    It parses the VCF file entries and returns the sequence context
    for the reference and the alternative allele, respectively.

    Parameters
    -----------
    bioseq : :class:`Bioseq`
        Bioseq container containing a reference genome.
        Make sure that the reference genome was loaded with store_whole_genome=True.
    variants : str
        VCF file name. Contains the variants
    binsize : int
        Context region size must be compatible with the network architecture.
    batch_size : int
        Batch size for parsing the VCF file.
    filter_region : None or str
        BED file name. If None, all VCF file entries are considered.
    """
    def __init__(self, bioseq, variants, binsize, batch_size):
        self.bioseq = bioseq
        self.variants = variants
        self.binsize = binsize
        self.batch_size = batch_size
        self.logger = logging.getLogger('variantstreamer')

    def is_compatible(self, rec):
        """ Check compatibility of variant.

        If the variant is not compatible the method returns False,
        otherwise True.
        """
        if rec.alts is None or len(rec.alts) != 1 or len(rec.alts[0]) != 1:
            return False

        if not (rec.alts[0].upper() in NMAP and rec.ref.upper() in NMAP):
            return False

        return True

    def get_variant_count(self):
        """Obtains the number of admissible variants"""
        ncounts = 0
        for rec in VariantFile(self.variants).fetch():
            start = rec.pos-self.binsize//2 + (1 if self.binsize%2 == 0 else 0) - 1
            if start < 0:
                continue
            if self.is_compatible(rec):
                ncounts += 1
        return ncounts

    def flow(self):
        """Data flow generator."""

        refs = np.zeros((self.batch_size, self.binsize - self.bioseq.garray.order + 1, 1,
                         pow(self.bioseq._alphabetsize, self.bioseq.garray.order)))
        alts = np.zeros_like(refs)

        vcf = VariantFile(self.variants).fetch()

        try:
            while True:
                # construct genomic region
                names = []
                chroms = []
                poss = []
                rallele = []
                aallele = []

                ibatch = 0

                while ibatch < self.batch_size:
                    rec = next(vcf)


                    if not self.is_compatible(rec):
                        continue

                    start = rec.pos-self.binsize//2 + (1 if self.binsize%2 == 0 else 0) - 1
                    end = rec.pos+self.binsize//2

                    if start < 0:
                        continue

                    names.append(rec.id if rec.id is not None else '')
                    chroms.append(rec.chrom)
                    poss.append(rec.pos - 1)
                    rallele.append(rec.ref.upper())
                    aallele.append(rec.alts[0].upper())

                    iref = self.bioseq._getsingleitem(Interval(rec.chrom, start, end)).copy()

                    ref = as_onehot(iref[None, :], self.bioseq.garray.order,
                                    self.bioseq._alphabetsize)
                    refs[ibatch] = ref
                    #mutate the position with the variant

                    # only support single nucleotide variants at the moment
                    for o in range(self.bioseq.garray.order):

                        irefbase = iref[self.binsize//2 + o - self.bioseq.garray.order +
                                        (0 if self.binsize%2 == 0 else 1)]
                        irefbase = irefbase // pow(self.bioseq._alphabetsize, o)
                        irefbase = irefbase % self.bioseq._alphabetsize

                        if NMAP[rec.ref.upper()] != irefbase:
                            self.logger.info('VCF reference and reference genome not compatible.'
                                             'Expected reference {}, but VCF indicates {}.'.format(
                                                 irefbase, NMAP[rec.ref.upper()]) +
                                             'VCF-Record: {}:{}-{}>{};{}. Skipped.'.format(
                                                 rec.chrom, rec.pos, rec.ref,
                                                 rec.alts[0], rec.id))
                        else:

                            replacement = (NMAP[rec.alts[0].upper()] -
                                           NMAP[rec.ref.upper()]) * \
                                          pow(self.bioseq._alphabetsize, o)

                            iref[self.binsize//2 + o - self.bioseq.garray.order +
                                 (0 if self.binsize%2 == 0 else 1)] += replacement

                    alt = as_onehot(iref[None, :], self.bioseq.garray.order,
                                    self.bioseq._alphabetsize)

                    alts[ibatch] = alt

                    ibatch += 1
                yield names, chroms, poss, rallele, aallele, refs, alts

        except StopIteration:
            refs = refs[:ibatch]
            alts = alts[:ibatch]

            yield names, chroms, poss, rallele, aallele, refs, alts
