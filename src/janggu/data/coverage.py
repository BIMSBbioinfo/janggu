"""Coverage dataset"""

import os

import numpy as np
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _get_genomic_reader
from janggu.utils import _iv_to_str
from janggu.utils import _str_to_iv
from janggu.utils import get_genome_size_from_regions
from janggu.utils import get_chrom_length

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

try:
    import pyBigWig
except ImportError:  # pragma: no cover
    pyBigWig = None
try:
    import pysam
except ImportError:  # pragma: no cover
    pysam = None


class Cover(Dataset):
    """Cover class.

    This datastructure holds coverage information across the genome.
    The coverage can conveniently fetched from a list of bam-files,
    bigwig-file, bed-files or gff-files.

    Parameters
    -----------
    name : str
        Name of the dataset
    garray : :class:`GenomicArray`
        A genomic array that holds the coverage data
    gindexer : :class:`GenomicIndexer` or None
        A genomic indexer translates an integer index to a
        corresponding genomic coordinate.
        It can be None the genomic indexer is supplied later.
    padding_value : int or float
        Padding value used to pad variable size fragments. Default: 0.
    dimmode : str
        Dimension mode can be 'first' or 'all'.
        'first' returns only the first element of the array
        representing the interval. By default, 'all' returns
        all elements.
    """

    _flank = None
    _gindexer = None

    def __init__(self, name, garray,
                 gindexer,  # indices of pointing to region start
                 padding_value,
                 dimmode, channel_last):  # padding value

        self.garray = garray
        self.gindexer = gindexer
        self.padding_value = padding_value
        self.dimmode = dimmode
        self._channel_last = channel_last

        Dataset.__init__(self, name)

    @classmethod
    def create_from_bam(cls, name,  # pylint: disable=too-many-locals
                        bamfiles,
                        regions=None,
                        genomesize=None,
                        conditions=None,
                        min_mapq=None,
                        binsize=None, stepsize=None,
                        flank=0,
                        resolution=1,
                        storage='ndarray',
                        dtype='int',
                        stranded=True,
                        overwrite=False,
                        pairedend='5prime',
                        template_extension=0,
                        aggregate=None,
                        datatags=None,
                        cache=False,
                        channel_last=True,
                        store_whole_genome=False):
        """Create a Cover class from a bam-file (or files).

        This constructor can be used to obtain coverage from BAM files.
        For single-end reads the read will be counted at the 5 prime end.
        Paired-end reads can be counted relative to the 5 prime ends of the read
        (default) or with respect to the midpoint.


        Parameters
        -----------
        name : str
            Name of the dataset
        bamfiles : str or list
            bam-file or list of bam files.
        regions : str or None
            Bed-file defining the region of interest.
            If set to None, the coverage will be
            fetched from the entire genome and a
            genomic indexer must be attached later.
        genomesize : dict or None
            Dictionary containing the genome size.
            If `genomesize=None`, the genome size
            is determined from the bam header.
            If `store_whole_genome=False`, this option does not have an effect.
        conditions : list(str) or None
            List of conditions.
            If `conditions=None`,
            the conditions are obtained from
            the filenames (without the directories
            and file-ending).
        min_mapq : int
            Minimal mapping quality.
            Reads with lower mapping quality are
            filtered out. If None, all reads are used.
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
            Flanking size increases the interval size at both ends by
            flank base pairs. Default: 0
        resolution : int
            Resolution in base pairs divides the region of interest
            in windows of length resolution.
            This effectively reduces the storage for coverage data.
            The resolution must be selected such that min(stepsize, binsize)
            is a multiple of resolution.
            Default: 1.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray', 'hdf5' or 'sparse'. Default: 'ndarray'.
        dtype : str
            Typecode to be used for storage the data.
            Default: 'int'.
        stranded : boolean
            Indicates whether to extract stranded or
            unstranded coverage. For unstranded
            coverage, reads aligning to both strands will be aggregated.
        overwrite : boolean
            Overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        pairedend : str
            Indicates whether to count reads at the '5prime' end or at
            the 'midpoint' for paired-end reads. Default: '5prime'.
        template_extension : int
            Elongates intervals by template_extension which allows to properly count
            template mid-points whose reads lie outside of the interval.
            This option is only relevant for paired-end reads counted at the
            'midpoint' and if the coverage is not obtained from the
            whole genome, e.g. regions is not None.
        aggregate : callable or None
            Aggregation operation for loading genomic array. If None,
            the coverage amounts to the raw counts.
            Default: None
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        store_whole_genome : boolean
            Indicates whether the whole genome or only selected regions
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False
        """

        if pysam is None:  # pragma: no cover
            raise Exception('pysam not available. '
                            '`create_from_bam` requires pysam to be installed.')

        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        if isinstance(bamfiles, str):
            bamfiles = [bamfiles]

        if conditions is None:
            conditions = [os.path.splitext(os.path.basename(f))[0] for f in bamfiles]

        if min_mapq is None:
            min_mapq = 0

        full_genome_index = store_whole_genome

        if not full_genome_index and not gindexer:
            raise ValueError('Either regions must be supplied or store_whole_genome must be True')

        if not full_genome_index:
            # if whole genome should not be loaded
            gsize = {_iv_to_str(iv.chrom, iv.start,
                                iv.end): iv.end-iv.start for iv in gindexer}

        else:
            # otherwise the whole genome will be fetched, or at least
            # a set of full length chromosomes
            if genomesize is not None:
                # if a genome size has specifically been given, use it.
                gsize = genomesize.copy()
            else:
                header = pysam.AlignmentFile(bamfiles[0], 'r')  # pylint: disable=no-member
                gsize = {}
                for chrom, length in zip(header.references, header.lengths):
                    gsize[chrom] = length

        def _bam_loader(garray, files):
            print("load from bam")
            for i, sample_file in enumerate(files):
                print('Counting from {}'.format(sample_file))
                aln_file = pysam.AlignmentFile(sample_file, 'rb')  # pylint: disable=no-member
                for chrom in gsize:

                    array = np.zeros((get_chrom_length(gsize[chrom], resolution),
                                     2), dtype=dtype)

                    locus = _str_to_iv(chrom, template_extension=template_extension)
                    if len(locus) == 1:
                        locus = (locus[0], 0, gsize[chrom])
                    # locus = (chr, start, end)
                    # or locus = (chr, )

                    for aln in aln_file.fetch(*locus):

                        if aln.is_unmapped:
                            continue

                        if aln.mapq < min_mapq:
                            continue

                        if aln.is_read2:
                            # only consider read1 so as not to double count
                            # fragments for paired end reads
                            # read2 will also be false for single end
                            # reads.
                            continue

                        if aln.is_paired:
                            # if paired end read, consider the midpoint
                            if not (aln.is_proper_pair and
                                    aln.reference_name == aln.next_reference_name):
                                # only consider paired end reads if both mates
                                # are properly mapped and they map to the
                                # same reference_name
                                continue
                            # if the next reference start >= 0,
                            # the read is considered as a paired end read
                            # in this case we consider the mid point
                            if pairedend == 'midpoint':
                                pos = min(aln.reference_start,
                                          aln.next_reference_start) + \
                                          abs(aln.template_length) // 2
                            else:
                                if aln.is_reverse:
                                    # last position of the downstream read
                                    pos = max(aln.reference_end,
                                              aln.next_reference_start +
                                              aln.query_length)
                                else:
                                    # first position of the upstream read
                                    pos = min(aln.reference_start,
                                              aln.next_reference_start)
                        else:
                            # here we consider single end reads
                            # whose 5 prime end is determined strand specifically
                            if aln.is_reverse:
                                pos = aln.reference_end
                            else:
                                pos = aln.reference_start

                        if not garray._full_genome_stored:
                            # if we get here, a region was given,
                            # otherwise, the entire chromosome is read.
                            pos -= locus[1] + template_extension

                            if pos < 0 or pos >= locus[2] - locus[1]:
                                # if the read 5 p end or mid point is outside
                                # of the region of interest, the read is discarded
                                continue

                        # compute divide by the resolution
                        pos //= resolution

                        # fill up the read strand specifically
                        if aln.is_reverse:
                            array[pos, 1] += 1
                        else:
                            array[pos, 0] += 1
                    # apply the aggregation
                    if aggregate is not None:
                        array = aggregate(array)

                    if stranded:
                        lp = locus + ('+',)
                        garray[GenomicInterval(*lp), i] = array[:, 0]
                        lm = locus + ('-',)
                        garray[GenomicInterval(*lm), i] = array[:, 1]
                    else:
                        # if unstranded, aggregate the reads from
                        # both strands
                        garray[GenomicInterval(*locus), i] = array.sum(axis=1)

            return garray

        datatags = [name] + datatags if datatags else [name]

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        cover = create_genomic_array(gsize, stranded=stranded,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     store_whole_genome=store_whole_genome,
                                     resolution=resolution,
                                     loader=_bam_loader,
                                     loader_args=(bamfiles,))

        return cls(name, cover, gindexer, padding_value=0, dimmode='all',
                   channel_last=channel_last)

    @classmethod
    def create_from_bigwig(cls, name,  # pylint: disable=too-many-locals
                           bigwigfiles,
                           regions=None,
                           genomesize=None,
                           conditions=None,
                           binsize=None, stepsize=None,
                           resolution=1,
                           flank=0, storage='ndarray',
                           dtype='float32',
                           overwrite=False,
                           dimmode='all',
                           aggregate=np.mean,
                           datatags=None, cache=False,
                           store_whole_genome=False,
                           channel_last=True,
                           nan_to_num=True):
        """Create a Cover class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bigwigfiles : str or list
            bigwig-file or list of bigwig files.
        regions : str or None
            Bed-file defining the region of interest.
            If set to None, the coverage will be
            fetched from the entire genome and a
            genomic indexer must be attached later.
            Otherwise, the coverage is only determined
            for the region of interest.
        genomesize : dict or None
            Dictionary containing the genome size.
            If `genomesize=None`, the genome size
            is determined from the bigwig file.
            If `store_whole_genome=False`, this option does not have an effect.
        conditions : list(str) or None
            List of conditions.
            If `conditions=None`,
            the conditions are obtained from
            the filenames (without the directories
            and file-ending).
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
        resolution : int
            Resolution in base pairs divides the region of interest
            in windows of length resolution.
            This effectively reduces the storage for coverage data.
            The resolution must be selected such that min(stepsize, binsize)
            is a multiple of resolution.
            Default: 1.
        flank : int
            Flanking size increases the interval size at both ends by
            flank bins. Note that the binsize is defined by the resolution parameter.
            Default: 0.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray', 'hdf5' or 'sparse'. Default: 'ndarray'.
        dtype : str
            Typecode to define the datatype to be used for storage.
            Default: 'float32'.
        dimmode : str
            Dimension mode can be 'first' or 'all'. If 'first', only
            the first element of size resolution is returned. Otherwise,
            all elements of size resolution spanning the interval are returned.
            Default: 'all'.
        overwrite : boolean
            Overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        aggregate : callable
            Aggregation operation for loading genomic array.
            Default: numpy.mean
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        store_whole_genome : boolean
            Indicates whether the whole genome or only selected regions
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        nan_to_num : boolean
            Indicates whether NaN values contained in the bigwig files should
            be interpreted as zeros. Default: True
        """
        if pyBigWig is None:  # pragma: no cover
            raise Exception('pyBigWig not available. '
                            '`create_from_bigwig` requires pyBigWig to be installed.')
        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        if isinstance(bigwigfiles, str):
            bigwigfiles = [bigwigfiles]

        if not store_whole_genome and not gindexer:
            raise ValueError('Either regions must be supplied or store_whole_genome must be True')

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = {_iv_to_str(iv.chrom, iv.start,
                                iv.end): iv.end-iv.start for iv in gindexer}

        else:
            # otherwise the whole genome will be fetched, or at least
            # a set of full length chromosomes
            if genomesize is not None:
                # if a genome size has specifically been given, use it.
                gsize = genomesize.copy()
            else:
                bwfile = pyBigWig.open(bigwigfiles[0], 'r')
                gsize = bwfile.chroms()

        if conditions is None:
            conditions = [os.path.splitext(os.path.basename(f))[0] for f in bigwigfiles]

        def _bigwig_loader(garray, aggregate):
            print("load from bigwig")
            for i, sample_file in enumerate(bigwigfiles):
                bwfile = pyBigWig.open(sample_file)

                for chrom in gsize:

                    vals = np.zeros((get_chrom_length(gsize[chrom], resolution), ),
                                    dtype=dtype)

                    locus = _str_to_iv(chrom, template_extension=0)
                    if len(locus) == 1:
                        locus = locus + (0, gsize[chrom])

                    # when only to load parts of the genome
                    for start in range(locus[1], locus[2], resolution):

                        if garray._full_genome_stored:
                            # be careful not to overshoot at the chromosome end
                            end = min(start+resolution, gsize[chrom])
                        else:
                            end = start + resolution

                        x = np.asarray(bwfile.values(
                            locus[0],
                            int(start),
                            int(end)))
                        if nan_to_num:
                            x = np.nan_to_num(x, copy=False)
                        vals[(start - locus[1])//resolution] = aggregate(x)

                    garray[GenomicInterval(*locus), i] = vals
            return garray

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     resolution=resolution,
                                     store_whole_genome=store_whole_genome,
                                     typecode=dtype,
                                     loader=_bigwig_loader,
                                     loader_args=(aggregate,))

        return cls(name, cover, gindexer,
                   padding_value=0, dimmode=dimmode, channel_last=channel_last)

    @classmethod
    def create_from_bed(cls, name,  # pylint: disable=too-many-locals
                        bedfiles,
                        regions=None,
                        genomesize=None,
                        conditions=None,
                        binsize=None, stepsize=None,
                        resolution=1,
                        flank=0, storage='ndarray',
                        dtype='int',
                        dimmode='all',
                        mode='binary',
                        store_whole_genome=False,
                        overwrite=False,
                        channel_last=True,
                        datatags=None, cache=False):
        """Create a Cover class from a bed-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bedfiles : str or list
            bed-file or list of bed files.
        regions : str or None
            Bed-file defining the region of interest.
            If set to None a genomesize must be supplied and
            a genomic indexer must be attached later.
        genomesize : dict or None
            Dictionary containing the genome size to fetch the coverage from.
            If `genomesize=None`, the genome size
            is fetched from the region of interest.
        conditions : list(str) or None
            List of conditions.
            If `conditions=None`,
            the conditions are obtained from
            the filenames (without the directories
            and file-ending).
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
        resolution : int
            Resolution in base pairs divides the region of interest
            in windows of length resolution.
            This effectively reduces the storage for coverage data.
            The resolution must be selected such that min(stepsize, binsize)
            is a multiple of resolution.
            Default: 1.
        flank : int
            Flanking size increases the interval size at both ends by
            flank bins. Note that the binsize is defined by the resolution parameter.
            Default: 0.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray', 'hdf5' or 'sparse'. Default: 'ndarray'.
        dtype : str
            Typecode to define the datatype to be used for storage.
            Default: 'int'.
        dimmode : str
            Dimension mode can be 'first' or 'all'. If 'first', only
            the first element of size resolution is returned. Otherwise,
            all elements of size resolution spanning the interval are returned.
            Default: 'all'.
        mode : str
            Mode of the dataset may be 'binary', 'score' or 'categorical'.
            Default: 'binary'.
        overwrite : boolean
            Overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        store_whole_genome : boolean
            Indicates whether the whole genome or only selected regions
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        """

        if regions is None and genomesize is None:
            raise ValueError('Either regions or genomesize must be specified.')

        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = {_iv_to_str(iv.chrom, iv.start,
                                iv.end): iv.end-iv.start for iv in gindexer}

        else:
            # otherwise the whole genome will be fetched, or at least
            # a set of full length chromosomes
            if genomesize is not None:
                # if a genome size has specifically been given, use it.
                gsize = genomesize.copy()
            else:
                gsize = get_genome_size_from_regions(regions)

        if isinstance(bedfiles, str):
            bedfiles = [bedfiles]

        if mode == 'categorical':
            if len(bedfiles) > 1:
                raise ValueError('Only one bed-file is '
                                 'allowed with mode=categorical')
            sample_file = bedfiles[0]
            regions_ = _get_genomic_reader(sample_file)

            max_class = 0
            for reg in regions_:
                if reg.score > max_class:
                    max_class = reg.score
            if conditions is None:
                conditions = [str(i) for i in range(int(max_class + 1))]
        if conditions is None:
            conditions = [os.path.splitext(os.path.basename(f))[0]
                          for f in bedfiles]

        def _bed_loader(garray, bedfiles, genomesize, mode):
            print("load from bed")
            for i, sample_file in enumerate(bedfiles):
                regions_ = _get_genomic_reader(sample_file)

                for region in regions_:
                    gidx = GenomicIndexer.create_from_region(
                        region.iv.chrom,
                        region.iv.start,
                        region.iv.end, region.iv.strand,
                        binsize, stepsize, flank)
                    for greg in gidx:

                        if region.score is None and mode in ['score',
                                                             'categorical']:
                            raise ValueError(
                                'No Score available. Score field must '
                                'present in {}'.format(sample_file) + \
                                'for mode="{}"'.format(mode))
                        # if region score is not defined, take the mere
                        # presence of a range as positive label.
                        if mode == 'score':
                            garray[greg, i] = np.dtype(dtype).type(region.score)
                        elif mode == 'categorical':
                            garray[greg,
                                   int(region.score)] = np.dtype(dtype).type(1)
                        elif mode == 'binary':
                            garray[greg, i] = np.dtype(dtype).type(1)
            return garray

        # At the moment, we treat the information contained
        # in each bed-file as unstranded

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=_bed_loader,
                                     loader_args=(bedfiles, gsize, mode))

        return cls(name, cover, gindexer,
                   padding_value=0, dimmode=dimmode, channel_last=channel_last)

    @classmethod
    def create_from_array(cls, name,  # pylint: disable=too-many-locals
                          array,
                          gindexer,
                          genomesize=None,
                          conditions=None,
                          resolution=1,
                          storage='ndarray',
                          overwrite=False,
                          datatags=None,
                          cache=False,
                          channel_last=True,
                          store_whole_genome=False):
        """Create a Cover class from a numpy.array.

        The purpose of this function is to convert output prediction from
        keras which are in numpy.array format into a Cover object.

        Parameters
        -----------
        name : str
            Name of the dataset
        array : numpy.array
            A 4D numpy array that will be re-interpreted as genomic array.
        gindexer : GenomicIndexer
            Genomic indices associated with the values contained in array.
        genomesize : dict or None
            Dictionary containing the genome size to fetch the coverage from.
            If `genomesize=None`, the genome size is automatically determined
            from the GenomicIndexer. If `store_whole_genome=False` this
            option does not have an effect.
        conditions : list(str) or None
            List of conditions.
            If `conditions=None`,
            the conditions are obtained from
            the filenames (without the directories
            and file-ending).
        resolution : int
            Resolution in base pairs divides the region of interest
            in windows of length resolution.
            This effectively reduces the storage for coverage data.
            The resolution must be selected such that min(stepsize, binsize)
            is a multiple of resolution.
            Default: 1.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray', 'hdf5' or 'sparse'. Default: 'ndarray'.
        overwrite : boolean
            Overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Together with the dataset name,
            the datatags are used to construct a cache file.
            If :code:`cache=False`, this option does not have an effect.
            Default: None.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        store_whole_genome : boolean
            Indicates whether the whole genome or only selected regions
            should be loaded. Default: False.
        channel_last : boolean
            This tells the constructor how to interpret the array dimensions.
            It indicates whether the condition axis is the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        """

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = {_iv_to_str(iv.chrom, iv.start,
                                iv.end): iv.end-iv.start for iv in gindexer}
        elif genomesize:
            gsize = genomesize.copy()
        else:
            # if not supplied, determine the genome size automatically
            # based on the gindexer intervals.
            gsize = get_genome_size_from_regions(gindexer)

        if not channel_last:
            array = np.transpose(array, (0, 3, 1, 2))

        if conditions is None:
            conditions = ["Cond_{}".format(i) for i in range(array.shape[-1])]

        # check if dimensions of gindexer and array match
        if len(gindexer) != array.shape[0]:
            raise ValueError("Data incompatible: "
                "The number intervals in gindexer"
                " must match the number of datapoints in the array "
                "(len(gindexer) != array.shape[0])")

        if store_whole_genome:
            # in this case the intervals must be non-overlapping
            # in order to obtain unambiguous data.
            if gindexer.binsize > gindexer.stepsize:
                raise ValueError("Overlapping intervals: "
                    "With overlapping intervals the mapping between "
                    "the array and genomic-array values is ambiguous. "
                    "Please ensure that binsize <= stepsize.")

        # determine the resolution
        resolution = gindexer[0].length // array.shape[1]

        # determine strandedness
        stranded = True if array.shape[2] == 2 else False

        def _array_loader(garray, array, gindexer):
            print("load from array")

            for i, region in enumerate(gindexer):
                iv = region
                for cond in range(array.shape[-1]):
                    if stranded:
                        iv.strand = '+'
                        garray[iv, cond] = array[i, :, 0, cond].astype(dtype)
                        iv.strand = '-'
                        garray[iv, cond] = array[i, :, 1, cond].astype(dtype)
                    else:
                        garray[iv, cond] = array[i, :, 0, cond]

            return garray

        # At the moment, we treat the information contained
        # in each bw-file as unstranded

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        cover = create_genomic_array(gsize, stranded=stranded,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=array.dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=_array_loader,
                                     loader_args=(array, gindexer))

        return cls(name, cover, gindexer,
                   padding_value=0, dimmode='all', channel_last=channel_last)

    @property
    def gindexer(self):
        """GenomicIndexer property"""
        if self._gindexer is None:
            raise ValueError('GenomicIndexer has not been set yet. Please specify an indexer.')
        return self._gindexer

    @gindexer.setter
    def gindexer(self, gindexer):
        if gindexer is None:
            self._gindexer = None
            return

        if (min(gindexer.stepsize,
                gindexer.binsize) % self.garray.resolution) != 0:
            raise ValueError('min(binsize, stepsize) must be divisible by resolution')
        self._gindexer = gindexer

    def __repr__(self):  # pragma: no cover
        return "Cover('{}') ".format(self.name)

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            if self.garray._full_genome_stored == True:
                # interpret idxs as genomic interval
                idxs = GenomicInterval(*idxs)
            else:
                chrom = idxs[0]
                start = idxs[1]
                end = idxs[2]
                gindexer_new = self.gindexer.filter_by_region(include=chrom, start=start, end=end)
                data = np.zeros((1, ((end - start) // self.garray.resolution) + (2 * (gindexer_new.stepsize) // self.garray.resolution)) + self.shape[2:])
                if self.padding_value != 0:
                    data.fill(self.padding_value)
                step_size = gindexer_new.stepsize
                for interval in gindexer_new:
                    rel_pos = (interval.start - (start - step_size)) // self.garray.resolution
                    tmp_data = np.array(self._getsingleitem(interval))
                    tmp_data = tmp_data.reshape((1,) + tmp_data.shape)
                    data[:, rel_pos: rel_pos + (step_size // self.garray.resolution), :, :] = tmp_data
                data = data[:,(1*(step_size) // self.garray.resolution): -1 * (1*(step_size) // self.garray.resolution),:,:]
                return data

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
            # accept a genomic interval directly
            #data = np.zeros((1,) + self.shape[1:])
            data = self._getsingleitem(idxs)
            data = data.reshape((1,) + data.shape)
            for transform in self.transformations:
                data = transform(data)
            if not self._channel_last:
                data = np.transpose(data, (0, 3, 1, 2))

            return data

        try:
            iter(idxs)
        except TypeError:
            raise IndexError('Cover.__getitem__: index must be iterable')

        data = np.zeros((len(idxs),) + self.shape_static[1:])
        if self.padding_value != 0:
            data.fill(self.padding_value)

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            data[i, :, :, :] = self._getsingleitem(interval)

        for transform in self.transformations:
            data = transform(data)

        if not self._channel_last:
            data = np.transpose(data, (0, 3, 1, 2))

        return data

    def _getsingleitem(self, pinterval):

        if pinterval.strand == '-':
            data = np.asarray(self.garray[pinterval])[::-1, ::-1, :]
        else:
            data = np.asarray(self.garray[pinterval])

        if self.dimmode == 'first':
            data = data[:1, :, :]

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""

        if self._channel_last:
            return self.shape_static
        else:
            return tuple(self.shape_static[x] for x in [0, 3, 1, 2])

    @property
    def shape_static(self):
        """Shape of the dataset"""
        if self.dimmode == 'all':
            blen = (self.gindexer.binsize) // self.garray.resolution
            seqlen = 2*self.gindexer.flank // self.garray.resolution + \
                (blen if blen > 0 else 1)
        elif self.dimmode == 'first':
            seqlen = 1
        return (len(self),
                seqlen,
                2 if self.garray.stranded else 1,
                len(self.garray.condition))

    @property
    def conditions(self):
        """Conditions"""
        return [s.decode('utf-8') for s in self.garray.condition]

    def export_to_bigwig(self, output_dir, genomesize=None):
        """ This method exports the coverage as bigwigs.

        This allows to use a standard genome browser to explore the
        predictions or features derived from a neural network.

        The bigwig files are named after the container name and the condition
        names.

        NOTE:
            This function expects that the regions that need to be
            exported are non-overlapping. That is the gindexer
            binsize must be smaller or equal than stepsize.

        Parameters
        ----------
        output_dir : str
            Output directory to which the bigwig files will be exported to.
        genomesize : dict or None
            Dictionary containing the genome size.
            If `genomesize=None`, the genome size
            is determined from the gindexer if `store_whole_genome=False`,
            or from the garray-size of `store_whole_genome=True`.
        """

        if pyBigWig is None:  # pragma: no cover
            raise Exception('pyBigWig not available. '
                            '`export_to_bigwig` requires pyBigWig to be installed.')

        resolution = self.garray.resolution

        if genomesize is not None:
            gsize = genomesize
        elif self.garray._full_genome_stored:
            gsize = {k: self.garray.handle[k].shape[0] * resolution \
                     for k in self.garray.handle}
        else:
            gsize = get_genome_size_from_regions(self.gindexer)

        bw_header = [(chrom, gsize[chrom])
                     for chrom in gsize]

        print('check header:')
        print(bw_header)

        for idx, condition in enumerate(self.conditions):
            print(condition)
            bw_file = pyBigWig.open(os.path.join(
                output_dir,
                '{name}.{condition}.bigwig'.format(
                    name=self.name, condition=condition)), 'w')

            bw_file.addHeader(bw_header)

            # we need the new filter_by_region method here
            #for chrom, size in bw_header:
                # ngindexer = self.gindexer.filter_by_region()
                # then use the new ngindexer to loop over as below

            for ridx, region in enumerate(self.gindexer):
                print(region)
                cov = self[ridx][0, :, :, idx].sum(axis=1)
                #print(region)

                bw_file.addEntries(str(region.chrom),
                                   int(region.start),
                                   values=cov,
                                   span=int(resolution),
                                   step=int(resolution))
            bw_file.close()


def plotGenomeTrack(covers, chr, start, end):

    """plotGenomeTrack shows plots of a specific interval from cover objects data.

    It takes a list of cover objects, number of chromosome, the start and the end of
    a required interval and shows a plot of the same interval for each condition of each cover.

    Parameters
    ----------
    covers : list(str)
        List of cover objects.
    chr : str
        chromosome name.
    start : int
        The start of the required interval.
    end : int
        The end of the required interval.

    Returns
    -------
    Figure
        A matplotlib figure built for the required interval for each condition of each cover objects.
        It is possible to show that figure with show() function integrated in matplotlib or even save it
        with the 'savefig()' function of the same library.
    """
    if not isinstance(covers, list):
        covers = [covers]

    n_covers = len(covers)
    color = iter(cm.rainbow(np.linspace(0, 1, n_covers)))
    data = covers[0][chr, start, end]
    len_files = [len(cover.conditions) for cover in covers]
    nfiles = np.sum(len_files)
    grid = plt.GridSpec(2 + (nfiles * 3) + (n_covers - 1), 10, wspace=0.4, hspace=0.3)
    fig = plt.figure(figsize=(1 + nfiles * 3, 2*nfiles))

    title = fig.add_subplot(grid[0, 1:])
    title.set_title(chr)
    plt.xlim([0, len(data[0, :, 0, 0])])
    title.spines['right'].set_visible(False)
    title.spines['top'].set_visible(False)
    title.spines['left'].set_visible(False)
    plt.xticks([0, len(data[0, :, 0, 0])], [start, end])
    plt.yticks(())
    cover_start = 2
    abs_cont = 0
    lat_titles = [None]*len(covers)
    plots = []
    for j, cover in enumerate(covers):
        color_ = next(color)
        lat_titles[j] = fig.add_subplot(grid[(cover_start + j):(cover_start + len_files[j]*3) + j, 0])
        cover_start += (len_files[j]*3)
        lat_titles[j].set_xticks(())
        lat_titles[j].spines['right'].set_visible(False)
        lat_titles[j].spines['top'].set_visible(False)
        lat_titles[j].spines['bottom'].set_visible(False)
        lat_titles[j].set_yticks([0.5])
        lat_titles[j].set_yticklabels([cover.name], color=color_)
        cont = 0
        for i in cover.conditions:
            plots.append(fig.add_subplot(grid[(cont + abs_cont) * 3 + 2 +j:(cont + abs_cont) * 3 + 5+j, 1:]))
            plots[-1].plot(data[0, :, 0, cont], linewidth=2, color = color_)
            plots[-1].set_yticks(())
            plots[-1].set_xticks(())
            plots[-1].set_xlim([0, len(data[0, :, 0, 0])])
            plots[-1].set_ylabel(i, labelpad=12)
            plots[-1].spines['right'].set_visible(False)
            plots[-1].spines['top'].set_visible(False)
            cont = cont + 1
        abs_cont += cont
    return (fig)
