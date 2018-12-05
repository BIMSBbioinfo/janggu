"""Coverage dataset"""

import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _get_genomic_reader
from janggu.utils import _iv_to_str
from janggu.utils import _str_to_iv
from janggu.utils import get_genome_size_from_regions

try:
    import pyBigWig
except ImportError:  # pragma: no cover
    pyBigWig = None
try:
    import pysam
except ImportError:  # pragma: no cover
    pysam = None

class BamLoader:
    """BamLoader class.

    This class loads the GenomicArray with read count coverage
    extracted from BAM files.

    Parameters
    ----------
    files : str or list(str)
        Bam file locations.
    gsize : dict
        Dictionary of genome sizes.
    template_extension : int
        Extension of intervals by template_extension for counting
        paired-end midpoints correctly.
        It may be possible that both read ends are located outside
        of the given interval, but the mid-points inside.
        template_extension extends the interval in order to correcly determine
        the mid-points.
    min_mapq : int
        Minimum mapping quality to be considered.
    pairedend : str
        Paired-end mode 'midpoint' or '5prime'.
    """
    def __init__(self, files, gsize, template_extension,
                 min_mapq, pairedend):
        self.files = files
        self.gsize = gsize
        self.template_extension = template_extension
        self.min_mapq = min_mapq
        self.pairedend = pairedend

    def __call__(self, garray):
        files = self.files
        gsize = self.gsize
        template_extension = self.template_extension
        resolution = garray.resolution
        dtype = garray.typecode
        min_mapq = self.min_mapq
        pairedend = self.pairedend

        print("load from bam")
        for i, sample_file in enumerate(files):
            print('Counting from {}'.format(sample_file))
            aln_file = pysam.AlignmentFile(sample_file, 'rb')  # pylint: disable=no-member
            for chrom in gsize:

                locus = _str_to_iv(chrom,
                                   template_extension=template_extension)
                if len(locus) == 1:
                    locus = (locus[0], 0, gsize[chrom])

                if resolution is None:
                    length = locus[2] - locus[1]
                else:
                    length = garray.get_iv_end(locus[2] -
                                               locus[1]) * resolution

                array = np.zeros((length, 2), dtype=dtype)

                # locus = (chr, start, end)
                start = locus[1]
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
                        pos -= start + template_extension

                        if pos < 0 or pos >= length:
                            # if the read 5 p end or mid point is outside
                            # of the region of interest, the read is discarded
                            continue

                    # compute divide by the resolution
                    #pos = garray.get_iv_start(pos)

                    # fill up the read strand specifically
                    if aln.is_reverse:
                        array[pos, 1] += 1
                    else:
                        array[pos, 0] += 1

                garray[GenomicInterval(*locus), i] = array

        return garray



class BigWigLoader:
    """BigWigLoader class.

    This class loads the GenomicArray with signal coverage
    extracted from BIGWIG files.

    Parameters
    ----------
    files : str or list(str)
        Bigwig file locations.
    gsize : dict
        Dictionary of genome sizes.
    nan_to_num : bool
        Whether to convert NAN's to zeros or not. Default: True.
    """
    def __init__(self, files, gsize, nan_to_num):
        self.files = files
        self.gsize = gsize
        self.nan_to_num = nan_to_num

    def __call__(self, garray):
        files = self.files
        gsize = self.gsize
        resolution = garray.resolution
        dtype = garray.typecode
        nan_to_num = self.nan_to_num

        print("load from bigwig")
        for i, sample_file in enumerate(files):
            bwfile = pyBigWig.open(sample_file)

            for chrom in gsize:

                locus = _str_to_iv(chrom)
                if len(locus) == 1:
                    locus = (locus[0], 0, gsize[chrom])

                if resolution is None:
                    length = locus[2] - locus[1]
                else:
                    length = garray.get_iv_end(locus[2]-locus[1]) * resolution

                array = np.zeros((length, 1), dtype=dtype)

                values = np.asarray(bwfile.values(locus[0],
                                                  int(locus[1]),
                                                  int(locus[2])))
                if nan_to_num:
                    values = np.nan_to_num(values, copy=False)

                array[:len(values), 0] = values

                garray[GenomicInterval(*locus), i] = array
        return garray


class BedLoader:
    """BedLoader class.

    This class loads the GenomicArray with signal coverage
    extracted from BED files.

    Parameters
    ----------
    files : str or list(str)
        Bed file locations.
    gindexer : GenomicIndexer
        GenomicIndexer object.
    mode : str
        Mode might be 'binary', 'score' or 'categorical'.
    """
    def __init__(self, files, gindexer, mode):
        self.files = files
        self.gindexer = gindexer
        self.mode = mode


    def __call__(self, garray):
        files = self.files
        dtype = garray.typecode
        gindexer = self.gindexer
        mode = self.mode

        print("load from bed")
        for i, sample_file in enumerate(files):
            regions_ = _get_genomic_reader(sample_file)

            for region in regions_:

                gidx = gindexer.filter_by_region(
                    include=region.iv.chrom,
                    start=region.iv.start,
                    end=region.iv.end)
                for greg in gidx:
                    if region.score is None and mode in ['score',
                                                         'categorical']:
                        raise ValueError(
                            'No Score available. Score field must '
                            'present in {}'.format(sample_file) + \
                            'for mode="{}"'.format(mode))
                    # if region score is not defined, take the mere
                    # presence of a range as positive label.

                    score = region.score if mode == 'score' else 1
                    array = np.repeat(
                        np.dtype(dtype).type(score).reshape((1, 1)),
                        greg.length, axis=0)

                    if greg.start < region.iv.start:
                        # set the beginning of array to zero
                        array[:(region.iv.start - greg.start), 0] = 0
                    if greg.end > region.iv.end:
                        # set the end of the array to zero
                        array[-(greg.end - region.iv.end):, 0] = 0

                    #if garray.resolution is not None:
                    array = np.maximum(array,
                                       np.repeat(garray[greg][:, :, i],
                                       greg.length if garray.resolution is None \
                                       else garray.resolution, axis=0))

                    #    array = np.maximum(array, np.repeat(garray[greg][:, :, i], greg.length, axis=0)

                    if mode == 'score':
                        garray[greg, i] = array
                    elif mode == 'categorical':
                        garray[greg, int(region.score)] = array
                    elif mode == 'binary':
                        garray[greg, i] = array
        return garray

class ArrayLoader:
    """ArrayLoader class.

    This class loads the GenomicArray with signal coverage
    extracted from a numpy array.

    Parameters
    ----------
    array : np.ndarray
        A numpy array that should be converted to a Cover object
    gindexer : GenomicIndexer
        A GenomicIndexer that holds the corresponding genomic intervals
        for the predictions in the array.
    """
    def __init__(self, array, gindexer):
        self.array = array
        self.gindexer = gindexer

    def __call__(self, garray):
        array = self.array
        gindexer = self.gindexer
        resolution = garray.resolution

        print("load from array")

        for i, region in enumerate(gindexer):
            interval = region
            for cond in range(array.shape[-1]):
                if resolution is None:
                    garray[interval, cond] = np.repeat(array[i, :, :, cond],
                                                       interval.length, axis=0)
                else:
                    garray[interval, cond] = np.repeat(array[i, :, :, cond],
                                                       resolution, axis=0)

        return garray


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
    """

    _flank = None
    _gindexer = None

    def __init__(self, name, garray,
                 gindexer,  # indices of pointing to region start
                 channel_last):  # padding value

        self.garray = garray
        self.gindexer = gindexer
        self._channel_last = channel_last
        Dataset.__init__(self, name)

    @classmethod
    def create_from_bam(cls, name,  # pylint: disable=too-many-locals
                        bamfiles,
                        roi=None,
                        genomesize=None,
                        conditions=None,
                        min_mapq=None,
                        binsize=None, stepsize=None,
                        flank=0,
                        resolution=1,
                        storage='ndarray',
                        dtype='float32',
                        stranded=True,
                        overwrite=False,
                        pairedend='5prime',
                        template_extension=0,
                        datatags=None,
                        cache=False,
                        channel_last=True,
                        normalizer=None,
                        zero_padding=True,
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
        roi : str or None
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
            the binsize will be determined from the bed-file.
            If resolution is of type integer, this
            requires that all intervals in the bed-file are of equal
            length. If resolution is None, the intervals in the bed-file
            may be of variable size.
            Default: None.
        stepsize : int or None
            stepsize in basepairs for traversing the genome.
            If stepsize is None, it will be set equal to binsize.
            Default: None.
        flank : int
            Flanking size increases the interval size at both ends by
            flank base pairs. Default: 0
        resolution : int or None
            If resolution represents an interger, it determines
            the base pairs resolution by which an interval should be divided.
            This requires equally sized bins or zero padding and
            effectively reduces the storage for coverage data.
            If resolution=None, the intervals will be represented by
            a collapsed summary score.
            For example, gene expression may be expressed by TPM in that manner.
            In the latter case, variable size intervals are permitted
            and zero padding does not have an effect.
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
            whole genome, e.g. roi is not None.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        zero_padding : boolean
            Indicates if variable size intervals should be zero padded.
            Zero padding is only supported with a specified
            binsize. If zero padding is false, intervals shorter than binsize will
            be skipped.
            Default: True.
        normalizer : None, str or callable
            This option specifies the normalization that can be applied.
            If None, no normalization is applied. If 'zscore', 'zscorelog', 'rpkm'
            then zscore transformation, zscore transformation on log transformed data
            and rpkm normalization are performed, respectively.
            If callable, a function with signature `norm(garray)` should be
            provided that performs the normalization on the genomic array.
            Default: None.
        store_whole_genome : boolean
            Indicates whether the whole genome or only ROI
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False
        """

        if pysam is None:  # pragma: no cover
            raise Exception('pysam not available. '
                            '`create_from_bam` requires pysam to be installed.')

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank,
                                                       zero_padding, collapse)
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
            raise ValueError('Either roi must be supplied or store_whole_genome must be True')

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


        bamloader = BamLoader(bamfiles, gsize, template_extension,
                              min_mapq, pairedend)

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
                                     loader=bamloader,
                                     normalizer=normalizer,
                                     collapser='sum')

        return cls(name, cover, gindexer,
                   channel_last=channel_last)

    @classmethod
    def create_from_bigwig(cls, name,  # pylint: disable=too-many-locals
                           bigwigfiles,
                           roi=None,
                           genomesize=None,
                           conditions=None,
                           binsize=None, stepsize=None,
                           resolution=1,
                           flank=0, storage='ndarray',
                           dtype='float32',
                           overwrite=False,
                           datatags=None, cache=False,
                           store_whole_genome=False,
                           channel_last=True,
                           zero_padding=True,
                           normalizer=None,
                           collapser=None,
                           nan_to_num=True):
        """Create a Cover class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bigwigfiles : str or list
            bigwig-file or list of bigwig files.
        roi : str or None
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
            the binsize will be determined from the bed-file.
            If resolution is of type integer, this
            requires that all intervals in the bed-file are of equal
            length. If resolution is None, the intervals in the bed-file
            may be of variable size.
            Default: None.
        stepsize : int or None
            stepsize in basepairs for traversing the genome.
            If stepsize is None, it will be set equal to binsize.
            Default: None.
        resolution : int or None
            If resolution represents an interger, it determines
            the base pairs resolution by which an interval should be divided.
            This requires equally sized bins or zero padding and
            effectively reduces the storage for coverage data.
            If resolution=None, the intervals will be represented by
            a collapsed summary score.
            For example, gene expression may be expressed by TPM in that manner.
            In the latter case, variable size intervals are permitted
            and zero padding does not have an effect.
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
            Indicates whether the whole genome or only ROI
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        zero_padding : boolean
            Indicates if variable size intervals should be zero padded.
            Zero padding is only supported with a specified
            binsize. If zero padding is false, intervals shorter than binsize will
            be skipped.
            Default: True.
        normalizer : None, str or callable
            This option specifies the normalization that can be applied.
            If None, no normalization is applied. If 'zscore', 'zscorelog', 'rpkm'
            then zscore transformation, zscore transformation on log transformed data
            and rpkm normalization are performed, respectively.
            If callable, a function with signature `norm(garray)` should be
            provided that performs the normalization on the genomic array.
            Default: None.
        collapser : None, str or callable
            This option defines how the genomic signal should be summarized when resolution
            is None or greater than one. It is possible to choose a number of options by
            name, including 'sum', 'mean', 'max'. In addtion, a function may be supplied
            that defines a custom aggregation method. If collapser is None,
            'mean' aggregation will be used.
            Default: None.
        nan_to_num : boolean
            Indicates whether NaN values contained in the bigwig files should
            be interpreted as zeros. Default: True
        """
        if pyBigWig is None:  # pragma: no cover
            raise Exception('pyBigWig not available. '
                            '`create_from_bigwig` requires pyBigWig to be installed.')

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank, zero_padding, collapse)
        else:
            gindexer = None

        if isinstance(bigwigfiles, str):
            bigwigfiles = [bigwigfiles]

        if not store_whole_genome and not gindexer:
            raise ValueError('Either roi must be supplied or store_whole_genome must be True')

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


        bigwigloader = BigWigLoader(bigwigfiles, gsize, nan_to_num)
        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        collapser_ = collapser if collapser is not None else 'mean'

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     resolution=resolution,
                                     store_whole_genome=store_whole_genome,
                                     typecode=dtype,
                                     loader=bigwigloader,
                                     collapser=collapser_,
                                     normalizer=normalizer)

        return cls(name, cover, gindexer,
                   channel_last=channel_last)

    @classmethod
    def create_from_bed(cls, name,  # pylint: disable=too-many-locals
                        bedfiles,
                        roi=None,
                        genomesize=None,
                        conditions=None,
                        binsize=None, stepsize=None,
                        resolution=1,
                        flank=0, storage='ndarray',
                        dtype='float32',
                        mode='binary',
                        store_whole_genome=False,
                        overwrite=False,
                        channel_last=True,
                        zero_padding=True,
                        normalizer=None,
                        collapser=None,
                        datatags=None, cache=False):
        """Create a Cover class from a bed-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bedfiles : str or list
            bed-file or list of bed files.
        roi : str or None
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
            the binsize will be determined from the bed-file.
            If resolution is of type integer, this
            requires that all intervals in the bed-file are of equal
            length. If resolution is None, the intervals in the bed-file
            may be of variable size.
            Default: None.
        stepsize : int or None
            stepsize in basepairs for traversing the genome.
            If stepsize is None, it will be set equal to binsize.
            Default: None.
        resolution : int or None
            If resolution represents an interger, it determines
            the base pairs resolution by which an interval should be divided.
            This requires equally sized bins or zero padding and
            effectively reduces the storage for coverage data.
            If resolution=None, the intervals will be represented by
            a collapsed summary score.
            For example, gene expression may be expressed by TPM in that manner.
            In the latter case, variable size intervals are permitted
            and zero padding does not have an effect.
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
            Indicates whether the whole genome or only ROI
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False.
        channel_last : boolean
            Indicates whether the condition axis should be the last dimension
            or the first. For example, tensorflow expects the channel at the
            last position. Default: True.
        zero_padding : boolean
            Indicates if variable size intervals should be zero padded.
            Zero padding is only supported with a specified
            binsize. If zero padding is false, intervals shorter than binsize will
            be skipped.
            Default: True.
        normalizer : None, str or callable
            This option specifies the normalization that can be applied.
            If None, no normalization is applied. If 'zscore', 'zscorelog', 'tpm'
            then zscore transformation, zscore transformation on log transformed data
            and rpkm normalization are performed, respectively.
            If callable, a function with signature `norm(garray)` should be
            provided that performs the normalization on the genomic array.
            Default: None.
        collapser : None, str or callable
            This option defines how the genomic signal should be summarized when resolution
            is None or greater than one. It is possible to choose a number of options by
            name, including 'sum', 'mean', 'max'. In addtion, a function may be supplied
            that defines a custom aggregation method. If collapser is None,
            'max' aggregation will be used.
            Default: None.
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
        """

        if roi is None and genomesize is None:
            raise ValueError('Either roi or genomesize must be specified.')

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank, zero_padding, collapse)
            binsize = gindexer.binsize
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
                gsize = get_genome_size_from_regions(roi)

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

        bedloader = BedLoader(bedfiles, gindexer, mode)
        # At the moment, we treat the information contained
        # in each bed-file as unstranded

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        collapser_ = collapser if collapser is not None else 'max'
        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=bedloader,
                                     collapser=collapser_,
                                     normalizer=normalizer)

        return cls(name, cover, gindexer,
                   channel_last=channel_last)

    @classmethod
    def create_from_array(cls, name,  # pylint: disable=too-many-locals
                          array,
                          gindexer,
                          genomesize=None,
                          conditions=None,
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
            Indicates whether the whole genome or only ROI
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
                             " must match the number of datapoints in "
                             "the array (len(gindexer) != array.shape[0])")

        if store_whole_genome:
            # in this case the intervals must be non-overlapping
            # in order to obtain unambiguous data.
            if gindexer.binsize > gindexer.stepsize:
                raise ValueError("Overlapping intervals: With overlapping "
                                 "intervals the mapping between the array and "
                                 "genomic-array values is ambiguous. "
                                 "Please ensure that binsize <= stepsize.")

        # determine the resolution
        if gindexer.binsize is None:
            # binsize will not be set if gindexer was loaded in collapse mode
            resolution = None
        else:
            resolution = gindexer.binsize // array.shape[1]

        # determine strandedness
        stranded = True if array.shape[2] == 2 else False


        arrayloader = ArrayLoader(array, gindexer)
        # At the moment, we treat the information contained
        # in each bw-file as unstranded

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        # define a dummy collapser

        def _dummy_collapser(values):
            # should be 3D
            # seqlen, resolution, strand
            return values[:, 0, :]

        cover = create_genomic_array(gsize, stranded=stranded,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=array.dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=arrayloader,
                                     collapser=_dummy_collapser)

        return cls(name, cover, gindexer,
                   channel_last=channel_last)

    @property
    def gindexer(self):
        """GenomicIndexer property"""
        if self._gindexer is None:
            raise ValueError('GenomicIndexer has not been set yet. Please specify an indexer.')
        return self._gindexer

    @gindexer.setter
    def gindexer(self, gindexer):
        self._gindexer = gindexer

    def __repr__(self):  # pragma: no cover
        return "Cover('{}') ".format(self.name)

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            idxs = GenomicInterval(*idxs)

        if isinstance(idxs, int):
            idxs = [idxs]
        elif isinstance(idxs, slice):
            idxs = range(idxs.start if idxs.start else 0,
                         idxs.stop if idxs.stop else len(self),
                         idxs.step if idxs.step else 1)
        elif isinstance(idxs, GenomicInterval):
            if self.garray._full_genome_stored:
                # accept a genomic interval directly
                data = self._getsingleitem(idxs)
                data = data.reshape((1,) + data.shape)
                for transform in self.transformations:
                    data = transform(data)

            else:
                chrom = idxs.chrom
                start = idxs.start
                end = idxs.end
                strand = idxs.strand
                gindexer_new = self.gindexer.filter_by_region(include=chrom,
                                                              start=start,
                                                              end=end)

                data = np.zeros((1, (end - start)) + self.shape[2:])

                for interval in gindexer_new:
                    tmp_data = np.array(self._getsingleitem(interval))
                    tmp_data = tmp_data.reshape((1,) + tmp_data.shape)

                    if interval.strand == '-':
                        # invert the data so that is again relative
                        # to the positive strand,
                        # this avoids having to change the rel_pos computation
                        tmp_data = tmp_data[:, ::-1, ::-1, :]

                    #determine upsampling factor
                    # so both tmp_data and data represent signals on
                    # nucleotide resolution
                    factor = interval.length // tmp_data.shape[1]
                    tmp_data = tmp_data.repeat(factor, axis=1)

                    if start - interval.start > 0:
                        tmp_start = start - interval.start
                        ref_start = 0
                    else:
                        tmp_start = 0
                        ref_start = interval.start - start

                    if interval.end - end > 0:
                        tmp_end = tmp_data.shape[1] - (interval.end - end)
                        ref_end = data.shape[1]
                    else:
                        tmp_end = tmp_data.shape[1]
                        ref_end = data.shape[1] - (end - interval.end)

                    data[:, ref_start:ref_end, :, :] = \
                        tmp_data[:, tmp_start:tmp_end, :, :]

                # issue with different resolution
                # it might not be optimal to return the data on a base-pair
                # resolution scale. If the user has loaded the array with resolution > 1
                # it might be expected to be applied to the returned dataset as well.
                # however, when store_whole_genome=False, resolution may also
                # be None. In this case, it is much more easy to project the variable
                # size intervals onto a common reference.
                #
                # A compromise for the future would be to apply downscaling
                # if resolution is > 1. But then also the dataset must be resized and reshaped.
                # We leave this change for the future, if it seems to matter.

                if strand == '-':
                    # invert it back relative to minus strand
                    data = data[:, ::-1, ::-1, :]

            if not self._channel_last:
                data = np.transpose(data, (0, 3, 1, 2))

            return data

        try:
            iter(idxs)
        except TypeError:
            raise IndexError('Cover.__getitem__: index must be iterable')

        data = np.zeros((len(idxs),) + self.shape_static[1:])

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            dat = self._getsingleitem(interval)
            data[i, :len(dat), :, :] = dat

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

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""

        if self._channel_last:
            return self.shape_static

        return tuple(self.shape_static[x] for x in [0, 3, 1, 2])

    @property
    def shape_static(self):
        """Shape of the dataset"""
        stranded = (2 if self.garray.stranded else 1, )
        if self.garray.resolution is not None:
            blen = (self.gindexer.binsize) // self.garray.resolution
            seqdims = (2*self.gindexer.flank // self.garray.resolution + \
                (blen if blen > 0 else 1), )
        else:
            seqdims = (1,)
        return (len(self),) + seqdims + stranded + (len(self.garray.condition),)

    @property
    def ndim(self):
        return len(self.shape)

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

        for idx, condition in enumerate(self.conditions):
            bw_file = pyBigWig.open(os.path.join(
                output_dir,
                '{name}.{condition}.bigwig'.format(
                    name=self.name, condition=condition)), 'w')

            bw_file.addHeader(bw_header)

            # we need to add data to the bigwig file handle
            # in the same order as given by bw_header.
            # therefore, we process each chromosome in order below

            for chrom, _ in bw_header:
                idxs = self.gindexer.idx_by_region(include=chrom)

                for ridx in idxs:
                    region = self.gindexer[int(ridx)]
                    cov = self[int(ridx)][0, :, :, idx].sum(axis=1)

                    bw_file.addEntries(str(region.chrom),
                                       int(region.start),
                                       values=cov,
                                       span=int(resolution),
                                       step=int(resolution))
            bw_file.close()


def plotGenomeTrack(covers, chrom, start, end):

    """plotGenomeTrack shows plots of a specific interval from cover objects data.

    It takes one or more cover objects as well as a genomic interval consisting
    of chromosome name, start and end and creates
    a genome browser-like plot.

    Parameters
    ----------
    covers : janggu.data.Cover or list(janggu.data.Cover)
        One or more coverge objects.
    chrom : str
        chromosome name.
    start : int
        The start of the required interval.
    end : int
        The end of the required interval.

    Returns
    -------
    matplotlib Figure
        A matplotlib figure illustrating the genome browser-view of the coverage
        objects for the given interval.
        To depict and save the figure the native matplotlib functions show()
        and savefig() can be used.
    """
    if not isinstance(covers, list):
        covers = [covers]

    n_covers = len(covers)
    color = iter(cm.rainbow(np.linspace(0, 1, n_covers)))
    #data = covers[0][chr, start, end]
    len_files = [len(cover.conditions) for cover in covers]
    nfiles = np.sum(len_files)
    grid = plt.GridSpec(2 + (nfiles * 3) + (n_covers - 1),
                        10, wspace=0.4, hspace=0.3)
    fig = plt.figure(figsize=(1 + nfiles * 3,
                              2*nfiles))

    title = fig.add_subplot(grid[0, 1:])

    title.set_title(chrom)
    plt.xlim([0, end - start])
    title.spines['right'].set_visible(False)
    title.spines['top'].set_visible(False)
    title.spines['left'].set_visible(False)
    plt.xticks([0, end-start], [start, end])
    plt.yticks(())
    cover_start = 2
    abs_cont = 0
    lat_titles = [None] * len(covers)
    plots = []
    for j, cover in enumerate(covers):
        color_ = next(color)
        lat_titles[j] = fig.add_subplot(grid[(cover_start + j):
                                             (cover_start +
                                              len_files[j]*3) + j, 0])
        cover_start += (len_files[j]*3)
        lat_titles[j].set_xticks(())
        lat_titles[j].spines['right'].set_visible(False)
        lat_titles[j].spines['top'].set_visible(False)
        lat_titles[j].spines['bottom'].set_visible(False)
        lat_titles[j].set_yticks([0.5])
        lat_titles[j].set_yticklabels([cover.name], color=color_)
        cont = 0
        for i in cover.conditions:
            plots.append(fig.add_subplot(grid[(cont + abs_cont) * 3 +
                                              2 +j:(cont + abs_cont) * 3 + 5+j,
                                              1:]))
            plots[-1].plot(cover[chrom, start, end][0, :, 0, cont],
                           linewidth=2, color=color_)
            plots[-1].set_yticks(())
            plots[-1].set_xticks(())
            plots[-1].set_xlim([0, len(cover[chrom, start, end][0, :, 0, 0])])
            plots[-1].set_ylabel(i, labelpad=12)
            plots[-1].spines['right'].set_visible(False)
            plots[-1].spines['top'].set_visible(False)
            cont = cont + 1
        abs_cont += cont
    return (fig)
