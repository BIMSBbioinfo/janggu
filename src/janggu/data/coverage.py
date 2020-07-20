"""Coverage dataset"""

import copy
import os
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
import pyBigWig
import pysam
import pandas as pd
from progress.bar import Bar
from pybedtools import BedTool
from pybedtools import Interval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomic_indexer import check_gindexer_compatibility
from janggu.data.genomicarray import create_genomic_array
from janggu.data.genomicarray import create_sha256_cache
from janggu.utils import _check_valid_files
from janggu.utils import _get_genomic_reader
from janggu.utils import _to_list
from janggu.utils import get_genome_size_from_regions
from janggu.version import dataversion as version


def _condition_from_filename(files, conditions):
    if conditions is None:
        conditions = [os.path.splitext(os.path.basename(f))[0]
                      for f in files]
    return conditions

class BedGenomicSizeLazyLoader:
    """BedGenomicSizeLazyLoader class

    This class facilitates lazy loading of BED files.
    It reads all BED files and determines the chromosome lengths
    as the maximum length observed.

    The call method is invoked for constructing a new genomic array
    with the correct shape.
    """
    def __init__(self, bedfiles, store_whole_genome, gindexer, genomesize,
                 binsize, stepsize, flank):
        self.bedfiles = bedfiles
        self.store_whole_genome = store_whole_genome
        self.external_gindexer = gindexer
        self.genomesize = genomesize
        self.gsize_ = None
        self.gindexer_ = None
        self.binsize = binsize
        self.stepsize = stepsize
        self.flank = flank

    def load_gsize(self):
        """loads the gsize if first required."""

        if not self.store_whole_genome:
            if self.genomesize is not None:
                gsize = GenomicIndexer.create_from_genomesize(self.genomesize.copy())
            else:
                gsize = self.external_gindexer
            self.gsize_ = gsize
            self.gindexer_ = self.external_gindexer
            return

        gsize = OrderedDict()

        for bedfile in self.bedfiles:
            bed = BedTool(bedfile).sort().merge()
            for region in bed:
                if region.chrom not in gsize:
                    gsize[region.chrom] = region.end
                    continue
                if gsize[region.chrom] < region.end:
                    gsize[region.chrom] = region.end

        gsize_ = GenomicIndexer.create_from_genomesize(gsize)

        self.gsize_ = gsize_

        # New gindexer for the entire genome
        gind = GenomicIndexer(self.binsize, self.stepsize,
                              self.flank, zero_padding=True, collapse=False)
        gind.add_gindexer(gsize_)
        self.gindexer_ = gind

    @property
    def gsize(self):
        """gsize"""
        if self.gsize_ is None:
            self.load_gsize()
        return self.gsize_

    @property
    def gindexer(self):
        """gindexer"""
        if self.gindexer_ is None:  # pragma: no cover
            self.load_gsize()
        return self.gindexer_

    def __call__(self):
        return self.gsize

    def tostr(self):
        """string representation"""
        return "full_genome_lazy_loading"


class BamLoader:
    """BamLoader class.

    This class loads the GenomicArray with read count coverage
    extracted from BAM files.

    Parameters
    ----------
    files : str or list(str)
        Bam file locations.
    gsize : GenomicIndexer
        GenomicIndexer representing the genomic region that should be loaded.
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
    verbose : boolean
        Default: False
    """
    def __init__(self, files, gsize, template_extension,
                 min_mapq, pairedend, verbose=False):
        self.files = files
        self.gsize = gsize
        self.template_extension = template_extension
        self.min_mapq = min_mapq
        self.pairedend = pairedend
        self.verbose = verbose

    def __call__(self, garray):
        files = self.files
        gsize = self.gsize
        template_extension = self.template_extension
        resolution = garray.resolution
        dtype = garray.typecode
        min_mapq = self.min_mapq
        pairedend = self.pairedend

        if self.verbose: bar = Bar('Loading bam files'.format(len(files)), max=len(files))
        for i, sample_file in enumerate(files):
            aln_file = pysam.AlignmentFile(sample_file, 'rb')  # pylint: disable=no-member

            unique_chroms = list(set(gsize.chrs))
            for process_chrom in unique_chroms:
                if process_chrom not in set(aln_file.header.references):
                    continue
                tmp_gsize = gsize.filter_by_region(include=process_chrom)
                length = aln_file.header.get_reference_length(process_chrom) + tmp_gsize.flank

                array = np.zeros((length, 2), dtype=dtype)

                for aln in aln_file.fetch(str(process_chrom)):

                    if aln.is_unmapped:
                        continue

                    if aln.mapq < min_mapq:
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
                            if aln.is_read2:
                                # only consider read1 so as not to double count
                                # fragments for paired end reads
                                # read2 will also be false for single end
                                # reads.
                                continue
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
                        #pos -= start + template_extension

                        if pos < 0 or pos >= length:
                            # if the read 5 p end or mid point is outside
                            # of the region of interest, the read is discarded
                            continue

                    # fill up the read strand specifically
                    if aln.is_reverse:
                        array[pos, 1] += 1
                    else:
                        array[pos, 0] += 1

                for interval in tmp_gsize:
                    garray[interval, i] = array[interval.start:interval.end, :]
            if self.verbose: bar.next()
        if self.verbose: bar.finish()
        return garray



class BigWigLoader:
    """BigWigLoader class.

    This class loads the GenomicArray with signal coverage
    extracted from BIGWIG files.

    Parameters
    ----------
    files : str or list(str)
        Bigwig file locations.
    gsize : GenomicIndexer
        GenomicIndexer representing the genomic region that should be loaded.
    nan_to_num : bool
        Whether to convert NAN's to zeros or not. Default: True.
    verbose : boolean
        Default: False
    """
    def __init__(self, files, gsize, nan_to_num, verbose=False):
        self.files = files
        self.gsize = gsize
        self.nan_to_num = nan_to_num
        self.verbose = verbose

    def __call__(self, garray):
        files = self.files
        gsize = self.gsize
        resolution = garray.resolution
        dtype = garray.typecode
        nan_to_num = self.nan_to_num

        if self.verbose: bar = Bar('Loading bigwig files'.format(len(files)), max=len(files))
        for i, sample_file in enumerate(files):
            bwfile = pyBigWig.open(sample_file)

            unique_chroms = list(set(gsize.chrs))
            for process_chrom in unique_chroms:

                tmp_gsize = gsize.filter_by_region(include=process_chrom)
                length = max(tmp_gsize.ends) + tmp_gsize.flank

                array = np.zeros((length, 1), dtype=dtype)

                if process_chrom not in bwfile.chroms():
                    continue
                values = np.asarray(bwfile.values(str(process_chrom),
                                                  int(0),
                                                  min(int(length),
                                                      bwfile.chroms()[process_chrom])))
                if nan_to_num:
                    values = np.nan_to_num(values, copy=False)

                array[:len(values), 0] = values

                for interval in tmp_gsize:
                    garray[interval, i] = array[int(interval.start):int(interval.end), :]
            if self.verbose: bar.next()
        if self.verbose: bar.finish()
        return garray


class BedLoader:
    """BedLoader class.

    This class loads the GenomicArray with signal coverage
    extracted from BED files.

    Parameters
    ----------
    files : str or list(str)
        Bed file locations.
    lazyloader : BedGenomicSizeLazyLoader
        BedGenomicSizeLazyLoader object.
    mode : str
        Mode might be 'binary', 'score', 'categorical', 'bedgraph'.
    minoverlap : float or None
        Minimum fraction of overlap of a given feature with a roi bin.
        Default: None (already a single base-pair overlap is considered)
    conditions : list
        List of condition names
    verbose : boolean
        Default: False
    """
    def __init__(self, files, lazyloader, mode,
                 minoverlap, conditions, verbose=False):
        self.files = files
        self.lazyloader = lazyloader
        self.mode = mode
        self.minoverlap = minoverlap
        self.verbose = verbose
        self.conditions = conditions
        self.conditionindex = {c: i for i, c in enumerate(conditions)}

    def __call__(self, garray):
        files = self.files
        dtype = garray.typecode
        # Gindexer contains all relevant intervals
        # We use the bedtools intersect method to
        # project the feature overlaps with the ROI intervals.
        gindexer = self.lazyloader.gindexer
        mode = self.mode

        tmpdir = tempfile.mkdtemp()
        predictable_filename = 'gindexerdump'
        tmpfilename = os.path.join(tmpdir, predictable_filename)

        gindexer.export_to_bed(tmpfilename)

        roifile = BedTool(tmpfilename)
        nfields_a = len(roifile[0].fields)

        if self.verbose: bar = Bar('Loading bed files', max=len(files))
        gsize = self.lazyloader.gsize

        gs = (pd.DataFrame({'chrom': gsize.chrs,
                           'end': gsize.ends})
                 .groupby('chrom')
                 .aggregate({'end':'max'}))

        for i, sample_file in enumerate(files):
            regions_ = _get_genomic_reader(sample_file)

            if regions_[0].score == '.' and mode in ['score',
                                                     'categorical',
                                                     'score_category',
                                                     'name_category']:
                raise ValueError(
                    'No Score available. Score field must '
                    'present in {}'.format(sample_file) + \
                    'for mode="{}"'.format(mode))

            # init whole genome array
            arrays = {j: np.zeros((row['end'], 2), dtype=dtype) for j, row in gs.iterrows()}

            # load data from signal coverage
            for region in regions_:
                if mode == 'bedgraph':
                    score = float(region.fields[-1])
                elif mode == 'score':
                    score = int(region.score)
                elif mode == 'binary':
                    score = 1
                elif mode in ['categorical', 'score_category']:
                    score = self.conditionindex[str(region.score)] if str(region.score) in self.conditionindex else None
                else:
                    score = self.conditionindex[region.name] if region.name in self.conditionindex else None

                if region.chrom in arrays and score is not None:
                    # first dim, score value, second dim, mask
                    arrays[region.chrom][region.start:region.end, 0] = score
                    arrays[region.chrom][region.start:region.end, 1] = 1

            # map data to rois
            roiregs = roifile.intersect(regions_, wa=True, u=True)
            for roireg in roiregs:
                if roireg.end <= arrays[roireg.chrom].shape[0]:
                    tmp_array = arrays[roireg.chrom][roireg.start:roireg.end]
                else:
                    tmp_array = np.zeros((roireg.length, 2))
                    tmp_array[:arrays[roireg.chrom][roireg.start:].shape[0]] = \
                        arrays[roireg.chrom][roireg.start:]
                if self.minoverlap is not None:
                    if tmp_array[:, :1].nonzero()[0].shape[0]/roireg.length < \
                        self.minoverlap:
                        # minimum overlap not achieved, skip
                        continue

                if mode in ['categorical', 'score_category', 'name_category']:
                    tmp_cat = np.zeros((roireg.length, 1, int(tmp_array.max())+1), dtype=dtype)
                    tmp_cat[np.arange(roireg.length), 0, tmp_array[:, 0].astype('int')] = tmp_array[:, 1]

                    for r in range(tmp_cat.shape[-1]):
                        garray[roireg, r] = tmp_cat[:, :, r]

                else:
                    garray[roireg, i] = tmp_array[:, :1]
            if self.verbose: bar.next()

        if self.verbose: bar.finish()
        os.remove(tmpfilename)
        os.rmdir(tmpdir)

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
    verbose : boolean
        Default: False
    """
    def __init__(self, array, gindexer, verbose=False):
        self.array = array
        self.gindexer = gindexer
        self.verbose = verbose

    def __call__(self, garray):
        array = self.array
        gindexer = self.gindexer

        if self.verbose: bar = Bar('Loading from array', max=len(gindexer))
        for i, region in enumerate(gindexer):
            interval = region
            new_item = array[i]
            if new_item.ndim < 3:
                garray[interval, :] = new_item[None, None, :]
            else:
                garray[interval, :] = new_item[:]
            if self.verbose: bar.next()
        if self.verbose: bar.finish()

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
                 gindexer):

        self.garray = garray
        self.gindexer = gindexer
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
                        normalizer=None,
                        zero_padding=True,
                        random_state=None,
                        store_whole_genome=False,
                        verbose=False):
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
        roi : str, list(Interval), BedTool, pandas.DataFrame or None
            Region of interest over which to iterate.
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
            Normalization is ignored when using storage='sparse'.
            Default: None.
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
        store_whole_genome : boolean
            Indicates whether the whole genome or only ROI
            should be loaded. If False, a bed-file with regions of interest
            must be specified. Default: False
        verbose : boolean
            Verbosity. Default: False
        """

        if overwrite:  # pragma: no cover
            warnings.warn('overwrite=True is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)
        if datatags is not None:  # pragma: no cover
            warnings.warn('datatags is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank,
                                                       zero_padding,
                                                       collapse,
                                                       random_state=random_state)
        else:
            gindexer = None

        check_gindexer_compatibility(gindexer, resolution, store_whole_genome)

        bamfiles = _check_valid_files(_to_list(bamfiles))

        conditions = _condition_from_filename(bamfiles, conditions)

        if min_mapq is None:
            min_mapq = 0

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = gindexer
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
            gsize = GenomicIndexer.create_from_genomesize(gsize)

        bamloader = BamLoader(bamfiles, gsize, template_extension,
                              min_mapq, pairedend, verbose)

        datatags = [name]
        normalizer = _to_list(normalizer)

        if cache:
            files = copy.copy(bamfiles)

            parameters = [gsize.tostr(), min_mapq,
                          resolution, storage, dtype, stranded,
                          pairedend, zero_padding,
                          store_whole_genome, version]
            if not store_whole_genome:
                files += [roi]
                parameters += [binsize, stepsize, flank,
                               template_extension, random_state]
            if storage == 'hdf5':
                parameters += normalizer
            cache_hash = create_sha256_cache(files, parameters)
        else:
            cache_hash = None
        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        cover = create_genomic_array(gsize, stranded=stranded,
                                     storage=storage, datatags=datatags,
                                     cache=cache_hash,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     store_whole_genome=store_whole_genome,
                                     resolution=resolution,
                                     loader=bamloader,
                                     normalizer=normalizer,
                                     collapser='sum',
                                     verbose=verbose)

        return cls(name, cover, gindexer)

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
                           zero_padding=True,
                           normalizer=None,
                           collapser=None,
                           random_state=None,
                           nan_to_num=True,
                           verbose=False):
        """Create a Cover class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bigwigfiles : str or list
            bigwig-file or list of bigwig files.
        roi : str, list(Interval), BedTool, pandas.DataFrame or None
            Region of interest over which to iterate.
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
            Normalization is ignored when using storage='sparse'.
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
        verbose : boolean
            Verbosity. Default: False
        """

        if overwrite:  # pragma: no cover
            warnings.warn('overwrite=True is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)
        if datatags is not None:  # pragma: no cover
            warnings.warn('datatags is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank,
                                                       zero_padding,
                                                       collapse,
                                                       random_state=random_state)
        else:
            gindexer = None

        check_gindexer_compatibility(gindexer, resolution, store_whole_genome)

        bigwigfiles = _check_valid_files(_to_list(bigwigfiles))

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = gindexer
        else:
            # otherwise the whole genome will be fetched, or at least
            # a set of full length chromosomes
            if genomesize is not None:
                # if a genome size has specifically been given, use it.
                gsize = genomesize.copy()
            else:
                bwfile = pyBigWig.open(bigwigfiles[0], 'r')
                gsize = bwfile.chroms()
            gsize = GenomicIndexer.create_from_genomesize(gsize)

        conditions = _condition_from_filename(bigwigfiles, conditions)

        bigwigloader = BigWigLoader(bigwigfiles, gsize, nan_to_num, verbose)
        datatags = [name]

        collapser_ = collapser if collapser is not None else 'mean'
        normalizer = _to_list(normalizer)

        if cache:
            files = copy.copy(bigwigfiles)
            parameters = [gsize.tostr(),
                          resolution, storage, dtype,
                          zero_padding,
                          collapser.__name__ if hasattr(collapser, '__name__') else collapser,
                          store_whole_genome, nan_to_num, version]
            if not store_whole_genome:
                files += [roi]
                parameters += [binsize, stepsize, flank, random_state]
            if storage == 'hdf5':
                parameters += normalizer
            cache_hash = create_sha256_cache(files, parameters)
        else:
            cache_hash = None

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache_hash,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     resolution=resolution,
                                     store_whole_genome=store_whole_genome,
                                     typecode=dtype,
                                     loader=bigwigloader,
                                     collapser=collapser_,
                                     normalizer=normalizer,
                                     verbose=verbose)

        return cls(name, cover, gindexer)

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
                        zero_padding=True,
                        normalizer=None,
                        collapser=None,
                        minoverlap=None,
                        random_state=None,
                        datatags=None, cache=False,
                        verbose=False):
        """Create a Cover class from a bed-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bedfiles : str or list
            bed-file or list of bed files.
        roi : str, list(Interval), BedTool, pandas.DataFrame or None
            Region of interest over which to iterate.
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
            Determines how the BED-like file should be interpreted, e.g. as class labels or
            scores.
            'binary' is used for presence/absence representation of features
            for a binary classification setting. Regions in the
            :code:`bedfiles` that intersect the ROI are considered positive examples, while
            the remaining ROI intervals are negative examples.
            'score' allows to use the score-value associated with the intervals (e.g. for regression).
            'score_category' (formerly 'categorical') allows to interpret the integer-valued score as class-label for categorical labels. The labels will be one-hot encoded.
            'name_category' allows to interpret the name field as class-label for categorical labels. The labels will be one-hot encoded.
            'bedgraph' indicates that the input file is in bedgraph format and reads out the associated score for each interval.
            Mode of the dataset may be 'binary', 'score', 'score_category' (or 'categorical'), 'name_category' or 'bedgraph'.
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
            Normalization is ignored when using storage='sparse'.
            Default: None.
        collapser : None, str or callable
            This option defines how the genomic signal should be summarized when resolution
            is None or greater than one. It is possible to choose a number of options by
            name, including 'sum', 'mean', 'max'. In addtion, a function may be supplied
            that defines a custom aggregation method. If collapser is None,
            'max' aggregation will be used.
            Default: None.
        minoverlap : float or None
            Minimum fraction of overlap of a given feature with a ROI bin.
            If None, any overlap (e.g. a single base-pair overlap) is
            considered as overlap.
            Default: None
        cache : boolean
            Indicates whether to cache the dataset. Default: False.
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
        verbose : boolean
            Verbosity. Default: False
        """

        if overwrite:  # pragma: no cover
            warnings.warn('overwrite=True is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)
        if datatags is not None:  # pragma: no cover
            warnings.warn('datatags is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)

        collapse = True if resolution is None else False

        if roi is not None:
            gindexer = GenomicIndexer.create_from_file(roi, binsize,
                                                       stepsize, flank,
                                                       zero_padding,
                                                       collapse,
                                                       random_state=random_state)
            binsize = gindexer.binsize
        else:
            gindexer = None

        check_gindexer_compatibility(gindexer, resolution, store_whole_genome)

        bedfiles = _check_valid_files(_to_list(bedfiles))

        gsize = BedGenomicSizeLazyLoader(bedfiles,
                                         store_whole_genome,
                                         gindexer, genomesize,
                                         binsize, stepsize, flank)

        if conditions is None and \
           mode in ['categorical', 'score_category', 'name_category']:
            if len(bedfiles) > 1:
                raise ValueError('Only one bed-file is '
                                 'allowed with mode=categorical, '
                                 'but got multiple files.')
            sample_file = bedfiles[0]
            regions_ = _get_genomic_reader(sample_file)

            categories = set()
            for reg in regions_:
                categories.add(reg.name if mode == 'name_category' \
                               else str(reg.score))
            conditions = sorted(list(categories))
        conditions = _condition_from_filename(bedfiles, conditions)

        bedloader = BedLoader(bedfiles, gsize, mode,
                              minoverlap, conditions, verbose)

        datatags = [name]

        collapser_ = collapser if collapser is not None else 'max'

        normalizer = _to_list(normalizer)

        if cache:
            files = copy.copy(bedfiles)
            parameters = [gsize.tostr(),
                          resolution, storage, dtype,
                          zero_padding, mode,
                          collapser.__name__ if hasattr(collapser, '__name__') else collapser,
                          store_whole_genome, version, minoverlap]
            # Because different binsizes may affect loading e.g. if a min overlap is required.
            parameters += [binsize, stepsize, flank]
            if not store_whole_genome:
                files += [roi]
                parameters += [random_state]
            if storage == 'hdf5':
                parameters += normalizer
            cache_hash = create_sha256_cache(files, parameters)
        else:
            cache_hash = None

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache_hash,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=bedloader,
                                     collapser=collapser_,
                                     normalizer=normalizer,
                                     verbose=verbose)

        return cls(name, cover, gindexer)

    @classmethod
    def create_from_array(cls, name,  # pylint: disable=too-many-locals
                          array,
                          gindexer,
                          genomesize=None,
                          conditions=None,
                          resolution=None,
                          storage='ndarray',
                          overwrite=False,
                          cache=False,
                          datatags=None,
                          padding_value=0.0,
                          store_whole_genome=False,
                          verbose=False):
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
        store_whole_genome : boolean
            Indicates whether the whole genome or only ROI
            should be loaded. Default: False.
        padding_value : float
            Padding value. Default: 0.
        verbose : boolean
            Verbosity. Default: False
        """

        if overwrite:  # pragma: no cover
            warnings.warn('overwrite=True is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)
        if datatags is not None:  # pragma: no cover
            warnings.warn('datatags is without effect '
                          'due to revised caching functionality.'
                          'The argument will be removed in the future.',
                          FutureWarning)

        if not store_whole_genome:
            # if whole genome should not be loaded
            gsize = gindexer
        elif genomesize:
            gsize = genomesize.copy()
            gsize = GenomicIndexer.create_from_genomesize(gsize)
        else:
            # if not supplied, determine the genome size automatically
            # based on the gindexer intervals.
            gsize = get_genome_size_from_regions(gindexer)
            gsize = GenomicIndexer.create_from_genomesize(gsize)

        if conditions is None:
            conditions = ["Cond_{}".format(i) for i in range(array.shape[-1])]

        # check if dimensions of gindexer and array match
        if len(gindexer) != array.shape[0]:
            raise ValueError("Data incompatible: "
                             "Number of regions must match with "
                             "the number of datapoints. "
                             "(len(gindexer)={} != array.shape[0]={})".
                             format(len(gindexer), array.shape[0]))

        if store_whole_genome:
            # in this case the intervals must be non-overlapping
            # in order to obtain unambiguous data.

            if not gindexer.collapse and gindexer.binsize > gindexer.stepsize:
                raise ValueError("Overlapping intervals: With overlapping "
                                 "intervals the mapping between the array and "
                                 "genomic-array values is ambiguous. "
                                 "Please ensure that binsize <= stepsize.")

        if resolution is None:
            # determine the resolution
            if gindexer.collapse:
                # binsize will not be set if gindexer was loaded in collapse mode
                resolution = None
            else:
                resolution = max(1, gindexer.binsize // array.shape[1])

        # determine strandedness
        stranded = True if array.ndim == 3 and array.shape[2] == 2 else False

        arrayloader = ArrayLoader(array, gindexer, verbose)
        # At the moment, we treat the information contained
        # in each bw-file as unstranded

        datatags = [name]

        # define a dummy collapser

        def _dummy_collapser(values):
            # should be 3D
            # seqlen, resolution, strand
            return values[:, 0, :]

        if cache:
            files = [array]
            parameters = [genomesize, gindexer.binsize,
                          resolution, storage, stranded,
                          _dummy_collapser.__name__, version,
                          store_whole_genome] + [str(reg_) for reg_ in gindexer]
            cache_hash = create_sha256_cache(files, parameters)
        else:
            cache_hash = None

        cover = create_genomic_array(gsize, stranded=stranded,
                                     storage=storage, datatags=datatags,
                                     cache=cache_hash,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=array.dtype,
                                     store_whole_genome=store_whole_genome,
                                     loader=arrayloader,
                                     padding_value=padding_value,
                                     collapser=_dummy_collapser,
                                     verbose=verbose)

        return cls(name, cover, gindexer)

    @property
    def gindexer(self):
        """GenomicIndexer property"""
        if self._gindexer is None:
            raise ValueError('No GenomicIndexer available. Please specify an gindexer.')
        return self._gindexer

    @gindexer.setter
    def gindexer(self, gindexer):
        self._gindexer = gindexer

    def __repr__(self):  # pragma: no cover
        return "Cover('{}')".format(self.name)

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            if len(idxs) == 3:
                idxs = Interval(*idxs)
            else:
                idxs = Interval(*idxs[:-1], strand=idxs[-1])

        if isinstance(idxs, int):
            idxs = [idxs]
        elif isinstance(idxs, slice):
            idxs = range(idxs.start if idxs.start else 0,
                         idxs.stop if idxs.stop else len(self),
                         idxs.step if idxs.step else 1)
        elif isinstance(idxs, Interval):
            if self.garray._full_genome_stored:
                # accept a genomic interval directly
                # but upscale the length to nucleotide resolution
                # because the plotting functionality expects that.
                data = self._getsingleitem(idxs)
                data = data.reshape((1,) + data.shape)
                resolution = self.garray.resolution
                data = data.repeat(resolution, axis=1)

                # the actual genomic coordinates must no be mapped onto the
                # rescaled data.
                data_start_offset = idxs.start - (idxs.start//resolution)*resolution
                data_end_offset = data_start_offset + idxs.length
                data = data[:, data_start_offset:data_end_offset, :, :]

            else:
                chrom = str(idxs.chrom)
                start = idxs.start
                end = idxs.end
                strand = str(idxs.strand)
                gindexer_new = self.gindexer.filter_by_region(include=chrom,
                                                              start=start,
                                                              end=end)

                if self.garray.padding_value == 0.0:
                    data = np.zeros((1, (end - start)) + self.shape[2:])
                else:
                    data = np.ones((1, (end - start)) + self.shape[2:]) * self.garray.padding_value

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
        return self.shape_static

    @property
    def shape_static(self):
        """Shape of the dataset"""
        stranded = (2 if self.garray.stranded else 1, )
        if self.garray.resolution is not None:
            seqdims = (int(np.ceil((self.gindexer.binsize + \
                       2*self.gindexer.flank)/self.garray.resolution)),)
        else:
            seqdims = (1,)
        return (len(self),) + seqdims + stranded + (len(self.garray.condition),)

    @property
    def ndim(self):  # pragma: no cover
        """ndim"""
        return len(self.shape)

    @property
    def conditions(self):
        """Conditions"""
        return self.garray.condition

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
        os.makedirs(output_dir, exist_ok=True)

        resolution = self.garray.resolution
        if resolution is None:
            resolution = self.gindexer[0].length

        if genomesize is not None:
            gsize = genomesize
        elif self.garray._full_genome_stored:
            gsize = {k: self.garray.handle[k].shape[0] * resolution \
                     for k in self.garray.handle}
        else:
            gsize = get_genome_size_from_regions(self.gindexer)

        chrorder = OrderedDict.fromkeys(self.gindexer.chrs)

        bw_header = [(str(chrom), gsize[chrom])
                     for chrom in chrorder]

        # approch suggested by remo
        multi = int(np.ceil(self.gindexer[0].length / resolution))
        chroms = np.repeat(self.gindexer.chrs, multi).tolist()
        starts = [iv.start + resolution*m for iv in self.gindexer for m in range(multi)]
        ends = [iv.start + (resolution)*m for iv in self.gindexer for m in range(1, multi+1)]
        cov = self[:].reshape((-1,)+ self.shape[2:]).sum(axis=1)

        for idx, condition in enumerate(self.conditions):
            bw_file = pyBigWig.open(os.path.join(
                output_dir,
                '{name}.{condition}.bigwig'.format(
                    name=self.name, condition=condition)), 'w')

            bw_file.addHeader(bw_header)

            bw_file.addEntries(chroms,
                               starts,
                               ends=ends,
                               values=cov[:, idx].tolist())

            bw_file.close()
