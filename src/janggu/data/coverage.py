"""Coverage dataset"""

import os

import numpy as np
try:
    import pyBigWig
except ImportError:  # pragma: no cover
    pyBigWig = None
try:
    import pysam
except ImportError:  # pragma: no cover
    pysam = None
from HTSeq import GenomicInterval

from janggu.data.data import Dataset
from janggu.data.genomic_indexer import GenomicIndexer
from janggu.data.genomicarray import create_genomic_array
from janggu.utils import _get_genomic_reader
from janggu.utils import _iv_to_str
from janggu.utils import _str_to_iv
from janggu.utils import get_genome_size_from_bed


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
        A genomic index mapper that translates an integer index to a
        genomic coordinate. Can be None, if the Dataset is only loaded.
    padding_value : int or float
        Padding value used to pad variable size fragments. Default: 0.
    """

    _flank = None
    _gindexer = None

    def __init__(self, name, garray,
                 gindexer,  # indices of pointing to region start
                 padding_value,
                 dimmode):  # padding value

        self.garray = garray
        self.gindexer = gindexer
        self.padding_value = padding_value
        self.dimmode = dimmode

        Dataset.__init__(self, name)

    @classmethod
    def create_from_bam(cls, name,  # pylint: disable=too-many-locals
                        bamfiles,
                        regions=None,
                        genomesize=None,
                        conditions=None,
                        min_mapq=None,
                        binsize=200, stepsize=200,
                        flank=0,
                        resolution=1,
                        storage='ndarray',
                        dtype='int',
                        stranded=True,
                        overwrite=False,
                        pairedend='5prime',
                        template_extension=0,
                        aggregate=None,
                        datatags=None, cache=False):
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
            Bed-file defining the regions that comprise the dataset.
            If set to None, a genomic indexer must be attached later.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size
            is fetched from the bam header. Otherwise, the supplied genome
            size is used.
        conditions : list(str) or None
            List of conditions. If `conditions=None`, the filenames
            are used as conditions directly.
        min_mapq : int
            Minimal mapping quality. Reads with lower mapping quality are
            discarded. If None, all reads are used.
        binsize : int
            Binsize in basepairs. Default: 200.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 200.
        flank : int
            Flanking size increases the interval size at both ends by
            flank base pairs. Default: 0
        resolution : int
            Resolution in base pairs. This is used to collect the mean signal
            over the window lengths defined by the resolution.
            This value must be chosen to be divisible by binsize and stepsize.
            Default: 1.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray', 'hdf5' or 'sparse'. Default: 'ndarray'.
        dtype : str
            Typecode to define the datatype to be used for storage.
            Default: 'int'.
        stranded : boolean
            Whether to extract stranded or unstranded coverage. For unstranded
            coverage, reads aligning to both strands will be aggregated.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Default: None.
        pairedend : str
            Indicates whether to count the region at the '5prime' end or at
            the 'midpoint' between the two reads.
        template_extension : int
            Elongates the region of interest by the template_extension in
            order to fetch paired end reads that do not directly overlap
            with the region but whose inferred template does.
            If template_extension is not set with pairedend='midpoint', reads
            might potentially be missed.
        aggregate : callable or None
            Aggregation operation for loading genomic array for a given resolution.
            Default: None
        cache : boolean
            Whether to cache the dataset. Default: False.
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

        if genomesize is not None:
            # if a genome size has specifically been given, use it.
            gsize = genomesize.copy()
            full_genome_index = True
        elif gindexer is not None:
            # if a gindexer has been supplied, load the array only for the
            # region of interest
            gsize = {_iv_to_str(iv.chrom, iv.start,
                                iv.end): iv.end-iv.start for iv in gindexer}
            full_genome_index = False
        else:
            header = pysam.AlignmentFile(bamfiles[0], 'r')  # pylint: disable=no-member
            gsize = {}
            for chrom, length in zip(header.references, header.lengths):
                gsize[chrom] = length
            full_genome_index = True

        def _bam_loader(garray, files):
            print("load from bam")
            for i, sample_file in enumerate(files):
                print('Counting from {}'.format(sample_file))
                aln_file = pysam.AlignmentFile(sample_file, 'rb')  # pylint: disable=no-member
                for chrom in gsize:

                    array = np.zeros((gsize[chrom]//resolution, 2), dtype=dtype)

                    locus = _str_to_iv(chrom, template_extension=template_extension)

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

                        if len(locus) == 3:
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
                        garray[GenomicInterval(chrom, 0, gsize[chrom],
                                               '+'), i] = array[:, 0]
                        garray[GenomicInterval(chrom, 0, gsize[chrom],
                                               '-'), i] = array[:, 1]
                    else:
                        # if unstranded, aggregate the reads from
                        # both strands
                        garray[GenomicInterval(chrom, 0, gsize[chrom],
                                               '.'), i] = array.sum(axis=1)

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
                                     resolution=resolution,
                                     loader=_bam_loader,
                                     loader_args=(bamfiles,))
        cover._full_genome_stored = full_genome_index

        return cls(name, cover, gindexer, padding_value=0, dimmode='all')

    @classmethod
    def create_from_bigwig(cls, name,  # pylint: disable=too-many-locals
                           bigwigfiles,
                           regions=None,
                           genomesize=None,
                           conditions=None,
                           binsize=200, stepsize=200,
                           resolution=200,
                           flank=0, storage='ndarray',
                           dtype='float32',
                           overwrite=False,
                           dimmode='all',
                           aggregate=np.mean,
                           datatags=None, cache=False):
        """Create a Cover class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bigwigfiles : str or list
            bigwig-file or list of bigwig files.
        regions : str or None
            Bed-file defining the regions that comprise the dataset.
            If set to None, a genomic indexer must be attached later.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size is fetched from the regions defined by the bed-file.
            Otherwise, the supplied genome size is used.
        conditions : list(str) or None
            List of conditions. If `conditions=None`, the filenames
            are used as conditions directly.
        binsize : int
            Binsize in basepairs. Default: 200.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 200.
        resolution : int
            Resolution in base pairs. This is used to collect the mean signal
            over the window lengths defined by the resolution.
            This value must be chosen to be divisible by binsize and stepsize.
            Default: 200.
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
            the first element of size resolution is used. Otherwise,
            all elements of size resolution spanning the binsize are returned.
            Default: 'all'.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Default: None.
        aggregate : callable
            Aggregation operation for loading genomic array for a given resolution.
            Default: numpy.mean
        cache : boolean
            Whether to cache the dataset. Default: False.
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

        if genomesize is not None:
            # if a genome size has specifically been given, use it.
            gsize = genomesize.copy()
            full_genome_index = True
        elif gindexer is not None:
            # if a gindexer has been supplied, load the array only for the
            # region of interest
            gsize = {_iv_to_str(iv.chrom, iv.start, iv.end): iv.end-iv.start for iv in gindexer}
            full_genome_index = False
        else:
            bwfile = pyBigWig.open(bigwigfiles[0], 'r')
            gsize = bwfile.chroms()
            full_genome_index = True

        if conditions is None:
            conditions = [os.path.splitext(os.path.basename(f))[0] for f in bigwigfiles]

        def _bigwig_loader(garray, aggregate):
            print("load from bigwig")
            for i, sample_file in enumerate(bigwigfiles):
                bwfile = pyBigWig.open(sample_file)

                for chrom in gsize:

                    vals = np.empty((gsize[chrom]//resolution),
                                    dtype=dtype)

                    locus = _str_to_iv(chrom, template_extension=0)

                    if len(locus) == 1:
                        for start in range(0, gsize[chrom], resolution):

                            vals[start//resolution] = aggregate(np.asarray(bwfile.values(
                                chrom,
                                int(start),
                                int(min((start+resolution), gsize[chrom])))))

                    else:
                        for start in range(locus[1], locus[2], resolution):
                            vals[(start - locus[1])//resolution] = aggregate(np.asarray(bwfile.values(
                                locus[0], int(start), int(start+resolution))))
                        # not sure what to do with nan yet.

                    garray[GenomicInterval(chrom, 0, gsize[chrom]), i] = vals
            return garray

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     overwrite=overwrite,
                                     resolution=resolution,
                                     typecode=dtype,
                                     loader=_bigwig_loader,
                                     loader_args=(aggregate,))
        cover._full_genome_stored = full_genome_index

        return cls(name, cover, gindexer,
                   padding_value=0, dimmode=dimmode)

    @classmethod
    def create_from_bed(cls, name,  # pylint: disable=too-many-locals
                        bedfiles,
                        regions=None,
                        genomesize=None,
                        conditions=None,
                        binsize=200, stepsize=200,
                        resolution=200,
                        flank=0, storage='ndarray',
                        dtype='int',
                        dimmode='all',
                        mode='binary',
                        overwrite=False,
                        datatags=None, cache=False):
        """Create a Cover class from a bed-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bedfiles : str or list
            bed-file or list of bed files.
        regions : str or None
            Bed-file defining the regions that comprise the dataset.
            If set to None, a genomic indexer must be attached later.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size is fetched from the regions defined by the bed-file.
            Otherwise, the supplied genome size is used.
        conditions : list(str) or None
            List of conditions. If `conditions=None`, the filenames
            are used as conditions directly.
        binsize : int
            Binsize in basepairs. Default: 200.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 200.
        resolution : int
            Resolution in base pairs. This is used to collect the mean signal
            over the window lengths defined by the resolution.
            This value should be chosen to be divisible by binsize and stepsize.
            Default: 200.
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
            the first element of size resolution is used. Otherwise,
            all elements of size resolution spanning the binsize are returned.
            Default: 'all'.
        mode : str
            Mode of the dataset may be 'binary', 'score' or 'categorical'.
            Default: binary.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        datatags : list(str) or None
            List of datatags. Default: None.
        cache : boolean
            Whether to cache the dataset. Default: False.
        """

        if regions is None and genomesize is None:
            raise ValueError('Either regions or gsize must be specified.')

        if regions is not None:
            gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                       stepsize, flank)
        else:
            gindexer = None

        # automatically determine genomesize from largest region
        if not genomesize:
            gsize = get_genome_size_from_bed(regions)
        else:
            gsize = genomesize.copy()

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
                print(sample_file)
                regions_ = _get_genomic_reader(sample_file)

                for region in regions_:
                    if region.iv.chrom not in genomesize:
                        continue

                    if genomesize[region.iv.chrom] <= region.iv.start:
                        print('Region {} outside of '.format(region.iv) +
                              'genome size - skipped')
                    else:
                        if region.score is None and mode in ['score',
                                                             'categorical']:
                            raise ValueError(
                                'score field must '
                                'be available with mode="{}"'.format(mode))
                        # if region score is not defined, take the mere
                        # presence of a range as positive label.
                        if mode == 'score':
                            garray[region.iv,
                                   i] = np.dtype(dtype).type(region.score)
                        elif mode == 'categorical':
                            garray[region.iv,
                                   int(region.score)] = np.dtype(dtype).type(1)
                        elif mode == 'binary':
                            garray[region.iv,
                                   i] = np.dtype(dtype).type(1)
            return garray

        # At the moment, we treat the information contained
        # in each bw-file as unstranded

        datatags = [name] + datatags if datatags else [name]
        datatags += ['resolution{}'.format(resolution)]

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, datatags=datatags,
                                     cache=cache,
                                     conditions=conditions,
                                     resolution=resolution,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     loader=_bed_loader,
                                     loader_args=(bedfiles, gsize, mode))

        return cls(name, cover, gindexer,
                   padding_value=-1, dimmode=dimmode)

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

        if (gindexer.stepsize % self.garray.resolution) != 0:
            raise ValueError('gindexer.stepsize must be divisible by resolution')
        self._gindexer = gindexer

    def __repr__(self):  # pragma: no cover
        return "Cover('{}') ".format(self.name)

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
            raise IndexError('Cover.__getitem__: '
                             + 'index must be iterable')

        data = np.empty((len(idxs),) + self.shape[1:])
        data.fill(self.padding_value)

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            pinterval = interval.copy()

            pinterval.start = interval.start

            if self.dimmode == 'all':
                pinterval.end = interval.end
            elif self.dimmode == 'first':
                pinterval.end = pinterval.start + self.garray.resolution

            data[i, :((pinterval.end-pinterval.start)//self.garray.resolution), :, :] = \
                np.asarray(self.garray[pinterval])

            if interval.strand == '-':
                # if the region is on the negative strand,
                # flip the order  of the coverage track
                data[i, :, :, :] = data[i, ::-1, ::-1, :]
        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        if self.dimmode == 'all':
            blen = (self.gindexer.binsize) // self.garray.resolution
            seqlen = 2*self.gindexer.flank // self.garray.resolution + \
                (blen if blen > 0 else 1)
        elif self.dimmode == 'first':
            seqlen = 1
        return (len(self),
                seqlen, 2 if self.garray.stranded else 1, len(self.garray.condition))

    @property
    def conditions(self):
        """Conditions"""
        return [s.decode('utf-8') for s in self.garray.condition]
