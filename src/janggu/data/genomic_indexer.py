"""GenomicIndexer module"""

import numpy as np
from pybedtools import Interval
from sklearn.utils import check_random_state

from janggu.utils import _get_genomic_reader


class GenomicIndexer(object):  # pylint: disable=too-many-instance-attributes
    """GenomicIndexer maps a set of integer indices to respective
    genomic intervals.

    The genomic intervals can be directly used to obtain data from a genomic
    array.
    """

    _stepsize = None
    _binsize = None
    _flank = None
    zero_padding = None
    collapse = None
    _randomidx = None
    chrs = None
    starts = None
    strand = None
    ends = None
    _random_state = None

    @property
    def randomidx(self):
        """randomidx property"""
        if self.random_state is not None and self._randomidx is None:
            self._randomidx = check_random_state(self.random_state).permutation(len(self))
        return self._randomidx

    @property
    def random_state(self):
        """random_state property"""
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._randomidx = None
        self._random_state = value

    @classmethod
    def create_from_file(cls, regions,  # pylint: disable=too-many-locals
                         binsize, stepsize, flank=0,
                         zero_padding=True, collapse=False,
                         random_state=None):
        """Creates a GenomicIndexer object.

        This method constructs a GenomicIndexer from
        a given BED or GFF file.

        Parameters
        ----------
        regions : str or list(Interval)
            Path to a BED or GFF file.
        binsize : int or None
            Binsize in base pairs. If None, the binsize is obtained from
            the interval lengths in the bed file, which requires intervals
            to be of equal length.
        stepsize : int or None
            Stepsize in base pairs. If stepsize is None,
            stepsize is set to equal to binsize.
        flank : int
            flank size in bp to be attached to
            both ends of a region. Default: 0.
        zero_padding : boolean
            zero_padding indicate if variable sequence
            lengths are used in conjunction with zero-padding.
            If zero_padding is True, a binsize must be specified.
            Default: True.
        collapse : boolean
            collapse indicates that the genomic interval will be represented by a
            scalar summary value. For example, the gene expression value in TPM.
            In this case, zero_padding does not have an effect. Intervals
            may be of fixed or variable lengths.
            Default: False.
        random_state : None or int
            random_state for shuffling intervals. Default: None
        """

        regions_ = _get_genomic_reader(regions)

        if binsize is None and not collapse:
            binsize_ = None
            # binsize will be inferred from bed file
            # the maximum interval length will be used
            for reg in regions_:
                if binsize_ is None:
                    binsize_ = reg.length
                if binsize_ < reg.length:
                    binsize_ = reg.length
            binsize = binsize_

        if stepsize is None:
            if binsize is None:
                stepsize = 1
            else:
                stepsize = binsize

        gind = cls(binsize, stepsize, flank,
                   zero_padding=zero_padding,
                   collapse=collapse, random_state=random_state)

        for reg in regions_:

            gind.add_interval(
                str(reg.chrom),
                int(reg.start), int(reg.end), str(reg.strand))

        return gind

    @classmethod
    def create_from_genomesize(cls, gsize):
        """Creates a GenomicIndexer object.

        This method constructs a GenomicIndexer from
        a given BED or GFF file.

        Parameters
        ----------
        gsize : dict
            Dictionary with keys and values representing chromosome names
            and lengths, respectively.
        """

        gind = cls(None, None, 0, zero_padding=False, collapse=False)

        for chrom in gsize:

            gind.add_interval(chrom, 0, gsize[chrom], '.')

        return gind

    @classmethod
    def create_from_region(cls, chrom, start, end, strand,
                           binsize, stepsize, flank=0,
                           zero_padding=True):
        """Creates a GenomicIndexer object.

        This method constructs a GenomicIndexer from
        a given genomic interval.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start : int
            Interval start
        end : int
            Interval end
        binsize : int or None
            Binsize in base pairs. If None, the binsize is obtained from
            the interval lengths in the bed file, which requires intervals
            to be of equal length.
        stepsize : int or None
            Stepsize in base pairs. If stepsize is None,
            stepsize is set to equal to binsize.
        flank : int
            flank size in bp to be attached to
            both ends of a region. Default: 0.
        zero_padding : boolean
            zero_padding indicate if variable sequence
            lengths are used in conjunction with zero-padding.
            If zero_padding is True, a binsize must be specified.
            Default: True.
        """

        if binsize is None:
            binsize = end - start

        if stepsize is None:
            stepsize = binsize

        gind = cls(binsize, stepsize, flank, zero_padding=zero_padding)

        gind.add_interval(chrom, start, end, strand)
        return gind

    def add_interval(self, chrom, start, end, strand):
        """Adds an interal to a GenomicIndexer object.

        This method adds another interal to a GenomicIndexer.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start : int
            Interval start
        end : int
            Interval end
        """

        binsize = self.binsize
        stepsize = self.stepsize
        zero_padding = self.zero_padding

        if binsize is None:
            binsize = end - start

        if stepsize is None:
            stepsize = binsize

        if stepsize <= binsize:
            val = (end - start - binsize + stepsize)
        else:
            val = (end - start)

        reglen = val // stepsize
        chrs = [chrom] * reglen
        starts = [x for x in range(start, start+(stepsize*reglen), stepsize)]

        ends = [x + binsize for x in starts]
        strands = [strand] * reglen
        # if there is a variable length fragment at the end,
        # we record the remaining fragment length
        if zero_padding and val % stepsize > 0:
            chrs += [chrom]
            starts += [start+(stepsize*reglen)]
            ends += [end]
            strands += [strand]

        self.chrs += chrs
        self.starts += starts
        self.strand += strands
        self.ends += ends
        return self

    def add_gindexer(self, othergindexer):
        """Adds intervals from another GenomicIndexer object.

        Parameters
        ----------
        othergindexer : GenomicIndexer
            GenomicIndexer object.
        """

        # add_interval ensures that the intervals from the other indexer
        # are adapted according to the binsize and stepsize used in self
        for region in othergindexer:
            self.add_interval(region.chrom, region.start, region.end, region.strand)
        return self

    def __init__(self, binsize, stepsize, flank=0, zero_padding=True,
                 collapse=False, random_state=None):

        self.binsize = binsize
        self.stepsize = stepsize
        self.flank = flank
        self.chrs = []
        self.starts = []
        self.strand = []
        self.ends = []
        self.zero_padding = zero_padding
        self.collapse = collapse
        self.random_state = random_state


    def __len__(self):
        return len(self.chrs)

    def __repr__(self):  # pragma: no cover
        return "GenomicIndexer(<regions>, " \
            + "binsize={}, stepsize={}, flank={})".format(self.binsize,
                                                          self.stepsize,
                                                          self.flank)

    def __getitem__(self, index_):
        if isinstance(index_, int):
            if self.randomidx is not None:
                index = self.randomidx[index_]
            else:
                index = index_
            start = self.starts[index]
            end = self.ends[index]
            if end == start:
                end += 1
            return Interval(self.chrs[index], max(0, start - self.flank),
                            end + self.flank,
                            strand=self.strand[index])

        raise IndexError('Cannot interpret index: {}'.format(type(index_)))

    @property
    def binsize(self):
        """binsize of the intervals"""
        return self._binsize

    @binsize.setter
    def binsize(self, value):
        if value is not None and value <= 0:
            raise ValueError('binsize>0 required')
        self._binsize = value

    @property
    def stepsize(self):
        """stepsize (step size)"""
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value is not None and value <= 0:
            raise ValueError('stepsize>0 required')
        self._stepsize = value

    @property
    def flank(self):
        """Flanking bins"""
        return self._flank

    @flank.setter
    def flank(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError('flank>=0 required')
        self._flank = value

    def tostr(self):
        """Returns representing the region."""
        return ['{}:{}-{}'.format(iv.chrom, iv.start, iv.end) for iv in self]

    def idx_by_region(self, include=None, exclude=None, start=None, end=None):

        """idx_by_region filters for chromosome and region ids.

        It takes a list of chromosome ids which should be
        included or excluded, the start and the end of
        a required interval as integers and returns a new GenomicIndexer
        associated with the compatible intervals after filtering.

        Parameters
        ----------
        include : list(str) or None
            List of chromosome names to be included. Default: None means
            all chromosomes are included.
        exclude : list(str)
            List of chromosome names to be excluded. Default: None.
        start : int or None
            The start of the required interval.
        end : int or None
            The end of the required interval.

        Returns
        -------
        idxs
            Containing a list of filtered regions indexes.
        """

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]

        if not include:
            idxs = set(range(len(self)))
        else:
            idxs = set()
            for inc in include:
                idxs.update(np.where(np.asarray(self.chrs) == inc)[0])

        if exclude:
            for exc in exclude:
                idxs = idxs.difference(
                    np.where(np.asarray(self.chrs) == exc)[0])

        if start is not None:
            regionmatch = ((np.array(self.ends) + self.flank) > start)
            indexmatch = np.where(regionmatch)[0]
            idxs = idxs.intersection(indexmatch)

        if end is not None:
            regionmatch = ((np.array(self.starts) - self.flank) < end)
            indexmatch = np.where(regionmatch)[0]
            idxs = idxs.intersection(indexmatch)

        idxs = list(idxs)
        idxs.sort()
        return idxs


    def filter_by_region(self, include=None, exclude=None, start=None, end=None):

        """filter_by_region filters for chromosome and region ids.

        It takes a list of chromosome ids which should be
        included or excluded, the start and the end of
        a required interval as integers and returns a new GenomicIndexer
        associated with the compatible intervals after filtering.

        Parameters
        ----------
        include : list(str) or str
            List of chromosome names to be included. Default: [] means
            all chromosomes are included.
        exclude : list(str) or str
            List of chromosome names to be excluded. Default: [].
        start : int
            The start of the required interval.
        end : int
            The end of the required interval.

        Returns
        -------
        GenomicIndexer
            Containing the filtered regions.
        """

        idxs = self.idx_by_region(include=include, exclude=exclude,
                                  start=start, end=end)

        new_gindexer = GenomicIndexer(self.binsize,
                                      self.stepsize,
                                      self.flank,
                                      zero_padding=self.zero_padding,
                                      collapse=self.collapse,
                                      random_state=self.random_state)
        new_gindexer.chrs = [self.chrs[i] for i in idxs]
        new_gindexer.starts = [self.starts[i] for i in idxs]
        new_gindexer.strand = [self.strand[i] for i in idxs]
        new_gindexer.ends = [self.ends[i] for i in idxs]

        return new_gindexer

    def export_to_bed(self, filename):
        """Export the intervals to bed format.

        Parameters
        ----------
        filename : str
            Bed file name
        """

        with open(filename, 'w') as handle:
            for i in range(len(self)):
                handle.write('{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n'.format(
                    chrom=self.chrs[i],
                    start=max(0, self.starts[i] - self.flank),
                    end=self.ends[i] + self.flank,
                    name='-',
                    score='-',
                    strand=self.strand[i]))


def check_gindexer_compatibility(gindexer, resolution, store_whole_genome):
    """Sanity check for gindexer.

    This function tests if the gindexer is compatible with
    other properties of the dataset, including the resolution and
    the store_whole_genome argument

    A ValueError is thrown if the gindexer is not valid.
    """

    if resolution is not None and resolution > 1 and store_whole_genome:
        # check resolution compatible
        if gindexer is not None and (gindexer.binsize % resolution) > 0:
            raise ValueError(
                'binsize must be an integer-multipe of resolution. '
                'Got binsize={} and resolution={}'.format(gindexer.binsize, resolution))

        for iv_ in gindexer or []:
            if (iv_.start % resolution) > 0:

                raise ValueError(
                    'Please ensure that all interval starts line up '
                    'with the resolution-sized bins. '
                    'This is necessary to prevent rounding issues. '
                    'Interval ({}:{}-{}) not compatible with resolution {}. '.format(
                        iv_.chrom, iv_.start, iv_.end, resolution) +\
                    'Consider using '
                    '"janggu-trim <input_roi> <trun_output> -divisible_by {resolution}"'
                    .format(resolution=resolution))

    if not store_whole_genome and gindexer is None:
        raise ValueError('Either specify roi or store_whole_genome=True')
