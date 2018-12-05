"""Genomic Indexer"""

import numpy as np
from HTSeq import GenomicInterval

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
    chrs = None
    starts = None
    strand = None
    ends = None

    @classmethod
    def create_from_file(cls, regions,  # pylint: disable=too-many-locals
                         binsize, stepsize, flank=0,
                         zero_padding=True, collapse=False):
        """Creates a GenomicIndexer object.

        This method constructs a GenomicIndexer from
        a given BED or GFF file.

        Parameters
        ----------
        regions : str
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
        """

        regions_ = _get_genomic_reader(regions)

        if binsize is None and not collapse:
            binsize_ = None
            # binsize will be inferred from bed file
            for reg in regions_:
                if binsize_ is None:
                    binsize_ = reg.iv.length
                if reg.iv.length != binsize_:
                    raise ValueError('An interval length must be specified if collapse=False.')
            binsize = binsize_

        if stepsize is None:
            if binsize is None:
                stepsize = 1
            else:
                stepsize = binsize

        gind = cls(binsize, stepsize, flank)

        gind.chrs = []
        gind.starts = []
        gind.strand = []
        gind.ends = []

        for reg in regions_:

            tmp_gidx = cls.create_from_region(
                reg.iv.chrom,
                reg.iv.start, reg.iv.end, reg.iv.strand,
                binsize, stepsize, flank, zero_padding)

            gind.chrs += tmp_gidx.chrs
            gind.starts += tmp_gidx.starts
            gind.strand += tmp_gidx.strand
            gind.ends += tmp_gidx.ends

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

        gind = cls(binsize, stepsize, flank)

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

        gind.chrs = chrs
        gind.starts = starts
        gind.strand = strands
        gind.ends = ends
        return gind

    def __init__(self, binsize, stepsize, flank=0):

        self.binsize = binsize
        self.stepsize = stepsize
        self.flank = flank

    def __len__(self):
        return len(self.chrs)

    def __repr__(self):  # pragma: no cover
        return "GenomicIndexer(<regions>, " \
            + "binsize={}, stepsize={}, flank={})".format(self.binsize,
                                                          self.stepsize,
                                                          self.flank)

    def __getitem__(self, index):
        if isinstance(index, int):
            start = self.starts[index]
            end = self.ends[index]
            if end == start:
                end += 1
            return GenomicInterval(self.chrs[index], start - self.flank,
                                   end + self.flank, self.strand[index])

        raise IndexError('Index support only for "int". Given {}'.format(
            type(index)))

    @property
    def binsize(self):
        """binsize of the intervals"""
        return self._binsize

    @binsize.setter
    def binsize(self, value):
        if value is not None and value <= 0:
            raise ValueError('binsize must be positive')
        self._binsize = value

    @property
    def stepsize(self):
        """stepsize (step size)"""
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        if value is not None and value <= 0:
            raise ValueError('stepsize must be positive')
        self._stepsize = value

    @property
    def flank(self):
        """Flanking bins"""
        return self._flank

    @flank.setter
    def flank(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError('_flank must be a non-negative integer')
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
        include : list(str)
            List of chromosome names to be included. Default: [] means
            all chromosomes are included.
        exclude : list(str)
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
        idxs = self.idx_by_region(include=include, exclude=exclude, start=start, end=end)
        #  construct the filtered gindexer
        new_gindexer = GenomicIndexer(self.binsize, self.stepsize, self.flank)
        new_gindexer.chrs = [self.chrs[i] for i in idxs]
        new_gindexer.starts = [self.starts[i] for i in idxs]
        new_gindexer.strand = [self.strand[i] for i in idxs]
        new_gindexer.ends = [self.ends[i] for i in idxs]

        return new_gindexer
