import os

import numpy as np
from HTSeq import GenomicInterval
from pandas import DataFrame

from bluewhalecore.data.utils import read_bed


class BwGenomicIndexer(object):
    """BwGenomicIndexer maps genomic positions to the respective indices

    Indexing a BwGenomicIndexer object returns a GenomicInterval
    for the associated index.

    Parameters
    ----------
    regions : str or pandas.DataFrame
        Bed-filename or content of a bed-file as a pandas.DataFrame.
    resolution : int
        Interval size in basepairs.
    stride : int
        Stride (step size) for traversing the genome in basepairs.
    """

    _stride = None
    _resolution = None

    def __init__(self, regions, resolution, stride):

        self.resolution = resolution
        self.stride = stride

        # fill up int8 rep of DNA
        # load dna, region index, and within region index
        if isinstance(regions, str) and os.path.exists(regions):
            regions_ = read_bed(regions)
        elif isinstance(regions, DataFrame):
            regions_ = regions.copy()
        else:
            raise Exception('regions must be a bed-filename \
                            or a pandas.DataFrame \
                            containing the content of a bed-file')

        # Create iregion index
        reglens = ((regions_.end - regions_.start - resolution + stride) //
                   stride).values

        self.chrs = []
        self.offsets = []
        for i, reglen in enumerate(reglens):
            self.chrs += [regions_.chr[i]] * reglen
            self.offsets += [regions_.start[i]] * reglen

        self.inregionidx = []
        for reglen in reglens:
            self.inregionidx += range(reglen)

    def __len__(self):
        return len(self.chrs)

    def __repr__(self):  # pragma: no cover
        return "BwGenomicIndexer(<regions>, " \
            + "resolution={}, stride={})".format(self.resolution,
                                                 self.stride)

    def __getitem__(self, index):
        if isinstance(index, int):
            start = self.offsets[index] + \
                    self.inregionidx[index]*self.stride
            return GenomicInterval(self.chrs[index], start,
                                   start + self.resolution, '.')

        raise IndexError('Index support only for "int". Given {}'.format(
            type(index)))

    @property
    def resolution(self):
        """Resolution of the intervals"""
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if value <= 0:
            raise ValueError('resolution must be positive')
        self._resolution = value

    @property
    def stride(self):
        """Stride (step size)"""
        return self._stride

    @stride.setter
    def stride(self, value):
        if value <= 0:
            raise ValueError('stride must be positive')
        self._stride = value

    def idx_by_chrom(self, include=None, exclude=None):
        """idx_by_chrom filters for chromosome ids.

        It takes a list of chromosome ids which should be included
        or excluded and returns the indices associated with the
        compatible intervals after filtering.

        Parameters
        ----------
        include : list(str)
            List of chromosome names to be included. Default: [] means
            all chromosomes are included.
        exclude : list(str)
            List of chromosome names to be excluded. Default: [].

        Returns
        -------
        list(int)
            List of compatible indices.
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

        return list(idxs)
