import os

import numpy as np
from HTSeq import GenomicInterval
from HTSeq import BED_Reader
from HTSeq import GFF_Reader


class BwGenomicIndexer(object):
    """BwGenomicIndexer maps genomic positions to the respective indices

    Indexing a BwGenomicIndexer object returns a GenomicInterval
    for the associated index.

    Parameters
    ----------
    regions : str
        Bed- or GFF-filename.
    resolution : int
        Interval size in basepairs.
    stride : int
        Stride (step size) for traversing the genome in basepairs.
    """

    _stride = None
    _resolution = None
    chrs = None
    offsets = None
    inregionidx = None
    strand = None

    @classmethod
    def create_from_file(cls, regions, resolution, stride):
        """Creates a BwGenomicIndexer object.

        This method constructs a BwGenomicIndexer from
        a given BED or GFF file.
        """

        if isinstance(regions, str) and regions.endswith('.bed'):
            regions_ = BED_Reader(regions)
        elif isinstance(regions, str) and (regions.endswith('.gff') or
                                           regions.endswith('.gtf')):
            regions_ = GFF_Reader(regions)
        else:
            raise Exception('Regions must be a bed, gff or gtf-file.')

        gind = cls(resolution, stride)

        chrs = []
        offsets = []
        inregionidx = []
        strand = []
        for reg in regions_:
            reglen = (reg.iv.end - reg.iv.start -
                      resolution + stride) // stride
            chrs += [reg.iv.chrom] * reglen
            offsets += [reg.iv.start] * reglen
            strand += [reg.iv.strand] * reglen
            inregionidx += range(reglen)

        gind.chrs = chrs
        gind.offsets = offsets
        gind.inregionidx = inregionidx
        gind.strand = strand
        return gind
#        return cls(resolution, stride, chrs, offsets, inregionidx, strand)

    def __init__(self, resolution, stride):

        self.resolution = resolution
        self.stride = stride

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
                                   start + self.resolution, self.strand[index])

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
