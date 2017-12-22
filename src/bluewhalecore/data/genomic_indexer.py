import os

import numpy as np
from genomeutils.regions import readBed
from HTSeq import GenomicInterval
from pandas import DataFrame


class BwGenomicIndexer:
    """ Maps genomic positions to the respective indices"""

    def __init__(self, regions, resolution, stride):

        self.resolution = resolution
        self.stride = stride

        # fill up int8 rep of DNA
        # load dna, region index, and within region index
        if isinstance(regions, str) and os.path.exists(regions):
            regions_ = readBed(regions)
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
        for i in range(len(reglens)):
            self.chrs += [regions_.chr[i]] * reglens[i]
            self.offsets += [regions_.start[i]] * reglens[i]

        self.inregionidx = []
        for i in range(len(reglens)):
            self.inregionidx += range(reglens[i])

    def __len__(self):
        return len(self.chrs)

    def __repr__(self):
        return "BwGenomicIndexer(<regions>, " \
            + "resolution={}, stride={})".format(self.resolution,
                                                 self.stride)

    def __getitem__(self, index):
        if isinstance(index, int):
            start = self.offsets[index] + \
                    self.inregionidx[index]*self.stride
            return GenomicInterval(self.chrs[index], start,
                                   start + self.resolution, '.')

        raise NotImplementedError('GenomicIndexer indexing with "int" only')

    def idxByChrom(self, include=[], exclude=[]):

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]

        if include == []:
            idxs = set(range(len(self)))
        else:
            idxs = set()
            for inc in include:
                idxs.update(np.where(np.asarray(self.chrs) == inc)[0])

        for exc in exclude:
            idxs = idxs.difference(np.where(np.asarray(self.chrs) == exc)[0])

        return list(idxs)
