import os
import itertools

import numpy as np
from HTSeq import BAM_Reader
from htseq_extension import BwGenomicArray
from data import BwDataset
from genomeutils.regions import readBed
from pandas import DataFrame


class CoverageBwDataset(BwDataset):

    def __init__(self, name, BwGenomicArray,
                 offsets,  # indices of pointing to region start
                 resolution,  # bp resolution
                 flank=4,  # flanking region to consider
                 stranded=True,  # strandedness to consider
                 cachedir=None):

        if isinstance(cachedir, str):
            self.cachedir = cachedir

        BwDataset.__init__(self, '{}'.format(name))

    @classmethod
    def coverageFromBam(cls, name, bam, regions, genomesize,
                        resolution=50,
                        flank=4, stranded=True, storage='memmap',
                        overwrite=False,
                        cachedir=None):

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
        reglens = ((regions_.end - regions_.start) //
                   resolution + 2*flank).values

        chrs = []
        for i in range(len(reglens)):
            chrs += [i] * reglens[i]


#        offsets = []
#        for ireg in xrange(len(reglens)):
#            offsets += [regions_.start[ireg] + (i - flank)*resolution for i in xrange(reglens[ireg])]

#        offsets = np.asarray([regions_.start - (flank + idx)*resolution
#                              for reglen in reglens for idx in range(reglen)])

        if isinstance(bam, str):
            bam = [bam]

        covers = []
        for sample_file in bam:

            memmap_dir = os.path.join(cachedir, name, os.path.basename(sample_file))
            if not os.path.exists(memmap_dir):
                os.makedirs(memmap_dir)

            ps = [os.path.join(memmap_dir,
                                                '{}{}.nmm'
                                                .format(x[0], x[1]))
                        for x in itertools.product(genomesize.keys(),
                                                   ['-', '+'] if stranded
                                                   else ['.'])]
            print(ps)
            nmms = [os.path.exists(p) for p in ps]


            cover = BwGenomicArray(genomesize, stranded=stranded,
                                   storage=storage, memmap_dir=memmap_dir,
                                   overwrite=overwrite)

            print('nmms={}, overwrite={}'.format(all(nmms), overwrite))
            if all(nmms) and not overwrite:
                print('Reload BwGenomicArray from {}'.format(memmap_dir))
            else:
                print('Counting from {}'.format(sample_file))
                aln_file = BAM_Reader(sample_file)

                for aln in aln_file:
                    if aln.aligned:
                        cover[aln.iv.start_d_as_pos] += 1

            covers.append(cover)

        return cls(name, covers, offsets, resolution, flank,
                   stranded, cachedir)

    @classmethod
    def coverageFromBigWig(cls, name, fastafile, order=1, cachedir=None):
        raise NotImplementedError('coverageFromBigWig')

    def load(self):
        # fill up int8 rep of DNA
        # load dna, region index, and within region index
        pass

    def __repr__(self):
        return 'CoverageBwDataset("{}", <BwGenomicArray>, <offsets>, \
                resolution={}, flank={}, stranded={}, \
                cachedir={})'\
                .format(self.name, self.resolution, self.flank, self.stranded,
                        self.cachedir)

    def getData(self, idxs):
        data = self.extractCoverage(idxs)

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.offsets)

    @property
    def shape(self):
        return (self.nstrands, 2*self.flank + 1, self.nsamples)

    def flank():
        doc = "The flank property."

        def fget(self):
            return self._flank

        def fset(self, value):
            if not isinstance(value, int) or value < 0:
                raise Exception('_flank must be a non-negative integer')
            self._flank = value

        def fdel(self):
            del self._flank
        return locals()

    flank = property(**flank())
