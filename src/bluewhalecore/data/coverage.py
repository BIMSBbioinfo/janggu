import itertools
import os

import numpy as np
import pyBigWig
from data import BwDataset
from genomic_indexer import BwGenomicIndexer
from HTSeq import BAM_Reader
from htseq_extension import BwGenomicArray


class CoverageBwDataset(BwDataset):

    def __init__(self, name, covers,
                 gindexer,  # indices of pointing to region start
                 flank=4,  # flanking region to consider
                 stranded=True,  # strandedness to consider
                 cachedir=None):

        self.covers = covers
        self.gindexer = gindexer
        self.flank = flank
        self.stranded = stranded
        self.cachedir = cachedir

        BwDataset.__init__(self, '{}'.format(name))

    @classmethod
    def fromBam(cls, name, bam, regions, genomesize,
                resolution=50, stride=50,
                flank=4, stranded=True, storage='memmap',
                overwrite=False,
                cachedir=None):

        gindexer = BwGenomicIndexer(regions, resolution, stride)

        if isinstance(bam, str):
            bam = [bam]

        covers = []
        for sample_file in bam:

            if storage == 'memmap':
                memmap_dir = os.path.join(cachedir, name,
                                          os.path.basename(sample_file))
                if not os.path.exists(memmap_dir):
                    os.makedirs(memmap_dir)

                ps = [os.path.join(memmap_dir, '{}{}.nmm'.format(x[0], x[1]))
                      for x in
                      itertools.product(genomesize.keys(),
                                        ['-', '+'] if stranded
                                        else ['.'])]
                nmms = [os.path.exists(p) for p in ps]
            else:
                nmms = [False]
                memmap_dir = ''

            cover = BwGenomicArray(genomesize, stranded=stranded,
                                   storage=storage, memmap_dir=memmap_dir,
                                   overwrite=overwrite)

            if all(nmms) and not overwrite:
                print('Reload BwGenomicArray from {}'.format(memmap_dir))
            else:
                print('Counting from {}'.format(sample_file))
                aln_file = BAM_Reader(sample_file)

                for aln in aln_file:
                    if aln.aligned:
                        cover[aln.iv.start_d_as_pos] += 1

            covers.append(cover)

        return cls(name, covers, gindexer, flank,
                   stranded, cachedir)

    @classmethod
    def fromBigWig(cls, name, bigwigfiles, regions, genomesize,
                   resolution=50, stride=50,
                   flank=4, stranded=True, storage='memmap',
                   overwrite=False,
                   cachedir=None):

        gindexer = BwGenomicIndexer(regions, resolution, stride)

        if isinstance(bigwigfiles, str):
            bigwigfiles = [bigwigfiles]

        covers = []
        for sample_file in bigwigfiles:

            if storage == 'memmap':
                memmap_dir = os.path.join(cachedir, name,
                                          os.path.basename(sample_file))
                if not os.path.exists(memmap_dir):
                    os.makedirs(memmap_dir)

                ps = [os.path.join(memmap_dir, '{}{}.nmm'.format(x[0], x[1]))
                      for x in
                      itertools.product(genomesize.keys(),
                                        ['-', '+'] if stranded
                                        else ['.'])]
                nmms = [os.path.exists(p) for p in ps]
            else:
                nmms = False
                memmap_dir = ''

            # At the moment, we treat the information contained
            # in each bw-file as unstranded
            cover = BwGenomicArray(genomesize, stranded=False,
                                   storage=storage, memmap_dir=memmap_dir,
                                   overwrite=overwrite)

            if all(nmms) and not overwrite:
                print('Reload BwGenomicArray from {}'.format(memmap_dir))
            else:
                print('Scoring from {}'.format(sample_file))
                bw = pyBigWig(sample_file)

                for i in range(len(gindexer)):
                    iv = gindexer[i]
                    cover[iv.start_as_pos] += bw.values(iv.chrom,
                                                        int(iv.start),
                                                        int(iv.end))

            covers.append(cover)

        return cls(name, covers, gindexer, flank,
                   stranded, cachedir)

    def __repr__(self):
        return 'CoverageBwDataset("{}", <BwGenomicArray>, <BwGenomicIndexer>, \
                flank={}, stranded={}, \
                cachedir={})'\
                .format(self.name, self.flank, self.stranded,
                        self.cachedir)

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = [idxs]
        if not isinstance(idxs, list):
            raise IndexError('CoverageBwDataset.__getitem__ '
                             + 'requires "int" or "list"')

        data = np.empty((len(idxs), 2 if self.stranded else 1,
                         1 + 2*self.flank, len(self.covers)))

        sign = ['+',  '-']
        for i in idxs:
            iv = self.gindexer[i]
            for b in range(-self.flank, self.flank + 1):
                try:
                    for s in range(2 if self.stranded else 1):

                        piv = iv.copy()
                        piv.start = iv.start + b * self.gindexer.stride
                        piv.end = iv.end + b * self.gindexer.stride
                        piv.strand = sign[s] if self.stranded else '.'
                        for c in range(len(self.covers)):
                            data[i, s, b+self.flank, c] = \
                                self.covers[c][piv].sum()
                except IndexError:
                    data[i, :, b+self.flank, c] = 0

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        return (len(self), 2 if self.stranded else 1,
                2*self.flank + 1, len(self.covers))

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
