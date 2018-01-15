import itertools
import os

import numpy as np
import pyBigWig
from HTSeq import BAM_Reader

from bluewhalecore.data.data import BwDataset
from bluewhalecore.data.genomic_indexer import BwGenomicIndexer
from bluewhalecore.data.htseq_extension import BwGenomicArray


class CoverageBwDataset(BwDataset):
    """CoverageBwDataset class.

    This datastructure holds coverage information across the genome.
    The coverage can conveniently fetched from a bam-file, a bigwig-file,
    or a list of files. E.g. a list of bam-files.
    For convenience, the

    Parameters
    -----------
    name : str
        Name of the dataset
    covers : :class:`BwGenomicArray`
        A genomic array that holds the coverage data
    gindxer : :class:`BwGenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Number of flanking regions to take into account. Default: 4.
    stranded : boolean
        Consider strandedness of coverage. Default: True.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.

    Attributes
    -----------
    name : str
        Name of the dataset
    covers : :class:`BwGenomicArray`
        A genomic array that holds the coverage data
    gindxer : :class:`BwGenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Number of flanking regions to take into account. Default: 4.
    stranded : boolean
        Consider strandedness of coverage. Default: True.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.
    """

    _flank = None

    def __init__(self, name, covers,
                 samplenames,
                 gindexer,  # indices of pointing to region start
                 flank=4,  # flanking region to consider
                 stranded=True,  # strandedness to consider
                 cachedir=None):

        self.covers = covers
        self.samplenames = samplenames
        self.gindexer = gindexer
        self.flank = flank
        self.stranded = stranded
        self.cachedir = cachedir

        BwDataset.__init__(self, '{}'.format(name))

    @staticmethod
    def _cacheexists(memmap_dir, chroms, stranded, storage):
        """Returns if the cachefiles exist."""

        if storage == 'memmap' or storage == 'hdf5':
            suffix = 'nmm' if storage == 'memmap' else 'h5'

            paths = [os.path.join(memmap_dir, '{}{}.{}'.format(x[0], x[1],
                                                               suffix))
                     for x in
                     itertools.product(chroms, ['-', '+'] if stranded
                                       else ['.'])]
            files_exist = [os.path.exists(p) for p in paths]
        else:
            files_exist = [False]

        return files_exist

    @classmethod
    def from_bam(cls, name, bam, regions, genomesize,
                 samplenames=None,
                 resolution=50, stride=50,
                 flank=4, stranded=True, storage='hdf5',
                 overwrite=False,
                 cachedir=None):
        """Create a CoverageBwDataset class from a bam-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bam : str or list
            bam-file or list of bam files.
        gindxer : pandas.DataFrame or str
            bed-filename or content of a bed-file
            (in terms of a pandas.DataFrame).
        genomesize : dict
            Dictionary containing the genome size.
        samplenames : list
            List of samplenames. Default: None means that the filenames
            are used as samplenames as well.
        resolution : int
            Resolution in basepairs. Default: 50.
        stride : int
            Stride in basepairs. This defines the step size for traversing
            the genome. Default: 50.
        flank : int
            Adjacent flanking bins to use, where the bin size is determined
            by the resolution. Default: 4.
        stranded : boolean
            Consider strandedness of coverage. Default: True.
        storage : str
            Storage mode for storing the coverage data can be
            'step', 'ndarray', 'memmap' or 'hdf5'. Default: 'hdf5'.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        cachedir : str or None
            Directory in which the cachefiles are located. Default: None.
        """

        gindexer = BwGenomicIndexer(regions, resolution, stride)

        if isinstance(bam, str):
            bam = [bam]

        if not samplenames:
            samplenames = bam

        covers = []
        for sample_file in bam:
            if storage in ['memmap', 'hdf5']:
                memmap_dir = os.path.join(cachedir, name,
                                          os.path.basename(sample_file))
                if not os.path.exists(memmap_dir):
                    os.makedirs(memmap_dir)
            else:
                memmap_dir = ''

            nmms = cls._cacheexists(memmap_dir, genomesize.keys(), stranded,
                                    storage)

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

        return cls(name, covers, samplenames, gindexer, flank,
                   stranded, cachedir)

    @classmethod
    def from_bigwig(cls, name, bigwigfiles, regions, genomesize,
                    samplenames=None,
                    resolution=50, stride=50,
                    flank=4, stranded=True, storage='hdf5',
                    overwrite=False,
                    cachedir=None):
        """Create a CoverageBwDataset class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bam : str or list
            bam-file or list of bam files.
        gindxer : pandas.DataFrame or str
            bed-filename or content of a bed-file
            (in terms of a pandas.DataFrame).
        genomesize : dict
            Dictionary containing the genome size.
        samplenames : list
            List of samplenames. Default: None means that the filenames
            are used as samplenames as well.
        resolution : int
            Resolution in basepairs. Default: 50.
        stride : int
            Stride in basepairs. This defines the step size for traversing
            the genome. Default: 50.
        flank : int
            Adjacent flanking bins to use, where the bin size is determined
            by the resolution. Default: 4.
        stranded : boolean
            Consider strandedness of coverage. Default: True.
        storage : str
            Storage mode for storing the coverage data can be
            'step', 'ndarray', 'memmap' or 'hdf5'. Default: 'hdf5'.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        cachedir : str or None
            Directory in which the cachefiles are located. Default: None.
        """

        gindexer = BwGenomicIndexer(regions, resolution, stride)

        if isinstance(bigwigfiles, str):
            bigwigfiles = [bigwigfiles]

        if not samplenames:
            samplenames = bigwigfiles

        covers = []
        for sample_file in bigwigfiles:
            if storage in ['memmap', 'hdf5']:
                memmap_dir = os.path.join(cachedir, name,
                                          os.path.basename(sample_file))
                if not os.path.exists(memmap_dir):
                    os.makedirs(memmap_dir)
            else:
                memmap_dir = ''

            nmms = cls._cacheexists(memmap_dir, genomesize.keys(), stranded,
                                    storage)

            # At the moment, we treat the information contained
            # in each bw-file as unstranded
            cover = BwGenomicArray(genomesize, stranded=False,
                                   storage=storage, memmap_dir=memmap_dir,
                                   overwrite=overwrite)

            if all(nmms) and not overwrite:
                pass
            else:
                bwfile = pyBigWig.open(sample_file)

                for i in range(len(gindexer)):
                    interval = gindexer[i]
                    cover[interval.start_as_pos] += \
                        bwfile.values(interval.chrom, int(interval.start),
                                      int(interval.end))

            covers.append(cover)

        return cls(name, covers, samplenames, gindexer, flank,
                   stranded, cachedir)

    def __repr__(self):  # pragma: no cover
        return 'CoverageBwDataset("{}", <BwGenomicArray>, <BwGenomicIndexer>, \
                flank={}, stranded={}, \
                cachedir={})'\
                .format(self.name, self.flank, self.stranded,
                        self.cachedir)

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
            raise IndexError('CoverageBwDataset.__getitem__: '
                             + 'index must be iterable')

        data = np.empty((len(idxs), 2 if self.stranded else 1,
                         1 + 2*self.flank, len(self.covers)))

        sign = ['+', '-']
        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]
            for iflank in range(-self.flank, self.flank + 1):
                try:
                    for istrand in range(2 if self.stranded else 1):

                        pinterval = interval.copy()
                        pinterval.start = interval.start + \
                            iflank * self.gindexer.stride
                        pinterval.end = interval.end + \
                            iflank * self.gindexer.stride
                        pinterval.strand = sign[istrand] \
                            if self.stranded else '.'
                        for icov, cover in enumerate(self.covers):
                            data[i, istrand, iflank+self.flank, icov] = \
                                cover[pinterval].sum()
                except IndexError:
                    data[i, :, iflank+self.flank, :] = 0

        for transform in self.transformations:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.gindexer)

    @property
    def shape(self):
        """Shape of the dataset"""
        return (len(self), 2 if self.stranded else 1,
                2*self.flank + 1, len(self.covers))

    @property
    def flank(self):
        """Flanking bins"""
        return self._flank

    @flank.setter
    def flank(self, value):
        if not isinstance(value, int) or value < 0:
            raise Exception('_flank must be a non-negative integer')
        self._flank = value
