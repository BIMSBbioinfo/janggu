import itertools
import os

import numpy as np
import pyBigWig
from data import BwDataset
from genomic_indexer import BwGenomicIndexer
from HTSeq import BAM_Reader
from htseq_extension import BwGenomicArray


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

            ps = [os.path.join(memmap_dir, '{}{}.{}'.format(x[0], x[1],
                                                            suffix))
                  for x in
                  itertools.product(chroms,
                                    ['-', '+'] if stranded
                                    else ['.'])]
            files_exist = [os.path.exists(p) for p in ps]
        else:
            files_exist = [False]

        return files_exist

    @classmethod
    def fromBam(cls, name, bam, regions, genomesize,
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
    def fromBigWig(cls, name, bigwigfiles, regions, genomesize,
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
                # print('Reload BwGenomicArray from {}'.format(memmap_dir))
            else:
                # print('Scoring from {}'.format(sample_file))
                bw = pyBigWig(sample_file)

                for i in range(len(gindexer)):
                    iv = gindexer[i]
                    cover[iv.start_as_pos] += bw.values(iv.chrom,
                                                        int(iv.start),
                                                        int(iv.end))

            covers.append(cover)

        return cls(name, covers, samplenames, gindexer, flank,
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
        for i, idx in enumerate(idxs):
            iv = self.gindexer[idx]
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
