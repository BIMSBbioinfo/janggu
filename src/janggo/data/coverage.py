import os

import numpy as np
import pyBigWig
import pysam
from HTSeq import GenomicInterval
from HTSeq import BED_Reader
from HTSeq import GFF_Reader

from janggo.data.data import Dataset
from janggo.data.genomic_indexer import GenomicIndexer
from janggo.data.genomicarray import create_genomic_array
from janggo.utils import get_genome_size_from_bed


class CoverageDataset(Dataset):
    """CoverageDataset class.

    This datastructure holds coverage information across the genome.
    The coverage can conveniently fetched from a list of bam-files,
    bigwig-file, bed-files or gff-files.

    Parameters
    -----------
    name : str
        Name of the dataset
    covers : :class:`BlgGenomicArray`
        A genomic array that holds the coverage data
    gindxer : :class:`GenomicIndexer`
        A genomic index mapper that translates an integer index to a
        genomic coordinate.
    flank : int
        Number of flanking regions to take into account. Default: 4.
    stranded : boolean
        Consider strandedness of coverage. Default: True.
    padding_value : int or float
        Padding value used to pad variable size fragments. Default: 0.
    """

    _flank = None

    def __init__(self, name, covers,
                 gindexer,  # indices of pointing to region start
                 flank,  # flanking region to consider
                 stranded,  # strandedness to consider
                 padding_value,
                 dimmode):  # padding value

        self.covers = covers
        self.gindexer = gindexer
        self.flank = flank
        self.stranded = stranded
        self.padding_value = padding_value
        self.dimmode = dimmode

        Dataset.__init__(self, name)

    @classmethod
    def create_from_bam(cls, name, bamfiles, regions, genomesize=None,
                        samplenames=None,
                        min_mapq=None,
                        binsize=50, stepsize=50,
                        flank=0, storage='hdf5',
                        dtype='int',
                        overwrite=False,
                        cachedir=None):
        """Create a CoverageDataset class from a bam-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bamfiles : str or list
            bam-file or list of bam files.
        regions : str
            Bed-file defining the regions that comprise the dataset.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size
            is fetched from the bam header. Otherwise, the supplied genome
            size is used.
        samplenames : list
            List of samplenames. If `samplenames=None`, the filenames
            are used as samplenames directly.
        min_mapq : int
            Minimal mapping quality. Reads with lower mapping quality are
            discarded. If None, all reads are used.
        binsize : int
            Binsize in basepairs. Default: 50.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 50.
        flank : int
            Flanking size increases the interval size at both ends by
            flank base pairs. Default: 0
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray' or 'hdf5'. Default: 'hdf5'.
        dtype : str
            Typecode to define the datatype to be used for storage.
            Default: 'int'.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        cachedir : str or None
            Directory in which the cachefiles are located. Default: None.
        """

        gindexer = GenomicIndexer.create_from_file(regions, binsize, stepsize)

        if isinstance(bamfiles, str):
            bamfiles = [bamfiles]

        if not samplenames:
            samplenames = bamfiles

        if not min_mapq:
            min_mapq = 0

        if not genomesize:
            header = pysam.AlignmentFile(bamfiles[0], 'r')
            gsize = {}
            for chrom, length in zip(header.references, header.lengths):
                gsize[chrom] = length
        else:
            gsize = genomesize.copy()

        def _bam_loader(cover, files):
            print("load from bam")
            for i, sample_file in enumerate(files):
                print('Counting from {}'.format(sample_file))
                aln_file = pysam.AlignmentFile(sample_file, 'rb')
                for chrom in gsize:

                    array = np.zeros((gsize[chrom], 2), dtype=dtype)

                    try:
                        it_ = aln_file.fetch(chrom)
                    except ValueError:
                        print("Contig '{}' abscent in bam".format(chrom))
                        continue
                    for aln in it_:
                        if aln.mapq < min_mapq:
                            continue

                        if aln.is_reverse:
                            array[aln.reference_end - 1 if aln.reference_end
                                  else aln.reference_start, 1] += 1
                        else:
                            array[aln.reference_start, 0] += 1

                    cover[GenomicInterval(chrom, 0, gsize[chrom],
                                          '+'), i] = array[:, 0]
                    cover[GenomicInterval(chrom, 0, gsize[chrom],
                                          '-'), i] = array[:, 1]

            return cover

        if cachedir:
            memmap_dir = os.path.join(cachedir, name)
        else:
            memmap_dir = None

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        cover = create_genomic_array(gsize, stranded=True,
                                     storage=storage, memmap_dir=memmap_dir,
                                     conditions=samplenames,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     loader=_bam_loader,
                                     loader_args=(bamfiles,))

        return cls(name, cover, gindexer, flank, stranded=True,
                   padding_value=0, dimmode='all')

    @classmethod
    def create_from_bigwig(cls, name, bigwigfiles, regions, genomesize=None,
                           samplenames=None,
                           binsize=200, stepsize=50,
                           resolution=50,
                           flank=0, storage='hdf5',
                           dtype='float32',
                           overwrite=False,
                           dimmode='all',
                           cachedir=None):
        """Create a CoverageDataset class from a bigwig-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bigwigfiles : str or list
            bigwig-file or list of bigwig files.
        regions : str
            Bed-file defining the regions that comprise the dataset.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size is fetched from the regions defined by the bed-file.
            Otherwise, the supplied genome size is used.
        samplenames : list
            List of samplenames. If `samplenames=None`, the filenames
            are used as samplenames directly.
        binsize : int
            Binsize in basepairs. Default: 50.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 50.
        resolution : int
            Resolution in base pairs. This is used to collect the mean signal
            over the window lengths defined by the resolution. Default: 50.
            This value should be chosen to be divisible by binsize and stepsize.
        flank : int
            Flanking size increases the interval size at both ends by
            flank bins. Note that the binsize is defined by the resolution parameter.
            Default: 0.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray' or 'hdf5'. Default: 'hdf5'.
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
        cachedir : str or None
            Directory in which the cachefiles are located. Default: None.
        """

        gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                   stepsize,
                                                   resolution=resolution)

        # automatically determine genomesize from largest region
        if not genomesize:
            gsize = get_genome_size_from_bed(regions, flank*resolution)
        else:
            gsize = genomesize.copy()

        if isinstance(bigwigfiles, str):
            bigwigfiles = [bigwigfiles]

        if not samplenames:
            samplenames = bigwigfiles

        def _bigwig_loader(cover, bigwigfiles, gindexer):
            print("load from bigwig")
            for i, sample_file in enumerate(bigwigfiles):
                bwfile = pyBigWig.open(sample_file)

                for interval in gindexer:
                    cover[interval.start_as_pos, i] = \
                        np.mean(
                            bwfile.values(
                                interval.chrom,
                                int(interval.start*gindexer.resolution),
                                int(interval.end*gindexer.resolution)))
            return cover

        if cachedir:
            memmap_dir = os.path.join(cachedir, name)
        else:
            memmap_dir = None

        for chrom in gsize:
            gsize[chrom] //= resolution

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, memmap_dir=memmap_dir,
                                     conditions=samplenames,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     loader=_bigwig_loader,
                                     loader_args=(bigwigfiles, gindexer))

        return cls(name, cover, gindexer, flank, stranded=False,
                   padding_value=0, dimmode=dimmode)


    @classmethod
    def create_from_bed(cls, name, bedfiles, regions, genomesize=None,
                        samplenames=None,
                        binsize=200, stepsize=50,
                        resolution=50,
                        flank=0, storage='hdf5',
                        dtype='int',
                        dimmode='all',
                        overwrite=False,
                        cachedir=None):
        """Create a CoverageDataset class from a bed-file (or files).

        Parameters
        -----------
        name : str
            Name of the dataset
        bedfiles : str or list
            bed-file or list of bed files.
        regions : str
            Bed-file defining the regions that comprise the dataset.
        genomesize : dict or None
            Dictionary containing the genome size. If `genomesize=None`,
            the genome size is fetched from the regions defined by the bed-file.
            Otherwise, the supplied genome size is used.
        samplenames : list
            List of samplenames. If `samplenames=None`, the filenames
            are used as samplenames directly.
        binsize : int
            Binsize in basepairs. Default: 50.
        stepsize : int
            Stepsize in basepairs. This defines the step size for traversing
            the genome. Default: 50.
        resolution : int
            Resolution in base pairs. This is used to collect the mean signal
            over the window lengths defined by the resolution. Default: 50.
            This value should be chosen to be divisible by binsize and stepsize.
        flank : int
            Flanking size increases the interval size at both ends by
            flank bins. Note that the binsize is defined by the resolution parameter.
            Default: 0.
        storage : str
            Storage mode for storing the coverage data can be
            'ndarray' or 'hdf5'. Default: 'hdf5'.
        dtype : str
            Typecode to define the datatype to be used for storage.
            Default: 'int'.
        dimmode : str
            Dimension mode can be 'first' or 'all'. If 'first', only
            the first element of size resolution is used. Otherwise,
            all elements of size resolution spanning the binsize are returned.
            Default: 'all'.
        overwrite : boolean
            overwrite cachefiles. Default: False.
        cachedir : str or None
            Directory in which the cachefiles are located. Default: None.
        """

        gindexer = GenomicIndexer.create_from_file(regions, binsize,
                                                   stepsize,
                                                   resolution=resolution)

        # automatically determine genomesize from largest region
        if not genomesize:
            gsize = get_genome_size_from_bed(regions, flank*resolution)
        else:
            gsize = genomesize.copy()

        if isinstance(bedfiles, str):
            bedfiles = [bedfiles]

        if not samplenames:
            samplenames = bedfiles

        def _bed_loader(cover, bedfiles, genomesize):
            print("load from bed")
            for i, sample_file in enumerate(bedfiles):
                print(sample_file)
                if isinstance(sample_file, str) and sample_file.endswith('.bed'):
                    regions_ = BED_Reader(sample_file)
                elif isinstance(regions, str) and (sample_file.endswith('.gff') or
                                                   sample_file.endswith('.gtf')):
                    regions_ = GFF_Reader(sample_file)
                else:
                    raise Exception('Regions must be a bed, gff or gtf-file.')

                for region in regions_:
                    region.iv.start //= resolution
                    region.iv.end //= resolution
                    if genomesize[region.iv.chrom] <= region.iv.start:
                        print("Region {} outside of genome size - skipped".format(region.iv))
                    else:
                        # if region score is not defined, take the mere
                        # presence of a range as positive label.
                        cover[region.iv.start_as_pos, i] = \
                        np.dtype(dtype).type(region.score if region.score else 1)
            return cover

        # At the moment, we treat the information contained
        # in each bw-file as unstranded
        if cachedir:
            memmap_dir = os.path.join(cachedir, name)
        else:
            memmap_dir = None

        for chrom in gsize:
            gsize[chrom] //= resolution

        cover = create_genomic_array(gsize, stranded=False,
                                     storage=storage, memmap_dir=memmap_dir,
                                     conditions=samplenames,
                                     overwrite=overwrite,
                                     typecode=dtype,
                                     loader=_bed_loader,
                                     loader_args=(bedfiles, gsize))

        return cls(name, cover, gindexer, flank, stranded=False,
                   padding_value=-1, dimmode=dimmode)

    def __repr__(self):  # pragma: no cover
        return "CoverageDataset('{}', ".format(self.name) \
               + "<GenomicArray>, " \
               + "<GenomicIndexer>, " \
               + "flank={}, stranded={})".format(self.flank, self.stranded)

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
            raise IndexError('CoverageDataset.__getitem__: '
                             + 'index must be iterable')

        data = np.empty((len(idxs),) + self.shape[1:])
        data.fill(self.padding_value)

        for i, idx in enumerate(idxs):
            interval = self.gindexer[idx]

            pinterval = interval.copy()

            pinterval.start = interval.start - self.flank

            if self.dimmode == 'all':
                pinterval.end = interval.end + self.flank
            elif self.dimmode == 'first':
                pinterval.end = pinterval.start + 1

            data[i, :(pinterval.end-pinterval.start), :, :] = \
                    np.asarray(self.covers[pinterval])

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
            blen = (self.gindexer.binsize) // self.gindexer.resolution
            seqlen = 2*self.flank + (blen if blen > 0 else 1)
        elif self.dimmode == 'first':
            seqlen = 1
        return (len(self),
                seqlen, 2 if self.stranded else 1, len(self.covers.condition))

    @property
    def samplenames(self):
        return self.covers.condition

    @property
    def flank(self):
        """Flanking bins"""
        return self._flank

    @flank.setter
    def flank(self, value):
        if not isinstance(value, int) or value < 0:
            raise Exception('_flank must be a non-negative integer')
        self._flank = value
