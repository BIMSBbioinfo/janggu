import sys
import os
import h5py

import numpy

from HTSeq import GenomicInterval

strand_plus = "+"
strand_minus = "-"
strand_nostrand = "."


class GenomicArray(object):
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, memmap or hdf5.

    Parameters
    ----------
    chroms : dict
        Dictionary with chromosome names as keys and chromosome lengths
        as values.
    stranded : bool
        Consider stranded profiles. Default: True.
    conditions : list(str) or None
        List of cell-type or condition labels associated with the corresponding
        array dimensions. Default: None means a one-dimensional array is produced.
    typecode : str
        Datatype. Default: 'd'.
    storage : str
        Storage type can be 'ndarray', 'memmap' or 'hdf5'.
        The first loads the data into a numpy array directly, while
        the latter two can be used to fetch the data from disk.
    memmap_dir : str
        Directory in which to store the cachefiles. Used only with
        'memmap' and 'hdf5'. Default: "".
    """

        def __init__(self, chroms, stranded=True, conditions=None, typecode='d'):

            self.shape = (chrlen, 2 if stranded else 1, len(conditions))

        def __setitem__(self, interval, value):
            pass

        def __getitem__(self, interval):
            pass


class Hdf5GenomicArray(object):
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, memmap or hdf5.

    Parameters
    ----------
    chroms : dict
     Dictionary with chromosome names as keys and chromosome lengths
     as values.
    stranded : bool
     Consider stranded profiles. Default: True.
    conditions : list(str) or None
     List of cell-type or condition labels associated with the corresponding
     array dimensions. Default: None means a one-dimensional array is produced.
    typecode : str
     Datatype. Default: 'd'.
    storage : str
     Storage type can be 'ndarray', 'memmap' or 'hdf5'.
     The first loads the data into a numpy array directly, while
     the latter two can be used to fetch the data from disk.
    memmap_dir : str
     Directory in which to store the cachefiles. Used only with
     'memmap' and 'hdf5'. Default: "".
    """

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d',
                 memmap_dir="", readonly=True):
        super(Hdf5GenomicArray, self).__init__(chroms, stranded, conditions, typecode)

        self.handle = h5py.File(os.path.join(memmap_dir, "storage.h5"),
                      'r' if readonly else 'r+')

        # one dataset per chrom
        if not readonly:
            for chrom in chroms:
                self.handle.create_dataset(chrom, self.shape, dtype=typecode)

    def __setitem__(self, interval, condition, value):
        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        dset = self.handle.get(chr)
        dset[start:end, 1 if strand=='-' else 0, condition] = value

    def __getitem__(self, index):
        # GenomicInterval
        interval = index

        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        return self.handle.get(chr)[start:end, :, :]


class NDArrayGenomicArray(object):
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, memmap or hdf5.

    Parameters
    ----------
    chroms : dict
     Dictionary with chromosome names as keys and chromosome lengths
     as values.
    stranded : bool
     Consider stranded profiles. Default: True.
    conditions : list(str) or None
     List of cell-type or condition labels associated with the corresponding
     array dimensions. Default: None means a one-dimensional array is produced.
    typecode : str
     Datatype. Default: 'd'.
    storage : str
     Storage type can be 'ndarray', 'memmap' or 'hdf5'.
     The first loads the data into a numpy array directly, while
     the latter two can be used to fetch the data from disk.
    memmap_dir : str
     Directory in which to store the cachefiles. Used only with
     'memmap' and 'hdf5'. Default: "".
    """

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d'):
        super(NDArrayGenomicArray, self).__init__(chroms, stranded, conditions, typecode)

        self.handle = {chrom: np.empty(shape=self.shape, dtype=typecode) for chrom in chroms}

    def __setitem__(self, interval, condition, value):
        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        self.handle[chr][start:end, 1 if strand=='-' else 0, condition] = value

    def __getitem__(self, index):
        # GenomicInterval
        interval = index

        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        return self.handle[chr][start:end, :, :]


class MemmapGenomicArray(object):
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, memmap or hdf5.

    Parameters
    ----------
    chroms : dict
     Dictionary with chromosome names as keys and chromosome lengths
     as values.
    stranded : bool
     Consider stranded profiles. Default: True.
    conditions : list(str) or None
     List of cell-type or condition labels associated with the corresponding
     array dimensions. Default: None means a one-dimensional array is produced.
    typecode : str
     Datatype. Default: 'd'.
    storage : str
     Storage type can be 'ndarray', 'memmap' or 'hdf5'.
     The first loads the data into a numpy array directly, while
     the latter two can be used to fetch the data from disk.
    memmap_dir : str
     Directory in which to store the cachefiles. Used only with
     'memmap' and 'hdf5'. Default: "".
    """

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d',
                 memmap_dir="", readonly=True):
        super(MemmapGenomicArray, self).__init__(chroms, stranded, conditions, typecode)

        self.handle = h5py.File(os.path.join(memmap_dir, "storage.h5"),
                      'r' if readonly else 'r+')

        # one dataset per chrom
        if not readonly:
            for chrom in chroms:
                self.handle.create_dataset(chrom, self.shape, dtype=typecode)

    def __setitem__(self, interval, condition, value):
        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        dset = self.handle.get(chr)
        dset[strand, start:end, condition] = value

    def __getitem__(self, index):
        # GenomicInterval
        interval = index

        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        return self.handle.get(chr)[start:end, :, :]

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d'):
        super(NDArrayGenomicArray, self).__init__(chroms, stranded, conditions, typecode)

        self.handle = {chrom: np.memmap(shape=self.shape, dtype=typecode) for chrom in chroms}

    def __setitem__(self, interval, condition, value):
        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        self.handle[chr][start:end, 1 if strand=='-' else 0, condition] = value

    def __getitem__(self, index):
        # GenomicInterval
        interval = index

        chr = interval.chrom
        start = interval.start
        end = interval.end
        strand = interval.strand
        return self.handle[chr][start:end, :, :]
