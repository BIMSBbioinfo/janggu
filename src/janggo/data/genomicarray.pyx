import sys
import os
import h5py

import numpy
import dask.array as da

from HTSeq import GenomicInterval


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
    handle = None

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d'):
        self.stranded = stranded

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]
        if isinstance(interval, GenomicInterval):
            chrom = interval.chrom
            start = interval.start
            end = interval.end
            strand = interval.strand
            self.handle[chrom][start:end, 1 if self.stranded and strand=='-' else 0, condition] = value
        else:
            raise IndexError("Index must be a GenomicInterval and a condition index")

    def __getitem__(self, index):
        interval = index[0]
        condition = index[1]
        if isinstance(interval, GenomicInterval):
            chrom = interval.chrom
            start = interval.start
            end = interval.end
            strand = interval.strand
            return da.from_array(self.handle[chrom], chunks=1024**2)[start:end, 1 if self.stranded and strand=='-' else 0, condition].compute()
        else:
            raise IndexError("Index must be a GenomicInterval and a condition index")


class HDF5GenomicArray(GenomicArray):
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
                 memmap_dir="", overwrite=False, loader=None, loader_args=None):
        super(HDF5GenomicArray, self).__init__(chroms, stranded,
                                               conditions, typecode)

        if not conditions:
            conditions = ['sample']

        if not os.path.exists(os.path.join(memmap_dir, "storage.h5")) or overwrite:
            os.makedirs(memmap_dir, exist_ok=True)
            self.handle = h5py.File(os.path.join(memmap_dir, "storage.h5"), 'w')

            for chrom in chroms:
                shape = (chroms[chrom], 2 if stranded else 1, len(conditions))
                self.handle.create_dataset(chrom, shape, dtype=typecode, compression='lzf')
            self.handle.attrs['conditions'] = [numpy.string_(x) for x in conditions]

            # invoke the loader
            if loader:
                loader(self, *loader_args)
            self.handle.close()

        self.handle = h5py.File(os.path.join(memmap_dir, "storage.h5"), 'r')

    @property
    def condition(self):
        return self.handle.attrs['conditions']

    @condition.setter
    def condition(self, conditions):
        self.handle.attrs['conditions'] = conditions


class NPGenomicArray(GenomicArray):
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
                 loader=None, loader_args=None):

        super(NPGenomicArray, self).__init__(chroms, stranded,
                                             conditions, typecode)

        if not conditions:
            conditions = ['sample']
        self.conditions = conditions

        self.handle = {chrom: numpy.empty(shape=(chroms[chrom],
                                                 2 if stranded else 1,
                                                 len(conditions)),
                                          dtype=typecode) for chrom in chroms}

        # invoke the loader
        if loader:
            loader(self, *loader_args)


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='d',
                         storage='hdf5', memmap_dir="", overwrite=False,
                         loader=None, loader_args=None):
    if storage == 'hdf5':
        return HDF5GenomicArray(chroms, stranded, conditions, typecode,
                                memmap_dir, overwrite, loader, loader_args)
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded, conditions, typecode,
                              loader, loader_args)
