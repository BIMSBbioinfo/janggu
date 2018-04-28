import os

import h5py
import numpy
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
        Storage type can be 'ndarray' or 'hdf5'.
        The first loads the data into a numpy array directly, while
        the latter two can be used to fetch the data from disk.
    memmap_dir : str
        Directory in which to store the cachefiles. Used only with
        'memmap' and 'hdf5'. Default: "".
    """
    handle = None
    _condition = None

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d'):
        self.stranded = stranded
        if not conditions:
            conditions = ['sample']

        self.condition = conditions
        self.typecode = typecode

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]
        if isinstance(interval, GenomicInterval) and isinstance(condition, int):
            chrom = interval.chrom
            start = interval.start
            end = interval.end
            strand = interval.strand
            self.handle[chrom][start:end, 1 if self.stranded and strand == '-' else 0, condition] = value
        else:
            raise IndexError("Index must be a GenomicInterval and a condition index")

    def __getitem__(self, index):
        # for now lets ignore everything except for chrom, start and end.
        if isinstance(index, GenomicInterval):
            interval = index
            chrom = interval.chrom
            start = interval.start
            end = interval.end

            return self.handle[chrom][start:end]
        else:
            raise IndexError("Index must be a GenomicInterval")

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, conditions):
        self._condition = conditions


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

        filename = 'storage.stranded.h5' if stranded else 'storage.unstranded.h5'
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        if not os.path.exists(os.path.join(memmap_dir, filename)) or overwrite:
            print('create {}'.format(os.path.join(memmap_dir, filename)))
            self.handle = h5py.File(os.path.join(memmap_dir, filename), 'w')

            for chrom in chroms:
                shape = (chroms[chrom] + 1, 2 if stranded else 1, len(self.condition))
                self.handle.create_dataset(chrom, shape,
                                           dtype=self.typecode, compression='lzf',
                                           data=numpy.zeros(shape, dtype=self.typecode))

            self.handle.attrs['conditions'] = [numpy.string_(x) for x in self.condition]

            # invoke the loader
            if loader:
                loader(self, *loader_args)
            self.handle.close()
        print('reload {}'.format(os.path.join(memmap_dir, filename)))
        self.handle = h5py.File(os.path.join(memmap_dir, filename), 'r',
                                driver='stdio')

        self.condition = self.handle.attrs['conditions']


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

        self.handle = {chrom: numpy.zeros(shape=(chroms[chrom] + 1,
                                                 2 if stranded else 1,
                                                 len(self.condition)),
                                          dtype=self.typecode) for chrom in chroms}

        # invoke the loader
        if loader:
            loader(self, *loader_args)


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='int',
                         storage='hdf5', memmap_dir="", overwrite=False,
                         loader=None, loader_args=None):
    if storage == 'hdf5':
        return HDF5GenomicArray(chroms, stranded, conditions, typecode,
                                memmap_dir, overwrite, loader, loader_args)
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded, conditions, typecode,
                              loader, loader_args)
    else:
        raise Exception("Storage type must be 'hdf5' or 'ndarray'")
