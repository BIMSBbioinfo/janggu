"""Genomic arrays"""

import os

import h5py
import numpy
from HTSeq import GenomicInterval
from janggu.utils import _get_output_data_location


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
    handle = dict()
    _condition = None

    def __init__(self, stranded=True, conditions=None, typecode='d'):
        self.stranded = stranded
        if conditions is None:
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
            self.handle[chrom][start:end,
                               1 if self.stranded and strand == '-' else 0,
                               condition] = value
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
        """condition"""
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
    datatags : list(str) or None
        Tags describing the dataset. This is used to store the cache file.
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d',
                 datatags=None, cache=True,
                 overwrite=False, loader=None, loader_args=None):
        super(HDF5GenomicArray, self).__init__(stranded, conditions, typecode)

        if not cache:
            raise ValueError('HDF5 format requires cache=True')

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'storage.h5'

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
    datatags : list(str) or None
        Tags describing the dataset. This is used to store the cache file.
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    def __init__(self, chroms, stranded=True, conditions=None, typecode='d',
                 datatags=None, cache=True,
                 overwrite=False, loader=None, loader_args=None):

        super(NPGenomicArray, self).__init__(stranded, conditions, typecode)

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'storage.npz'
        if cache and not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)

        print(os.path.join(memmap_dir, filename))
        print(os.path.exists(os.path.join(memmap_dir, filename)))
        if cache and not os.path.exists(os.path.join(memmap_dir, filename)) \
            or overwrite or not cache:
            print('create {}'.format(os.path.join(memmap_dir, filename)))
            data = {chrom: numpy.zeros(shape=(chroms[chrom] + 1,
                                              2 if stranded else 1,
                                              len(self.condition)),
                                       dtype=self.typecode) for chrom in chroms}
            self.handle = data

            # invoke the loader
            if loader:
                loader(self, *loader_args)

            condition = [numpy.string_(x) for x in self.condition]
            names = [x for x in data]
            data['conditions'] = condition

            if cache:
                numpy.savez(os.path.join(memmap_dir, filename), **data)

        if cache:
            print('reload {}'.format(os.path.join(memmap_dir, filename)))
            data = numpy.load(os.path.join(memmap_dir, filename))
            names = [x for x in data.files if x != 'conditions']
            condition = data['conditions']

        # here we get either the freshly loaded data or the reloaded
        # data from numpy.load.
        self.handle = {key: data[key] for key in names}

        self.condition = condition


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='int',
                         storage='hdf5', datatags=None, cache=True, overwrite=False,
                         loader=None, loader_args=None):
    """Factory function for creating a GenomicArray."""
    print('datatags',  datatags)
    if storage == 'hdf5':
        return HDF5GenomicArray(chroms, stranded=stranded,
                                conditions=conditions,
                                typecode=typecode,
                                datatags=datatags,
                                cache=cache,
                                overwrite=overwrite,
                                loader=loader,
                                loader_args=loader_args)
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded=stranded,
                              conditions=conditions,
                              typecode=typecode,
                              datatags=datatags,
                              cache=cache,
                              overwrite=overwrite,
                              loader=loader,
                              loader_args=loader_args)
    else:
        raise Exception("Storage type must be 'hdf5' or 'ndarray'")
