"""Genomic arrays"""

import os

import h5py
import numpy
from HTSeq import GenomicInterval
from scipy import sparse

from janggu.utils import _get_output_data_location
from janggu.utils import _iv_to_str


class GenomicArray(object):  # pylint: disable=too-many-instance-attributes
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, hdf5 or as sparse dataset.

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
    resolution : int
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    """
    handle = dict()
    _condition = None
    _resolution = None
    _order = None
    _full_genome_stored = True

    def __init__(self, stranded=True, conditions=None, typecode='d',
                 resolution=1, order=1):
        self.stranded = stranded
        if conditions is None:
            conditions = ['sample']

        self.condition = conditions
        self.order = order
        if not isinstance(order, int) or order < 1:
            raise Exception('order must be a positive integer')
        if order > 4:
            raise Exception('order support only up to order=4.')
        self.resolution = resolution
        self.typecode = typecode

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]
        if isinstance(interval, GenomicInterval) and isinstance(condition, int):
            chrom = interval.chrom
            start = interval.start // self.resolution
            end = interval.end // self.resolution
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
            start = interval.start // self.resolution
            end = interval.end // self.resolution

            # original length
            length = end-start

            if not self._full_genome_stored:
                # correcting for the overshooting starts and ends is not necessary
                # for partially loaded data
                return self._reshape(self.handle[_iv_to_str(chrom, interval.start,
                                                            interval.end)][:(length)],  
                                     (length, 2 if self.stranded else 1, len(self.condition)))

            if start >= 0 and end <= self.handle[chrom].shape[0]:
                # this is a short-cut, which does not require zero-padding
                return self._reshape(self.handle[chrom][start:end], (end-start,  2 if self.stranded else 1,
                                 len(self.condition)))

            # below is some functionality for zero-padding, in case the region
            # reaches out of the chromosome size
       
            data = numpy.zeros((length, 2 if self.stranded else 1, len(self.condition)),
                               dtype=self.handle[chrom].dtype)

            dstart = 0
            dend = length
            # if start of interval is negative, due to flank, discard the start
            if start < 0:
                dstart = -start
                start = 0

            # if end of interval reached out of the chromosome, clip it
            if self.handle[chrom].shape[0] < end:
                dend -= end - self.handle[chrom].shape[0]
                end = self.handle[chrom].shape[0]

            # dstart and dend are offset by the number of positions
            # the region reaches out of the chromosome
            data[dstart:dend, :, :] = self._reshape(self.handle[chrom][start:end], 
                                                    (end-start,  2 if self.stranded else 1,
                                                     len(self.condition)))
            return data

        raise IndexError("Index must be a GenomicInterval")

    @property
    def condition(self):
        """condition"""
        return self._condition

    @condition.setter
    def condition(self, conditions):
        self._condition = conditions

    @property
    def resolution(self):
        """resolution"""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if resolution <= 0:
            raise ValueError('resolution must be greater than zero')
        self._resolution = resolution

    def _reshape(self, data, shape):
        return data

    @property
    def order(self):
        """order"""
        return self._order

    @order.setter
    def order(self, order):
        if order <= 0:
            raise ValueError('order must be greater than zero')
        self._order = order


class HDF5GenomicArray(GenomicArray):
    """HDF5GenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.

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
    resolution : int
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1, cache=True,
                 overwrite=False, loader=None, loader_args=None):
        super(HDF5GenomicArray, self).__init__(stranded, conditions, typecode,
                                               resolution,
                                               order)

        if not cache:
            raise ValueError('HDF5 format requires cache=True')

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'storage.h5'

        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        if not os.path.exists(os.path.join(memmap_dir, filename)) or overwrite:
            self.handle = h5py.File(os.path.join(memmap_dir, filename), 'w')

            for chrom in chroms:
                shape = (chroms[chrom] // self.resolution + 1,
                         2 if stranded else 1, len(self.condition))
                self.handle.create_dataset(chrom, shape,
                                           dtype=self.typecode, compression='gzip',
                                           data=numpy.zeros(shape, dtype=self.typecode))

            self.handle.attrs['conditions'] = [numpy.string_(x) for x in self.condition]
            self.handle.attrs['order'] = self.order
            self.handle.attrs['resolution'] = self.resolution

            # invoke the loader
            if loader:
                loader(self, *loader_args)
            self.handle.close()
        print('reload {}'.format(os.path.join(memmap_dir, filename)))
        self.handle = h5py.File(os.path.join(memmap_dir, filename), 'r',
                                driver='stdio')

        self.condition = self.handle.attrs['conditions']
        self.order = self.handle.attrs['order']
        self.resolution = self.handle.attrs['resolution']


class NPGenomicArray(GenomicArray):
    """NPGenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.
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
    resolution : int
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    cache : boolean
        Specifies whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1, cache=True,
                 overwrite=False, loader=None, loader_args=None):

        super(NPGenomicArray, self).__init__(stranded, conditions, typecode,
                                             resolution,
                                             order)

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'storage.npz'
        if cache and not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)

        if cache and not os.path.exists(os.path.join(memmap_dir, filename)) \
                or overwrite or not cache:
            data = {chrom: numpy.zeros(shape=(chroms[chrom] // self.resolution + 1,
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
            data['order'] = order
            data['resolution'] = resolution

            if cache:
                numpy.savez(os.path.join(memmap_dir, filename), **data)

        if cache:
            print('reload {}'.format(os.path.join(memmap_dir, filename)))
            data = numpy.load(os.path.join(memmap_dir, filename))
            names = [x for x in data.files if x not in ['conditions', 'order', 'resolution']]
            condition = data['conditions']
            order = data['order']
            resolution = data['resolution']

        # here we get either the freshly loaded data or the reloaded
        # data from numpy.load.
        self.handle = {key: data[key] for key in names}

        self.condition = condition
        self.resolution = resolution
        self.order = order


class SparseGenomicArray(GenomicArray):
    """SparseGenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.

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
    resolution : int
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1, cache=True,
                 overwrite=False, loader=None, loader_args=None):
        super(SparseGenomicArray, self).__init__(stranded, conditions,
                                                 typecode,
                                                 resolution,
                                                 order)

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'sparse.npz'
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        if cache and not os.path.exists(os.path.join(memmap_dir, filename)) \
            or overwrite or not cache:
            data = {chrom: sparse.dok_matrix((chroms[chrom] // self.resolution + 1,
                                              (2 if stranded else 1) *
                                              len(self.condition)),
                                             dtype=self.typecode)
                    for chrom in chroms}
            self.handle = data

            # invoke the loader
            if loader:
                loader(self, *loader_args)

            data = self.handle

            data = {chrom: data[chrom].tocoo() for chrom in data}

            condition = [numpy.string_(x) for x in self.condition]

            names = [x for x in data]

            storage = {chrom: numpy.column_stack([data[chrom].data,
                                                  data[chrom].row,
                                                  data[chrom].col]) for chrom in data}
            storage.update({'shape.'+chrom: numpy.asarray(data[chrom].shape) for chrom in data})
            storage['conditions'] = condition
            storage['order'] = order
            storage['resolution'] = resolution

            if cache:
                numpy.savez(os.path.join(memmap_dir, filename), **storage)

        if cache:
            print('reload {}'.format(os.path.join(memmap_dir, filename)))
            storage = numpy.load(os.path.join(memmap_dir, filename))

            names = [x for x in storage.files if
                     x not in ['conditions', 'order', 'resolution'] and x[:6] != 'shape.']
            condition = storage['conditions']
            order = storage['order']
            resolution = storage['resolution']

        self.handle = {key: sparse.coo_matrix((storage[key][:, 0],
                                               (storage[key][:, 1].astype('int'),
                                                storage[key][:, 2].astype('int'))),
                                              shape=tuple(storage['shape.' +
                                                                  key])).tocsr()
                       for key in names}

        self.condition = condition
        self.resolution = resolution
        self.order = order

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]
        if isinstance(interval, GenomicInterval) and isinstance(condition, int):
            chrom = interval.chrom
            start = interval.start // self.resolution
            end = interval.end // self.resolution
            strand = interval.strand
            sind = 1 if self.stranded and strand == '-' else 0

            for idx, iarray in enumerate(range(start, end)):
                if hasattr(value, '__len__'):
                    # value is a numpy array or a list
                    val = value[idx]
                else:
                    # value is a scalar value
                    val = value

                if val > 0:
                    self.handle[chrom][iarray,
                                       sind * len(self.condition)
                                       + condition] = val

            return
        raise IndexError("Index must be a GenomicInterval and a condition index")

    def _reshape(self, data, shape):
        return data.toarray().reshape(shape)


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='int',
                         storage='hdf5', resolution=1,
                         order=1,
                         datatags=None, cache=True, overwrite=False,
                         loader=None, loader_args=None):
    """Factory function for creating a GenomicArray.

    This function creates a genomic array for a given storage mode.

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
        Storage type can be 'ndarray', 'hdf5' or 'sparse'.
        Numpy loads the entire dataset into the memory. HDF5 keeps
        the data on disk and loads the mini-batches from disk.
        Sparse maintains sparse matrix representation of the dataset
        in the memory.
        Usage of numpy will require high memory consumption, but allows fast
        slicing operations on the dataset. HDF5 requires low memory consumption,
        but fetching the data from disk might be time consuming.
        sparse will be a good compromise if the data is indeed sparse. In this
        case, memory consumption will be low while slicing will still be fast.
    datatags : list(str) or None
        Tags describing the dataset. This is used to store the cache file.
    resolution : int
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    loader_args : tuple or None
        Arguments for loader.
    """

    if storage == 'hdf5':
        return HDF5GenomicArray(chroms, stranded=stranded,
                                conditions=conditions,
                                typecode=typecode,
                                datatags=datatags,
                                resolution=resolution,
                                order=order,
                                cache=cache,
                                overwrite=overwrite,
                                loader=loader,
                                loader_args=loader_args)
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded=stranded,
                              conditions=conditions,
                              typecode=typecode,
                              datatags=datatags,
                              resolution=resolution,
                              order=order,
                              cache=cache,
                              overwrite=overwrite,
                              loader=loader,
                              loader_args=loader_args)
    elif storage == 'sparse':
        return SparseGenomicArray(chroms, stranded=stranded,
                                  conditions=conditions,
                                  typecode=typecode,
                                  datatags=datatags,
                                  resolution=resolution,
                                  order=order,
                                  cache=cache,
                                  overwrite=overwrite,
                                  loader=loader,
                                  loader_args=loader_args)

    raise Exception("Storage type must be 'hdf5', 'ndarray' or 'sparse'")
