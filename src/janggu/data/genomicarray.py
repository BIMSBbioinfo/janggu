"""Genomic arrays"""

import hashlib
import os
from collections import OrderedDict

import h5py
import numpy as np
from pybedtools import Interval
from scipy import sparse

from janggu.utils import _get_output_data_location
from janggu.utils import _iv_to_str
from janggu.utils import _str_to_iv


def _get_iv_length(length, resolution):
    """obtain the chromosome length for a given resolution."""
    if resolution is None:
        return 1
    return int(np.ceil(float(length)/resolution))


def get_collapser(method):
    """Get collapse method."""

    if method is None:
        return None
    elif  callable(method):
        return method
    elif method == 'mean':
        return lambda x: x.mean(axis=1)
    elif method == 'sum':
        return lambda x: x.sum(axis=1)
    elif method == 'max':
        return lambda x: x.max(axis=1)

    raise ValueError('Unknown method: {}'.format(method))


def create_sha256_cache(data, parameters):
    """Cache file determined from files and parameters."""

    sha256_hash = hashlib.sha256()

    # add file content to hash
    for datum in data or []:
        if isinstance(datum, str) and os.path.exists(datum):
            with open(datum, 'rb') as file_:
                for bblock in iter(lambda: file_.read(1024**2), b""):
                    sha256_hash.update(bblock)
        elif isinstance(datum, np.ndarray):
            sha256_hash.update(datum.tobytes())
        else:
            sha256_hash.update(str(datum).encode('utf-8'))

    # add parameter settings to hash
    sha256_hash.update(str(parameters).encode('utf-8'))

    return sha256_hash.hexdigest()


def _get_cachefile(cachestr, tags, fileending):
    """ Determine cache file location """
    filename = None
    if cachestr is not None:
        memmap_dir = _get_output_data_location(tags)
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)

        filename = str(cachestr) + fileending
        filename = os.path.join(memmap_dir, filename)
        return filename
    return None

def _load_data(cachestr, tags, fileending):
    """ loading data from scratch or from cache """
    filename = _get_cachefile(cachestr, tags, fileending)
    if filename is not None and os.path.exists(filename):
        return False
    return True

class GenomicArray(object):  # pylint: disable=too-many-instance-attributes
    """GenomicArray stores multi-dimensional genomic information.

    It acts as a dataset for holding genomic data. For instance,
    coverage along an entire genome composed of arbitrary length chromosomes
    as well as for multiple cell-types and conditions simultaneously.
    Inspired by the HTSeq analog, the array can hold the data in different
    storage modes, including ndarray, hdf5 or as sparse dataset.

    Parameters
    ----------
    stranded : bool
        Consider stranded profiles. Default: True.
    conditions : list(str) or None
        List of cell-type or condition labels associated with the corresponding
        array dimensions. Default: None means a one-dimensional array is produced.
    typecode : str
        Datatype. Default: 'd'.
    resolution : int or None
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    padding_value : float
        Padding value. Default: 0.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    collapser : None or callable
        Method to aggregate values along a given interval.
    """
    handle = OrderedDict()
    _condition = None
    _resolution = None
    _order = None
    region2index = None

    def __init__(self, stranded=True, conditions=None, typecode='d',
                 resolution=1, padding_value=0.,
                 order=1, store_whole_genome=True, collapser=None):
        self.stranded = stranded
        if conditions is None:
            conditions = ['sample']

        self.condition = conditions
        self.order = order
        self.padding_value = padding_value
        if not isinstance(order, int) or order < 1:
            raise Exception('order must be a positive integer')

        self.resolution = resolution
        self.typecode = typecode
        self._full_genome_stored = store_whole_genome
        self.collapser = collapser

    def _get_indices(self, interval, arraylen):
        """Given the original genomic coordinates,
           the array indices of the reference dataset (garray.handle)
           and the array indices of the view are returned.

        Parameters
        ----------
        interval : Interval
            Interval containing (chr, start, end)
        arraylen : int
            Length of the numpy target array.

        Returns
        -------
        tuple
            Tuple of indices corresponding to the slice in the arrays
            (ref_start, ref_end, array_start, array_end).
            ref_[start,end] indicate the slice in garray.handle
            while array_[start,end] indicates the slice in the
            target array / view.
        """
        chrom = interval.chrom
        start = self.get_iv_start(interval.start)
        end = self.get_iv_end(interval.end) - self.order + 1

        if start >= self.handle[chrom].shape[0]:
            return 0, 0, 0, 0
        else:
            array_start = 0
            ref_start = start

        if end > self.handle[chrom].shape[0]:
            array_end = arraylen - (end - self.handle[chrom].shape[0])
            ref_end = self.handle[chrom].shape[0]
        else:
            array_end = arraylen
            ref_end = end

        return ref_start, ref_end, array_start, array_end

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]
        if isinstance(condition, slice) and value.ndim != 3:
            raise ValueError('Expected 3D array with condition slice.')
        if isinstance(condition, slice):
            condition = slice(None, value.shape[-1], None)

        if self.stranded and value.shape[1] != 2:
            raise ValueError('If genomic array is in stranded mode, shape[-1] == 2 is expected')

        if not self.stranded and value.shape[1] != 1:
            value = value.sum(axis=1, keepdims=True)

        if isinstance(interval, Interval) and isinstance(condition, (int, slice)):
            start = self.get_iv_start(interval.start)
            end = self.get_iv_end(interval.end)

            length = end - start - self.order + 1
            # value should be a 2 dimensional array
            # it will be reshaped to a 2D array where the collapse operation is performed

            # along the second dimension.
            value = self._do_collapse(interval, value)

            try:
                self._setitem(interval, condition, length, value)

            except KeyError:
                # we end up here if the peak regions are not a subset of
                # the regions of interest. that might be the case if
                # peaks from the holdout proportion of the genome are tried
                # to be added.
                # unfortunately, it is also possible that store_whole_genome=False
                # and the peaks and regions of interest are just not synchronized
                # in which case nothing (or too few peaks) are added. in the latter
                # case an error would help actually, but I am not sure how to
                # check if the first or the second is the case here.
                pass
        else:
            raise IndexError("Cannot interpret interval and condition: {}".format((interval, condition)))

    def _setitem(self, interval, condition, length, value):
        if not self._full_genome_stored:
            idx = self.region2index[_iv_to_str(interval.chrom, interval.start,
                                               interval.end)]

            # correcting for the overshooting starts and ends is not necessary
            # for partially loaded data
            self.handle['data'][idx, :length, :, condition] = value

        else:
            ref_start, ref_end, array_start, \
                array_end = self._get_indices(interval, value.shape[0])
            self.handle[interval.chrom][ref_start:ref_end, :, condition] = \
                               value[array_start:array_end]

    def _do_collapse(self, interval, value):
        if self.collapser is not None:

            if self.resolution is None and value.shape[0] == 1 or \
                self.resolution is not None and \
                value.shape[0] == interval.length//self.resolution:
                # collapsing becomes obsolete, because the data has already
                # the expected shape (after collapsing)
                pass
            else:
                if self.resolution is None:
                    # collapse along the entire interval
                    value = value.reshape((1,) + value.shape)
                else:
                    # if the array shape[0] is a multipe of resolution,
                    # it can simply be reshaped. otherwise,
                    # it needs to be resized before.
                    if value.shape[0] % self.resolution > 0:
                        value = np.resize(value, (
                            int(np.ceil(value.shape[0]/float(self.resolution))*self.resolution),) +
                                          value.shape[1:])
                    # collapse in bins of size resolution
                    value = value.reshape((value.shape[0]//min(self.resolution,
                                                               value.shape[0]),
                                           min(self.resolution, value.shape[0]),) + \
                                          value.shape[1:])

                value = self.collapser(value)
        return value

    def __getitem__(self, index):
        # for now lets ignore everything except for chrom, start and end.
        if isinstance(index, Interval):
            interval = index
            chrom = interval.chrom
            start = self.get_iv_start(interval.start)
            end = self.get_iv_end(interval.end)

            # original length
            length = end-start - self.order + 1

            if not self._full_genome_stored:
                idx = self.region2index[_iv_to_str(chrom, interval.start, interval.end)]
                # correcting for the overshooting starts and ends is not necessary
                # for partially loaded data
                return self._reshape(self.handle['data'][idx],
                                     (length, 2 if self.stranded else 1,
                                      len(self.condition)))

            if chrom not in self.handle:
                return np.ones((length, 2 if self.stranded else 1,
                                len(self.condition)),
                               dtype=self.typecode) * self.padding_value

            if start >= 0 and end <= self.handle[chrom].shape[0]:
                end = end - self.order + 1
                # this is a short-cut, which does not require zero-padding
                return self._reshape(self.handle[chrom][start:end],
                                     (end-start, 2 if self.stranded else 1,
                                      len(self.condition)))

            # below is some functionality for zero-padding, in case the region
            # reaches out of the chromosome size

            if self.padding_value == 0.0:
                data = np.zeros((length, 2 if self.stranded else 1,
                                 len(self.condition)),
                                dtype=self.typecode)
            else:
                data = np.ones((length, 2 if self.stranded else 1,
                                len(self.condition)),
                               dtype=self.typecode) * self.padding_value

            ref_start, ref_end, array_start, array_end = self._get_indices(interval, data.shape[0])

            data[array_start:array_end, :, :] = self._reshape(self.handle[chrom][ref_start:ref_end],
                                                              (ref_end - ref_start,
                                                               2 if self.stranded else 1,
                                                               len(self.condition)))
            return data

        raise IndexError("Cannot interpret interval: {}".format(index))

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
    def resolution(self, value):
        if value is not None and value <= 0:
            raise ValueError('resolution>0 required')
        self._resolution = value

    def _reshape(self, data, shape):
        # shape not necessary here,
        # data should just fall through
        return data

    def interval_length(self, chrom):
        """Method returns the interval lengths."""
        # extract the length by the interval length
        # or by the array shape
        locus = _str_to_iv(chrom)
        if len(locus) > 1:
            return locus[2] - locus[1]

        return self.resolution

    def scale_by_region_length(self):
        """ This method scales the regions by the region length ."""
        for chrom in self.handle:
            if self._full_genome_stored:
                self.handle[chrom][:] /= self.interval_length(chrom)
            else:
                for rstr, idx in self.region2index.items():
                    self.handle[chrom][idx] /= self.interval_length(rstr)

    def weighted_mean(self):
        """ Base pair resolution mean weighted by interval length
        """

        # summing the signal
        sums = [self.sum(chrom) for chrom in self.handle]
        sums = np.asarray(sums).sum(axis=0)

        # weights are determined by interval and chromosome length
        weights = [np.prod(self.handle[chrom].shape[:-1]) \
                   for chrom in self.handle]
        weights = np.asarray(weights).sum()
        return sums / weights

    def shift(self, means):
        """Centering the signal by the weighted mean"""
        #means = self.weighted_mean()

        for chrom in self.handle:
            # adjust base pair resoltion mean to interval length
            self.handle[chrom][:] -= means

    def rescale(self, scale):
        """ Method to rescale the signal """
        for chrom in self.handle:
            self.handle[chrom][:] /= scale

    def sum(self, chrom=None):
        """Sum signal across chromosomes."""
        if chrom is not None:
            return self.handle[chrom][:]\
                .sum(axis=tuple(range(self.handle[chrom].ndim - 1)))

        return np.asarray([self.handle[chrom][:]\
            .sum(axis=tuple(range(self.handle[chrom].ndim - 1)))
                           for chrom in self.handle])

    def weighted_sd(self):
        """ Interval scaled standard deviation """

        # summing the squared signal signal
        sums = [np.square(self.handle[chrom][:, :, :]).sum(
            axis=tuple(range(self.handle[chrom].ndim - 1))) \
            for chrom in self.handle]
        sums = np.asarray(sums).sum(axis=0)

        # weights are determined by interval and chromosome length
        weights = [np.prod(self.handle[chrom].shape[:-1]) \
                   for chrom in self.handle]
        weights = np.asarray(weights).sum()
        return np.sqrt(sums / (weights - 1.))

    @property
    def order(self):
        """order"""
        return self._order

    @order.setter
    def order(self, order):
        if order <= 0:
            raise ValueError('order>0 required')
        self._order = order

    def get_iv_end(self, end):
        """obtain the chromosome length for a given resolution."""
        return _get_iv_length(end, self.resolution)

    def get_iv_start(self, start):
        """obtain the chromosome length for a given resolution."""
        if self.resolution is None:
            return 0
        return start // self.resolution

def init_with_padding_value(padding_value, shape, dtype):
    """ create array with given padding value. """
    if padding_value == 0.0:
        return np.zeros(shape, dtype)
    else:
        return np.ones(shape, dtype) * padding_value

class HDF5GenomicArray(GenomicArray):
    """HDF5GenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.

    Parameters
    ----------
    gsize : GenomicIndexer or callable
        GenomicIndexer containing the genome sizes or a callable that
        returns a GenomicIndexer to enable lazy loading.
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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    padding_value : float
        Padding value. Default: 0.
    cache : str or None
        Hash string of the data and parameters to cache the dataset. If None,
        caching is deactivated. Default: None.
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    collapser : None or callable
        Method to aggregate values along a given interval.
    verbose : boolean
        Verbosity. Default: False
    """

    def __init__(self, gsize,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 padding_value=0.,
                 store_whole_genome=True,
                 cache=None,
                 overwrite=False, loader=None,
                 normalizer=None,
                 collapser=None,
                 verbose=False):
        super(HDF5GenomicArray, self).__init__(stranded, conditions, typecode,
                                               resolution,
                                               order=order,
                                               padding_value=padding_value,
                                               store_whole_genome=store_whole_genome,
                                               collapser=collapser)

        if cache is None:
            raise ValueError('cache=True required for HDF format')

        gsize_ = None

        if not store_whole_genome:
            gsize_ = gsize() if callable(gsize) else gsize
            self.region2index = {_iv_to_str(region.chrom,
                                            region.start,
                                            region.end): i \
                                                for i, region in enumerate(gsize_)}

        cachefile = _get_cachefile(cache, datatags, '.h5')
        load_from_file = _load_data(cache, datatags, '.h5')

        if load_from_file:
            if gsize_ is None:
                gsize_ = gsize() if callable(gsize) else gsize

            h5file = h5py.File(cachefile, 'w')

            if store_whole_genome:
                for region in gsize_:
                    shape = (_get_iv_length(region.length - self.order + 1, self.resolution),
                             2 if stranded else 1, len(self.condition))
                    h5file.create_dataset(str(region.chrom), shape,
                                          dtype=self.typecode,
                                          data=init_with_padding_value(padding_value,
                                                                       shape,
                                                                       self.typecode))
                    self.handle = h5file
            else:
                shape = (len(gsize_),
                         _get_iv_length(gsize_.binsize + 2*gsize_.flank - self.order + 1,
                                        self.resolution),
                         2 if stranded else 1, len(self.condition))
                h5file.create_dataset('data', shape,
                                      dtype=self.typecode,
                                      data=init_with_padding_value(padding_value,
                                                                   shape,
                                                                   self.typecode))
                self.handle = h5file
            # invoke the loader
            if loader:
                loader(self)

            for norm in normalizer or []:
                get_normalizer(norm)(self)
            h5file.close()
        if verbose: print('reload {}'.format(cachefile))
        h5file = h5py.File(cachefile, 'a', driver='stdio')

        self.handle = h5file


class NPGenomicArray(GenomicArray):
    """NPGenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.
    Parameters
    ----------
    gsize : GenomicIndexer or callable
        GenomicIndexer containing the genome sizes or a callable that
        returns a GenomicIndexer to enable lazy loading.
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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    padding_value : float
        Padding value. Default: 0.
    cache : str or None
        Hash string of the data and parameters to cache the dataset. If None,
        caching is deactivated. Default: None.
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    normalizer : callable or None
        Normalization to be applied. This argumenet can be None,
        if no normalization is applied, or a callable that takes
        a garray and returns a normalized garray.
        Default: None.
    collapser : None or callable
        Method to aggregate values along a given interval.
    verbose : boolean
        Verbosity. Default: False
    """

    def __init__(self, gsize,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 padding_value=0.0,
                 store_whole_genome=True,
                 cache=None,
                 overwrite=False, loader=None,
                 normalizer=None, collapser=None,
                 verbose=False):

        super(NPGenomicArray, self).__init__(stranded, conditions, typecode,
                                             resolution,
                                             order=order,
                                             padding_value=padding_value,
                                             store_whole_genome=store_whole_genome,
                                             collapser=collapser)

        gsize_ = None

        if not store_whole_genome:

            gsize_ = gsize() if callable(gsize) else gsize

            self.region2index = {_iv_to_str(region.chrom,
                                            region.start,
                                            region.end): i \
                                                for i, region in enumerate(gsize_)}

        cachefile = _get_cachefile(cache, datatags, '.npz')
        load_from_file = _load_data(cache, datatags, '.npz')

        if load_from_file:
            if gsize_ is None:
                gsize_ = gsize() if callable(gsize) else gsize

            if store_whole_genome:
                data = {str(region.chrom): init_with_padding_value(
                    padding_value,
                    shape=(_get_iv_length(region.length - self.order + 1,
                                          self.resolution),
                           2 if stranded else 1,
                           len(self.condition)),
                    dtype=self.typecode) for region in gsize_}
                names = [str(region.chrom) for region in gsize_]
                self.handle = data
            else:
                data = {'data': init_with_padding_value(
                    padding_value,
                    shape=(len(gsize_),
                           _get_iv_length(gsize_.binsize + 2*gsize_.flank - self.order + 1,
                                          self.resolution) if self.resolution is not None else 1,
                           2 if stranded else 1,
                           len(self.condition)),
                    dtype=self.typecode)}
                names = ['data']
                self.handle = data

            # invoke the loader
            if loader:
                loader(self)


            if cachefile is not None:
                np.savez(cachefile, **data)


        if cachefile is not None:
            if verbose: print('reload {}'.format(cachefile))
            data = np.load(cachefile)
            names = [x for x in data]

        # here we get either the freshly loaded data or the reloaded
        # data from np.load.
        self.handle = {key: data[key] for key in names}

        for norm in normalizer or []:
            get_normalizer(norm)(self)


class SparseGenomicArray(GenomicArray):
    """SparseGenomicArray stores multi-dimensional genomic information.

    Implements GenomicArray.

    Parameters
    ----------
    gsize : GenomicIndexer or callable
        GenomicIndexer containing the genome sizes or a callable that
        returns a GenomicIndexer to enable lazy loading.
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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    padding_value : float
        Padding value. Default: 0.
    cache : str or None
        Hash string of the data and parameters to cache the dataset. If None,
        caching is deactivated. Default: None.
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    normalizer : callable or None
        Normalization to be applied. This argumenet can be None,
        if no normalization is applied, or a callable that takes
        a garray and returns a normalized garray.
        Default: None.
    collapser : None or callable
        Method to aggregate values along a given interval.
    verbose : boolean
        Verbosity. Default: False
    """

    def __init__(self, gsize,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 store_whole_genome=True,
                 cache=None,
                 padding_value=0.0,
                 overwrite=False,
                 loader=None,
                 collapser=None,
                 verbose=False):
        super(SparseGenomicArray, self).__init__(stranded, conditions,
                                                 typecode,
                                                 resolution,
                                                 order=order,
                                                 padding_value=padding_value,
                                                 store_whole_genome=store_whole_genome,
                                                 collapser=collapser)

        cachefile = _get_cachefile(cache, datatags, '.npz')
        load_from_file = _load_data(cache, datatags, '.npz')

        gsize_ = None

        if not store_whole_genome:

            gsize_ = gsize() if callable(gsize) else gsize

            self.region2index = {_iv_to_str(region.chrom,
                                            region.start,
                                            region.end): i \
                                                for i, region in enumerate(gsize_)}

        if load_from_file:
            if gsize_ is None:
                gsize_ = gsize() if callable(gsize) else gsize

            if store_whole_genome:
                data = {str(region.chrom): sparse.dok_matrix(
                    (_get_iv_length(region.length - self.order + 1,
                                    resolution),
                     (2 if stranded else 1) * len(self.condition)),
                    dtype=self.typecode)
                        for region in gsize_}
            else:
                data = {'data': sparse.dok_matrix(
                    (len(gsize_),
                     (_get_iv_length(gsize_.binsize + 2*gsize_.flank - self.order + 1,
                                     self.resolution) if self.resolution is not None else 1) *
                     (2 if stranded else 1) * len(self.condition)),
                    dtype=self.typecode)}
            self.handle = data

            # invoke the loader
            if loader:
                loader(self)

            data = self.handle

            data = {chrom: data[chrom].tocoo() for chrom in data}

            storage = {chrom: np.column_stack([data[chrom].data,
                                               data[chrom].row,
                                               data[chrom].col]) \
                                               for chrom in data}
            for region in gsize_:
                if store_whole_genome:
                    storage[region.chrom + '__length__'] = region.length

            names = [name for name in storage]

            if cachefile is not None:
                np.savez(cachefile, **storage)

        if cachefile is not None:
            if verbose: print('reload {}'.format(cachefile))
            storage = np.load(cachefile)

        names = [name for name in storage if '__length__' not in name]

        if store_whole_genome:
            self.handle = {name: sparse.coo_matrix(
                (storage[name][:, 0],
                 (storage[name][:, 1].astype('int'),
                  storage[name][:, 2].astype('int'))),
                shape=(_get_iv_length(storage[str(name)+'__length__'], resolution),
                       (2 if stranded else 1) * len(self.condition))).tocsr()
                           for name in names}
        else:
            # gsize_ is always available for store_whole_genome=False
            self.handle = {name: sparse.coo_matrix(
                (storage[name][:, 0],
                 (storage[name][:, 1].astype('int'),
                  storage[name][:, 2].astype('int'))),
                shape=(len(gsize_),
                       (_get_iv_length(gsize_.binsize + 2*gsize_.flank, resolution)
                        if self.resolution is not None else 1) *
                       (2 if stranded else 1) * len(self.condition))).tocsr()
                           for name in names}

    def _reshape(self, data, shape):
        # what to do with zero padding
        data = data.toarray()

        if self._full_genome_stored:
            return data.reshape(data.shape[0], data.shape[1]//(shape[-1]), shape[-1])
        else:
            return data.reshape(data.shape[1]//(shape[-2]*shape[-1]), shape[-2], shape[-1])

    def _setitem(self, interval, condition, length, value):
        if not self._full_genome_stored:
            regidx = self.region2index[_iv_to_str(interval.chrom, interval.start, interval.end)]
            nconditions = len(self.condition)
            ncondstrand = len(self.condition) * value.shape[-1]
            #end = end - self.order + 1
            idxs = np.where(value > 0)
            for idx in zip(*idxs):
                basepos = idx[0] * ncondstrand
                strand = idx[1] * nconditions
                cond = condition if isinstance(condition, int) else idx[2]
                self.handle['data'][regidx,
                                    basepos + strand + cond] = value[idx]
        else:
            ref_start, ref_end, array_start, _ = self._get_indices(interval, value.shape[0])
            idxs = np.where(value > 0)
            iarray = np.arange(ref_start, ref_end)
            for idx in zip(*idxs):
                cond = condition if isinstance(condition, int) else idx[2]
                self.handle[interval.chrom][iarray[idx[0]],
                                            idx[1] * len(self.condition)
                                            + cond] = value[idx[0] + array_start][idx[1:]]

class PercentileTrimming(object):
    """Percentile trimming normalization.

    This class performs percentile trimming of a GenomicArray to aleviate
    the effect of outliers.
    All values that exceed the value associated with the given percentile
    are set to be equal to the percentile.

    Parameters
    ----------
    percentile : float
        Percentile at which to perform chromosome-level trimming.
    """
    def __init__(self, percentile):
        self.percentile = percentile

    def __call__(self, garray):

        quants = np.percentile(np.concatenate(np.asarray([garray.handle[chrom] for \
                                      chrom in garray.handle]), axis=0),
                               self.percentile, axis=(0, 1))

        for icond, quant in enumerate(quants):
            for chrom in garray.handle:
                arr = garray.handle[chrom][:, :, icond]
                arr[arr > quant] = quant
                garray.handle[chrom][:, :, icond] = arr
        return garray

    def __str__(self):  # pragma: no cover
        return 'PercentileTrimming({})'.format(self.percentile)

    def __repr__(self):  # pragma: no cover
        return str(self)


class RegionLengthNormalization(object):
    """ Normalization for variable-region length.

    This class performs region length normalization of a GenomicArray.
    This is relevant when genomic features are of variable size, e.g.
    enhancer regions of different width or when using variable length genes.

    Parameters
    ----------
    regionmask : str or GenomicIndexer, None
        A bed file or a genomic indexer that contains the masking region
        that is considered for the signal. For instance, when normalizing
        gene expression to TPM, the mask contains exons. Otherwise, the
        TPM would normalize for the full length gene annotation.
        If None, no mask is included.
    """
    def __init__(self, regionmask=None):
        self.regionmask = regionmask

    def __call__(self, garray):
        # length scaling

        garray.scale_by_region_length()

        return garray

    def __str__(self):  # pragma: no cover
        return 'RegionLengthNormalization({})'.format(self.regionmask)

    def __repr__(self):  # pragma: no cover
        return str(self)


class ZScore(object):
    """ZScore normalization.

    This class performs ZScore normalization of a GenomicArray.
    It automatically adjusts for variable interval lenths.

    Parameters
    ----------
    means : float or None
        Provided means will be applied for zero-centering.
        If None, the means will be determined
        from the GenomicArray and then applied.
        Default: None.
    stds : float or None
        Provided standard deviations will be applied for scaling.
        If None, the stds will be determined
        from the GenomicArray and then applied.
        Default: None.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, garray):

        # determine the mean signal per condition
        if self.mean is None:
            self.mean = garray.weighted_mean()

        # centering to zero-mean
        garray.shift(self.mean)

        # determines standard deviation per contition
        if self.std is None:
            self.std = garray.weighted_sd()

        # rescale by standard deviation
        garray.rescale(self.std)

        return garray

    def __str__(self):  # pragma: no cover
        return 'ZScore({},{})'.format(self.mean, self.std)

    def __repr__(self):  # pragma: no cover
        return str(self)


class LogTransform(object):
    """Log transformation of intput signal.

    This class performs log-transformation
    of a GenomicArray using log(x + 1.) to avoid NAN's from zeros.

    """
    def __init__(self):
        pass

    def __call__(self, garray):

        for chrom in garray.handle:
            garray.handle[chrom][:] = np.log(garray.handle[chrom][:] + 1.)
        return garray

    def __str__(self):  # pragma: no cover
        return 'Log'

    def __repr__(self):  # pragma: no cover
        return str(self)


class ZScoreLog(object):
    """ZScore normalization after log transformation.

    This class performs ZScore normalization after log-transformation
    of a GenomicArray using log(x + 1.) to avoid NAN's from zeros.
    It automatically adjusts for variable interval lenths.

    Parameters
    ----------
    means : float or None
        Provided means will be applied for zero-centering.
        If None, the means will be determined
        from the GenomicArray and then applied.
        Default: None.
    stds : float or None
        Provided standard deviations will be applied for scaling.
        If None, the stds will be determined
        from the GenomicArray and then applied.
        Default: None.
    """
    def __init__(self, mean=None, std=None):
        self.logtr = LogTransform()
        self.zscore = ZScore(mean, std)

    def __call__(self, garray):
        return self.zscore(self.logtr(garray))

    def __str__(self):  # pragma: no cover
        return str(self.zscore) + str(self.logtr)

    def __repr__(self):  # pragma: no cover
        return str(self)


def normalize_garray_tpm(garray):
    """This function performs TPM normalization
    for a given GenomicArray.

    """

    # rescale by region lengths in bp
    garray = RegionLengthNormalization()(garray)

    # recale to kb
    garray.rescale(1e-3)

    # compute scaling factor
    scale = garray.sum() # per chromsome sum
    scale = scale.sum(axis=0) # sum across chroms
    scale /= 1e6 # rescale by million

    # divide by scaling factor
    garray.rescale(scale)

    return garray


def get_normalizer(normalizer):
    """ maps built-in normalizers by name and
    returns the respective function """

    if callable(normalizer):
        return normalizer
    elif normalizer == 'zscore':
        return ZScore()
    elif normalizer == 'zscorelog':
        return ZScoreLog()
    elif normalizer == 'binsizenorm':
        return RegionLengthNormalization()
    elif normalizer == 'perctrim':
        return PercentileTrimming(99)
    elif normalizer == 'tpm':
        return normalize_garray_tpm

    raise ValueError('unknown normalizer: {}'.format(normalizer))


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='float32',
                         storage='hdf5', resolution=1,
                         order=1,
                         padding_value=0.0,
                         store_whole_genome=True,
                         datatags=None, cache=None, overwrite=False,
                         loader=None,
                         normalizer=None, collapser=None,
                         verbose=False):
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
        Datatype. Default: 'float32'.
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
    padding_value : float
        Padding value. Default: 0.0
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    cache : str or None
        Hash string of the data and parameters to cache the dataset. If None,
        caching is deactivated. Default: None.
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    normalizer : callable or None
        Normalization to be applied. This argumenet can be None,
        if no normalization is applied, or a callable that takes
        a garray and returns a normalized garray.
        Default: None.
    collapser : str, callable or None
        Collapse method defines how the signal is aggregated for resolution>1 or resolution=None.
        For example, by summing the signal over a given interval.
    verbose : boolean
        Verbosity. Default: False
    """

    # check if collapser available
    if (resolution is None or resolution > 1) and collapser is None:
        raise ValueError('Requiring collapser due to resolution=None or resolution>1')

    # force store_whole_genome=False if resolution=None
    if resolution is None and store_whole_genome:
        print('store_whole_genome=True ignored, because it is not compatible'
              'with resolution=None. store_whole_genome=False is used instead.')
        store_whole_genome = False

    if storage == 'hdf5':
        return HDF5GenomicArray(chroms, stranded=stranded,
                                conditions=conditions,
                                typecode=typecode,
                                datatags=datatags,
                                resolution=resolution,
                                order=order,
                                store_whole_genome=store_whole_genome,
                                cache=cache,
                                padding_value=padding_value,
                                overwrite=overwrite,
                                loader=loader,
                                normalizer=normalizer,
                                collapser=get_collapser(collapser),
                                verbose=verbose)
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded=stranded,
                              conditions=conditions,
                              typecode=typecode,
                              datatags=datatags,
                              resolution=resolution,
                              order=order,
                              store_whole_genome=store_whole_genome,
                              cache=cache,
                              padding_value=padding_value,
                              overwrite=overwrite,
                              loader=loader,
                              normalizer=normalizer,
                              collapser=get_collapser(collapser),
                              verbose=verbose)
    elif storage == 'sparse':
        return SparseGenomicArray(chroms, stranded=stranded,
                                  conditions=conditions,
                                  typecode=typecode,
                                  datatags=datatags,
                                  resolution=resolution,
                                  order=order,
                                  store_whole_genome=store_whole_genome,
                                  cache=cache,
                                  padding_value=padding_value,
                                  overwrite=overwrite,
                                  loader=loader,
                                  collapser=get_collapser(collapser),
                                  verbose=verbose)

    raise Exception("Storage type must be 'hdf5', 'ndarray' or 'sparse'")
