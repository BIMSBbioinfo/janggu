"""Genomic arrays"""

import os

import h5py
import numpy as np
from HTSeq import GenomicInterval
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
    elif isinstance(method, str):
        if method == 'mean':
            return lambda x: x.mean(axis=1)
        elif method == 'sum':
            return lambda x: x.sum(axis=1)
        elif method == 'max':
            return lambda x: x.max(axis=1)
    elif callable(method):
        return method

    raise ValueError('Unknown method: {}'.format(method))

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
    resolution : int or None
        Resolution for storing the genomic array. Only relevant for the use
        with Cover Datasets. Default: 1.
    order : int
        Order of the alphabet size. Only relevant for Bioseq Datasets. Default: 1.
    collapser : None or callable
        Method to aggregate values along a given interval.
    """
    handle = dict()
    _condition = None
    _resolution = None
    _order = None

    def __init__(self, stranded=True, conditions=None, typecode='d',
                 resolution=1, order=1, store_whole_genome=True, collapser=None):
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
        self._full_genome_stored = store_whole_genome
        self.collapser = collapser

    def __setitem__(self, index, value):
        interval = index[0]
        condition = index[1]

        if self.stranded and value.shape[-1] != 2:
            raise ValueError('If genomic array is in stranded mode, shape[-1] == 2 is expected')

        if not self.stranded and value.shape[-1] != 1:
            value = value.sum(axis=1).reshape(-1, 1)

        if isinstance(interval, GenomicInterval) and isinstance(condition, int):
            chrom = interval.chrom
            start = self.get_iv_start(interval.start)
            end = self.get_iv_end(interval.end)

            # value should be a 2 dimensional array
            # it will be reshaped to a 2D array where the collapse operation is performed
            # along the second dimension.
            if self.collapser is not None:
                if self.resolution is None:
                    # collapse along the entire interval
                    value = value.reshape((1, len(value), value.shape[-1]))
                else:
                    # collapse in bins of size resolution
                    value = value.reshape((len(value)//self.resolution,
                                           self.resolution, value.shape[-1]))

                value = self.collapser(value)

            try:
                if not self._full_genome_stored:
                    length = end-start
                    # correcting for the overshooting starts and ends is not necessary
                    # for partially loaded data

                    self.handle[_iv_to_str(chrom, interval.start,
                                           interval.end)][:(length), :, condition] = value

                else:
                    if start < 0:
                        tmp_start = -start
                        ref_start = 0
                    else:
                        tmp_start = 0
                        ref_start = start

                    if end > self.handle[chrom].shape[0]:
                        tmp_end = value.shape[0] - (end - self.handle[chrom].shape[0])
                        ref_end = self.handle[chrom].shape[0]
                    else:
                        tmp_end = value.shape[0]
                        ref_end = end

                    #start_offset = max(start, 0)
                    #end_offset = min(end, self.handle[chrom].shape[0])
                    #dstart = start_offset - start
                    #dend = end_offset - end
                    #cend = end + (dend)
                    #if dend < 0:
                    self.handle[chrom][ref_start:ref_end, :, condition] = \
                                       value[tmp_start:tmp_end, :]

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
            raise IndexError("Index must be a GenomicInterval and a condition index")

    def __getitem__(self, index):
        # for now lets ignore everything except for chrom, start and end.
        if isinstance(index, GenomicInterval):
            interval = index
            chrom = interval.chrom
            start = self.get_iv_start(interval.start)
            end = self.get_iv_end(interval.end)

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
                return self._reshape(self.handle[chrom][start:end],
                                     (end-start, 2 if self.stranded else 1,
                                      len(self.condition)))

            # below is some functionality for zero-padding, in case the region
            # reaches out of the chromosome size

            data = np.zeros((length, 2 if self.stranded else 1,
                             len(self.condition)),
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
                                                    (end-start,
                                                     2 if self.stranded else 1,
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
    def resolution(self, value):
        if value is not None and value <= 0:
            raise ValueError('resolution must be greater than zero')
        self._resolution = value

    def _reshape(self, data, shape):
        # shape not necessary here,
        # data should just fall through
        return data

    def _interval_length(self, chrom):
        # extract the length by the interval length
        # or by the array shape
        locus = _str_to_iv(chrom)
        if len(locus) > 1:
            return locus[2] - locus[1]

        return self.resolution

    def scale_by_region_length(self):
        """ This method scales the regions by the region length ."""
        for chrom in self.handle:
            self.handle[chrom][:] /= self._interval_length(chrom)

    def weighted_mean(self):
        """ Base pair resolution mean weighted by interval length
        """

        # summing the signal
        sums = [self.sum(chrom) * self._interval_length(chrom) \
                for chrom in self.handle]
        sums = np.asarray(sums).sum(axis=0)

        # weights are determined by interval and chromosome length
        weights = [np.prod(self.handle[chrom].shape[:-1]) * \
                   self._interval_length(chrom) \
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
            axis=tuple(range(self.handle[chrom].ndim - 1))) * \
            self._interval_length(chrom) \
            for chrom in self.handle]
        sums = np.asarray(sums).sum(axis=0)

        # weights are determined by interval and chromosome length
        weights = [np.prod(self.handle[chrom].shape[:-1]) * \
                   self._interval_length(chrom) \
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
            raise ValueError('order must be greater than zero')
        self._order = order

    def get_iv_end(self, end):
        """obtain the chromosome length for a given resolution."""
        return _get_iv_length(end, self.resolution)

    def get_iv_start(self, start):
        """obtain the chromosome length for a given resolution."""
        if self.resolution is None:
            return 0
        return start // self.resolution


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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    cache : boolean
        Whether to cache the dataset. Default: True
    overwrite : boolean
        Whether to overwrite the cache. Default: False
    loader : callable or None
        Function to be called for loading the genomic array.
    collapser : None or callable
        Method to aggregate values along a given interval.
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 store_whole_genome=True,
                 cache=True,
                 overwrite=False, loader=None,
                 normalizer=None,
                 collapser=None):
        super(HDF5GenomicArray, self).__init__(stranded, conditions, typecode,
                                               resolution,
                                               order, store_whole_genome, collapser)

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
                shape = (_get_iv_length(chroms[chrom], self.resolution),
                         2 if stranded else 1, len(self.condition))
                self.handle.create_dataset(chrom, shape,
                                           dtype=self.typecode, compression='gzip',
                                           data=np.zeros(shape, dtype=self.typecode))

            self.handle.attrs['conditions'] = [np.string_(x) for x in self.condition]
            self.handle.attrs['order'] = self.order
            self.handle.attrs['resolution'] = resolution if resolution is not None else 0

            # invoke the loader
            if loader:
                loader(self)

            if normalizer:
                normalizer(self)

            self.handle.close()
        print('reload {}'.format(os.path.join(memmap_dir, filename)))
        self.handle = h5py.File(os.path.join(memmap_dir, filename), 'r',
                                driver='stdio')

        self.condition = self.handle.attrs['conditions']
        self.order = self.handle.attrs['order']
        self.resolution = self.handle.attrs['resolution'] \
            if self.handle.attrs['resolution'] > 0 else None

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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    cache : boolean
        Specifies whether to cache the dataset. Default: True
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
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 store_whole_genome=True,
                 cache=True,
                 overwrite=False, loader=None,
                 normalizer=None, collapser=None):

        super(NPGenomicArray, self).__init__(stranded, conditions, typecode,
                                             resolution,
                                             order, store_whole_genome, collapser)

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'storage.npz'
        if cache and not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)

        if cache and not os.path.exists(os.path.join(memmap_dir, filename)) \
                or overwrite or not cache:
            data = {chrom: np.zeros(shape=(_get_iv_length(chroms[chrom],
                                                          self.resolution),
                                           2 if stranded else 1,
                                           len(self.condition)),
                                    dtype=self.typecode) for chrom in chroms}
            self.handle = data

            # invoke the loader
            if loader:
                loader(self)

            if normalizer:
                normalizer(self)

            condition = [np.string_(x) for x in self.condition]
            names = [x for x in data]
            data['conditions'] = condition
            data['order'] = order
            data['resolution'] = resolution if resolution is not None else 0

            if cache:
                np.savez(os.path.join(memmap_dir, filename), **data)

        if cache:
            print('reload {}'.format(os.path.join(memmap_dir, filename)))
            data = np.load(os.path.join(memmap_dir, filename))
            names = [x for x in data.files if x not in ['conditions', 'order', 'resolution']]
            condition = data['conditions']
            order = data['order']
            resolution = data['resolution'] if data['resolution'] > 0 else None

        # here we get either the freshly loaded data or the reloaded
        # data from np.load.
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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    cache : boolean
        Whether to cache the dataset. Default: True
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
    """

    def __init__(self, chroms,  # pylint: disable=too-many-locals
                 stranded=True,
                 conditions=None,
                 typecode='d',
                 datatags=None,
                 resolution=1,
                 order=1,
                 store_whole_genome=True,
                 cache=True,
                 overwrite=False,
                 loader=None,
                 collapser=None):
        super(SparseGenomicArray, self).__init__(stranded, conditions,
                                                 typecode,
                                                 resolution,
                                                 order, store_whole_genome, collapser)

        if stranded:
            datatags = datatags + ['stranded'] if datatags else ['stranded']

        memmap_dir = _get_output_data_location(datatags)

        filename = 'sparse.npz'
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        if cache and not os.path.exists(os.path.join(memmap_dir, filename)) \
            or overwrite or not cache:
            data = {chrom: sparse.dok_matrix((_get_iv_length(chroms[chrom],
                                                             self.resolution),
                                              (2 if stranded else 1) *
                                              len(self.condition)),
                                             dtype=self.typecode)
                    for chrom in chroms}
            self.handle = data

            # invoke the loader
            if loader:
                loader(self)

            data = self.handle

            data = {chrom: data[chrom].tocoo() for chrom in data}

            condition = [np.string_(x) for x in self.condition]

            names = [x for x in data]

            storage = {chrom: np.column_stack([data[chrom].data,
                                               data[chrom].row,
                                               data[chrom].col]) \
                                               for chrom in data}
            storage.update({'shape.'+chrom: \
                np.asarray(data[chrom].shape) for chrom in data})
            storage['conditions'] = condition
            storage['order'] = order
            storage['resolution'] = resolution if resolution is not None else 0

            if cache:
                np.savez(os.path.join(memmap_dir, filename), **storage)

        if cache:
            print('reload {}'.format(os.path.join(memmap_dir, filename)))
            storage = np.load(os.path.join(memmap_dir, filename))

            names = [x for x in storage.files if
                     x not in ['conditions', 'order', 'resolution'] and x[:6] != 'shape.']
            condition = storage['conditions']
            order = storage['order']
            resolution = storage['resolution'] if storage['resolution'] > 0 else None

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
            start = self.get_iv_start(interval.start)
            end = self.get_iv_end(interval.end)
            #strand = interval.strand
            #sind = 1 if self.stranded and strand == '-' else 0

            if self.stranded and value.shape[-1] != 2:
                raise ValueError('If genomic array is in stranded mode, shape[-1] == 2 is expected')

            if not self.stranded and value.shape[-1] != 1:
                value = value.sum(axis=1).reshape(-1, 1)

            # value should be a 2 dimensional array
            # it will be reshaped to a 2D array where the collapse operation is performed
            # along the second dimension.
            if self.collapser is not None:
                if self.resolution is None:
                    # collapse along the entire interval
                    value = value.reshape((1, len(value), value.shape[-1]))
                else:
                    # collapse in bins of size resolution
                    value = value.reshape((len(value)//self.resolution,
                                           self.resolution, value.shape[-1]))

                value = self.collapser(value)

            try:
                for sind in range(value.shape[-1]):
                    if not self._full_genome_stored:
                        for idx, iarray in enumerate(range(start, end)):
                            val = value[idx, sind]

                            if val > 0:

                                self.handle[
                                    _iv_to_str(chrom, interval.start,
                                               interval.end)][
                                                   idx, sind * len(self.condition)
                                                   + condition] = val
                    else:
                        if start < 0:
                            tmp_start = -start
                            ref_start = 0
                        else:
                            tmp_start = 0
                            ref_start = start

                        if end > self.handle[chrom].shape[0]:
                            tmp_end = value.shape[0] - (end - self.handle[chrom].shape[0])
                            ref_end = self.handle[chrom].shape[0]
                        else:
                            tmp_end = value.shape[0]
                            ref_end = end

                        for idx, iarray in enumerate(range(ref_start, ref_end)):
                            val = value[idx + tmp_start, sind]
                            if val > 0:
                                self.handle[chrom][iarray,
                                                   sind * len(self.condition)
                                                   + condition] = val

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
            return
        raise IndexError("Index must be a GenomicInterval and a condition index")

    def _reshape(self, data, shape):
        return data.toarray().reshape(shape)

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
        # length scaling

        garray.scale_by_region_length()

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


class ZScoreLog(ZScore):
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
    def __init__(self, means=None, stds=None):
        super(ZScoreLog, self).__init__(means, stds)

    def __call__(self, garray):

        # overall mean
        # first log transform
        for chrom in garray.handle:
            garray.handle[chrom][:] = np.log(garray.handle[chrom][:] + 1.)

        return super(ZScoreLog, self).__call__(garray)


def normalize_garray_tpm(garray):
    """This function performs TPM normalization
    for a given GenomicArray.

    """

    # rescale by region lengths in bp
    garray.scale_by_region_length()

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
    if normalizer is None:
        return normalizer
    elif isinstance(normalizer, str):
        if normalizer == 'zscore':
            return ZScore()
        elif normalizer == 'zscorelog':
            return ZScoreLog()
        elif normalizer == 'tpm':
            return normalize_garray_tpm
    elif callable(normalizer):
        return normalizer
    raise ValueError('unknown normalizer: {}'.format(normalizer))


def create_genomic_array(chroms, stranded=True, conditions=None, typecode='float32',
                         storage='hdf5', resolution=1,
                         order=1,
                         store_whole_genome=True,
                         datatags=None, cache=True, overwrite=False,
                         loader=None,
                         normalizer=None, collapser=None):
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
    store_whole_genome : boolean
        Whether to store the entire genome or only the regions of interest.
        Default: True
    cache : boolean
        Whether to cache the dataset. Default: True
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
    """

    # check if collapser available
    if (resolution is None or resolution > 1) and collapser is None:
        raise ValueError('A collapse method must be specified when'
                         'resolution is None or greater than one, but'
                         'collapser=None.')

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
                                overwrite=overwrite,
                                loader=loader,
                                normalizer=get_normalizer(normalizer),
                                collapser=get_collapser(collapser))
    elif storage == 'ndarray':
        return NPGenomicArray(chroms, stranded=stranded,
                              conditions=conditions,
                              typecode=typecode,
                              datatags=datatags,
                              resolution=resolution,
                              order=order,
                              store_whole_genome=store_whole_genome,
                              cache=cache,
                              overwrite=overwrite,
                              loader=loader,
                              normalizer=get_normalizer(normalizer),
                              collapser=get_collapser(collapser))
    elif storage == 'sparse':
        if normalizer is not None:
            print("Dataset normalization is not supported "
                  "for sparse genomic data yet. Argument ignored.")
        return SparseGenomicArray(chroms, stranded=stranded,
                                  conditions=conditions,
                                  typecode=typecode,
                                  datatags=datatags,
                                  resolution=resolution,
                                  order=order,
                                  store_whole_genome=store_whole_genome,
                                  cache=cache,
                                  overwrite=overwrite,
                                  loader=loader,
                                  collapser=get_collapser(collapser))

    raise Exception("Storage type must be 'hdf5', 'ndarray' or 'sparse'")
