"""Array dataset"""
import copy

import numpy as np

from janggu.data.data import Dataset


class Wrapper(Dataset):
    def __init__(self, data, *args):
        super(Wrapper, self).__init__(data.name)
        self.data = copy.copy(data)
        self.args = args

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data)

    def __len__(self):
        return len(self.data)

    @property
    def conditions(self):
        """conditions"""
        return self.data.conditions if hasattr(self.data, "conditions") else None

    @property
    def shape(self):
        """shape"""
        return self.data.shape

    @property
    def ndim(self):
        return len(self.shape)

    def __copy__(self):
        ar = tuple([copy.copy(e) for e in (self.data,)+self.args])
        obj = self.__class__(*ar)
        return obj

    @property
    def gindexer(self):  # pragma: no cover
        """gindexer"""
        if hasattr(self.data, 'gindexer'):
            return self.data.gindexer
        raise ValueError('No gindexer available.')

    @gindexer.setter
    def gindexer(self, gindexer):  # pragma: no cover
        if hasattr(self.data, 'gindexer'):
            self.data.gindexer = gindexer
            return
        raise ValueError('No gindexer available.')

    @property
    def garray(self):  # pragma: no cover
        """gindexer"""
        if hasattr(self.data, 'garray'):
            return self.data.garray
        raise ValueError('No garray available.')


class Array(Dataset):
    """Array class.

    This datastructure wraps arbitrary numpy.arrays for a
    deep learning application with Janggu.
    The main difference to an ordinary numpy.array is that
    Array has a name attribute.

    Parameters
    -----------
    name : str
        Name of the dataset
    array : :class:`numpy.array`
        Numpy array.
    conditions : list(str) or None
        Conditions or label names of the dataset.
    """

    def __init__(self, name, array, conditions=None):

        if conditions is not None:
            assert array.shape[-1] == len(conditions), \
                "array shape and condition number does not agree: {} != {}" \
                .format(array.shape[-1], len(conditions))

        self.data = copy.copy(array)
        self.conditions = conditions

        Dataset.__init__(self, '{}'.format(name))

    def __repr__(self):  # pragma: no cover
        return 'Array("{}", <np.array>)'.format(self.name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idxs):
        data = self.data[idxs]

        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        if len(self.data.shape) == 1:
            return self.data.shape + (1,)
        return self.data.shape

    @property
    def ndim(self):  # pragma: no cover
        "ndim"
        return len(self.shape)

    def __copy__(self):
        obj = type(self)(self.name, self.data, self.conditions)
        obj.__dict__.update(self.__dict__)
        return obj


class ReduceDim(Wrapper):
    """ReduceDim class.

    This class wraps an 4D coverage object and reduces
    the middle two dimensions by applying the aggregate function.
    Therefore, it transforms the 4D object into a table-like 2D representation

    Example
    -------
    .. code-block:: python

      # given some dataset, e.g. a Cover object
      # originally, the cover object is a 4D-object.
      cover.shape
      cover = ReduceDim(cover, aggregator='mean')
      cover.shape
      # Afterwards, the cover object is 2D, where the second and
      # third dimension have been averaged out.


    Parameters
    -----------
    array : Dataset
        Dataset
    aggregator : str or callable
        Aggregator used for reducing the intermediate dimensions.
        Available aggregators are 'sum', 'mean', 'max' for performing
        summation, averaging or obtaining the maximum value.
        It is also possible to supply a callable directly that performs the
        operation.
        Default: 'sum'
    axis : None or tuple(ints)
        Dimensions over which to perform aggregation. Default: None
        aggregates with :code:`axis=(1, 2)`
    """

    def __init__(self, array, aggregator=None, axis=None):
        super(ReduceDim, self).__init__(array, *(aggregator, axis))

        if aggregator is None:
            aggregator = 'sum'

        def _get_aggregator(name):
            if callable(name):
                return name
            if name == 'sum':
                return np.sum
            elif name == 'mean':
                return np.mean
            elif name == 'max':
                return np.max
            raise ValueError('ReduceDim aggregator="{}" not known.'.format(name) +
                             'Must be "sum", "mean" or "max" or a callable.')
        self.aggregator = _get_aggregator(aggregator)
        self.axis = axis if axis is not None else (1, 2)
        #Dataset.__init__(self, array.name)

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        data = self.aggregator(self.data[idxs], axis=self.axis)
        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        shape = (self.data.shape[0],)
        for idx in range(1, self.data.ndim):
            if idx in self.axis:
                continue
            shape += (self.data.shape[idx],)
        return shape


class SqueezeDim(Wrapper):
    """SqueezeDim class.

    This class wraps an 4D coverage object and reduces
    the middle two dimensions by applying the aggregate function.
    Therefore, it transforms the 4D object into a table-like 2D representation

    Parameters
    -----------
    array : Dataset
        Dataset
    axis : None or tuple(ints)
        Dimensions over which to perform aggregation. Default: None
        aggregates with :code:`axis=(1, 2)`
    """

    def __init__(self, array, axis=None):

        super(SqueezeDim, self).__init__(array, *(axis,))

        self.axis = axis

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        data = np.squeeze(self.data[idxs], axis=self.axis)
        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        shape = tuple()
        for idx in range(self.data.ndim):
            if self.data.shape[idx] == 1:
                if self.axis is None or idx in self.axis:
                    continue
            shape += (self.data.shape[idx],)
        return shape



class Transpose(Wrapper):
    """Transpose class.

    This class can be used to shuffle the dimensions.
    For example, if the channel is expected to be at a specific location.

    Parameters
    -----------
    array : Dataset
        Dataset
    axis : tuple(ints)
        Order to the dimensions.
    """

    def __init__(self, array, axis):
        super(Transpose, self).__init__(array, *(axis,))

        self.axis = axis

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        data = np.transpose(self.data[idxs], self.axis)
        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        shape = tuple()
        for idx in self.axis:
            shape += (self.data.shape[idx],)
        return shape


class NanToNumConverter(Wrapper):
    """NanToNumConverter class.

    This wrapper dataset converts NAN's in the dataset to
    zeros.

    Example
    -------
    .. code-block:: python

      # given some dataset, e.g. a Cover object
      cover
      cover = NanToNumConverter(cover)

      # now all remaining NaNs will be converted to zeros.

    Parameters
    -----------
    array : Dataset
        Dataset
    """

    def __init__(self, array):
        super(NanToNumConverter, self).__init__(array, *())

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        data = np.nan_to_num(self.data[idxs])
        return data


# Wrappers for data augmentation
class RandomSignalScale(Wrapper):
    """RandomSignalScale class.

    This wrapper performs
    performs random uniform scaling of the original input.
    For example, this can be used to randomly change the peak or signal
    heights during training.

    Parameters
    -----------
    array : Dataset
        Dataset object
    deviance : float
        The signal is rescaled using (1 + uniform(-deviance, deviance)) x original signal.
    """

    def __init__(self, array, deviance):
        super(RandomSignalScale, self).__init__(array, *(deviance,))
        self.deviance = deviance

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        data = self.data[idxs]

        scales = np.random.rand(data.shape[0], data.shape[-1])

        scales = 1 - (scales - self.deviance) / (2*self.deviance)

        data = data * scales[:, None, None, :]

        return data


class RandomOrientation(Wrapper):
    """RandomOrientation class.

    This wrapper randomly inverts the directionality of
    the signal tracks.
    For example a signal track is randomely presented in 5' to 3' and 3' to 5'
    orientation. Furthermore, if the dataset is stranded, the strand is switched
    as well.

    Parameters
    -----------
    array : Dataset
        Dataset object must be 4D.
    """

    def __init__(self, array):
        super(RandomOrientation, self).__init__(array, *())

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            data = self.data[slice(idxs, idxs+1)]
        else:
            data = self.data[idxs]

        for i, _ in enumerate(data):
            if np.random.randint(2, size=1):
                data[i] = data[i, ::-1, ::-1, :]

        return data



class RandomShift(Wrapper):
    """Randomshift class.

    This wrapper randomly shifts the input sequence by a random number of
    up to 'shift' bases in either direction. Meant for use with BioSeq.

    This form of data-augmentation has been shown to reduce overfitting
    in a number of settings.

    The sequence is zero-padded in order to remain the same length.

    When 'batchwise' is set to True it will shift all the sequences retrieved
    by a single call to __getitem__ by the same amount (useful for
    computationally efficient batching).

    Parameters
    -----------
    array : Dataset
        Dataset object must be 4D.
    """

    def __init__(self, array, shift, batchwise=False):
        super(RandomShift, self).__init__(array, *(shift, batchwise))
        self.batchwise = batchwise
        self.shift = shift

    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            data = self.data[slice(idxs, idxs+1)]
        else:
            data = self.data[idxs]

        if self.batchwise:
            rshift = np.random.randint(-self.shift, self.shift)
            if rshift < 0:
                data[:, :rshift, :, :] = data[:, -rshift:, :, :]
                data[:, rshift:, :, :] = 0.
            elif rshift > 0:
                data[:, rshift:, :, :] = data[:, :-rshift, :, :]
                data[:, :rshift, :, :] = 0.
        else:
            for i, _ in enumerate(data):
                rshift = np.random.randint(-self.shift, self.shift)
                if rshift < 0:
                    data[i, :rshift, :, :] = data[i, -rshift:, :, :]
                    data[i, rshift:, :, :] = 0.
                elif rshift > 0:
                    data[i, rshift:, :, :] = data[i, :-rshift, :, :]
                    data[i, :rshift, :, :] = 0.
        return data

