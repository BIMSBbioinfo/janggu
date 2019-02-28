"""Array dataset"""
import numpy as np

from janggu.data.data import Dataset


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

        self.data = array
        if conditions is not None and isinstance(conditions, list):
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
    def ndim(self):
        "ndim"
        return len(self.shape)


class ReduceDim(Dataset):
    """ReduceDim class.

    This class wraps an 4D coverage object and reduces
    the middle two dimensions by applying the aggregate function.
    Therefore, it transforms the 4D object into a table-like 2D representation

    Parameters
    -----------
    array : Dataset
        Dataset
    aggregator : str
        Aggregator used for reducing the intermediate dimensions.
        Available aggregators are 'sum', 'mean', 'max' for performing
        summation, averaging or obtaining the maximum value.
        Default: 'sum'
    """

    def __init__(self, array, aggregator=None):

        self.data = copy.copy(array)
        if aggregator is None:
            aggregator = 'sum'

        def _get_aggregator(name):
            if name == 'sum':
                return np.sum
            elif name == 'mean':
                return np.mean
            elif name == 'max':
                return np.max
            else:
                raise ValueError('ReduceDim aggregator="{}" not known. Must be "sum", "mean" or "max".'.format(name))
        self.aggregator = _get_aggregator(aggregator)
        Dataset.__init__(self, array.name)

    def __repr__(self):  # pragma: no cover
        return 'ReduceDim({})'.format(str(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idxs):
        data = self.aggregator(self.data[idxs], axis=(1, 2))
        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        shape = (self.data.shape[0], self.data.shape[-1])
        return shape

    @property
    def ndim(self):
        "ndim"
        return len(self.shape)
