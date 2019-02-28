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

    This class wraps an arbitrary array
    and removes single-dimensional entries from the shape
    using numpy.squeeze.
    It can be used to transform a 4D Cover object to
    a table-like representation.

    Parameters
    -----------
    array : Dataset
        Dataset
    """

    def __init__(self, array):

        self.data = array
        Dataset.__init__(self, array.name)

    def __repr__(self):  # pragma: no cover
        return 'ReduceDim("{}")'.format(self.name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idxs):
        data = np.squeeze(self.data[idxs])
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        shape = tuple(s for s in self.data.shape if s > 1)
        if len(shape) == 1:
            return shape + (1,)
        return shape

    @property
    def ndim(self):
        "ndim"
        return len(self.shape)
