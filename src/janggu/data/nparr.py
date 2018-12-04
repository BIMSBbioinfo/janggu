"""Array dataset"""
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

        for transform in self.transformations:
            data = transform(data)

        return data

    @property
    def shape(self):
        """Shape of the dataset"""
        if len(self.data.shape) == 1:
            return self.data.shape + (1,)
        return self.data.shape


    @property
    def ndim(self):
        return len(self.shape)
