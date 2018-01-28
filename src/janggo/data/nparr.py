from janggo.data.data import Dataset


class NumpyDataset(Dataset):
    """NumpyDataset class.

    This datastructure wraps arbitrary numpy.arrays for a
    deep learning application with Janggo.
    The main difference to an ordinary numpy.array is that
    NumpyDataset has a name attribute.

    Parameters
    -----------
    name : str
        Name of the dataset
    array : :class:`numpy.array`
        Numpy array.
    samplenames : list(str)
        Samplenames (optional). They are relevant if the dataset
        is used to hold labels for a deep learning applications.
        For instance, samplenames might correspond to category names.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.
    """

    def __init__(self, name, array, samplenames=None, cachedir=None):

        self.data = array
        if samplenames:
            if isinstance(samplenames, list):
                self.samplenames = samplenames
            else:
                self.samplenames = [samplenames]

        self.cachedir = cachedir

        Dataset.__init__(self, '{}'.format(name))

    def __repr__(self):  # pragma: no cover
        return 'NumpyDataset("{}", <np.array>)'.format(self.name)

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
        return self.data.shape
