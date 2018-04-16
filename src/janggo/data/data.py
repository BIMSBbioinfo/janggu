from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty


class Dataset:
    """Janggo Dataset interface.

    The Janggo dataset mimics a numpy array such that it can be
    seamlessly used in conjunction with keras.

    Parameters
    -----------
    name : str
        Name of the dataset

    Attributes
    ----------
    name : str
        Name of the dataset
    shape : tuple
        numpy-style shape of the dataset
    """

    __metaclass__ = ABCMeta

    # list of data augmentation transformations
    transformations = []
    _name = None

    def __init__(self, name):
        self.name = name

    @property
    def name(self):
        """Name of the Dataset"""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise Exception('name must be a string')
        self._name = value

    @abstractmethod
    def __getitem__(self, idxs):  # pragma: no cover
        pass

    def __len__(self):  # pragma: no cover
        pass

    @abstractproperty
    def shape(self):  # pragma: no cover
        """Shape of the dataset"""
        pass

    @property
    def ndim(self):
        """ndim"""
        return len(self.shape)


def input_props(data):
    """Extracts the shape of a provided Input-Dataset.

    Parameters
    ---------
    data : :class:`Dataset` or list(:class:`Dataset`)
        Dataset or list(Dataset).

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corrsponding
        shape as value.
    """
    if isinstance(data, Dataset):
        data = [data]

    if isinstance(data, list):
        dataprops = {}
        for datum in data:
            shape = datum.shape[1:]
            if shape == ():
                shape = (1,)
            dataprops[datum.name] = {'shape': shape}
        return dataprops
    else:
        raise Exception('inputSpace wrong argument: {}'.format(data))


def output_props(data, activation='sigmoid'):
    """Extracts the shape of a provided Output-Dataset.

    Parameters
    ---------
    data : :class:`Dataset` or list(:class:`Dataset`)
        Dataset or list(Dataset).
    activation : str
        Output activation function. Default: 'sigmoid'.

    Returns
    -------
    dict
        Dictionary description of the network output.
    """

    if isinstance(data, Dataset):
        data = [data]

    if isinstance(data, list):
        dataprops = {}
        for datum in data:
            shape = datum.shape[1:]
            if shape == ():
                shape = (1,)
            dataprops[datum.name] = {'shape': shape,
                                     'activation': activation}
        return dataprops
    else:
        raise Exception('outputSpace wrong argument: {}'.format(data))
