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
    def __getitem__(self, idxs):
        pass

    def __len__(self):
        pass

    @abstractproperty
    def shape(self):
        """Shape of the dataset"""
        pass

    @property
    def ndim(self):
        """ndim"""
        return len(self.shape)


def input_props(bwdata):
    """Extracts the shape of a provided Input-Dataset.

    Parameters
    ---------
    bwdata : :class:`Dataset` or list(:class:`Dataset`)
        Dataset or list(Dataset).

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corrsponding
        shape as value.
    """
    if isinstance(bwdata, Dataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        data = {}
        for bwdatum in bwdata:
            shape = bwdatum.shape[1:]
            if shape == ():
                shape = (1,)
            data[bwdatum.name] = {'shape': shape}
        return data
    else:
        raise Exception('inputSpace wrong argument: {}'.format(bwdata))


def output_props(bwdata, activation='sigmoid'):
    """Extracts the shape of a provided Output-Dataset.

    Parameters
    ---------
    bwdata : :class:`Dataset` or list(:class:`Dataset`)
        Dataset or list(Dataset).
    activation : str
        Output activation function. Default: 'sigmoid'.

    Returns
    -------
    dict
        Dictionary description of the network output.
    """

    if isinstance(bwdata, Dataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        data = {}
        for bwdatum in bwdata:
            shape = bwdatum.shape[1:]
            if shape == ():
                shape = (1,)
            data[bwdatum.name] = {'shape': shape,
                                  'activation': activation}
        return data
    else:
        raise Exception('outputSpace wrong argument: {}'.format(bwdata))
