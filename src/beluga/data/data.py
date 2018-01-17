from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty


class BlgDataset:
    """Beluga Dataset interface.

    The Beluga dataset mimics a numpy array such that it can be
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
