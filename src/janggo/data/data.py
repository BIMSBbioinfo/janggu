from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import numpy
from keras.utils import Sequence


class Dataset:
    """Dataset interface.

    The Dataset class mimics a numpy array which allows
    it to be directly supplied to keras.

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


def _data_props(data):
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


class JanggoSequence(Sequence):
    def __init__(self, batch_size, inputs, outputs=None, sample_weights=None,
                 shuffle=False):
        self.inputs = inputs
        self.outputs = outputs
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        for k in inputs:
            xlen = len(inputs[k])
            break

        self.indices = list(range(xlen))
        self.shuffle = shuffle

    def __len__(self):
        return int(numpy.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):

        inputs = {}

        for k in self.inputs:
            inputs[k] = self.inputs[k][
                self.indices[idx*self.batch_size:(idx+1)*self.batch_size]]

        ret = (inputs, )
        if self.outputs is not None:
            outputs = {}
            for k in self.outputs:
                outputs[k] = self.outputs[k][
                    self.indices[idx*self.batch_size:(idx+1)*self.batch_size]]
        else:
            outputs = None

        if self.sample_weights is not None:

            sweight = self.sample_weight[
                self.indices[idx*self.batch_size:(idx+1)*self.batch_size]]
        else:
            sweight = None
        ret += (outputs, sweight)

        print(len(ret))
        return ret

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
