"""Janggu specific dataset class."""

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import numpy
from keras.utils import Sequence


class Dataset:
    """Dataset interface.

    All dataset classes in janggu inherit from
    the Dataset class which mimics a numpy array
    and can be used directly with keras.

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
        """Dataset name"""
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
            dataprops[datum.name] = {'shape': datum.shape[1:]}
        return dataprops
    elif isinstance(data, dict):
        return data

    raise Exception('inputSpace wrong argument: {}'.format(data))


class JangguSequence(Sequence):
    """JangguSequence class.

    This class is a subclass of keras.utils.Sequence.
    It is used to serve the fit_generator, predict_generator
    and evaluate_generator.
    """
    def __init__(self, batch_size, inputs, outputs=None, sample_weights=None,
                 shuffle=False):

        self.inputs = inputs
        self.outputs = outputs
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        for k in inputs:
            xlen = len(inputs[k])
            break

        for k in inputs:
            if not len(inputs[k]) == xlen:
                raise ValueError('Datasets contain differing number of datapoints.')

        for k in outputs or []:
            if not len(outputs[k]) == xlen:
                raise ValueError('Datasets contain differing number of datapoints.')

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

            sweight = self.sample_weights[
                self.indices[idx*self.batch_size:(idx+1)*self.batch_size]]
        else:
            sweight = None
        ret += (outputs, sweight)

        return ret

    def on_epoch_end(self):
        """Stuff to do after epoch end."""
        if self.shuffle:
            numpy.random.shuffle(self.indices)
