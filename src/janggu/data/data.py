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
    It can also be used with the new keras interface (>=2.4.3).

    Parameters
    ---------
    inputs : :class:`Dataset`, list(:class:`Dataset`) or array-like
        Dataset or list(Dataset).
    outputs :  :class:`Dataset`, list(:class:`Dataset`), array-like or None
        Dataset or list(Dataset).
    sample_weights : array-like or None
        Array or list of sample weights.
    batch_size : int
        Batch size. Default: 32
    shuffle : boolean
        Whether to shuffle the data. Default: False
    as_dict : boolean
        Whether to return mini-batches as dict or unnamed tuples.
        In the latter case, the order of the input arguments
        reflects the order to the mini-batch tuples. Default: True
    """
    def __init__(self, inputs, outputs=None, sample_weights=None,
                 batch_size=32,
                 shuffle=False, as_dict=True):

        def _todict(x):
            if not isinstance(x, dict) and x is not None:
                if not isinstance(x, (list, tuple)):
                    x = [x]
                x = {ip.name: ip for ip in x}
            return x

        def _tolist(x):
            if isinstance(x, dict):
                raise ValueError('dict-like inputs/output not accepted with option as_dict=False')
            if not isinstance(x, list) and x is not None:
                x = [x]
            return x

        self.as_dict = as_dict
        inputs = _todict(inputs) if as_dict else _tolist(inputs)
        outputs = _todict(outputs) if as_dict else _tolist(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        for k in inputs:
            if isinstance(inputs, dict):
                xlen = len(inputs[k])
            else:
                # seems to be a list, so extract len of first elem
                xlen = len(k)
            break

        for k in inputs:
            if isinstance(inputs, dict):
                elem = inputs[k]
            else:
                # seems to be a list, so extract len of first elem
                elem = k
            if not len(elem) == xlen:
                raise ValueError('Datasets contain differing number of datapoints.')

        for k in outputs or []:
            if isinstance(inputs, dict):
                elem = outputs[k]
            else:
                # seems to be a list, so extract len of first elem
                elem = k
            if not len(elem) == xlen:
                raise ValueError('Datasets contain differing number of datapoints.')

        self.indices = list(range(xlen))
        self.shuffle = shuffle

    def __len__(self):
        return int(numpy.ceil(len(self.indices) / float(self.batch_size)))

    def _getitemlist(self, idx):

        inputs = []

        for inp in self.inputs:
            inputs.append(inp[
                self.indices[idx*self.batch_size:(idx+1)*self.batch_size]])

        ret = (inputs, )
        if self.outputs is not None:
            outputs = []
            for oup in self.outputs:
                outputs.append(oup[
                    self.indices[idx*self.batch_size:(idx+1)*self.batch_size]])
        else:
            outputs = None

        if self.sample_weights is not None:

            sweight = self.sample_weights[
                self.indices[idx*self.batch_size:(idx+1)*self.batch_size]]
        else:
            sweight = None
        ret += (outputs, sweight)

        return ret

    def _getitemdict(self, idx):

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

    def __getitem__(self, idx):
        return self._getitemdict(idx) if self.as_dict else self._getitemlist(idx)


    def on_epoch_end(self):
        """Stuff to do after epoch end."""
        if self.shuffle:
            numpy.random.shuffle(self.indices)
