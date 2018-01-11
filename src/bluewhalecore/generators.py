import threading

import numpy as np


def bluewhale_fit_generator(inputdata, outputdata, batch_size,
                            sample_weight=None, shuffle=False):
    """Generator for BlueWhale-model fitting.

    This generator is designed for the use with :meth:`BlueWhale.fit`
    or :meth:`BlueWhale.evaluate`.

    Parameters
    ----------
    inputdata : dict
        Dictionary with keys corresponding to the dataset names and
        values being a :class:`BwDataset`.
    outputdata : dict
        Dictionary with keys corresponding to the dataset names and
        values being a :class:`BwDataset`.
    batch_size : int
        Batchsize to use for enumerating the dataset.
    sample_weight : None or list
        List of sample-specific weights. Default: None means no
        sample_weight is used.
    shuffle : bool
        Shuffle the dataset once per epoch. Default: False.

    Yields
    ------
    tuple
        Either `(inputs, outputs, sample_weight)` per batch if
        sample_weight is used or `(inputs, outputs)` otherwise.
    """

    lock = threading.Lock()

    if not isinstance(inputdata, dict) or not isinstance(outputdata, dict):
        raise Exception('generate_fit_data expects data to be dicts')

    for k in inputdata:
        indices = range(len(inputdata[k]))
        break

    while 1:
        ibatch = 0
        if shuffle:
            np.random.shuffle(indices)

        if not indices:
            raise Exception("index list is empty")

        while ibatch < \
                (len(indices)//batch_size +
                 (1 if len(indices) % batch_size > 0 else 0)):

            with lock:
                tmpi = ibatch
                ibatch += 1

            inputs = {}

            for k in inputdata:
                inputs[k] = inputdata[k][indices[tmpi*batch_size:
                                                 (tmpi+1)*batch_size]]

            outputs = {}
            for k in outputdata:
                outputs[k] = outputdata[k][indices[tmpi*batch_size:
                                                   (tmpi+1)*batch_size]]

            if sample_weight:
                sweight = sample_weight[indices[tmpi*batch_size:
                                                (tmpi+1)*batch_size]]
                yield inputs, outputs, sweight
            else:
                yield inputs, outputs


def bluewhale_predict_generator(inputdata, batch_size):
    """Generator for BlueWhale-model prediction.

    This generator is designed for the use with :meth:`BlueWhale.predict`.

    Parameters
    ----------
    inputdata : dict
        Dictionary with keys corresponding to the dataset names and
        values being a :class:`BwDataset`.
    batch_size : int
        Batchsize to use for enumerating the dataset.

    Yields
    ------
    numpy.array or list(numpy.array)
        Per batch output of model.
    """

    lock = threading.Lock()

    if not isinstance(inputdata, dict):
        raise Exception('generate_predict_data expects inputdata to be a dict')

    for k in inputdata:
        indices = range(len(inputdata[k]))
        break

    if not indices:
        raise Exception("index list is empty")
    while 1:
        ibatch = 0
        while ibatch < \
                (len(indices)//batch_size +
                 (1 if len(indices) % batch_size > 0 else 0)):

            with lock:
                tmpi = ibatch
                ibatch += 1

            inputs = {}

            for k in inputdata:
                inputs[k] = inputdata[k][indices[tmpi*batch_size:
                                                 (tmpi+1)*batch_size]]

            yield inputs
