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
        Either `(input, output, sample_weight)` per batch if
        sample_weight is used or `(input, output)` otherwise.
    """

    lock = threading.Lock()

    if not isinstance(inputdata, dict) or not isinstance(outputdata, dict):
        raise Exception('generate_fit_data expects data to be dicts')

    for k in inputdata:
        indices = range(len(inputdata[k]))
        break

    while 1:
        ib = 0
        if shuffle:
            np.random.shuffle(indices)

        if len(indices) == 0:
            raise Exception("index list is empty")

        while ib < \
                (len(indices)//batch_size +
                    (1 if len(indices) % batch_size > 0 else 0)):

            with lock:
                tmpi = ib
                ib += 1

            input = {}

            for k in inputdata:
                input[k] = inputdata[k][indices[tmpi*batch_size:
                                                (tmpi+1)*batch_size]]

            output = {}
            for k in outputdata:
                output[k] = outputdata[k][indices[tmpi*batch_size:
                                                  (tmpi+1)*batch_size]]

            if sample_weight:
                sw = sample_weight[indices[tmpi*batch_size:
                                           (tmpi+1)*batch_size]]
                yield input, output, sw
            else:
                yield input, output
            # ib += 1


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

    if len(indices) == 0:
        raise Exception("index list is empty")
    while 1:
        ib = 0
        while ib < \
                (len(indices)//batch_size +
                    (1 if len(indices) % batch_size > 0 else 0)):

            with lock:
                tmpi = ib
                ib += 1

            input = {}

            for k in inputdata:
                input[k] = inputdata[k][indices[tmpi*batch_size:
                                                (tmpi+1)*batch_size]]

            yield input
