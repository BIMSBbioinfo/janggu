import threading

import numpy as np


def bluewhale_fit_generator(inputdata, outputdata, batch_size,
                            sample_weight=None, shuffle=False):

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
