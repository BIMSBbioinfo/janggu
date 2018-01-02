import threading

import numpy as np


# taken from the blog post:
# https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(gen):
    def g(*args, **kargs):
        return threadsafe_iter(gen(*args, **kargs))
    return g


@threadsafe_generator
def generate_fit_data(inputdata, outputdata, batchsize,
                      sample_weights=None, shuffle=False):
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
                (len(indices)//batchsize +
                    (1 if len(indices) % batchsize > 0 else 0)):

            input = {}

            for k in inputdata:
                input[k] = inputdata[k][indices[ib*batchsize:(ib+1)*batchsize]]

            output = {}
            for k in outputdata:
                output[k] = outputdata[k][indices[ib*batchsize:
                                                  (ib+1)*batchsize]]

            if sample_weights:
                sw = sample_weights[indices[ib*batchsize:(ib+1)*batchsize]]
                yield input, output, sw
            else:
                yield input, output

            ib += 1


@threadsafe_generator
def generate_predict_data(inputdata, batchsize):

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
                (len(indices)//batchsize +
                    (1 if len(indices) % batchsize > 0 else 0)):

            input = {}

            for k in inputdata:
                input[k] = inputdata[k][indices[ib*batchsize:(ib+1)*batchsize]]

            ib += 1

            yield input
