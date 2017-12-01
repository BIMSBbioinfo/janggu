import numpy as np


def generate_fit_data(inputdata, outputdata, indices, batchsize,
                      sample_weights=None):
    while 1:
        ib = 0
        np.random.shuffle(indices)

        if len(indices) == 0:
            raise Exception("index list is empty")

        while ib < \
                (len(indices)//batchsize +
                    (1 if len(indices) % batchsize > 0 else 0)):

            input = {}

            for data in inputdata:
                input[data.name] = data.getData(
                    indices[ib*batchsize:(ib+1)*batchsize]).copy()

            output = {}
            for data in outputdata:
                output[data.name] = data.getData(
                    indices[ib*batchsize:(ib+1)*batchsize]).copy()

            if isinstance(sample_weights, type(None)):
                sw = None
            else:
                sw = sample_weights[
                    indices[ib*batchsize:(ib+1)*batchsize]].copy()

            ib += 1

            yield input, output, sw


def generate_predict_data(inputdata, outputdata, indices, batchsize):
    while 1:
        ib = 0
        if len(indices) == 0:
            raise Exception("index list is empty")
        while ib < \
                (len(indices)//batchsize +
                    (1 if len(indices) % batchsize > 0 else 0)):

            input = {}

            for data in inputdata:
                input[data.name] = data.getData(
                    indices[ib*batchsize:(ib+1)*batchsize]).copy()

            ib += 1

            yield input
