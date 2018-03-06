import os

import pandas as pd

from janggo.data.data import Dataset


class TabDataset(Dataset):
    """TabDataset class.

    TabDataset allows to fetch data from a CSV or TSV file.
    It might be used to load a set of training labels.

    Parameters
    -----------
    name : str
        Unique name of the model.
    filename : str or list(str)
        Filename or list of filenames containing tables to load.
    samplenames : list(str)
        Samplenames (optional). They are relevant if the dataset
        is used to hold labels for a deep learning applications.
        For instance, samplenames might correspond to category names.
    cachedir : str or None
        Directory in which the cachefiles are located. Default: None.
    dtype : str
        Datatype. Default: 'int8'
    sep : str
        Item separator. Default: sep=','.
    """

    _filename = None

    def __init__(self, name, filename, samplenames=None,
                 cachedir=None, dtype='int8', sep=','):

        if not isinstance(filename, list):
            self.filename = [filename]
        else:
            self.filename = filename

        self.samplenames = samplenames
        if not samplenames:
            self.samplenames = filename

        data = []
        for _file in self.filename:
            data.append(pd.read_csv(_file, header=None,
                                    sep=sep, dtype=dtype))

            self.data = pd.concat(data, axis=1, ignore_index=True).values

        self.cachedir = cachedir

        Dataset.__init__(self, name)

    def __repr__(self):  # pragma: no cover
        return 'TabDataset("{}", "{}")'\
                .format(self.name, self.filename)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        """Shape of the dataset"""
        return self.data.shape

    def __getitem__(self, idxs):
        data = self.data[idxs]

        for transform in self.transformations:
            data = transform(data)

        return data

    @property
    def filename(self):
        """The filename property."""
        return self._filename

    @filename.setter
    def filename(self, values):
        if isinstance(values, str):
            values = [values]
        for value in values:
            if not os.path.exists(value):
                raise Exception('File does not exist \
                                not exists: {}'.format(value))
        self._filename = values
