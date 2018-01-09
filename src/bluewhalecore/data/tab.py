import os

import pandas as pd
from data import BwDataset


class TabBwDataset(BwDataset):
    """TabBwDataset class.

    TabBwDataset allows to fetch data from a CSV or TSV file.
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

    def __init__(self, name, filename, samplenames=None,
                 cachedir=None, dtype='int8', sep=','):

        self.filename = filename

        self.samplenames = samplenames
        if not samplenames:
            self.samplenames = filename

        self.sep = sep
        self.header = None
        self.dtype = dtype

        data = []
        for f in self.filename:
            data.append(pd.read_csv(f, header=self.header,
                                    sep=self.sep, dtype=self.dtype))

            self.data = pd.concat(data, axis=1, ignore_index=True).values

        self.cachedir = cachedir

        BwDataset.__init__(self, name)

    def __repr__(self):
        return 'TabBwDataset("{}", "{}", sep="{}")'\
                .format(self.name, self.filename, self.sep)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        __doc__ = super(TabBwDataset, self).__doc__  # noqa
        return self.data.shape

    @property
    def filename(self):
        """The filename property."""
        return self._filename

    @filename.setter
    def filename(self, value):
        if isinstance(value, str):
            value = [value]
        for el in value:
            if not os.path.exists(el):
                raise Exception('File does not exist \
                                not exists: {}'.format(el))
        self._filename = value
