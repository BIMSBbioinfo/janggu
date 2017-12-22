import os

import pandas as pd
from data import BwDataset


class TabBwDataset(BwDataset):
    """TabBwDataset class.
    The class :class:`TabBwDataset` allows to fetch data from a CSV or TSV
    file.

    Parameters
    -----------
    name : str
        Unique name of the model.
    filename : str or list of str
        Filename or list of filenames containing tables to load.
    cachefile : str
        cachefile to load the data from if present
    dtype : str
        Type. Default: 'int8'
    sep : str
        Item separator. Default: sep=','
    """

    def __init__(self, name, filename, cachedir=None, dtype='int8', sep=','):

        self.filename = filename
        self.sep = sep
        self.header = None
        self.dtype = dtype

        data = []
        for f in self.filename:
            data.append(pd.read_csv(f, header=self.header,
                                    sep=self.sep, dtype=self.dtype))

            self.data = pd.concat(data, axis=1, ignore_index=True).values

        if isinstance(cachedir, str):
            self.cachedir = cachedir

        BwDataset.__init__(self, name)

    def __repr__(self):
        return 'TabBwDataset("{}", "{}", sep="{}")'\
                .format(self.name, self.filename, self.sep)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def filename():
        doc = "The filename property."

        def fget(self):
            return self._filename

        def fset(self, value):
            if isinstance(value, str):
                value = [value]
            for el in value:
                if not os.path.exists(el):
                    raise Exception('File does not exist \
                                    not exists: {}'.format(el))
            self._filename = value

        def fdel(self):
            del self._filename
        return locals()

    filename = property(**filename())
