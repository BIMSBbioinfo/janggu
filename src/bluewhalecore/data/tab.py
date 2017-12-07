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
    sep : str
        Item separator. Default: sep=','
    """

    def __init__(self, name, filename, cachefile=None, sep=','):

        self.filename = filename
        self.sep = sep
        self.header = None

        if isinstance(cachefile, str):
            self.cachefile = cachefile

        BwDataset.__init__(self, name)

    def load(self):

        data = []
        for f in self.filename:
            data.append(pd.read_csv(f, header=self.header,
                                    sep=self.sep)).values

        data = pd.concat(data, axis=1, ignore_index=True).values

    def __repr__(self):
        return 'TabBwDataset("{}", {}})'.format(self.name, self.filename)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape[1:]

    def filename():
        doc = "The filename property."

        def fget(self):
            return self._filename

        def fset(self, value):
            if isinstance(value, str):
                value = list(value)
            for el in value:
                if not os.path.exists(el):
                    raise Exception('File does not exist \
                                    not exists: {}'.format(el))
            self._filename = value

        def fdel(self):
            del self._filename
        return locals()

    filename = property(**filename())
