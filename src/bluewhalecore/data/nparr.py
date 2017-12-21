from data import BwDataset


class NumpyBwDataset(BwDataset):

    def __init__(self, name, array, cachedir=None):

        self.data = array

        if isinstance(cachedir, str):
            self.cachedir = cachedir

        BwDataset.__init__(self, '{}'.format(name))

    def __repr__(self):
        return 'NumpyBwDataset("{}", <np.array>)'.format(self.name)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        if len(self.data.shape) > 1:
            return self.data.shape[1:]
        else:
            return (1, )
