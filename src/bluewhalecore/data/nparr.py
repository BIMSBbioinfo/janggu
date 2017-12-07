from data import BwDataset


class NumpyBwDataset(BwDataset):

    def __init__(self, name, array, cachefile=None):

        self.data = array

        if isinstance(cachefile, str):
            self.cachefile = cachefile

        self.sanityChecks()

        BwDataset.__init__(self, 'NP_{}'.format(name))

    def load(self):
        pass

    def __repr__(self):
        return 'NumpyBwDataset("{}", <np.array>)'.format(self.name)

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape[1:]
