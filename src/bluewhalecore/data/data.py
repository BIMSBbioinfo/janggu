
class BwDataset(object):
    # list of data augmentation transformations
    transformations = []

    def __init__(self, name):
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise Exception('name must be a string')
        self._name = value

    def __getitem__(self, idxs):

        data = self.data[idxs]

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        pass

    @property
    def shape(self):
        pass


def inputShape(bwdata):
    """Extracts the shape of a provided BwDataset."""
    if isinstance(bwdata, BwDataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        x = {}
        for el in bwdata:
            shape = el.shape[1:]
            if shape == ():
                shape = (1,)
            x[el.name] = {'shape': shape}
        return x
    else:
        raise Exception('inputSpace wrong argument: {}'.format(bwdata))


def outputShape(bwdata, loss, activation='sigmoid',
                loss_weight=1.):
    """Extracts the shape and specifies learning objectives."""
    if isinstance(bwdata, BwDataset):
        bwdata = [bwdata]

    if isinstance(bwdata, list):
        x = {}
        for el in bwdata:
            shape = el.shape[1:]
            if shape == ():
                shape = (1,)
            x[el.name] = {'shape': shape,
                          'loss': loss,
                          'loss_weight': loss_weight,
                          'activation': activation}
        return x
    else:
        raise Exception('outputSpace wrong argument: {}'.format(bwdata))
