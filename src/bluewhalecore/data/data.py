
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
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape[1:]


def inputShape(bwdata):
    """Extracts the shape of a provided BwDataset."""
    if isinstance(bwdata, BwDataset):
        return {bwdata.name: {'shape': bwdata.shape}}
    elif isinstance(bwdata, list):
        x = {}
        for el in bwdata:
            x[el.name] = {'shape': el.shape}
        return x
    else:
        raise Exception('inputSpace wrong argument: {}'.format(bwdata))


def outputShape(bwdata, loss, activation='sigmoid',
                loss_weight=1.):
    """Extracts the shape and specifies learning objectives."""
    if isinstance(bwdata, BwDataset):
        return {bwdata.name: {'shape': bwdata.shape,
                              'loss': loss,
                              'loss_weight': loss_weight,
                              'activation': activation}}
    elif isinstance(bwdata, list):
        x = {}
        for el in bwdata:
            x[el.name] = {'shape': el.shape,
                          'loss': loss,
                          'loss_weight': loss_weight,
                          'activation': activation}
        return x
    else:
        raise Exception('outputSpace wrong argument: {}'.format(bwdata))
