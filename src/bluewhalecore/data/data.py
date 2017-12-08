
class BwDataset(object):
    # list of data augmentation transformations
    transformations = []

    def __init__(self, name):
        self.name = name
        self.load()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise Exception('name must be a string')
        self._name = value

    def load(self):
        raise NotImplemented('load data must be implemented')

    def getData(self, idxs):

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
    return {bwdata.name: {'shape': bwdata.shape}}


def outputShape(bwdata, loss='binary_crossentropy',
                loss_weight=1., activation='sigmoid'):
    """Extracts the shape and specifies learning objectives."""
    return {bwdata.name: {'shape': bwdata.shape,
                          'loss': loss,
                          'loss_weight': loss_weight,
                          'activation': activation}}
