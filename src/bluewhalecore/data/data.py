
class BWDataset(object):
    # list of data augmentation transformations
    transformations = []

    def __init__(self, name):
        self.name = name
        self.load()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    def load(self):
        raise NotImplemented('load data must be implemented')

    def getData(self, idx=None):
        if isinstance(idx, type(None)):
            data = self.data
        else:
            data = self.data[idx]

        for tr in self.transformations:
            data = tr(data)

        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape[1:]


class BWOutput(BWDataset):

    def __init__(self, name, loss='binary_crossentropy', loss_weight=1.,
                 act_fct='sigmoid'):
        self.loss = loss
        self.loss_weight = loss_weight
        self.activation = act_fct

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, l):
        self._loss = l

    @property
    def loss_weight(self):
        return self._loss_weight

    @loss_weight.setter
    def loss_weight(self, l):
        self._loss_weight = l

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, l):
        self._activation = l
