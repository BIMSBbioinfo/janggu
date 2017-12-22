import os

import keras
import numpy as np
import pkg_resources
from keras import backend as K
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from bluewhalecore.data import DnaBwDataset
from bluewhalecore.data import NumpyBwDataset


np.random.seed(1234)

data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')
filename = os.path.join(data_path, 'oct4.fa')
bw_x_train = DnaBwDataset.fromFasta('train', fastafile=filename, order=1)
bw_y_train = NumpyBwDataset('y', np.random.randint(2, size=(len(bw_x_train), 1)))

print(len(bw_x_train))
print(len(bw_y_train))
print(bw_x_train.shape)
print(bw_y_train.shape)

# Option 2:
# Instantiate the model manually
def fmodel():
    input = Input(shape=(4, 200, 1))
    layer = Conv2D(30, (4, 21), activation='relu')(input)
    layer = GlobalAveragePooling2D()(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# For comparison, here is how the model would train without BlueWhale
K.clear_session()
np.random.seed(1234)
m = fmodel()
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)


K.clear_session()
m = fmodel()
h = m.fit(bw_x_train, bw_y_train, epochs=30, batch_size=1000,
          shuffle=False, verbose=0)
print('#' * 40)
print('loss: {}, acc: {}'.format(h.history['loss'][-1], h.history['acc'][-1]))
print('#' * 40)
