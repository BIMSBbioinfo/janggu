import os

import numpy as np
import pkg_resources
import pytest

from beluga.data import TabBlgDataset


def test_tab_reading():
    data_path = pkg_resources.resource_filename('beluga', 'resources/')

    ctcf = TabBlgDataset('train', filename=os.path.join(data_path,
                                                        'ctcf_sample.csv'))

    np.testing.assert_equal(len(ctcf), 14344)
    np.testing.assert_equal(ctcf.shape, (len(ctcf), 1,))

    jund = TabBlgDataset('train', filename=os.path.join(data_path,
                                                        'jund_sample.csv'))

    np.testing.assert_equal(len(jund), 14344)
    np.testing.assert_equal(jund.shape, (len(jund), 1,))

    # read both
    both = TabBlgDataset('train',
                         filename=[os.path.join(data_path, 'jund_sample.csv'),
                                   os.path.join(data_path, 'ctcf_sample.csv')])

    np.testing.assert_equal(len(both), 14344)
    np.testing.assert_equal(both.shape, both[:].shape)

    with pytest.raises(Exception):
        TabBlgDataset('train', filename='')
