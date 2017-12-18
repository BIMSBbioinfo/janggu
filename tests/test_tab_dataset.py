import os
import pkg_resources
import numpy as np
import pytest

from bluewhalecore.data.tab import TabBwDataset


def test_tab_reading():
    data_path = pkg_resources.resource_filename('bluewhalecore', 'resources/')

    ctcf = TabBwDataset('train', filename=os.path.join(data_path,
                                                       'ctcf_sample.csv'))

    np.testing.assert_equal(len(ctcf), 14344)
    np.testing.assert_equal(ctcf.shape, (1,))

    jund = TabBwDataset('train', filename=os.path.join(data_path,
                                                       'jund_sample.csv'))

    np.testing.assert_equal(len(jund), 14344)
    np.testing.assert_equal(jund.shape, (1,))

    # read both
    both = TabBwDataset('train',
                        filename=[os.path.join(data_path, 'jund_sample.csv'),
                                  os.path.join(data_path, 'ctcf_sample.csv')])

    np.testing.assert_equal(len(both), 14344)
    np.testing.assert_equal(both.shape, (2,))

    with pytest.raises(Exception):
        TabBwDataset('train', filename='')
