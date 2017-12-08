import numpy as np
import pytest

from bluewhalecore.data.tab import TabBwDataset


def test_tab_reading():

    ctcf = TabBwDataset('train', filename='resources/ctcf_sample.csv')

    np.testing.assert_equal(len(ctcf), 14344)
    np.testing.assert_equal(ctcf.shape, (1,))

    jund = TabBwDataset('train', filename='resources/jund_sample.csv')

    np.testing.assert_equal(len(jund), 14344)
    np.testing.assert_equal(jund.shape, (1,))

    # read both
    both = TabBwDataset('train', filename=['resources/jund_sample.csv',
                                           'resources/ctcf_sample.csv'])

    np.testing.assert_equal(len(both), 14344)
    np.testing.assert_equal(both.shape, (2,))

    with pytest.raises(Exception):
        TabBwDataset('train', filename='')
