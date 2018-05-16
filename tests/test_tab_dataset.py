import os

import numpy as np
import pkg_resources
import pytest

from janggu.data import Table


def test_tab_reading():
    data_path = pkg_resources.resource_filename('janggu', 'resources/')

    sample = Table('train', filename=os.path.join(data_path, 'sample.csv'))

    np.testing.assert_equal(len(sample), 100)
    np.testing.assert_equal(sample.shape, (100, 1,))

    # read both
    both = Table('train',
                 filename=[os.path.join(data_path, 'sample.csv'),
                           os.path.join(data_path, 'sample.csv')])

    np.testing.assert_equal(len(both), 100)
    np.testing.assert_equal(both.shape, (100, 2,))

    with pytest.raises(Exception):
        Table('train', filename='')
