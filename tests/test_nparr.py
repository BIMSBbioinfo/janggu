import numpy as np

from beluga.data import NumpyBlgDataset


def test_nparr(tmpdir):
    X = NumpyBlgDataset("X", np.random.random((1000, 100)))
    y = NumpyBlgDataset('y', np.random.randint(2, size=(1000, 1)))

    np.testing.assert_equal(len(X), len(y))
    np.testing.assert_equal(len(X), 1000)
    np.testing.assert_equal(X.shape, (1000, 100,))
    np.testing.assert_equal(y.shape, (1000, 1,))
