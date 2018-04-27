import numpy as np

from janggo.data import NumpyWrapper


def test_nparr(tmpdir):
    X = NumpyWrapper("X", np.random.random((1000, 100)))
    y = NumpyWrapper('y', np.random.randint(2, size=(1000, 1)))

    np.testing.assert_equal(len(X), len(y))
    np.testing.assert_equal(len(X), 1000)
    np.testing.assert_equal(X.shape, (1000, 100,))
    np.testing.assert_equal(y.shape, (1000, 1,))
