import numpy as np

from janggu.data import Array
from janggu.data import ReduceDim

def test_nparr(tmpdir):
    X = Array("X", np.random.random((1000, 100)))
    y = Array('y', np.random.randint(2, size=(1000,)))

    np.testing.assert_equal(len(X), len(y))
    np.testing.assert_equal(len(X), 1000)
    np.testing.assert_equal(X.shape, (1000, 100,))
    np.testing.assert_equal(y.shape, (1000, 1))
    assert y.ndim == 2
    assert y.shape == (1000, 1)


def test_reducedim():
    x_orig = np.zeros((3,1,1,2))
    
    x_reduce = ReduceDim(Array('test', x_orig))
    np.testing.assert_equal(len(x_reduce), 3)
    np.testing.assert_equal(x_reduce.shape, (3,2))

    assert x_reduce.ndim == 2
