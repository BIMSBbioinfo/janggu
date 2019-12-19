from copy import copy

import numpy as np
import pytest

from janggu.data import Array
from janggu.data import NanToNumConverter
from janggu.data import RandomOrientation
from janggu.data import RandomShift
from janggu.data import RandomSignalScale
from janggu.data import ReduceDim
from janggu.data import SqueezeDim


def test_nparr(tmpdir):
    X = Array("X", np.random.random((1000, 100)))
    y = Array('y', np.random.randint(2, size=(1000,)))

    np.testing.assert_equal(len(X), len(y))
    np.testing.assert_equal(len(X), 1000)
    np.testing.assert_equal(X.shape, (1000, 100,))
    np.testing.assert_equal(y.shape, (1000, 1))
    assert y.ndim == 2
    assert y.shape == (1000, 1)
    new_X = copy(X)


def test_reducedim():
    x_orig = np.zeros((3,1,1,2))

    np.testing.assert_equal(x_orig.ndim, 4)
    x_reduce = ReduceDim(Array('test', x_orig, conditions=["A", "B"]))
    x_reduce = ReduceDim(Array('test', x_orig, conditions=["A", "B"]), aggregator='mean')
    x_reduce = ReduceDim(Array('test', x_orig, conditions=["A", "B"]), aggregator='max')
    x_reduce = ReduceDim(Array('test', x_orig, conditions=["A", "B"]), aggregator=np.mean)
    with pytest.raises(ValueError):
        ReduceDim(Array('test', x_orig, conditions=["A", "B"]), aggregator='nonsense')

    np.testing.assert_equal(len(x_reduce), 3)
    np.testing.assert_equal(x_reduce.shape, (3,2))
    np.testing.assert_equal(x_reduce.ndim, 2)
    assert x_reduce[0].shape == (1, 2)
    assert x_reduce[:3].shape == (3, 2)
    assert x_reduce[[0,1]].shape == (2, 2)
    assert x_reduce.ndim == 2
    new_x = copy(x_reduce)
    assert x_reduce[0].shape == new_x[0].shape
    assert x_reduce.conditions == ["A", "B"]

def test_squeezedim():
    x_orig = np.zeros((3,1,1,2))

    np.testing.assert_equal(x_orig.ndim, 4)
    x_sq = SqueezeDim(Array('test', x_orig, conditions=["A", "B"]))

    np.testing.assert_equal(len(x_sq), 3)

    np.testing.assert_equal(x_sq.shape, (3,2))

    np.testing.assert_equal(x_sq.ndim, 2)
    assert x_sq[0].shape == (2,)
    assert x_sq[:3].shape == (3, 2)
    assert x_sq[[0,1]].shape == (2, 2)
    assert x_sq.ndim == 2
    new_x = copy(x_sq)
    assert x_sq[0].shape == new_x[0].shape
    assert x_sq.conditions == ["A", "B"]


def test_nantonumconverter():
    x_orig = np.zeros((3,1,1,2))
    x_orig[0,0,0,0] = np.nan
    arr = Array('test', x_orig, conditions=["A", "B"])
    assert np.isnan(arr[0].mean())

    x_tr = NanToNumConverter(Array('test', x_orig, conditions=["A", "B"]))
    assert x_tr[0].shape == (1, 1, 1, 2)
    assert x_tr[:3].shape == (3, 1, 1, 2)
    assert x_tr[[0,1]].shape == (2, 1, 1, 2)
    assert len(x_tr) == 3
    assert x_tr.shape == (3, 1, 1, 2)
    assert x_tr.ndim == 4
    assert not np.isnan(x_tr[0].mean())
    np.testing.assert_equal(x_tr[0], [[[[0,0]]]])
    new_x = copy(x_tr)
    assert x_tr[0].shape == new_x[0].shape
    assert x_tr.conditions == ["A", "B"]

def test_randomorientation():
    x_orig = np.zeros((3,1,1,2))

    x_tr = RandomOrientation(Array('test', x_orig, conditions=["A", "B"]))
    assert x_tr[0].shape == (1, 1, 1, 2)
    assert x_tr[:3].shape == (3, 1, 1, 2)
    assert x_tr[[0,1]].shape == (2, 1, 1, 2)
    np.testing.assert_equal(len(x_tr), 3)
    assert len(x_tr) == 3
    assert x_tr.shape == (3, 1, 1, 2)
    assert x_tr.ndim == 4
    np.testing.assert_equal(x_tr[0], [[[[0,0]]]])
    new_x = copy(x_tr)
    assert x_tr[0].shape == new_x[0].shape
    assert x_tr.conditions == ["A", "B"]


def test_randomsignalscale():
    x_orig = np.ones((3,1,1,2))

    x_tr = RandomSignalScale(Array('test', x_orig), .1)
    assert x_tr[0].shape == (1, 1, 1, 2)
    assert x_tr[:3].shape == (3, 1, 1, 2)
    assert x_tr[[0,1]].shape == (2, 1, 1, 2)
    np.testing.assert_equal(len(x_tr), 3)
    assert len(x_tr) == 3
    assert x_tr.shape == (3, 1, 1, 2)
    assert x_tr.ndim == 4
    new_x = copy(x_tr)
    assert x_tr[0].shape == new_x[0].shape
    assert x_tr.conditions == None


def test_randomshift():
    x_orig = np.zeros((1,4,1,4))
    x_orig[0, 0, 0,0] = 1
    x_orig[0, 1, 0,1] = 1
    x_orig[0, 2, 0,2] = 1
    x_orig[0, 3, 0,3] = 1

    x_tr = RandomShift(Array('test', x_orig), 1)
    assert x_tr[0].shape == (1, 4, 1, 4)
    np.testing.assert_equal(len(x_tr), 1)
    assert x_tr.shape == (1, 4, 1, 4)
    assert x_tr.ndim == 4
    new_x = copy(x_tr)
    assert x_tr[0].shape == new_x[0].shape
    assert x_tr.conditions == None
    
    x_tr = RandomShift(Array('test', x_orig), 1, True)
    assert x_tr[0].shape == (1, 4, 1, 4)
    np.testing.assert_equal(len(x_tr), 1)
    assert x_tr.shape == (1, 4, 1, 4)
    assert x_tr.ndim == 4
    new_x = copy(x_tr)
    assert x_tr[0].shape == new_x[0].shape
    assert x_tr.conditions == None
    
