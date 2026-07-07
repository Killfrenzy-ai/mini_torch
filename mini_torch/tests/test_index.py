import numpy as np
import pytest

from mini_torch.tensors import tensor
from mini_torch.parameter import Parameter

from mini_torch.autograd.engine import backward


# ==========================================================
# Forward
# ==========================================================

def test_index_single_element():

    x = tensor(np.array([1., 2., 3., 4.]))

    y = x[2]

    assert y.data == 3.


def test_index_slice():

    x = tensor(np.array([1., 2., 3., 4., 5.]))

    y = x[1:4]

    assert np.array_equal(
        y.data,
        np.array([2., 3., 4.]),
    )


def test_index_list():

    x = tensor(np.array([10., 20., 30., 40.]))

    y = x[[0, 3]]

    assert np.array_equal(
        y.data,
        np.array([10., 40.]),
    )


def test_index_numpy_array():

    x = tensor(np.array([5., 6., 7., 8.]))

    indices = np.array([1, 2])

    y = x[indices]

    assert np.array_equal(
        y.data,
        np.array([6., 7.]),
    )


def test_index_multidimensional():

    x = tensor(
        np.arange(12).reshape(3, 4)
    )

    y = x[1]

    assert np.array_equal(
        y.data,
        np.array([4, 5, 6, 7]),
    )


# ==========================================================
# Shapes
# ==========================================================

def test_index_shape():

    x = tensor(
        np.random.randn(10, 5)
    )

    y = x[[1, 3, 5]]

    assert y.shape == (3, 5)


def test_slice_shape():

    x = tensor(
        np.random.randn(8, 4)
    )

    y = x[2:6]

    assert y.shape == (4, 4)


# ==========================================================
# Backward
# ==========================================================

def test_index_backward_single():

    w = Parameter(
        np.random.randn(5, 3)
    )

    y = w[[2]]

    loss = y.sum()

    backward(loss)

    expected = np.zeros_like(w.data)

    expected[2] = 1

    assert np.array_equal(
        w.grad,
        expected,
    )


def test_index_backward_multiple():

    w = Parameter(
        np.random.randn(6, 2)
    )

    y = w[[1, 4]]

    loss = y.sum()

    backward(loss)

    expected = np.zeros_like(w.data)

    expected[1] = 1
    expected[4] = 1

    assert np.array_equal(
        w.grad,
        expected,
    )


def test_index_backward_repeated_indices():

    w = Parameter(
        np.random.randn(5, 2)
    )

    y = w[[2, 2, 2]]

    loss = y.sum()

    backward(loss)

    expected = np.zeros_like(w.data)

    expected[2] = 3

    assert np.array_equal(
        w.grad,
        expected,
    )


def test_index_backward_no_unused_gradients():

    w = Parameter(
        np.random.randn(4, 3)
    )

    y = w[[1]]

    loss = y.sum()

    backward(loss)

    mask = np.ones(4, dtype=bool)

    mask[1] = False

    assert np.all(
        w.grad[mask] == 0
    )


# ==========================================================
# Graph Construction
# ==========================================================

def test_index_has_parent():

    x = Parameter(
        np.random.randn(5)
    )

    y = x[2]

    assert len(y.parents) == 1

    assert y.parents[0] is x


def test_index_requires_grad():

    x = Parameter(
        np.random.randn(5)
    )

    y = x[1]

    assert y.requires_grad


# ==========================================================
# Metadata
# ==========================================================

def test_index_metadata_saved():

    x = Parameter(
        np.random.randn(6)
    )

    y = x[[1, 3]]

    assert hasattr(y, "index")

    assert y.index == [1, 3]


# ==========================================================
# Regression
# ==========================================================

def test_index_then_relu_backward():

    w = Parameter(
        np.array(
            [
                [-1., 2.],
                [3., -4.],
            ]
        )
    )

    y = w[[0, 1]]

    out = y.relu()

    loss = out.sum()

    backward(loss)

    expected = np.array(
        [
            [0., 1.],
            [1., 0.],
        ]
    )

    assert np.array_equal(
        w.grad,
        expected,
    )