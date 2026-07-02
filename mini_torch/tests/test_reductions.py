import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.gradcheck import gradcheck


# ==========================================================
# SUM
# ==========================================================

def test_sum_forward():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.sum()

    assert np.allclose(y.data, 10.0)


def test_sum_axis0():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.sum(axis=0)

    expected = np.array([4., 6.])

    assert np.allclose(y.data, expected)


def test_sum_axis1():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.sum(axis=1)

    expected = np.array([3., 7.])

    assert np.allclose(y.data, expected)


def test_sum_keepdims():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.sum(axis=1, keepdims=True)

    expected = np.array([[3.],
                         [7.]])

    assert np.allclose(y.data, expected)
    assert y.shape == (2, 1)


def test_sum_backward():
    x = tensor([[1., 2.],
                [3., 4.]],
               requires_grad=True)

    loss = x.sum()

    loss.backward()

    expected = np.ones((2, 2))

    assert np.allclose(x.grad, expected)


def test_sum_axis_backward():
    x = tensor([[1., 2.],
                [3., 4.]],
               requires_grad=True)

    loss = x.sum(axis=1).sum()

    loss.backward()

    expected = np.ones((2, 2))

    assert np.allclose(x.grad, expected)


def test_sum_gradcheck():
    x = tensor(np.random.randn(3, 4), requires_grad=True)

    def fn(x):
        return x.sum(axis=1).sum()

    assert gradcheck(fn, [x])


# ==========================================================
# MEAN
# ==========================================================

def test_mean_forward():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.mean()

    assert np.allclose(y.data, 2.5)


def test_mean_axis0():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.mean(axis=0)

    expected = np.array([2., 3.])

    assert np.allclose(y.data, expected)


def test_mean_axis1():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.mean(axis=1)

    expected = np.array([1.5, 3.5])

    assert np.allclose(y.data, expected)


def test_mean_keepdims():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.mean(axis=1, keepdims=True)

    expected = np.array([[1.5],
                         [3.5]])

    assert np.allclose(y.data, expected)
    assert y.shape == (2, 1)


def test_mean_backward():
    x = tensor([[1., 2.],
                [3., 4.]],
               requires_grad=True)

    loss = x.mean()

    loss.backward()

    expected = np.ones((2, 2)) / 4

    assert np.allclose(x.grad, expected)


def test_mean_axis_backward():
    x = tensor([[1., 2.],
                [3., 4.]],
               requires_grad=True)

    loss = x.mean(axis=1).sum()

    loss.backward()

    expected = np.array([[0.5, 0.5],
                         [0.5, 0.5]])

    assert np.allclose(x.grad, expected)


def test_mean_gradcheck():
    x = tensor(np.random.randn(3, 4), requires_grad=True)

    def fn(x):
        return x.mean(axis=1).sum()

    assert gradcheck(fn, [x])


# ==========================================================
# NEGATIVE AXIS
# ==========================================================

def test_sum_negative_axis():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.sum(axis=-1)

    expected = np.array([3., 7.])

    assert np.allclose(y.data, expected)


def test_mean_negative_axis():
    x = tensor([[1., 2.],
                [3., 4.]])

    y = x.mean(axis=-1)

    expected = np.array([1.5, 3.5])

    assert np.allclose(y.data, expected)