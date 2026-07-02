import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.gradcheck import gradcheck


# ==========================================================
# FORWARD
# ==========================================================

def test_softmax_forward():
    x = tensor([1.0, 2.0, 3.0])

    y = x.softmax()

    expected = np.exp(np.array([1.0, 2.0, 3.0]))
    expected /= expected.sum()

    assert np.allclose(y.data, expected)


def test_softmax_output_sum():
    x = tensor([1.0, 2.0, 3.0])

    y = x.softmax()

    assert np.allclose(y.data.sum(), 1.0)


def test_softmax_axis():
    x = tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])

    y = x.softmax(axis=1)

    row_sums = y.data.sum(axis=1)

    assert np.allclose(row_sums, np.ones(2))


def test_softmax_keep_shape():
    x = tensor(np.random.randn(4, 5))

    y = x.softmax(axis=1)

    assert y.shape == x.shape


# ==========================================================
# NUMERICAL STABILITY
# ==========================================================

def test_softmax_large_values():
    x = tensor([1000., 1001., 999.])

    y = x.softmax()

    assert np.isfinite(y.data).all()
    assert np.allclose(y.data.sum(), 1.0)


def test_softmax_negative_values():
    x = tensor([-1000., -999., -998.])

    y = x.softmax()

    assert np.isfinite(y.data).all()
    assert np.allclose(y.data.sum(), 1.0)


# ==========================================================
# BACKWARD
# ==========================================================

def test_softmax_backward():
    x = tensor(
        np.random.randn(3, 4),
        requires_grad=True,
    )

    loss = x.softmax(axis=1).sum()

    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ==========================================================
# GRADCHECK
# ==========================================================

def test_softmax_gradcheck():
    x = tensor(
        np.random.randn(2, 3),
        requires_grad=True,
    )

    def fn(x):
        return x.softmax(axis=1).sum()

    assert gradcheck(fn, [x])


# ==========================================================
# EDGE CASES
# ==========================================================

def test_softmax_single_element():
    x = tensor([42.])

    y = x.softmax()

    assert np.allclose(y.data, np.array([1.0]))


def test_softmax_uniform_input():
    x = tensor([5., 5., 5., 5.])

    y = x.softmax()

    expected = np.full(4, 0.25)

    assert np.allclose(y.data, expected)


def test_softmax_batch_rows_sum_to_one():
    x = tensor(np.random.randn(8, 10))

    y = x.softmax(axis=1)

    expected = np.ones(8)

    assert np.allclose(
        y.data.sum(axis=1),
        expected,
    )


# ==========================================================
# REGRESSION TESTS
# ==========================================================

def test_softmax_no_nan():
    x = tensor(np.random.randn(16, 32))

    y = x.softmax(axis=1)

    assert not np.isnan(y.data).any()


def test_softmax_no_inf():
    x = tensor(np.random.randn(16, 32))

    y = x.softmax(axis=1)

    assert np.isfinite(y.data).all()