import numpy as np

from mini_torch.tensors import tensor


# ==========================================================
# FORWARD
# ==========================================================

def test_max_forward():
    x = tensor([1., 5., 2.])

    y = x.max()

    assert y.data == 5.


def test_max_axis0():
    x = tensor([
        [1., 5., 2.],
        [4., 3., 8.]
    ])

    y = x.max(axis=0)

    expected = np.array([4., 5., 8.])

    assert np.allclose(y.data, expected)


def test_max_axis1():
    x = tensor([
        [1., 5., 2.],
        [4., 3., 8.]
    ])

    y = x.max(axis=1)

    expected = np.array([5., 8.])

    assert np.allclose(y.data, expected)


def test_max_negative_axis():
    x = tensor([
        [1., 5., 2.],
        [4., 3., 8.]
    ])

    y = x.max(axis=-1)

    expected = np.array([5., 8.])

    assert np.allclose(y.data, expected)


def test_max_keepdims():
    x = tensor([
        [1., 5., 2.],
        [4., 3., 8.]
    ])

    y = x.max(axis=1, keepdims=True)

    expected = np.array([
        [5.],
        [8.]
    ])

    assert np.allclose(y.data, expected)
    assert y.shape == (2, 1)


# ==========================================================
# BACKWARD
# ==========================================================

def test_max_backward():
    x = tensor(
        [1., 5., 2.],
        requires_grad=True,
    )

    loss = x.max()

    loss.backward()

    expected = np.array([0., 1., 0.])

    assert np.allclose(x.grad, expected)


def test_max_axis_backward():
    x = tensor(
        [
            [1., 5., 2.],
            [4., 3., 8.]
        ],
        requires_grad=True,
    )

    loss = x.max(axis=1).sum()

    loss.backward()

    expected = np.array([
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    assert np.allclose(x.grad, expected)


def test_max_keepdims_backward():
    x = tensor(
        [
            [1., 5., 2.],
            [4., 3., 8.]
        ],
        requires_grad=True,
    )

    loss = x.max(axis=1, keepdims=True).sum()

    loss.backward()

    expected = np.array([
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    assert np.allclose(x.grad, expected)


# ==========================================================
# EDGE CASES
# ==========================================================

def test_max_single_element():
    x = tensor([42.], requires_grad=True)

    y = x.max()

    y.backward()

    assert y.data == 42.
    assert np.allclose(x.grad, np.array([1.]))


def test_max_shape():
    x = tensor(np.random.randn(4, 5, 6))

    y = x.max(axis=2)

    assert y.shape == (4, 5)


def test_max_multiple_maxima():
    """
    Educational implementation:
    Every maximum receives the incoming gradient.
    """

    x = tensor(
        [5., 5., 2.],
        requires_grad=True,
    )

    y = x.max()

    y.backward()

    expected = np.array([1., 1., 0.])

    assert np.allclose(x.grad, expected)