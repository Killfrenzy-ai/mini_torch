import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.gradcheck import gradcheck


def test_clip_forward():
    x = tensor([-2.0, 0.5, 4.0])

    y = x.clip(0.0, 1.0)

    expected = np.array([0.0, 0.5, 1.0])

    assert np.allclose(y.data, expected)


def test_clip_backward():
    x = tensor(
        [-2.0, 0.5, 4.0],
        requires_grad=True,
    )

    loss = x.clip(0.0, 1.0).sum()

    loss.backward()

    expected = np.array([0.0, 1.0, 0.0])

    assert np.allclose(x.grad, expected)


def test_clip_gradcheck():
    x = tensor(
        [0.2, 0.4, 0.8],
        requires_grad=True,
    )

    def fn(x):
        return x.clip(0.0, 1.0).sum()

    assert gradcheck(fn, [x])


def test_clip_extreme_values():
    x = tensor([-100.0, 100.0])

    y = x.clip(-1.0, 1.0)

    expected = np.array([-1.0, 1.0])

    assert np.allclose(y.data, expected)


def test_clip_shape():
    x = tensor(np.random.randn(4, 5, 6))

    y = x.clip(-0.5, 0.5)

    assert y.shape == x.shape