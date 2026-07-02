import numpy as np

from mini_torch.tensors import tensor
from mini_torch.nn.activations import Sigmoid
from mini_torch.autograd.engine import backward
from mini_torch.autograd.gradcheck import gradcheck


# ==========================================================
# Sigmoid Forward
# ==========================================================

def test_sigmoid_forward():

    x = tensor(np.array([-1.0, 0.0, 1.0]))

    sigmoid = Sigmoid()

    y = sigmoid(x)

    expected = np.array([
        0.26894142,
        0.5,
        0.73105858,
    ])

    assert np.allclose(y.data, expected)


# ==========================================================
# Sigmoid Backward
# ==========================================================

def test_sigmoid_backward():

    x = tensor(0.0, requires_grad=True)

    sigmoid = Sigmoid()

    y = sigmoid(x)

    backward(y)

    # σ(0) = 0.5
    # σ'(0) = 0.5 * (1 - 0.5) = 0.25
    assert np.allclose(x.grad, 0.25)


# ==========================================================
# Sigmoid Gradcheck
# ==========================================================

def test_sigmoid_gradcheck():

    np.random.seed(42)

    x = tensor(
        np.random.randn(5),
        requires_grad=True,
    )

    sigmoid = Sigmoid()

    def fn(inp):
        return sigmoid(inp).sum()

    assert gradcheck(fn, [x])


# ==========================================================
# Output Range
# ==========================================================

def test_sigmoid_output_range():

    np.random.seed(42)

    x = tensor(np.random.randn(100))

    sigmoid = Sigmoid()

    y = sigmoid(x)

    assert np.all(y.data > 0.0)
    assert np.all(y.data < 1.0)


# ==========================================================
# Extreme Values
# ==========================================================

def test_sigmoid_extreme_values():

    x = tensor(np.array([-100.0, 100.0]))

    sigmoid = Sigmoid()

    y = sigmoid(x)

    expected = np.array([
        0.0,
        1.0,
    ])

    assert np.allclose(y.data, expected, atol=1e-6)


# ==========================================================
# Shape Preservation
# ==========================================================

def test_sigmoid_shape():

    x = tensor(np.random.randn(8, 4))

    sigmoid = Sigmoid()

    y = sigmoid(x)

    assert y.shape == x.shape