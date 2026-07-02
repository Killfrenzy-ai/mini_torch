import numpy as np

from mini_torch.tensors import tensor
from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.sequential import Sequential

from mini_torch.autograd.engine import backward
from mini_torch.autograd.gradcheck import gradcheck


# ==========================================================
# ReLU
# ==========================================================

def test_relu_forward():

    x = tensor(np.array([-2., -1., 0., 1., 2.]))

    relu = ReLU()

    y = relu(x)

    expected = np.array([0., 0., 0., 1., 2.])

    assert np.allclose(y.data, expected)


def test_relu_backward():

    x = tensor(
        np.array([-2., -1., 0., 1., 2.]),
        requires_grad=True,
    )

    relu = ReLU()

    loss = relu(x).sum()

    backward(loss)

    expected = np.array([0., 0., 0., 1., 1.])

    assert np.allclose(x.grad, expected)


def test_relu_gradcheck():

    x = tensor(
        np.random.randn(5),
        requires_grad=True,
    )

    relu = ReLU()

    def fn(inp):
        return relu(inp).sum()

    assert gradcheck(fn, [x])


# ==========================================================
# Sequential
# ==========================================================

def test_sequential_construction():

    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 2),
    )

    assert len(model.layers) == 3


def test_sequential_forward():

    np.random.seed(42)

    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 2),
    )

    x = tensor(np.random.randn(8, 3))

    y = model(x)

    assert y.shape == (8, 2)


def test_sequential_parameters():

    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 2),
    )

    params = list(model.parameters())

    # weight1, bias1, weight2, bias2
    assert len(params) == 4

    assert params[0] is model.layer0.weight
    assert params[1] is model.layer0.bias
    assert params[2] is model.layer2.weight
    assert params[3] is model.layer2.bias


def test_sequential_backward():

    np.random.seed(42)

    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 2),
    )

    x = tensor(
        np.random.randn(10, 3),
        requires_grad=True,
    )

    loss = model(x).sum()

    backward(loss)

    assert x.grad is not None

    for param in model.parameters():
        assert param.grad is not None


def test_sequential_gradcheck():

    np.random.seed(42)

    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    x = tensor(
        np.random.randn(4, 2),
        requires_grad=True,
    )

    def fn(inp):
        return model(inp).sum()

    assert gradcheck(fn, [x])