import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.layernorm import LayerNorm

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_layernorm_construction():

    layer = LayerNorm(8)

    assert layer.normalized_shape == 8
    assert layer.eps == 1e-5


def test_gamma_shape():

    layer = LayerNorm(16)

    assert layer.gamma.shape == (16,)


def test_beta_shape():

    layer = LayerNorm(16)

    assert layer.beta.shape == (16,)


def test_parameter_registration():

    layer = LayerNorm(32)

    params = list(layer.parameters())

    assert len(params) == 2
    assert params[0] is layer.gamma
    assert params[1] is layer.beta


# ==========================================================
# Forward
# ==========================================================

def test_forward_shape():

    layer = LayerNorm(4)

    x = tensor(
        np.random.randn(8, 4)
    )

    y = layer(x)

    assert y.shape == (8, 4)


def test_forward_batch_shape():

    layer = LayerNorm(16)

    x = tensor(
        np.random.randn(4, 10, 16)
    )

    y = layer(x)

    assert y.shape == (4, 10, 16)


def test_mean_is_zero():

    np.random.seed(42)

    layer = LayerNorm(8)

    x = tensor(
        np.random.randn(32, 8)
    )

    y = layer(x)

    mean = y.data.mean(axis=-1)

    assert np.allclose(
        mean,
        0,
        atol=1e-6,
    )


def test_variance_is_one():

    np.random.seed(42)

    layer = LayerNorm(8)

    x = tensor(
        np.random.randn(32, 8)
    )

    y = layer(x)

    variance = y.data.var(axis=-1)

    assert np.allclose(
        variance,
        1,
        rtol=1e-4,
    )


# ==========================================================
# Gamma / Beta
# ==========================================================

def test_gamma_scales_output():

    layer = LayerNorm(4)

    layer.gamma.data *= 2

    x = tensor(
        np.random.randn(6, 4)
    )

    y = layer(x)

    variance = y.data.var(axis=-1)

    assert np.allclose(
        variance,
        4,
        rtol=1e-4,
    )


def test_beta_shifts_output():

    layer = LayerNorm(4)

    layer.beta.data[:] = 3

    x = tensor(
        np.random.randn(10, 4)
    )

    y = layer(x)

    mean = y.data.mean(axis=-1)

    assert np.allclose(
        mean,
        3,
        atol=1e-5,
    )


# ==========================================================
# Numerical Stability
# ==========================================================

def test_constant_input():

    layer = LayerNorm(4)

    x = tensor(
        np.ones((5, 4))
    )

    y = layer(x)

    assert not np.isnan(y.data).any()
    assert not np.isinf(y.data).any()


def test_small_variance():

    layer = LayerNorm(8)

    x = tensor(
        np.full((6, 8), 5.0)
        + 1e-8
    )

    y = layer(x)

    assert not np.isnan(y.data).any()
    assert not np.isinf(y.data).any()


def test_custom_eps():

    layer = LayerNorm(
        4,
        eps=1e-3,
    )

    assert layer.eps == 1e-3


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    layer = LayerNorm(4)

    x = tensor(
        np.random.randn(3, 4),
        requires_grad=True,
    )

    y = layer(x)

    loss = y.sum()

    backward(loss)

    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None


# ==========================================================
# Optimizers
# ==========================================================

def test_sgd_step():

    layer = LayerNorm(4)

    optimizer = SGD(
        layer.parameters(),
        lr=0.1,
    )

    x = tensor(
        np.random.randn(5, 4),
        requires_grad=True,
    )

    loss = layer(x).sum()

    backward(loss)

    before = layer.gamma.data.copy()

    optimizer.step()

    assert not np.array_equal(
        before,
        layer.gamma.data,
    )


def test_adam_step():

    layer = LayerNorm(4)

    optimizer = Adam(
        layer.parameters(),
        lr=0.01,
    )

    x = tensor(
        np.random.randn(5, 4),
        requires_grad=True,
    )

    loss = layer(x).sum()

    backward(loss)

    before = layer.gamma.data.copy()

    optimizer.step()

    assert not np.array_equal(
        before,
        layer.gamma.data,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict():

    layer = LayerNorm(8)

    state = layer.state_dict()

    assert "gamma" in state
    assert "beta" in state


def test_load_state_dict():

    layer1 = LayerNorm(8)

    state = layer1.state_dict()

    layer2 = LayerNorm(8)

    layer2.load_state_dict(state)

    assert np.array_equal(
        layer1.gamma.data,
        layer2.gamma.data,
    )

    assert np.array_equal(
        layer1.beta.data,
        layer2.beta.data,
    )


# ==========================================================
# Regression
# ==========================================================

def test_output_contains_no_nan():

    layer = LayerNorm(32)

    x = tensor(
        np.random.randn(64, 32)
    )

    y = layer(x)

    assert not np.isnan(y.data).any()


def test_output_contains_no_inf():

    layer = LayerNorm(32)

    x = tensor(
        np.random.randn(64, 32)
    )

    y = layer(x)

    assert not np.isinf(y.data).any()