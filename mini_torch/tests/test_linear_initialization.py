import numpy as np
import pytest

from mini_torch.nn.linear import Linear
from mini_torch.nn.init import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
)


# ==========================================================
# Default Initialization
# ==========================================================

def test_linear_default_initializer():
    layer = Linear(4, 16)

    assert layer.weight.data.shape == (4, 16)
    assert layer.bias.data.shape == (16,)


# ==========================================================
# Xavier Uniform
# ==========================================================

def test_linear_xavier_uniform():
    layer = Linear(
        4,
        16,
        initialization="xavier_uniform",
    )

    fan_in = 4
    fan_out = 16

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    assert np.all(layer.weight.data <= limit)
    assert np.all(layer.weight.data >= -limit)


# ==========================================================
# Xavier Normal
# ==========================================================

def test_linear_xavier_normal():
    layer = Linear(
        1000,
        1000,
        initialization="xavier_normal",
    )

    expected_std = np.sqrt(2.0 / (1000 + 1000))

    assert np.isclose(
        layer.weight.data.std(),
        expected_std,
        atol=0.01,
    )


# ==========================================================
# He Uniform
# ==========================================================

def test_linear_he_uniform():
    layer = Linear(
        4,
        16,
        initialization="kaiming_uniform",
    )

    limit = np.sqrt(6.0 / 4)

    assert np.all(layer.weight.data <= limit)
    assert np.all(layer.weight.data >= -limit)


# ==========================================================
# He Normal
# ==========================================================

def test_linear_he_normal():
    layer = Linear(
        1000,
        1000,
        initialization="kaiming_normal",
    )

    expected_std = np.sqrt(2.0 / 1000)

    assert np.isclose(
        layer.weight.data.std(),
        expected_std,
        atol=0.01,
    )


# ==========================================================
# Bias Initialization
# ==========================================================

def test_linear_bias_is_zero():
    layer = Linear(8, 4)

    assert np.all(layer.bias.data == 0)


# ==========================================================
# Disable Bias
# ==========================================================

def test_linear_without_bias():
    layer = Linear(
        4,
        8,
        bias=False,
    )

    assert layer.bias is None


# ==========================================================
# Unknown Initializer
# ==========================================================

def test_unknown_initializer():
    with pytest.raises(ValueError):
        Linear(
            4,
            8,
            initialization="unknown_initializer",
        )


# ==========================================================
# Forward Still Works
# ==========================================================

def test_linear_forward_after_initialization():
    from mini_torch.tensors import tensor

    x = tensor(np.random.randn(5, 4))

    layer = Linear(
        4,
        3,
        initialization="kaiming_uniform",
    )

    y = layer(x)

    assert y.shape == (5, 3)


# ==========================================================
# Initializers Produce Different Weights
# ==========================================================

def test_initializers_are_different():

    xavier = Linear(
        64,
        64,
        initialization="xavier_uniform",
    )

    he = Linear(
        64,
        64,
        initialization="kaiming_uniform",
    )

    assert not np.array_equal(
        xavier.weight.data,
        he.weight.data,
    )


# ==========================================================
# Parameter Registration
# ==========================================================

def test_linear_parameters_registered():
    layer = Linear(
        4,
        8,
        initialization="kaiming_uniform",
    )

    params = list(layer.parameters())

    assert len(params) == 2


# ==========================================================
# Weight Shape
# ==========================================================

@pytest.mark.parametrize(
    "in_features,out_features",
    [
        (1, 1),
        (4, 8),
        (8, 4),
        (32, 64),
        (64, 32),
    ],
)
def test_weight_shapes(in_features, out_features):

    layer = Linear(
        in_features,
        out_features,
        initialization="kaiming_uniform",
    )

    assert layer.weight.shape == (
        in_features,
        out_features,
    )


# ==========================================================
# Bias Shape
# ==========================================================

@pytest.mark.parametrize(
    "out_features",
    [1, 4, 8, 16, 32],
)
def test_bias_shapes(out_features):

    layer = Linear(
        8,
        out_features,
    )

    assert layer.bias.shape == (
        out_features,
    )