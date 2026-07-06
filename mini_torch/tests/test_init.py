import numpy as np
import pytest

from mini_torch.nn.init import (
    calculate_fan_in_out,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
)


# ==========================================================
# Fan In / Fan Out
# ==========================================================

def test_calculate_fan_in_out():
    fan_in, fan_out = calculate_fan_in_out((4, 16))

    assert fan_in == 4
    assert fan_out == 16


def test_calculate_fan_in_out_square():
    fan_in, fan_out = calculate_fan_in_out((32, 32))

    assert fan_in == 32
    assert fan_out == 32


def test_calculate_fan_in_out_invalid_shape():
    with pytest.raises(ValueError):
        calculate_fan_in_out((10,))


# ==========================================================
# Xavier Uniform
# ==========================================================

def test_xavier_uniform_shape():
    weights = xavier_uniform((4, 16))

    assert weights.shape == (4, 16)


def test_xavier_uniform_range():
    shape = (4, 16)

    weights = xavier_uniform(shape)

    fan_in, fan_out = calculate_fan_in_out(shape)

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    assert np.all(weights >= -limit)
    assert np.all(weights <= limit)


def test_xavier_uniform_mean():
    weights = xavier_uniform((1000, 1000))

    assert abs(weights.mean()) < 0.01


# ==========================================================
# Xavier Normal
# ==========================================================

def test_xavier_normal_shape():
    weights = xavier_normal((8, 32))

    assert weights.shape == (8, 32)


def test_xavier_normal_mean():
    weights = xavier_normal((1000, 1000))

    assert abs(weights.mean()) < 0.01


def test_xavier_normal_std():
    shape = (1000, 1000)

    weights = xavier_normal(shape)

    fan_in, fan_out = calculate_fan_in_out(shape)

    expected_std = np.sqrt(
        2.0 / (fan_in + fan_out)
    )

    assert np.isclose(
        weights.std(),
        expected_std,
        atol=0.01,
    )


# ==========================================================
# He Uniform
# ==========================================================

def test_kaiming_uniform_shape():
    weights = kaiming_uniform((4, 16))

    assert weights.shape == (4, 16)


def test_kaiming_uniform_range():
    shape = (4, 16)

    weights = kaiming_uniform(shape)

    fan_in, _ = calculate_fan_in_out(shape)

    limit = np.sqrt(6.0 / fan_in)

    assert np.all(weights >= -limit)
    assert np.all(weights <= limit)


def test_kaiming_uniform_mean():
    weights = kaiming_uniform((1000, 1000))

    assert abs(weights.mean()) < 0.01


# ==========================================================
# Kaiming Normal
# ==========================================================

def test_kaiming_normal_shape():
    weights = kaiming_normal((16, 8))

    assert weights.shape == (16, 8)


def test_kaiming_normal_mean():
    weights = kaiming_normal((1000, 1000))

    assert abs(weights.mean()) < 0.01


def test_kaiming_normal_std():
    shape = (1000, 1000)

    weights = kaiming_normal(shape)

    fan_in, _ = calculate_fan_in_out(shape)

    expected_std = np.sqrt(
        2.0 / fan_in
    )

    assert np.isclose(
        weights.std(),
        expected_std,
        atol=0.01,
    )


# ==========================================================
# Randomness
# ==========================================================

def test_initializers_return_different_values():
    w1 = xavier_uniform((16, 16))
    w2 = xavier_uniform((16, 16))

    assert not np.array_equal(w1, w2)


# ==========================================================
# Dtype
# ==========================================================

def test_initializers_return_numpy_arrays():
    initializers = (
        xavier_uniform,
        xavier_normal,
        kaiming_uniform,
        kaiming_normal,
    )

    for initializer in initializers:

        weights = initializer((4, 4))

        assert isinstance(weights, np.ndarray)