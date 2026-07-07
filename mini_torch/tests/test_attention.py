import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.attention import ScaledDotProductAttention

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_attention_construction():

    attention = ScaledDotProductAttention()

    assert attention.softmax.axis == -1


# ==========================================================
# Forward
# ==========================================================

def test_output_shape():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 5, 8))
    k = tensor(np.random.randn(2, 5, 8))
    v = tensor(np.random.randn(2, 5, 8))

    output, weights = attention(q, k, v)

    assert output.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 5)


def test_single_batch():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(1, 4, 16))
    k = tensor(np.random.randn(1, 4, 16))
    v = tensor(np.random.randn(1, 4, 16))

    output, weights = attention(q, k, v)

    assert output.shape == (1, 4, 16)
    assert weights.shape == (1, 4, 4)


def test_single_token():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 1, 8))
    k = tensor(np.random.randn(2, 1, 8))
    v = tensor(np.random.randn(2, 1, 8))

    output, weights = attention(q, k, v)

    assert output.shape == (2, 1, 8)
    assert weights.shape == (2, 1, 1)


# ==========================================================
# Attention Weights
# ==========================================================

def test_attention_rows_sum_to_one():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(3, 6, 8))
    k = tensor(np.random.randn(3, 6, 8))
    v = tensor(np.random.randn(3, 6, 8))

    _, weights = attention(q, k, v)

    row_sums = weights.data.sum(axis=-1)

    assert np.allclose(
        row_sums,
        1,
        atol=1e-6,
    )


def test_attention_weights_non_negative():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 4, 8))
    k = tensor(np.random.randn(2, 4, 8))
    v = tensor(np.random.randn(2, 4, 8))

    _, weights = attention(q, k, v)

    assert np.all(weights.data >= 0)


def test_attention_no_nan():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 5, 8))
    k = tensor(np.random.randn(2, 5, 8))
    v = tensor(np.random.randn(2, 5, 8))

    output, weights = attention(q, k, v)

    assert not np.isnan(output.data).any()
    assert not np.isnan(weights.data).any()


def test_attention_no_inf():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 5, 8))
    k = tensor(np.random.randn(2, 5, 8))
    v = tensor(np.random.randn(2, 5, 8))

    output, weights = attention(q, k, v)

    assert not np.isinf(output.data).any()
    assert not np.isinf(weights.data).any()


# ==========================================================
# Backward
# ==========================================================

def test_backward_query():

    attention = ScaledDotProductAttention()

    q = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    k = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    v = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    output, _ = attention(q, k, v)

    loss = output.sum()

    backward(loss)

    assert q.grad is not None


def test_backward_key():

    attention = ScaledDotProductAttention()

    q = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    k = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    v = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    output, _ = attention(q, k, v)

    loss = output.sum()

    backward(loss)

    assert k.grad is not None


def test_backward_value():

    attention = ScaledDotProductAttention()

    q = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    k = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    v = tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    output, _ = attention(q, k, v)

    loss = output.sum()

    backward(loss)

    assert v.grad is not None


# ==========================================================
# Numerical Stability
# ==========================================================

def test_large_values():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 4, 8) * 100)
    k = tensor(np.random.randn(2, 4, 8) * 100)
    v = tensor(np.random.randn(2, 4, 8))

    output, weights = attention(q, k, v)

    assert not np.isnan(output.data).any()
    assert not np.isnan(weights.data).any()


def test_small_values():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 4, 8) * 1e-6)
    k = tensor(np.random.randn(2, 4, 8) * 1e-6)
    v = tensor(np.random.randn(2, 4, 8))

    output, weights = attention(q, k, v)

    assert not np.isnan(output.data).any()
    assert not np.isnan(weights.data).any()


# ==========================================================
# Optimizer Compatibility
# ==========================================================

def test_sgd_compatibility():

    attention = ScaledDotProductAttention()

    optimizer = SGD([], lr=0.1)

    optimizer.step()


def test_adam_compatibility():

    attention = ScaledDotProductAttention()

    optimizer = Adam([], lr=0.01)

    optimizer.step()


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict_empty():

    attention = ScaledDotProductAttention()

    state = attention.state_dict()

    assert state == {}


def test_load_state_dict():

    attention = ScaledDotProductAttention()

    attention.load_state_dict({})


# ==========================================================
# Regression
# ==========================================================

def test_output_dtype():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 3, 8))
    k = tensor(np.random.randn(2, 3, 8))
    v = tensor(np.random.randn(2, 3, 8))

    output, _ = attention(q, k, v)

    assert output.dtype == q.dtype


def test_weights_dtype():

    attention = ScaledDotProductAttention()

    q = tensor(np.random.randn(2, 3, 8))
    k = tensor(np.random.randn(2, 3, 8))
    v = tensor(np.random.randn(2, 3, 8))

    _, weights = attention(q, k, v)

    assert weights.dtype == q.dtype

