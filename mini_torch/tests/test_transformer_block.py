import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.transformer_block import TransformerBlock

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam
from mini_torch.nn.functional import casual_mask


# ==========================================================
# Construction
# ==========================================================

def test_construction():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    assert block.attention.embed_dim == 32
    assert block.attention.num_heads == 4


def test_custom_hidden_dim():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        ff_hidden_dim=96,
    )

    assert block.feedforward.hidden_dim == 96


def test_parameter_registration():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    params = list(block.parameters())

    assert len(params) > 0


# ==========================================================
# Forward
# ==========================================================

def test_output_shape():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 10, 32)
    )

    y, weights = block(x)

    assert y.shape == (2, 10, 32)
    assert weights.shape == (2, 4, 10, 10)


def test_single_token():

    block = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 1, 16)
    )

    y, weights = block(x)

    assert y.shape == (2, 1, 16)
    assert weights.shape == (2, 4, 1, 1)


def test_batch_size_one():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=8,
    )

    x = tensor(
        np.random.randn(1, 6, 32)
    )

    y, _ = block(x)

    assert y.shape == (1, 6, 32)


# ==========================================================
# Attention
# ==========================================================

def test_attention_rows_sum_to_one():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    _, weights = block(x)

    row_sum = weights.data.sum(axis=-1)

    assert np.allclose(
        row_sum,
        1,
        atol=1e-6,
    )


def test_casual_mask():

    block = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(1, 5, 16)
    )

    mask = casual_mask(5)

    _, weights = block(
        x,
        mask=mask,
    )

    upper = np.triu_indices(5, 1)

    masked = weights.data[0, :, upper[0], upper[1]]

    assert np.allclose(
        masked,
        0,
        atol=1e-6,
    )


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 6, 32),
        requires_grad=True,
    )

    y, _ = block(x)

    loss = y.sum()

    backward(loss)

    assert x.grad is not None


def test_parameter_gradients():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 6, 32),
        requires_grad=True,
    )

    y, _ = block(x)

    backward(y.sum())

    for p in block.parameters():
        assert p.grad is not None


# ==========================================================
# Dropout
# ==========================================================

def test_eval_mode():

    np.random.seed(42)

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        dropout=0.5,
    )

    block.eval()

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    y1, _ = block(x)
    y2, _ = block(x)

    assert np.allclose(
        y1.data,
        y2.data,
    )


def test_train_mode():

    np.random.seed(42)

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        dropout=0.5,
    )

    block.train()

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    y1, _ = block(x)
    y2, _ = block(x)

    assert not np.allclose(
        y1.data,
        y2.data,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict():

    block = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    state = block.state_dict()

    assert "attention.q_proj.weight" in state
    assert "attention.k_proj.weight" in state
    assert "attention.v_proj.weight" in state
    assert "attention.out_proj.weight" in state

    assert "feedforward.fc1.weight" in state
    assert "feedforward.fc2.weight" in state

    assert "norm1.gamma" in state
    assert "norm2.gamma" in state


def test_load_state_dict():

    block1 = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    state = block1.state_dict()

    block2 = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    block2.load_state_dict(state)

    for p1, p2 in zip(
        block1.parameters(),
        block2.parameters(),
    ):
        assert np.array_equal(
            p1.data,
            p2.data,
        )


# ==========================================================
# Optimizers
# ==========================================================

def test_sgd_step():

    block = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    optimizer = SGD(
        block.parameters(),
        lr=0.1,
    )

    x = tensor(
        np.random.randn(2, 6, 16),
        requires_grad=True,
    )

    y, _ = block(x)

    backward(y.sum())

    before = block.attention.q_proj.weight.data.copy()

    optimizer.step()

    after = block.attention.q_proj.weight.data

    assert not np.array_equal(
        before,
        after,
    )


def test_adam_step():

    block = TransformerBlock(
        embed_dim=16,
        num_heads=4,
    )

    optimizer = Adam(
        block.parameters(),
        lr=0.01,
    )

    x = tensor(
        np.random.randn(2, 6, 16),
        requires_grad=True,
    )

    y, _ = block(x)

    backward(y.sum())

    before = block.attention.q_proj.weight.data.copy()

    optimizer.step()

    after = block.attention.q_proj.weight.data

    assert not np.array_equal(
        before,
        after,
    )


# ==========================================================
# Regression
# ==========================================================

def test_no_nan():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    y, weights = block(x)

    assert not np.isnan(y.data).any()
    assert not np.isnan(weights.data).any()


def test_no_inf():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    y, weights = block(x)

    assert not np.isinf(y.data).any()
    assert not np.isinf(weights.data).any()


def test_dtype():

    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
    )

    x = tensor(
        np.random.randn(2, 8, 32)
    )

    y, _ = block(x)

    assert y.dtype == x.dtype