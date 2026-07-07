import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.position import PositionalEmbedding

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_construction():

    layer = PositionalEmbedding(
        max_length=128,
        embedding_dim=64,
    )

    assert layer.max_length == 128
    assert layer.embedding_dim == 64


def test_embedding_shape():

    layer = PositionalEmbedding(
        max_length=50,
        embedding_dim=32,
    )

    assert layer.embedding.weight.shape == (50, 32)


def test_parameter_registration():

    layer = PositionalEmbedding(
        max_length=20,
        embedding_dim=16,
    )

    params = list(layer.parameters())

    assert len(params) == 1

    assert params[0] is layer.embedding.weight


# ==========================================================
# Forward
# ==========================================================

def test_output_shape():

    layer = PositionalEmbedding(
        max_length=100,
        embedding_dim=32,
    )

    tokens = tensor(
        np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
    )

    out = layer(tokens)

    assert out.shape == (2, 4, 32)


def test_single_sequence():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=8,
    )

    tokens = tensor(
        np.array([[1, 2, 3]])
    )

    out = layer(tokens)

    assert out.shape == (1, 3, 8)


def test_sequence_length_one():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=4,
    )

    tokens = tensor(
        np.array([[7]])
    )

    out = layer(tokens)

    assert out.shape == (1, 1, 4)


# ==========================================================
# Position Generation
# ==========================================================

def test_same_positions_same_embeddings():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=6,
    )

    tokens = tensor(
        np.array([
            [4, 5, 6],
            [9, 8, 7],
        ])
    )

    out = layer(tokens)

    assert np.allclose(
        out.data[0],
        out.data[1],
    )


def test_position_zero_matches_embedding():

    layer = PositionalEmbedding(
        max_length=20,
        embedding_dim=5,
    )

    tokens = tensor(
        np.array([[5, 6, 7]])
    )

    out = layer(tokens)

    expected = layer.embedding.weight.data[0]

    assert np.allclose(
        out.data[0, 0],
        expected,
    )


def test_last_position_matches_embedding():

    layer = PositionalEmbedding(
        max_length=20,
        embedding_dim=5,
    )

    tokens = tensor(
        np.array([[5, 6, 7, 8]])
    )

    out = layer(tokens)

    expected = layer.embedding.weight.data[3]

    assert np.allclose(
        out.data[0, 3],
        expected,
    )


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=4,
    )

    tokens = tensor(
        np.array([
            [1, 2, 3],
        ])
    )

    out = layer(tokens)

    loss = out.sum()

    backward(loss)

    grad = layer.embedding.weight.grad

    assert grad is not None

    assert np.any(grad != 0)


def test_only_used_positions_receive_gradients():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=3,
    )

    tokens = tensor(
        np.array([
            [1, 2],
        ])
    )

    out = layer(tokens)

    loss = out.sum()

    backward(loss)

    grad = layer.embedding.weight.grad

    assert np.all(grad[0] == 1)
    assert np.all(grad[1] == 1)

    assert np.all(grad[2:] == 0)


def test_repeated_batches_accumulate_gradients():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=2,
    )

    tokens = tensor(
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
    )

    out = layer(tokens)

    loss = out.sum()

    backward(loss)

    grad = layer.embedding.weight.grad

    assert np.all(grad[0] == 2)
    assert np.all(grad[1] == 2)
    assert np.all(grad[2] == 2)


# ==========================================================
# Optimizers
# ==========================================================

def test_sgd_step():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=4,
    )

    optimizer = SGD(
        layer.parameters(),
        lr=0.1,
    )

    tokens = tensor(
        np.array([[1, 2, 3]])
    )

    loss = layer(tokens).sum()

    backward(loss)

    before = layer.embedding.weight.data.copy()

    optimizer.step()

    after = layer.embedding.weight.data

    assert not np.array_equal(
        before,
        after,
    )


def test_adam_step():

    layer = PositionalEmbedding(
        max_length=10,
        embedding_dim=4,
    )

    optimizer = Adam(
        layer.parameters(),
        lr=0.01,
    )

    tokens = tensor(
        np.array([[1, 2, 3]])
    )

    loss = layer(tokens).sum()

    backward(loss)

    before = layer.embedding.weight.data.copy()

    optimizer.step()

    after = layer.embedding.weight.data

    assert not np.array_equal(
        before,
        after,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict():

    layer = PositionalEmbedding(
        max_length=15,
        embedding_dim=6,
    )

    state = layer.state_dict()

    assert "embedding.weight" in state


def test_load_state_dict():

    layer1 = PositionalEmbedding(
        max_length=15,
        embedding_dim=6,
    )

    state = layer1.state_dict()

    layer2 = PositionalEmbedding(
        max_length=15,
        embedding_dim=6,
    )

    layer2.load_state_dict(state)

    assert np.array_equal(
        layer1.embedding.weight.data,
        layer2.embedding.weight.data,
    )


# ==========================================================
# Regression
# ==========================================================

def test_no_nan():

    layer = PositionalEmbedding(
        max_length=50,
        embedding_dim=16,
    )

    tokens = tensor(
        np.random.randint(
            0,
            50,
            size=(8, 10),
        )
    )

    out = layer(tokens)

    assert not np.isnan(out.data).any()


def test_no_inf():

    layer = PositionalEmbedding(
        max_length=50,
        embedding_dim=16,
    )

    tokens = tensor(
        np.random.randint(
            0,
            50,
            size=(8, 10),
        )
    )

    out = layer(tokens)

    assert not np.isinf(out.data).any()


def test_output_dtype():

    layer = PositionalEmbedding(
        max_length=20,
        embedding_dim=8,
    )

    tokens = tensor(
        np.array([[1, 2, 3]])
    )

    out = layer(tokens)

    assert out.dtype == layer.embedding.weight.dtype