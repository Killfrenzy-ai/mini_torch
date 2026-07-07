import numpy as np

from mini_torch.tensors import tensor
from mini_torch.parameter import Parameter

from mini_torch.nn.embedding import Embedding

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam

from mini_torch.autograd.engine import backward


# ==========================================================
# Construction
# ==========================================================

def test_embedding_construction():

    embedding = Embedding(100, 64)

    assert embedding.num_embeddings == 100
    assert embedding.embedding_dim == 64


def test_embedding_weight_shape():

    embedding = Embedding(50, 32)

    assert embedding.weight.shape == (50, 32)


def test_embedding_parameter_registration():

    embedding = Embedding(20, 16)

    params = list(embedding.parameters())

    assert len(params) == 1
    assert params[0] is embedding.weight


# ==========================================================
# Forward
# ==========================================================

def test_single_token_lookup():

    embedding = Embedding(10, 4)

    x = tensor(np.array([3]))

    y = embedding(x)

    assert y.shape == (1, 4)

    assert np.array_equal(
        y.data,
        embedding.weight.data[[3]],
    )


def test_multiple_token_lookup():

    embedding = Embedding(20, 8)

    x = tensor(np.array([1, 5, 9]))

    y = embedding(x)

    assert y.shape == (3, 8)

    assert np.array_equal(
        y.data,
        embedding.weight.data[[1, 5, 9]],
    )


def test_batch_lookup():

    embedding = Embedding(30, 16)

    x = tensor(
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
    )

    y = embedding(x)

    assert y.shape == (2, 3, 16)


def test_embedding_preserves_values():

    embedding = Embedding(5, 3)

    embedding.weight.data = np.array(
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
            [10., 11., 12.],
            [13., 14., 15.],
        ]
    )

    x = tensor(np.array([2, 4]))

    y = embedding(x)

    expected = np.array(
        [
            [7., 8., 9.],
            [13., 14., 15.],
        ]
    )

    assert np.array_equal(
        y.data,
        expected,
    )


# ==========================================================
# Backward
# ==========================================================

def test_single_token_backward():

    embedding = Embedding(6, 3)

    x = tensor(np.array([2]))

    out = embedding(x)

    loss = out.sum()

    backward(loss)

    expected = np.zeros_like(
        embedding.weight.data
    )

    expected[2] = 1

    assert np.array_equal(
        embedding.weight.grad,
        expected,
    )


def test_multiple_token_backward():

    embedding = Embedding(6, 2)

    x = tensor(np.array([1, 4]))

    out = embedding(x)

    loss = out.sum()

    backward(loss)

    expected = np.zeros_like(
        embedding.weight.data
    )

    expected[1] = 1
    expected[4] = 1

    assert np.array_equal(
        embedding.weight.grad,
        expected,
    )


def test_repeated_token_backward():

    embedding = Embedding(5, 4)

    x = tensor(np.array([3, 3, 3]))

    out = embedding(x)

    loss = out.sum()

    backward(loss)

    expected = np.zeros_like(
        embedding.weight.data
    )

    expected[3] = 3

    assert np.array_equal(
        embedding.weight.grad,
        expected,
    )


def test_unused_embeddings_receive_zero_gradient():

    embedding = Embedding(8, 5)

    x = tensor(np.array([2]))

    out = embedding(x)

    loss = out.sum()

    backward(loss)

    mask = np.ones(8, dtype=bool)

    mask[2] = False

    assert np.all(
        embedding.weight.grad[mask] == 0
    )


# ==========================================================
# Optimizer Compatibility
# ==========================================================

def test_embedding_with_sgd():

    embedding = Embedding(10, 4)

    optimizer = SGD(
        embedding.parameters(),
        lr=0.1,
    )

    x = tensor(np.array([1, 2]))

    loss = embedding(x).sum()

    backward(loss)

    before = embedding.weight.data.copy()

    optimizer.step()

    after = embedding.weight.data

    assert not np.array_equal(
        before,
        after,
    )


def test_embedding_with_adam():

    embedding = Embedding(10, 4)

    optimizer = Adam(
        embedding.parameters(),
        lr=0.01,
    )

    x = tensor(np.array([1, 2]))

    loss = embedding(x).sum()

    backward(loss)

    before = embedding.weight.data.copy()

    optimizer.step()

    after = embedding.weight.data

    assert not np.array_equal(
        before,
        after,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_embedding_state_dict():

    embedding = Embedding(12, 6)

    state = embedding.state_dict()

    assert "weight" in state

    assert state["weight"].shape == (
        12,
        6,
    )


def test_embedding_load_state_dict():

    embedding1 = Embedding(8, 3)

    state = embedding1.state_dict()

    embedding2 = Embedding(8, 3)

    embedding2.load_state_dict(state)

    assert np.array_equal(
        embedding1.weight.data,
        embedding2.weight.data,
    )


# ==========================================================
# Shape Regression
# ==========================================================

def test_embedding_output_shape():

    embedding = Embedding(100, 32)

    x = tensor(
        np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
    )

    y = embedding(x)

    assert y.shape == (
        3,
        2,
        32,
    )


# ==========================================================
# Lookup Consistency
# ==========================================================

def test_same_token_same_embedding():

    embedding = Embedding(20, 5)

    x = tensor(np.array([7, 7]))

    y = embedding(x)

    assert np.array_equal(
        y.data[0],
        y.data[1],
    )