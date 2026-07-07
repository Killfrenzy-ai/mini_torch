import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.feedforward import FeedForward

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_default_hidden_dim():

    ffn = FeedForward(embed_dim=32)

    assert ffn.hidden_dim == 128


def test_custom_hidden_dim():

    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=64,
    )

    assert ffn.hidden_dim == 64


def test_parameter_count():

    ffn = FeedForward(
        embed_dim=16,
        hidden_dim=32,
    )

    params = list(ffn.parameters())

    assert len(params) == 4


# ==========================================================
# Forward
# ==========================================================

def test_output_shape():

    ffn = FeedForward(32)

    x = tensor(
        np.random.randn(8, 20, 32)
    )

    y = ffn(x)

    assert y.shape == (8, 20, 32)


def test_single_token():

    ffn = FeedForward(16)

    x = tensor(
        np.random.randn(2, 1, 16)
    )

    y = ffn(x)

    assert y.shape == (2, 1, 16)


def test_batch_size_one():

    ffn = FeedForward(16)

    x = tensor(
        np.random.randn(1, 10, 16)
    )

    y = ffn(x)

    assert y.shape == (1, 10, 16)


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    ffn = FeedForward(16)

    x = tensor(
        np.random.randn(4, 6, 16),
        requires_grad=True,
    )

    y = ffn(x)

    loss = y.sum()

    backward(loss)

    assert x.grad is not None


def test_parameter_gradients():

    ffn = FeedForward(16)

    x = tensor(
        np.random.randn(2, 5, 16),
        requires_grad=True,
    )

    loss = ffn(x).sum()

    backward(loss)

    for p in ffn.parameters():
        assert p.grad is not None


# ==========================================================
# Dropout
# ==========================================================

def test_eval_mode():

    np.random.seed(42)

    ffn = FeedForward(
        embed_dim=16,
        dropout=0.5,
    )

    ffn.eval()

    x = tensor(
        np.random.randn(2, 5, 16)
    )

    y1 = ffn(x)
    y2 = ffn(x)

    assert np.allclose(
        y1.data,
        y2.data,
    )


def test_train_mode():

    np.random.seed(42)

    ffn = FeedForward(
        embed_dim=16,
        dropout=0.5,
    )

    ffn.train()

    x = tensor(
        np.random.randn(2, 5, 16)
    )

    y1 = ffn(x)
    y2 = ffn(x)

    assert not np.allclose(
        y1.data,
        y2.data,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict():

    ffn = FeedForward(16)

    state = ffn.state_dict()

    assert "fc1.weight" in state
    assert "fc1.bias" in state
    assert "fc2.weight" in state
    assert "fc2.bias" in state


def test_load_state_dict():

    model1 = FeedForward(16)

    state = model1.state_dict()

    model2 = FeedForward(16)

    model2.load_state_dict(state)

    for p1, p2 in zip(
        model1.parameters(),
        model2.parameters(),
    ):
        assert np.array_equal(
            p1.data,
            p2.data,
        )


# ==========================================================
# Optimizers
# ==========================================================

def test_sgd_step():

    model = FeedForward(16)

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    x = tensor(
        np.random.randn(2, 5, 16),
        requires_grad=True,
    )

    loss = model(x).sum()

    backward(loss)

    before = model.fc1.weight.data.copy()

    optimizer.step()

    after = model.fc1.weight.data

    assert not np.array_equal(
        before,
        after,
    )


def test_adam_step():

    model = FeedForward(16)

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )

    x = tensor(
        np.random.randn(2, 5, 16),
        requires_grad=True,
    )

    loss = model(x).sum()

    backward(loss)

    before = model.fc1.weight.data.copy()

    optimizer.step()

    after = model.fc1.weight.data

    assert not np.array_equal(
        before,
        after,
    )


# ==========================================================
# Regression
# ==========================================================

def test_no_nan():

    model = FeedForward(32)

    x = tensor(
        np.random.randn(8, 20, 32)
    )

    y = model(x)

    assert not np.isnan(y.data).any()


def test_no_inf():

    model = FeedForward(32)

    x = tensor(
        np.random.randn(8, 20, 32)
    )

    y = model(x)

    assert not np.isinf(y.data).any()


def test_dtype():

    model = FeedForward(32)

    x = tensor(
        np.random.randn(4, 5, 32)
    )

    y = model(x)

    assert y.dtype == x.dtype