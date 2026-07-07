import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.gpt import GPT

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_construction():

    model = GPT(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
    )

    assert model.vocab_size == 100
    assert model.embed_dim == 32
    assert model.num_layers == 2


def test_block_count():

    model = GPT(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=6,
        max_seq_len=16,
    )

    assert len(model.blocks) == 6


# ==========================================================
# Forward
# ==========================================================

def test_output_shape():

    model = GPT(
        vocab_size=50,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=10,
    )

    tokens = tensor(
        np.random.randint(
            0,
            50,
            size=(4, 10),
        )
    )

    logits = model(tokens)

    assert logits.shape == (4, 10, 50)


def test_batch_size_one():

    model = GPT(
        vocab_size=30,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    tokens = tensor(
        np.random.randint(
            0,
            30,
            size=(1, 8),
        )
    )

    logits = model(tokens)

    assert logits.shape == (1, 8, 30)


def test_single_token():

    model = GPT(
        vocab_size=25,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=1,
    )

    tokens = tensor(
        np.array([[5]])
    )

    logits = model(tokens)

    assert logits.shape == (1, 1, 25)


# ==========================================================
# Embeddings
# ==========================================================

def test_position_embedding_added():

    model = GPT(
        vocab_size=40,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=8,
    )

    tokens = tensor(
        np.random.randint(
            0,
            40,
            size=(2, 8),
        )
    )

    logits = model(tokens)

    assert logits.shape[-1] == 40


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    model = GPT(
        vocab_size=50,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    tokens = tensor(
        np.random.randint(
            0,
            50,
            size=(2, 8),
        ),
        requires_grad=False,
    )

    logits = model(tokens)

    loss = logits.sum()

    backward(loss)

    grads = [
        p.grad
        for p in model.parameters()
    ]

    assert all(
        g is not None
        for g in grads
    )


# ==========================================================
# Optimizers
# ==========================================================

def test_sgd_step():

    model = GPT(
        vocab_size=40,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    tokens = tensor(
        np.random.randint(
            0,
            40,
            size=(2, 8),
        )
    )

    logits = model(tokens)

    loss = logits.sum()

    backward(loss)

    before = model.lm_head.weight.data.copy()

    optimizer.step()

    after = model.lm_head.weight.data

    assert not np.array_equal(
        before,
        after,
    )


def test_adam_step():

    model = GPT(
        vocab_size=40,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )

    tokens = tensor(
        np.random.randint(
            0,
            40,
            size=(2, 8),
        )
    )

    logits = model(tokens)

    loss = logits.sum()

    backward(loss)

    before = model.lm_head.weight.data.copy()

    optimizer.step()

    after = model.lm_head.weight.data

    assert not np.array_equal(
        before,
        after,
    )


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict():

    model = GPT(
        vocab_size=50,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    state = model.state_dict()

    assert "token_embedding.weight" in state
    assert "position_embedding.embedding.weight" in state
    assert "blocks.0.attention.q_proj.weight" in state
    assert "blocks.1.feedforward.fc1.weight" in state
    assert "lm_head.weight" in state


def test_load_state_dict():

    model1 = GPT(
        vocab_size=30,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    state = model1.state_dict()

    model2 = GPT(
        vocab_size=30,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

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
# Train / Eval
# ==========================================================

def test_eval_mode():

    np.random.seed(42)

    model = GPT(
        vocab_size=40,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
        dropout=0.5,
    )

    model.eval()

    tokens = tensor(
        np.random.randint(
            0,
            40,
            size=(2, 8),
        )
    )

    out1 = model(tokens)
    out2 = model(tokens)

    assert np.allclose(
        out1.data,
        out2.data,
    )


def test_train_mode():

    np.random.seed(42)

    model = GPT(
        vocab_size=40,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
        dropout=0.5,
    )

    model.train()

    tokens = tensor(
        np.random.randint(
            0,
            40,
            size=(2, 8),
        )
    )

    out1 = model(tokens)
    out2 = model(tokens)

    assert not np.allclose(
        out1.data,
        out2.data,
    )


# ==========================================================
# Numerical Stability
# ==========================================================

def test_no_nan():

    model = GPT(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=10,
    )

    tokens = tensor(
        np.random.randint(
            0,
            100,
            size=(2, 10),
        )
    )

    logits = model(tokens)

    assert not np.isnan(
        logits.data
    ).any()


def test_no_inf():

    model = GPT(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=10,
    )

    tokens = tensor(
        np.random.randint(
            0,
            100,
            size=(2, 10),
        )
    )

    logits = model(tokens)

    assert not np.isinf(
        logits.data
    ).any()


def test_dtype():

    model = GPT(
        vocab_size=50,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
    )

    tokens = tensor(
        np.random.randint(
            0,
            50,
            size=(2, 8),
        )
    )

    logits = model(tokens)

    assert logits.dtype == np.float64