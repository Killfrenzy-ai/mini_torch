from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.sequential import Sequential
from mini_torch.nn.embedding import Embedding
from mini_torch.nn.layernorm import LayerNorm


# ==========================================================
# Default Mode
# ==========================================================

def test_module_defaults_to_training():

    layer = Linear(4, 8)

    assert layer.training is True


# ==========================================================
# train() / eval()
# ==========================================================

def test_eval_switches_mode():

    layer = Linear(4, 8)

    layer.eval()

    assert layer.training is False


def test_train_switches_back():

    layer = Linear(4, 8)

    layer.eval()

    layer.train()

    assert layer.training is True


# ==========================================================
# Recursive Propagation
# ==========================================================

def test_sequential_eval_propagates():

    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )

    model.eval()

    for module in model.modules():
        assert module.training is False


def test_sequential_train_propagates():

    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )

    model.eval()
    model.train()

    for module in model.modules():
        assert module.training is True


# ==========================================================
# Nested Modules
# ==========================================================

def test_nested_sequential_eval():

    inner = Sequential(
        Linear(8, 8),
        ReLU(),
    )

    model = Sequential(
        Linear(4, 8),
        inner,
        Linear(8, 2),
    )

    model.eval()

    for module in model.modules():
        assert module.training is False


def test_nested_sequential_train():

    inner = Sequential(
        Linear(8, 8),
        ReLU(),
    )

    model = Sequential(
        Linear(4, 8),
        inner,
        Linear(8, 2),
    )

    model.eval()
    model.train()

    for module in model.modules():
        assert module.training is True


# ==========================================================
# modules()
# ==========================================================

def test_modules_contains_self():

    model = Sequential(
        Linear(4, 8),
        ReLU(),
    )

    modules = list(model.modules())

    assert modules[0] is model


def test_modules_returns_all_modules():

    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )

    modules = list(model.modules())

    assert len(modules) == 4
    # Sequential + Linear + ReLU + Linear


# ==========================================================
# Method Chaining
# ==========================================================

def test_train_returns_self():

    layer = Linear(4, 8)

    returned = layer.train()

    assert returned is layer


def test_eval_returns_self():

    layer = Linear(4, 8)

    returned = layer.eval()

    assert returned is layer


def test_train_eval_chaining():

    model = Sequential(
        Linear(4, 8),
        ReLU(),
    )

    model.eval().train().eval()

    assert model.training is False

    for module in model.modules():
        assert module.training is False


# ==========================================================
# Parameter Integrity
# ==========================================================

def test_train_eval_does_not_modify_parameters():

    layer = Linear(4, 8)

    weight_before = layer.weight.data.copy()
    bias_before = layer.bias.data.copy()

    layer.eval()
    layer.train()
    layer.eval()

    assert (layer.weight.data == weight_before).all()
    assert (layer.bias.data == bias_before).all()


# ==========================================================
# Different Module Types
# ==========================================================

def test_embedding_mode_switch():

    embedding = Embedding(100, 32)

    embedding.eval()

    assert embedding.training is False

    embedding.train()

    assert embedding.training is True


def test_layernorm_mode_switch():

    layernorm = LayerNorm(16)

    layernorm.eval()

    assert layernorm.training is False

    layernorm.train()

    assert layernorm.training is True