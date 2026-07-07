import numpy as np
import pytest

from pathlib import Path

from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.sequential import Sequential


# ==========================================================
# Helpers
# ==========================================================

def create_model():
    return Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 3),
    )


# ==========================================================
# state_dict()
# ==========================================================

def test_state_dict_keys():

    model = create_model()

    state = model.state_dict()

    assert set(state.keys()) == {
        "layer0.weight",
        "layer0.bias",
        "layer2.weight",
        "layer2.bias",
    }


def test_state_dict_shapes():

    model = create_model()

    state = model.state_dict()

    assert state["layer0.weight"].shape == (4, 8)
    assert state["layer0.bias"].shape == (8,)
    assert state["layer2.weight"].shape == (8, 3)
    assert state["layer2.bias"].shape == (3,)


def test_state_dict_returns_copy():

    model = create_model()

    state = model.state_dict()

    original = state["layer0.weight"].copy()

    model._modules["layer0"].weight.data += 10

    assert np.array_equal(
        state["layer0.weight"],
        original,
    )


# ==========================================================
# load_state_dict()
# ==========================================================

def test_load_state_dict_restores_parameters():

    model = create_model()

    state = model.state_dict()

    for parameter in model.parameters():
        parameter.data += 100

    model.load_state_dict(state)

    restored = model.state_dict()

    for key in state:
        assert np.allclose(
            state[key],
            restored[key],
        )


def test_load_state_dict_shape_mismatch():

    model = create_model()

    state = model.state_dict()

    state["layer0.weight"] = np.random.randn(10, 10)

    with pytest.raises(ValueError):
        model.load_state_dict(state)


def test_load_state_dict_unknown_parameter():

    model = create_model()

    state = model.state_dict()

    state["layer0.invalid"] = np.zeros((1,))

    with pytest.raises(KeyError):
        model.load_state_dict(state)


# ==========================================================
# save() / load()
# ==========================================================

def test_save_creates_file(tmp_path):

    model = create_model()

    path = tmp_path / "model.npz"

    model.save(path)

    assert path.exists()


def test_save_load_round_trip(tmp_path):

    model1 = create_model()

    path = tmp_path / "model.npz"

    model1.save(path)

    model2 = create_model()

    model2.load(path)

    state1 = model1.state_dict()
    state2 = model2.state_dict()

    for key in state1:

        assert np.allclose(
            state1[key],
            state2[key],
        )


# ==========================================================
# Prediction Consistency
# ==========================================================

def test_predictions_identical_after_loading(tmp_path):

    from mini_torch.tensors import tensor

    x = tensor(
        np.random.randn(16, 4)
    )

    model1 = create_model()

    prediction1 = model1(x).data.copy()

    path = tmp_path / "model.npz"

    model1.save(path)

    model2 = create_model()

    model2.load(path)

    prediction2 = model2(x).data

    assert np.allclose(
        prediction1,
        prediction2,
    )


# ==========================================================
# Overwrite Existing Checkpoint
# ==========================================================

def test_save_overwrites_existing_file(tmp_path):

    model = create_model()

    path = tmp_path / "checkpoint.npz"

    model.save(path)
    model.save(path)

    assert path.exists()


# ==========================================================
# Multiple Saves
# ==========================================================

def test_multiple_save_load_cycles(tmp_path):

    model = create_model()

    path = tmp_path / "cycle.npz"

    for _ in range(3):

        model.save(path)
        model.load(path)

    assert path.exists()


# ==========================================================
# Empty State Dict
# ==========================================================

def test_empty_module_state_dict():

    class EmptyModule:

        def state_dict(self):
            return {}

    module = EmptyModule()

    assert module.state_dict() == {}