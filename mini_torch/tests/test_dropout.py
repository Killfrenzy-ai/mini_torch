import numpy as np
import pytest

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward

from mini_torch.nn.dropout import Dropout

from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam


# ==========================================================
# Construction
# ==========================================================

def test_dropout_construction():

    layer = Dropout()

    assert layer.p == 0.5


def test_custom_probability():

    layer = Dropout(0.2)

    assert layer.p == 0.2


@pytest.mark.parametrize(
    "p",
    [
        -0.1,
        1.0,
        1.5,
    ],
)
def test_invalid_probability(p):

    with pytest.raises(ValueError):
        Dropout(p)


# ==========================================================
# Forward (Training)
# ==========================================================

def test_forward_shape():

    layer = Dropout(0.5)

    x = tensor(
        np.random.randn(32, 16)
    )

    y = layer(x)

    assert y.shape == x.shape


def test_zero_probability():

    layer = Dropout(0.0)

    x = tensor(
        np.random.randn(10, 5)
    )

    y = layer(x)

    assert np.array_equal(
        x.data,
        y.data,
    )


def test_training_changes_output():

    np.random.seed(42)

    layer = Dropout(0.5)

    x = tensor(
        np.ones((1000,))
    )

    y = layer(x)

    assert not np.array_equal(
        x.data,
        y.data,
    )


def test_inverted_dropout_scaling():

    np.random.seed(42)

    layer = Dropout(0.5)

    x = tensor(
        np.ones((100000,))
    )

    y = layer(x)

    assert np.isclose(
        y.data.mean(),
        1.0,
        atol=0.03,
    )


# ==========================================================
# Evaluation Mode
# ==========================================================

def test_eval_returns_input():

    layer = Dropout(0.5)

    layer.eval()

    x = tensor(
        np.random.randn(20, 4)
    )

    y = layer(x)

    assert np.array_equal(
        x.data,
        y.data,
    )


def test_train_then_eval():

    layer = Dropout(0.5)

    x = tensor(
        np.random.randn(20, 4)
    )

    layer.train()
    train_output = layer(x)

    layer.eval()
    eval_output = layer(x)

    assert np.array_equal(
        eval_output.data,
        x.data,
    )

    assert not np.array_equal(
        train_output.data,
        eval_output.data,
    )


# ==========================================================
# Mask Properties
# ==========================================================

def test_mask_saved():

    layer = Dropout(0.5)

    x = tensor(
        np.ones((10,))
    )

    y = layer(x)

    assert hasattr(y, "mask")


def test_mask_shape():

    layer = Dropout(0.5)

    x = tensor(
        np.ones((5, 8))
    )

    y = layer(x)

    assert y.mask.shape == x.shape


# ==========================================================
# Backward
# ==========================================================

def test_backward():

    np.random.seed(42)

    layer = Dropout(0.5)

    x = tensor(
        np.ones((20,)),
        requires_grad=True,
    )

    y = layer(x)

    loss = y.sum()

    backward(loss)

    assert np.array_equal(
        x.grad,
        y.mask,
    )


# ==========================================================
# Optimizers
# ==========================================================

def test_dropout_with_sgd():

    layer = Dropout(0.5)

    optimizer = SGD(
        [],
        lr=0.1,
    )

    optimizer.step()


def test_dropout_with_adam():

    layer = Dropout(0.5)

    optimizer = Adam(
        [],
        lr=0.001,
    )

    optimizer.step()


# ==========================================================
# Serialization
# ==========================================================

def test_state_dict_empty():

    layer = Dropout(0.5)

    state = layer.state_dict()

    assert state == {}


def test_load_state_dict_empty():

    layer = Dropout()

    layer.load_state_dict({})


# ==========================================================
# Regression
# ==========================================================

def test_output_contains_no_nan():

    layer = Dropout(0.5)

    x = tensor(
        np.random.randn(1000)
    )

    y = layer(x)

    assert not np.isnan(
        y.data
    ).any()


def test_output_contains_no_inf():

    layer = Dropout(0.5)

    x = tensor(
        np.random.randn(1000)
    )

    y = layer(x)

    assert not np.isinf(
        y.data
    ).any()


def test_zero_fraction_close_to_probability():

    np.random.seed(42)

    p = 0.3

    layer = Dropout(p)

    x = tensor(
        np.ones((100000,))
    )

    y = layer(x)

    dropped = np.sum(
        y.data == 0
    )

    fraction = dropped / len(y.data)

    assert np.isclose(
        fraction,
        p,
        atol=0.02,
    )


def test_dropout_preserves_dtype():

    layer = Dropout(0.5)

    x = tensor(
        np.ones(
            (10,),
            dtype=np.float64,
        )
    )

    y = layer(x)

    assert y.dtype == x.dtype