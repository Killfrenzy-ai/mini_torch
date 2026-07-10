import numpy as np

from mini_torch.tensors import tensor
from mini_torch.nn.losses import CrossEntropyLoss
from mini_torch.autograd.gradcheck import gradcheck


# ==========================================================
# FORWARD
# ==========================================================

def test_cross_entropy_forward():
    prediction = tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
    ])

    target = tensor([
        [1,0],
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    expected = (
        -np.log(0.8)
        -np.log(0.7)
    ) / 2

    assert np.allclose(loss.data, expected)


def test_cross_entropy_single_sample():
    prediction = tensor([
        [0.2, 0.7, 0.1]
    ])

    target = tensor([
        [1]
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    expected = -np.log(0.7)

    assert np.allclose(loss.data, expected)


# ==========================================================
# BACKWARD
# ==========================================================

def test_cross_entropy_backward():
    prediction = tensor(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ],
        requires_grad=True,
    )

    target = tensor(
        [
            1,
            0
        ]
    )

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    loss.backward()

    assert prediction.grad is not None
    assert prediction.grad.shape == prediction.shape


# ==========================================================
# GRADCHECK
# ==========================================================

def test_cross_entropy_gradcheck():
    prediction = tensor(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.3, 0.1],
        ],
        requires_grad=True,
    )

    target = tensor(
        [
            1,0
        ]
    )

    criterion = CrossEntropyLoss()

    def fn(pred):
        return criterion(pred, target)

    assert gradcheck(fn, [prediction])


# ==========================================================
# EDGE CASES
# ==========================================================

def test_cross_entropy_perfect_prediction():
    prediction = tensor([
        [0.001, 0.998, 0.001]
    ])

    target = tensor([
        1
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    assert loss.data < 0.01


def test_cross_entropy_wrong_prediction():
    prediction = tensor([
        [0.998, 0.001, 0.001]
    ])

    target = tensor([
        1
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    assert loss.data > 5.0


def test_cross_entropy_uniform_prediction():
    prediction = tensor([
        [1/3, 1/3, 1/3]
    ])

    target = tensor([
        1
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    expected = -np.log(1/3)

    assert np.allclose(loss.data, expected)


# ==========================================================
# BATCH TESTS
# ==========================================================

def test_cross_entropy_batch():
    prediction = tensor([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ])

    target = tensor([
        0,
        1,
        2,
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    expected = (
        -np.log(0.8)
        -np.log(0.7)
        -np.log(0.7)
    ) / 3

    assert np.allclose(loss.data, expected)


# ==========================================================
# NUMERICAL STABILITY
# ==========================================================

def test_cross_entropy_no_nan():
    prediction = tensor([
        [1e-12, 1.0 - 2e-12, 1e-12]
    ])

    target = tensor([
        1
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    assert np.isfinite(loss.data)


def test_cross_entropy_no_inf():
    prediction = tensor([
        [0., 1., 0.]
    ])

    target = tensor([
        1
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    assert np.isfinite(loss.data)


# ==========================================================
# REGRESSION TESTS
# ==========================================================

def test_cross_entropy_output_scalar():
    prediction = tensor([
        [0.2, 0.7, 0.1],
        [0.8, 0.1, 0.1],
    ])

    target = tensor([
        1,
        0
    ])

    criterion = CrossEntropyLoss()

    loss = criterion(prediction, target)

    assert loss.shape == ()