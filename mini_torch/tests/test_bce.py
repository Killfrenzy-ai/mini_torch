import numpy as np

from mini_torch.tensors import tensor
from mini_torch.nn.losses import BCELoss
from mini_torch.autograd.gradcheck import gradcheck


def test_bce_forward():

    prediction = tensor([[0.9], [0.2]])

    target = tensor([[1], [0]])

    criterion = BCELoss()

    loss = criterion(prediction, target)

    expected = np.mean(
        -(
            np.array([[1.0], [0.0]]) * np.log(np.array([[0.9], [0.2]]))
            +
            (1 - np.array([[1.0], [0.0]]))
            * np.log(1 - np.array([[0.9], [0.2]]))
        )
    )

    assert np.allclose(loss.data, expected)


def test_bce_backward():

    prediction = tensor(
        [[0.8], [0.3]],
        requires_grad=True,
    )

    target = tensor([[1], [0]])

    criterion = BCELoss()

    loss = criterion(prediction, target)

    loss.backward()

    assert prediction.grad is not None
    assert prediction.grad.shape == prediction.shape


def test_bce_gradcheck():

    prediction = tensor(
        [[0.7], [0.3]],
        requires_grad=True,
    )

    target = tensor([[1], [0]])

    criterion = BCELoss()

    def fn(pred):
        return criterion(pred, target)

    assert gradcheck(fn, [prediction])


def test_bce_perfect_prediction():

    prediction = tensor([[0.999], [0.001]])

    target = tensor([[1], [0]])

    criterion = BCELoss()

    loss = criterion(prediction, target)

    assert loss.data < 0.01


def test_bce_wrong_prediction():

    prediction = tensor([[0.001], [0.999]])

    target = tensor([[1], [0]])

    criterion = BCELoss()

    loss = criterion(prediction, target)

    assert loss.data > 5