import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward
from mini_torch.autograd.gradcheck import gradcheck


def test_matmul_forward():
    A = tensor([[1., 2.],
                [3., 4.]])

    B = tensor([[5., 6.],
                [7., 8.]])

    C = A @ B

    expected = np.array([[19., 22.],
                         [43., 50.]])

    assert np.allclose(C.data, expected)


def test_matmul_backward():

    A = tensor([[1., 2.],
                [3., 4.]], requires_grad=True)

    B = tensor([[5., 6.],
                [7., 8.]], requires_grad=True)

    loss = (A @ B).sum()

    backward(loss)

    expected_grad_A = np.array([[11., 15.],
                                [11., 15.]])

    expected_grad_B = np.array([[4., 4.],
                                [6., 6.]])

    assert np.allclose(A.grad, expected_grad_A)
    assert np.allclose(B.grad, expected_grad_B)


def test_matmul_gradcheck():

    A = tensor([[1.5, 2.5],
                [3.5, 4.5]], requires_grad=True)

    B = tensor([[5.5, 6.5],
                [7.5, 8.5]], requires_grad=True)

    def fn(a, b):
        return (a @ b).sum()

    assert gradcheck(fn, [A, B])

def test_batched_matmul_backward():

    a = tensor(
        np.random.randn(2, 3, 4),
        requires_grad=True,
    )

    b = tensor(
        np.random.randn(2, 4, 5),
        requires_grad=True,
    )

    out = a @ b

    loss = out.sum()

    backward(loss)

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape