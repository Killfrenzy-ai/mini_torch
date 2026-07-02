import numpy as np

from mini_torch.tensors import tensor
from mini_torch.autograd.graph import topological_sort
from mini_torch.autograd.engine import backward


# ==========================================================
# Tensor Creation
# ==========================================================

def test_tensor_creation():
    t = tensor([1, 2, 3])

    assert t.shape == (3,)
    assert t.dtype == np.int64
    assert t.is_leaf
    assert t.requires_grad is False


# ==========================================================
# Forward Operations
# ==========================================================

def test_forward_operations():
    a = tensor(2)
    b = tensor(3)

    assert (a + b).data == 5
    assert (a - b).data == -1
    assert (a * b).data == 6
    assert (b / a).data == 1.5


# ==========================================================
# Graph Traversal
# ==========================================================

def test_topological_sort():

    a = tensor(2, requires_grad=True)
    b = tensor(3, requires_grad=True)

    c = a + b
    d = c * tensor(5)
    e = d - a

    order = topological_sort(e)

    # Root should be last.
    assert order[-1] is e

    # Every node should appear exactly once.
    assert len(order) == len(set(order))

    # Leaves should be marked correctly.
    assert a.is_leaf
    assert b.is_leaf


# ==========================================================
# Addition Backward
# ==========================================================

def test_add_backward():

    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)

    c = a + b

    backward(c)

    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)


# ==========================================================
# Multiplication Backward
# ==========================================================

def test_mul_backward():

    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)

    c = a * b

    backward(c)

    assert np.allclose(a.grad, 3.0)
    assert np.allclose(b.grad, 2.0)


# ==========================================================
# Chain Rule
# ==========================================================

def test_chain_rule():

    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)

    c = a + b
    d = c * tensor(4.0)

    backward(d)

    assert np.allclose(a.grad, 4.0)
    assert np.allclose(b.grad, 4.0)


# ==========================================================
# Gradient Accumulation
# ==========================================================

def test_gradient_accumulation():

    a = tensor(2.0, requires_grad=True)

    b = a * a

    backward(b)

    # d(a²)/da = 2a = 4
    assert np.allclose(a.grad, 4.0)