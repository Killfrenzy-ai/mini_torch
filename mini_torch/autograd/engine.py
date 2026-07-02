from .graph import topological_sort

import numpy as np


def backward(loss):
    """
    Perform reverse-mode automatic differentiation.

    Parameters
    ----------
    loss : Tensor
        Scalar output tensor.
    """
    if loss.data.size != 1:
        raise ValueError("Loss tensor must be a scalar (single value).")

    nodes = topological_sort(loss)
    loss.grad = np.ones_like(loss.data)

    for node in reversed(nodes):
        if node.op is None:
            continue
        if node.grad is None:
            continue


        grads = node.op.backward(node,node.grad)
        for parent, grad in zip(node.parents, grads):
            if not parent.requires_grad:
                continue
            if parent.grad is None:
                parent.grad = grad
            else:
                parent.grad += grad