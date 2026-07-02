import numpy as np


def unbroadcast(grad, shape):
    """
    Reduce a broadcasted gradient back to the original shape.
    """

    original_shape = shape

    # Align dimensions
    while len(shape) < grad.ndim:
        shape = (1,) + shape

    # Sum over broadcast axes
    for axis, (grad_dim, original_dim) in enumerate(zip(grad.shape, shape)):
        if original_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # Remove leading dimensions again
    grad = grad.reshape(original_shape)

    return grad