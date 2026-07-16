from turtle import left
from mini_torch.autograd.utils import unbroadcast
import numpy as np
import math
from mini_torch.autograd.operation import Operation
from mini_torch.backend import xp

def normalize_axis(axis, ndim):
    """
    Convert negative axes to positive indices.
    """
    if axis is None:
        return None

    if isinstance(axis, tuple):
        return tuple(a if a >= 0 else ndim + a for a in axis)

    return axis if axis >= 0 else ndim + axis


class Add(Operation):
    """Addition operation."""

    def backward(self, node, grad_output: np.ndarray):
        left, right = node.parents

        return (unbroadcast(grad_output, left.shape),
                unbroadcast(grad_output, right.shape))
    
class Sub(Operation):

    def backward(self, node, grad_output):
        left, right = node.parents
        grad_left = unbroadcast(grad_output, left.shape)
        grad_right = unbroadcast(-grad_output, right.shape)
        return (grad_left, grad_right)
    
class Mul(Operation):

    def backward(self, node, grad_output):
        left, right = node.parents
        grad_left = unbroadcast(grad_output * right.data, left.shape)
        grad_right = unbroadcast(grad_output * left.data, right.shape)

        return (grad_left,grad_right,)

class Div(Operation):

    def backward(self, node, grad_output):
        left, right = node.parents
        grad_left = unbroadcast(grad_output / right.data, left.shape)
        grad_right = unbroadcast(-grad_output * left.data / (right.data ** 2), right.shape)

        return (grad_left, grad_right)
    
class MatMul(Operation):
    """
    Backward rule for matrix multiplication.

    C = A @ B

    dL/dA = dL/dC @ Bᵀ
    dL/dB = Aᵀ @ dL/dC
    """

    def backward(self, node, grad_output):
        left, right = node.parents

        grad_left = grad_output @ np.swapaxes(right.data, -1, -2)
        grad_right = np.swapaxes(left.data, -1, -2) @ grad_output

        grad_left = unbroadcast(grad_left, left.shape)
        grad_right = unbroadcast(grad_right, right.shape)

        return grad_left, grad_right
    
class Sum(Operation):
    """
    Backward rule for tensor summation.
    """

    def backward(self, node, grad_output):

        parent, = node.parents

        axis = normalize_axis(node.axis, parent.data.ndim)
        keepdims = node.keepdims
        original_shape = node.original_shape

        grad = grad_output

        if axis is not None and not keepdims:
            grad = xp().expand_dims(
                grad,
                axis=axis,
            )

        grad = xp().broadcast_to(
            grad,
            original_shape,
        ).copy()

        return (grad,)
    
class Mean(Operation):

    def backward(self, node, grad_output):

        parent, = node.parents

        axis = normalize_axis(node.axis, parent.data.ndim)
        keepdims = node.keepdims
        original_shape = node.original_shape

        if axis is None:

            divisor = math.prod(original_shape)

        else:

            if isinstance(axis, tuple):

                divisor = math.prod(
                    [original_shape[a] for a in axis]
                )

            else:

                divisor = original_shape[axis]

        grad = grad_output / divisor

        if axis is not None and not keepdims:
            grad = xp().expand_dims(
                grad,
                axis=axis,
            )

        grad = xp().broadcast_to(
            grad,
            original_shape,
        ).copy()

        return (grad,)

class Reshape(Operation):
    """Backward rule for reshape."""

    def backward(self, node, grad_output):
        parent, = node.parents

        return (grad_output.reshape(parent.shape),)
    
class Transpose(Operation):
    """Backward rule for transpose."""

    def backward(self, node, grad_output):
        parent, = node.parents

        if not hasattr(node, "axes"):
            return (xp().transpose(grad_output),)

        inverse = tuple(np.argsort(node.axes))

        return (xp().transpose(grad_output, inverse),)
    
class Squeeze(Operation):
    """Backward rule for squeeze."""

    def backward(self, node, grad_output):
        parent, = node.parents

        if node.axis is None:
            grad = grad_output.reshape(parent.shape)
        else:
            grad = xp().expand_dims(grad_output, axis=node.axis)

        return (grad,)
    
class Unsqueeze(Operation):
    """Backward rule for unsqueeze."""

    def backward(self, node, grad_output):
        grad = xp().squeeze(grad_output, axis=node.axis)
        return (grad,)
    
class Pow(Operation):

    def backward(self, node, grad_output):

        parent, = node.parents

        exponent = node.exponent

        grad = (
            grad_output
            * exponent
            * xp().power(parent.data, exponent - 1)
        )

        return (grad,)

class Stack(Operation):

    def backward(self, node, grad_output):

        axis = node.axis

        gradients = []

        for i in range(len(node.parents)):

            grad = xp().take(
                grad_output,
                i,
                axis=axis,
            )

            gradients.append(grad)

        return tuple(gradients)
    
ADD = Add()
SUB = Sub()
MUL = Mul()
DIV = Div()
SUM = Sum()
MEAN = Mean()
MATMUL = MatMul()
TRANSPOSE = Transpose()
RESHAPE = Reshape()
SQUEEZE = Squeeze()
UNSQUEEZE = Unsqueeze()
POW = Pow()
STACK = Stack()