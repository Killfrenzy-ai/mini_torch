from turtle import left
from mini_torch.autograd.utils import unbroadcast
import numpy as np

from mini_torch.autograd.operation import Operation


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

        grad_left = grad_output @ right.data.T
        grad_right = left.data.T @ grad_output

        return grad_left, grad_right
    
class Sum(Operation):
    """
    Backward rule for tensor summation.
    """

    def backward(self, node, grad_output):
        parent, = node.parents

        grad = np.ones_like(parent.data) * grad_output

        return (grad,)
    
class Mean(Operation):

    def backward(self, node, grad_output):
        parent, = node.parents

        scale = parent.data.size

        grad = np.ones_like(parent.data) * (grad_output / scale)

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
            return (np.transpose(grad_output),)

        inverse = np.argsort(node.axes)

        return (np.transpose(grad_output, inverse),)
    
class Squeeze(Operation):
    """Backward rule for squeeze."""

    def backward(self, node, grad_output):
        parent, = node.parents

        if node.axis is None:
            grad = grad_output.reshape(parent.shape)
        else:
            grad = np.expand_dims(grad_output, axis=node.axis)

        return (grad,)
    
class Unsqueeze(Operation):
    """Backward rule for unsqueeze."""

    def backward(self, node, grad_output):
        grad = np.squeeze(grad_output, axis=node.axis)
        return (grad,)
    
class Pow(Operation):

    def backward(self, node, grad_output):

        parent, = node.parents

        exponent = node.exponent

        grad = (
            grad_output
            * exponent
            * np.power(parent.data, exponent - 1)
        )

        return (grad,)
    
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