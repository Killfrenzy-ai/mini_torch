import numpy as np

from mini_torch.autograd.operation import Operation


class Dropout(Operation):
    """
    Backward rule for inverted dropout.
    """

    def backward(self, node, grad_output):

        mask = node.mask

        grad = grad_output * mask

        return (grad,)