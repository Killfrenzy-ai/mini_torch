import numpy as np

from mini_torch.autograd.operation import Operation


class Max(Operation):
    """
    Backward rule for tensor maximum.
    """

    def backward(self, node, grad_output):

        parent, = node.parents

        axis = node.axis
        keepdims = node.keepdims
        original_shape = node.original_shape

        max_values = node.data

        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)
            max_values = np.expand_dims(max_values, axis)

        mask = (parent.data == max_values)

        grad = grad_output * mask

        return (grad,)


MAX = Max()