from mini_torch.autograd.operation import Operation
from mini_torch.backend import xp


class Max(Operation):
    """
    Backward rule for tensor maximum.
    """

    def backward(self, node, grad_output):

        parent, = node.parents

        axis = node.metadata["axis"]
        keepdims = node.metadata["keepdims"]
        original_shape = node.metadata["original_shape"]

        max_values = node.data

        if axis is not None and not keepdims:
            grad_output = xp().expand_dims(grad_output, axis)
            max_values = xp().expand_dims(max_values, axis)

        mask = (parent.data == max_values)

        grad = grad_output * mask

        return (grad,)


MAX = Max()