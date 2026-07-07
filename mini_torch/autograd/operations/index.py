import numpy as np

from mini_torch.autograd.operation import Operation


class Index(Operation):
    """
    Backward rule for tensor indexing.
    """

    def backward(self, node, grad_output):

        parent, = node.parents

        grad = np.zeros_like(parent.data)

        np.add.at(
            grad,
            node.index,
            grad_output,
        )

        return (grad,)