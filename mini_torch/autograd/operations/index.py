from mini_torch.backend import xp

from mini_torch.autograd.operation import Operation


class Index(Operation):
    """
    Backward rule for tensor indexing.
    """

    def backward(self, node, grad_output):

        parent, = node.parents

        grad = xp().zeros_like(parent.data)

        xp().add.at(
            grad,
            node.metadata["index"],
            grad_output,
        )

        return (grad,)