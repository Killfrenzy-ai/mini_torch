from mini_torch.autograd.operation import Operation
import numpy as np


class Clip(Operation):

    def backward(self, node, grad_output):

        x = node.parents[0].data

        minimum = node.metadata["minimum"]
        maximum = node.metadata["maximum"]

        mask = (
            (x >= minimum)
            &
            (x <= maximum)
        )

        grad = grad_output * mask

        return (grad,)


CLIP = Clip()