import numpy as np

from mini_torch.backend import xp


class CrossEntropyOperation:
    """
    Fused Cross Entropy from logits.

    Forward computes the loss directly from logits.

    Backward returns

        softmax(logits) - one_hot(targets)

    without traversing the internal graph.
    """

    @staticmethod
    def backward(node, grad):

        logits = node.parents[0].data
        targets = node.metadata["targets"]

        # -----------------------------------------
        # Stable softmax
        # -----------------------------------------

        shifted = logits - xp().max(
            logits,
            axis=-1,
            keepdims=True,
        )

        exp = xp().exp(shifted)

        probs = exp / xp().sum(
            exp,
            axis=-1,
            keepdims=True,
        )

        # -----------------------------------------
        # dL/dz
        # -----------------------------------------

        rows = xp().arange(
            targets.shape[0]
        )

        probs[rows, targets] -= 1

        probs /= targets.shape[0]

        probs *= grad

        return (probs,)

CROSS_ENTROPY = CrossEntropyOperation()