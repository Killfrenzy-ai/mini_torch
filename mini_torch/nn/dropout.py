import numpy as np

from mini_torch.nn.module import Module

from mini_torch.autograd.operations import DROPOUT


class Dropout(Module):
    """
    Inverted Dropout.

    Parameters
    ----------
    p : float
        Probability of dropping an activation.
    """

    def __init__(self, p=0.5):

        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(
                "Dropout probability must satisfy 0 <= p < 1."
            )

        self.p = p

    def forward(self, x):

        if not self.training:
            return x

        if self.p == 0:
            return x

        keep_probability = 1.0 - self.p

        mask = (
            np.random.rand(*x.shape)
            < keep_probability
        ).astype(x.dtype)

        mask /= keep_probability

        return x._attach_metadata(
            x._create_tensor(
                x.data * mask,
                parents=(x,),
                op=DROPOUT,
            ),
            mask=mask,
        )