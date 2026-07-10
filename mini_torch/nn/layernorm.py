from mini_torch.nn.module import Module
from mini_torch.parameter import Parameter
from mini_torch.backend import xp


class LayerNorm(Module):
    """
    Layer Normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Parameter(
            xp().ones(normalized_shape)
        )

        self.beta = Parameter(
            xp().zeros(normalized_shape)
        )

    def forward(self, x):
        """
        Apply Layer Normalization over the last dimension.
        """

        mean = x.mean(
            axis=-1,
            keepdims=True,
        )

        centered = x - mean

        variance = (
            centered ** 2
        ).mean(
            axis=-1,
            keepdims=True,
        )

        normalized = (
            centered
        ) / (
            variance + self.eps
        ) ** 0.5

        return (
            self.gamma * normalized
            + self.beta
        )