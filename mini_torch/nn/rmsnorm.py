from mini_torch.backend import xp
from mini_torch.nn.module import Module
from mini_torch.parameter import Parameter
from mini_torch.amp.autocast import is_autocast_enabled

class RMSNorm(Module):

    def __init__(self, embed_dim, eps=1e-6,):
        super().__init__()

        self.eps = eps

        self.weight = Parameter(xp().ones(embed_dim))

    def forward(self, x):

        original_dtype = x.dtype

        if is_autocast_enabled():
            x_compute = x.float()
        else:
            x_compute = x

        variance = (
            x_compute ** 2
        ).mean(
            axis=-1,
            keepdims=True,
        )

        normalized = (
            x_compute
            / (
                variance
                + self.eps
            ) ** 0.5
        )

        output = (
            normalized
            * self.weight
        )

        if is_autocast_enabled():
            output = output.astype(
                original_dtype
            )

        return output