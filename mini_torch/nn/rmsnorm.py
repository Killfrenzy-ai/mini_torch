from mini_torch.backend import xp
from mini_torch.nn.module import Module
from mini_torch.parameter import Parameter

class RMSNorm(Module):

    def __init__(self, embed_dim, eps=1e-6,):
        super().__init__()

        self.eps = eps

        self.weight = Parameter(xp().ones(embed_dim))

    def forward(self, x):

        rms = (
            (x ** 2).mean(axis=-1, keepdims=True,) + self.eps) ** 0.5

        return (x / rms ) * self.weight