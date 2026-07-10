from mini_torch.nn.module import Module
from mini_torch.nn.activations import Softmax

import math
import numpy as np
from mini_torch.backend import xp


class ScaledDotProductAttention(Module):
    """
    Scaled Dot Product Attention.

    Computes

        softmax(QKᵀ / √dₖ)V
    """

    def __init__(self):
        super().__init__()
        self.softmax = Softmax(axis=-1)

    def forward(self,query,key,value,mask=None,):
        
        d_k = query.shape[-1]
        scale = math.sqrt(d_k)
        scores = query @ key.transpose(-2, -1) / scale

        if mask is not None:
            scores.data = xp().where(mask, -1e9, scores.data)

        weights = self.softmax(scores)
        output = weights @ value

        return output, weights