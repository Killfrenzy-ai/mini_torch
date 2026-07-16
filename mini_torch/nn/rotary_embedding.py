from mini_torch.nn.module import Module
from mini_torch.backend import xp
from mini_torch.tensors import tensor, stack


class RotaryEmbedding(Module):

    def __init__(
        self,
        head_dim,
        max_seq_len=2048,
        base=10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even head dimension."
            )

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (
                xp().arange(0, head_dim, 2)
                / head_dim
            )
        )

        positions = xp().arange(max_seq_len)

        frequencies = (
            positions[:, None]
            * inv_freq[None, :]
        )

        self.cos = xp().cos(frequencies)
        self.sin = xp().sin(frequencies)

    def forward(self, x):

        B, H, T, D = x.shape

        if D != self.head_dim:
            raise ValueError(
                f"Expected head dimension "
                f"{self.head_dim}, got {D}."
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum {self.max_seq_len}."
            )

        cos = tensor(
            self.cos[:T]
        ).reshape(
            1,
            1,
            T,
            D // 2,
        )

        sin = tensor(self.sin[:T]).reshape( 1, 1, T, D // 2, )

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = ( x_even * cos - x_odd * sin )

        rotated_odd = ( x_even * sin + x_odd * cos)

        rotated = stack(
            [rotated_even, rotated_odd], axis=-1,)

        return rotated.reshape(B,H,T,D,)