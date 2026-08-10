from mini_torch.nn.module import Module
from mini_torch.backend import xp


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

        # ------------------------------------------
        # Compute inverse frequencies
        # ------------------------------------------

        inv_freq = 1.0 / (
            base ** (
                xp().arange(
                    0,
                    head_dim,
                    2,
                )
                / head_dim
            )
        )

        # ------------------------------------------
        # Position indices
        # ------------------------------------------

        positions = xp().arange(
            max_seq_len
        )

        frequencies = (
            positions[:, None]
            * inv_freq[None, :]
        )

        # ------------------------------------------
        # Precompute RoPE constants
        #
        # Keep these as raw NumPy/CuPy arrays.
        # They do not require gradients.
        # ------------------------------------------

        self.cos = xp().cos(
            frequencies
        ).reshape(
            1,
            max_seq_len,
            1,
            head_dim // 2,
        )

        self.sin = xp().sin(
            frequencies
        ).reshape(
            1,
            max_seq_len,
            1,
            head_dim // 2,
        )

    def forward(self, x):

        _, T, _, D = x.shape

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

        # ------------------------------------------
        # Select cached positions
        #
        # Raw backend slicing:
        # no Tensor operation
        # no autograd node
        # ------------------------------------------

        cos = self.cos[
            :,
            :T,
            :,
            :,
        ]

        sin = self.sin[
            :,
            :T,
            :,
            :,
        ]

        # ------------------------------------------
        # Single fused autograd operation
        # ------------------------------------------

        return x.rotary_embedding(
            cos,
            sin,
        )