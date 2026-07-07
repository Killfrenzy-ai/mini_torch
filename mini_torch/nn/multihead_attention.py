from mini_torch.nn.module import Module
from mini_torch.nn.linear import Linear
from mini_torch.nn.attention import ScaledDotProductAttention


class MultiHeadAttention(Module):
    """
    Multi-Head Self Attention.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (
            embed_dim // num_heads
        )

        self.q_proj = Linear(
            embed_dim,
            embed_dim,
        )

        self.k_proj = Linear(
            embed_dim,
            embed_dim,
        )

        self.v_proj = Linear(
            embed_dim,
            embed_dim,
        )

        self.attention = (
            ScaledDotProductAttention()
        )

        self.out_proj = Linear(
            embed_dim,
            embed_dim,
        )

    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        q = q.transpose(0,2,1,3)
        k = k.transpose(0,2,1,3)
        v = v.transpose(0,2,1,3)

        context, weights = self.attention(q, k, v, mask)

        context = context.transpose(0,2,1,3)
        context = context.reshape(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(context)

        return output, weights