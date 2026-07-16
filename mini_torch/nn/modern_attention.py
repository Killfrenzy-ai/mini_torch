from mini_torch.nn.module import Module
from mini_torch.nn.linear import Linear
from mini_torch.nn.attention import ScaledDotProductAttention
from mini_torch.nn.rotary_embedding import RotaryEmbedding


class ModernMultiHeadAttention(Module):
    """
    Multi-Head Self Attention with Rotary Positional Embeddings.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len=2048,
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

        # RoPE rotates pairs of dimensions.
        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE."
            )

        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.attention = (
            ScaledDotProductAttention()
        )

        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

    def forward(
        self,
        x,
        mask=None,
    ):

        batch_size, seq_len, _ = x.shape

        # ==========================================
        # Q, K, V projections
        # ==========================================

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ==========================================
        # Split into attention heads
        #
        # (B, T, D)
        #       ↓
        # (B, T, H, Dh)
        #       ↓
        # (B, H, T, Dh)
        # ==========================================

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

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # ==========================================
        # Apply Rotary Positional Embeddings
        #
        # Only Q and K are rotated.
        # ==========================================

        q = self.rope(q)
        k = self.rope(k)

        # ==========================================
        # Scaled dot-product attention
        # ==========================================

        context, weights = self.attention(
            q,
            k,
            v,
            mask,
        )

        # ==========================================
        # Merge attention heads
        #
        # (B, H, T, Dh)
        #       ↓
        # (B, T, H, Dh)
        #       ↓
        # (B, T, D)
        # ==========================================

        context = context.transpose(
            0,
            2,
            1,
            3,
        )

        context = context.reshape(
            batch_size,
            seq_len,
            self.embed_dim,
        )

        output = self.out_proj(
            context
        )

        return output, weights