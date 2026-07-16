from mini_torch.nn.module import Module
from mini_torch.nn.rmsnorm import RMSNorm
from mini_torch.nn.modern_attention import ModernMultiHeadAttention
from mini_torch.nn.swiglu import SwiGLU
from mini_torch.nn.dropout import Dropout


class ModernTransformerBlock(Module):
    """
    Pre-normalized Transformer block using:

    - RMSNorm
    - Multi-Head Attention with RoPE
    - SwiGLU feed-forward network
    - Residual connections
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        max_seq_len,
        dropout=0.1,
    ):
        super().__init__()

        # ==========================================
        # Attention block
        # ==========================================

        self.attention_norm = RMSNorm(
            embed_dim
        )

        self.attention = ModernMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.attention_dropout = Dropout(
            dropout
        )

        # ==========================================
        # Feed-forward block
        # ==========================================

        self.ffn_norm = RMSNorm(
            embed_dim
        )

        self.ffn = SwiGLU(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

        self.ffn_dropout = Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask=None,
    ):

        # ==========================================
        # Attention
        #
        # x = x + Attention(RMSNorm(x))
        # ==========================================

        residual = x

        normalized = self.attention_norm(x)

        attention_output, attention_weights = (
            self.attention(
                normalized,
                mask,
            )
        )

        attention_output = self.attention_dropout(
            attention_output
        )

        x = residual + attention_output

        # ==========================================
        # Feed Forward
        #
        # x = x + SwiGLU(RMSNorm(x))
        # ==========================================

        residual = x

        normalized = self.ffn_norm(x)

        ffn_output = self.ffn(
            normalized
        )

        ffn_output = self.ffn_dropout(
            ffn_output
        )

        x = residual + ffn_output

        return x, attention_weights