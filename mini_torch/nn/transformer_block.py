from mini_torch.nn.module import Module
from mini_torch.nn.layernorm import LayerNorm
from mini_torch.nn.dropout import Dropout
from mini_torch.nn.multihead_attention import MultiHeadAttention
from mini_torch.nn.feedforward import FeedForward


class TransformerBlock(Module):
    """
    GPT-style Pre-LayerNorm Transformer Block.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
        )

        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(embed_dim)

        self.feedforward = FeedForward(
            embed_dim,
            ff_hidden_dim,
            dropout,
        )

        self.dropout2 = Dropout(dropout)

    def forward(self, x, mask=None):
        #Attention layer
        residual = x
        x = self.norm1(x)
        attn_output, weights = self.attention(x, mask)
        attn_output = self.dropout1(attn_output)
        x = residual + attn_output

        #Feedforward layer
        residual = x
        x = self.norm2(x)
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        x = residual + ff_output

        return x, weights 