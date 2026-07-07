from mini_torch.nn.module import Module
from mini_torch.nn.embedding import Embedding
from mini_torch.nn.position import PositionalEmbedding
from mini_torch.nn.dropout import Dropout
from mini_torch.nn.layernorm import LayerNorm
from mini_torch.nn.linear import Linear
from mini_torch.nn.module_list import ModuleList
from mini_torch.nn.transformer_block import TransformerBlock
from mini_torch.nn.functional import casual_mask


class GPT(Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        ff_hidden_dim=None,
        dropout=0.1,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = Embedding(
            vocab_size,
            embed_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len,
            embed_dim,
        )

        self.dropout = Dropout(dropout)

        self.blocks = ModuleList()

        for _ in range(num_layers):

            self.blocks.append(

                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ff_hidden_dim,
                    dropout,
                )

            )

        self.norm = LayerNorm(embed_dim)

        self.lm_head = Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, tokens):

        batch_size, seq_len = tokens.shape

        x = self.token_embedding(tokens)

        positions = self.position_embedding(tokens)

        x = x + positions

        x = self.dropout(x)

        mask = casual_mask(seq_len)

        for block in self.blocks:

            x, _ = block(
                x,
                mask,
            )

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits