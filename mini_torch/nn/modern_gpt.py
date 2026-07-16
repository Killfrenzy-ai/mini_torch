from mini_torch.nn.module import Module
from mini_torch.nn.embedding import Embedding
from mini_torch.nn.dropout import Dropout
from mini_torch.nn.rmsnorm import RMSNorm
from mini_torch.nn.linear import Linear
from mini_torch.nn.module_list import ModuleList
from mini_torch.nn.modern_transformer_block import (
    ModernTransformerBlock
)
from mini_torch.nn.functional import casual_mask


class ModernGPT(Module):

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

        # ------------------------------------------
        # SwiGLU hidden dimension
        # ------------------------------------------

        if ff_hidden_dim is None:
            ff_hidden_dim = int(
                (8 / 3) * embed_dim
            )

        self.ff_hidden_dim = ff_hidden_dim

        # ------------------------------------------
        # Token embeddings
        #
        # No positional embeddings.
        # RoPE is applied inside attention.
        # ------------------------------------------

        self.token_embedding = Embedding(
            vocab_size,
            embed_dim,
        )

        self.dropout = Dropout(
            dropout
        )

        # ------------------------------------------
        # Transformer blocks
        # ------------------------------------------

        self.blocks = ModuleList()

        for _ in range(num_layers):

            self.blocks.append(

                ModernTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=ff_hidden_dim,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )

            )

        # ------------------------------------------
        # Final normalization
        # ------------------------------------------

        self.norm = RMSNorm(
            embed_dim
        )

        # ------------------------------------------
        # Language modeling head
        # ------------------------------------------

        self.lm_head = Linear(
            embed_dim,
            vocab_size,
            bias=True,
        )
        self.lm_head.weight = None

    def forward(self, tokens):

        batch_size, seq_len = tokens.shape

        if seq_len > self.max_seq_len:

            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum sequence length "
                f"{self.max_seq_len}."
            )

        # ------------------------------------------
        # Token embeddings
        #
        # No positional embedding addition.
        # ------------------------------------------

        x = self.token_embedding(
            tokens
        )

        x = self.dropout(x)

        # ------------------------------------------
        # Causal attention mask
        # ------------------------------------------

        mask = casual_mask(
            seq_len
        )

        # ------------------------------------------
        # Transformer blocks
        # ------------------------------------------

        for block in self.blocks:

            x, _ = block(
                x,
                mask,
            )

        # ------------------------------------------
        # Final normalization
        # ------------------------------------------

        x = self.norm(x)

        # ------------------------------------------
        # Vocabulary logits
        # ------------------------------------------

        logits = self.lm_head(x)

        return logits