import numpy as np

from mini_torch.tensors import tensor
from mini_torch.nn.module import Module
from mini_torch.nn.embedding import Embedding


class PositionalEmbedding(Module):
    """
    Learned positional embedding.
    """

    def __init__(
        self,
        max_length,
        embedding_dim,
    ):
        super().__init__()

        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.embedding = Embedding(
            max_length,
            embedding_dim,
        )

    def forward(self, tokens):
        """
        Generate learned positional embeddings.

        Parameters
        ----------
        tokens : Tensor
            Input token IDs of shape (batch_size, sequence_length).

        Returns
        -------
        Tensor
            Positional embeddings of shape
            (batch_size, sequence_length, embedding_dim).
        """

        batch_size, seq_len = tokens.shape

        positions = np.tile(
            np.arange(seq_len),
            (batch_size, 1),
        )

        positions = tensor(positions)

        return self.embedding(positions)