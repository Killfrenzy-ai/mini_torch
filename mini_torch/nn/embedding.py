from mini_torch.nn.module import Module
from mini_torch.parameter import Parameter

from mini_torch.nn.init import xavier_uniform


class Embedding(Module):
    """
    Learnable embedding lookup table.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            xavier_uniform(
                (
                    num_embeddings,
                    embedding_dim,
                )
            )
        )

    def forward(self, indices):
        """
        Lookup embedding vectors.

        Parameters
        ----------
        indices : tensor

        Returns
        -------
        tensor
        """

        return self.weight[indices.data.astype(int)]