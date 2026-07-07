from mini_torch.nn.module import Module
from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.dropout import Dropout


class FeedForward(Module):
    """
    Position-wise Feed Forward Network.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 4

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.fc1 = Linear(
            embed_dim,
            hidden_dim,
        )

        self.activation = ReLU()

        self.dropout = Dropout(dropout)

        self.fc2 = Linear(
            hidden_dim,
            embed_dim,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x