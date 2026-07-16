from mini_torch.nn.module import Module
from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import SiLU


class SwiGLU(Module):

    def __init__( self, embed_dim, hidden_dim,):
        super().__init__()

        self.gate_proj = Linear( embed_dim, hidden_dim, bias=False,)

        self.up_proj = Linear( embed_dim, hidden_dim, bias=False,)

        self.down_proj = Linear( hidden_dim, embed_dim, bias=False,)

        self.activation = SiLU()

    def forward(self, x):

        gate = self.gate_proj(x)

        gate = self.activation(gate)

        up = self.up_proj(x)

        x = gate * up

        return self.down_proj(x)