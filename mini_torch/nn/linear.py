from mini_torch.nn.module import Module
from mini_torch.parameter import Parameter
from mini_torch.tensors import tensor
from mini_torch.nn.init import get_initializer
from mini_torch.backend import xp

class Linear(Module):
    """
    A linear layer that applies a linear transformation to the input data.

    The layer computes the output as:
        output = input@weight + bias
    """

    def __init__(self, in_features , out_features, bias=True, initialization="kaiming_uniform"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias
        # Kaiming initialization for weights
        initializer = get_initializer(initialization)
        self.weight = Parameter(initializer((in_features, out_features)))

        if bias:
            self.bias = Parameter(xp().zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: tensor) -> tensor:
        """
        Forward pass of the linear layer.

        Args:
            x (tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            tensor: Output tensor of shape (batch_size, out_features).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features "
                f"but received {x.shape[-1]}."
            )
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def __repr__(self):
        bias = self.bias is not None

        return (
            f"Linear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={bias})"
        )