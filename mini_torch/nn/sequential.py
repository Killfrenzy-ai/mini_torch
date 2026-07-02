from mini_torch.nn.module import Module

class Sequential(Module):
    """
    A sequential container for stacking multiple layers.

    The layers are added in the order they are passed to the constructor.
    """

    def __init__(self, *modules):

        super().__init__()

        self.layers = []

        for index, module in enumerate(modules):

            setattr(self, f"layer{index}", module)

            self.layers.append(module)

    def forward(self, x):
        """
        Forward pass through the sequential container.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x