from mini_torch.nn.module import Module

class ReLU(Module):

    def forward(self, x):
        return x.relu()

class Sigmoid(Module):

    def forward(self, x):
        return x.sigmoid()

class Softmax(Module):
    """
    Softmax activation.
    """

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, input):
        return input.softmax(axis=self.axis)