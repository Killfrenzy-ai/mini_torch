from mini_torch.tensors import tensor


class Parameter(tensor):
    """
    A trainable tensor.

    Parameters always require gradients.
    """

    def __init__(self, data):
        super().__init__(data,requires_grad=True,)

    def __repr__(self):
        return f"Parameter({self.data}), shape={self.shape})"