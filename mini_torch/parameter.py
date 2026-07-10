from mini_torch.tensors import tensor
from mini_torch.backend import (to_cpu,to_gpu)


class Parameter(tensor):
    """
    A trainable tensor.

    Parameters always require gradients.
    """

    def __init__(self, data):
        super().__init__(data,requires_grad=True,)

    def __repr__(self):
        return f"Parameter({self.data}), shape={self.shape})"

    def cuda(self):
        self.data = to_gpu(self.data)

        if self.grad is not None:
            self.grad = to_gpu(self.grad)

        return self

    def cpu(self):
        self.data = to_cpu(self.data)

        if self.grad is not None:
            self.grad = to_cpu(self.grad)

        return self

    def to(self, device):

        if device == "cuda":
            return self.cuda()

        if device == "cpu":
            return self.cpu()

        raise ValueError(f"Unknown device '{device}'.")