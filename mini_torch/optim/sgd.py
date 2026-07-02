from mini_torch.optim.optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameters, lr=0.01):

        super().__init__(parameters)

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        self.lr = lr

    def step(self):

        for parameter in self.parameters:

            if parameter.grad is None:
                continue

            parameter.data -= self.lr * parameter.grad