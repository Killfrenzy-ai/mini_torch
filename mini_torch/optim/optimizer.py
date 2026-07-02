from abc import ABC, abstractmethod


class Optimizer(ABC):

    """
    Base class for all optimizers.
    """

    def __init__(self, parameters):

        self.parameters = list(parameters)

    def zero_grad(self):

        for parameter in self.parameters:
            parameter.grad = None

    @abstractmethod
    def step(self):
        """
        Update parameters.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(num_parameters={len(self.parameters)})"