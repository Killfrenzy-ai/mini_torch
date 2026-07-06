from abc import ABC, abstractmethod


class Optimizer(ABC):

    """
    Base class for all optimizers.
    """

    def __init__(self, parameters):

        self.parameters = tuple(parameters)

        # Optimizer-specific state
        # Only stateful optimizers (like Adam) will use this dictionary to store their state.

        self.state = {}

    def zero_grad(self):

        for parameter in self.parameters:
            parameter.grad = None

    def state_dict(self):
        """
        Returns the state of the optimizer as a dictionary.
        """
        return {"state":self.state,}
    
    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        self.state = state_dict["state"]

    @abstractmethod
    def step(self):
        """
        Update parameters.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(num_parameters={len(self.parameters)})"