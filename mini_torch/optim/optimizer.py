from abc import ABC, abstractmethod
import pickle

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

        parameter_states = []

        for parameter in self.parameters:

            state = self.state.get(id(parameter), {})

            parameter_states.append({key: value.copy() for key, value in state.items()})

        return {"parameter_states": parameter_states,}
    
    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """

        self.state = {}

        for parameter, saved_state in zip(self.parameters, state_dict["parameter_states"],):

            self.state[id(parameter)] = {key: value.copy() for key, value in saved_state.items()}

    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(self.state_dict(),f,)

    def load(self, path):

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)

    @abstractmethod
    def step(self):
        """
        Update parameters.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(num_parameters={len(self.parameters)})"