from mini_torch.parameter import Parameter
import numpy as np

DEFAULT_CHECKPOINT_PATH = {r"C:\bitbucket\slm_experiments\mini_torch\mini_torch\checkpoints\model_checkpoint.npz"}

class Module:

    def __init__(self):

        self.training = True
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):

        if isinstance(value, Parameter):
            self._parameters[name] = value

        if isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def parameters(self):

        for parameter in self._parameters.values():
            yield parameter

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=""):

        """
        Yield (name, parameter) pairs recursively.
        """

        for name, parameter in self._parameters.items():

            full_name = (
                f"{prefix}.{name}"
                if prefix
                else name
            )

            yield full_name, parameter

        for module_name, module in self._modules.items():

            module_prefix = (
                f"{prefix}.{module_name}"
                if prefix
                else module_name
            )

            yield from module.named_parameters(module_prefix)

    def state_dict(self):
        """
        Return a dictionary containing all model parameters.
        """

        state = {}

        for name, parameter in self.named_parameters():

            state[name] = parameter.data.copy()

        return state
    
    def load_state_dict(self, state_dict):
        """
        Load model parameters from a state dictionary.
        """

        parameter_map = dict(
            self.named_parameters()
        )

        for name, data in state_dict.items():

            if name not in parameter_map:
                raise KeyError(
                    f"Unexpected parameter '{name}'."
                )

            parameter = parameter_map[name]

            if parameter.data.shape != data.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {parameter.data.shape}, "
                    f"got {data.shape}."
                )

            parameter.data[...] = data

    def save(self, path=DEFAULT_CHECKPOINT_PATH):
        np.savez(
            path,
            **self.state_dict(),
        )

    def load(self, path=DEFAULT_CHECKPOINT_PATH):
        state = dict(np.load(path))
        self.load_state_dict(state)

    def zero_grad(self):

        for parameter in self.parameters():
            parameter.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward().")
    
    def train(self):
        """
        Put this module and all child modules into training mode.
        """

        self.training = True

        for module in self._modules.values():
            module.train()

        return self
    
    def eval(self):
        """
        Put this module and all child modules into evaluation mode.
        """

        self.training = False

        for module in self._modules.values():
            module.eval()

        return self
    
    def modules(self):
        """
        Yield this module and all child modules.
        """

        yield self

        for module in self._modules.values():
            yield from module.modules()