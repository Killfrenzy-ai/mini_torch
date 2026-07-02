from mini_torch.parameter import Parameter


class Module:

    def __init__(self):

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

    def zero_grad(self):

        for parameter in self.parameters():
            parameter.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward().")