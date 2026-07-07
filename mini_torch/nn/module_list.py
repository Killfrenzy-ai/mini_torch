from mini_torch.nn.module import Module


class ModuleList(Module):
    """
    Holds submodules in a list.

    Unlike Sequential, it does not define a forward pass.
    """

    def __init__(self, *modules):

        super().__init__()

        self._list = []

        for module in modules:
            self.append(module)

    def append(self, module):

        if not isinstance(module, Module):
            raise TypeError(
                "ModuleList only accepts Module instances."
            )

        index = len(self._list)

        self._list.append(module)

        setattr(
            self,
            str(index),
            module,
        )

    def __getitem__(self, index):
        return self._list[index]
    
    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(self._list)
    
    def forward(self, *args, **kwargs):

        raise RuntimeError(
        "ModuleList has no forward() method."
        )
    
    def extend(self, modules):

        for module in modules:
            self.append(module)