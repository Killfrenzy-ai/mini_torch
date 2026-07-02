from abc import ABC, abstractmethod
import numpy as np


class Operation(ABC):
    """Base class for all differentiable operations."""

    @abstractmethod
    def backward(self, node, grad_output: np.ndarray):
        """
        Compute gradients for each parent of `node`.

        Parameters
        ----------
        node : tensor
            The tensor produced by this operation.

        grad_output : np.ndarray
            Gradient flowing into this node.

        Returns
        -------
        tuple[np.ndarray, ...]
            One gradient for each parent tensor.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return self.__class__.__name__