from numpy import typing
import numpy as np
import operator
from mini_torch.autograd.operations import (
    ADD,
    SUB,
    MUL,
    DIV,
    SUM,
    MEAN,
    MATMUL,
    TRANSPOSE,
    RESHAPE,
    SQUEEZE,
    UNSQUEEZE,
    POW,
    NEG,
    EXP,
    LOG,
    RELU,
    SIGMOID,
)
class tensor:
    """ Multi-dimensional array used throughout the framework. """

    _next_id = 0  # Class variable to assign unique IDs to each tensor

    def __init__(self, data: typing.ArrayLike, *, parents=(), op=None, requires_grad=False):
        """ Creates a tensor from array-like numeric values"""

        array = np.asarray(data)
        self._id = tensor._next_id
        tensor._next_id += 1

        if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
            raise TypeError(f"Unsupported dtype '{array.dtype}'. Tensor only supports integer and floating-point data.")
        
        self.data = array
        self.parents = parents
        self.op = op
        self.requires_grad = requires_grad
        self.grad = None  # Gradient of the tensor, initialized to None

    @property
    def is_leaf(self):
        """ Returns True if the tensor is a leaf node in the computation graph. """
        return len(self.parents) == 0
    @property
    def shape(self):
        """ Returns the shape of the tensor as a tuple. """
        return self.data.shape
    
    @property
    def dtype(self):
        """ Returns the data type of the tensor. """
        return self.data.dtype
    
    @property
    def ndim(self): 
        """ Returns the number of dimensions of the tensor. """
        return self.data.ndim
    
    @property
    def size(self):
        """ Returns the total number of elements in the tensor. """
        return self.data.size
    
    def numpy(self):
        """ Returns the underlying NumPy array of the tensor. """
        return self.data
    
    def __repr__(self):
        """ Returns a string representation of the tensor. """    
        matrix = str(self.data)
        return f"tensor({matrix}), dtype={self.dtype}, id={self._id}"
    @staticmethod
    def _ensure_tensor(other):
        """ Ensures that the other object is a tensor. If not, converts it to a tensor. """
        if isinstance(other, tensor):
            return other
        return tensor(other)
          
    def _binary_op(self, other, operation):
        """ Performs a binary operation on two tensors. """
        try:
            result = operation(self.data, other.data)
            return result
        except ValueError as e:
            raise ValueError(f"TensorShapeError: Cannot perform operation on tensors with shapes {self.shape} and {other.shape}.") from e
        
    def _unary_op(self, operation, op_instance):
        result = operation(self.data)

        return tensor(
            result,
            parents=(self,),
            op=op_instance,
            requires_grad=self.requires_grad,
        )
    
    def __add__(self, other):
        """ Adds two tensors element-wise. """
        other = self._ensure_tensor(other)
        parents = (self, other)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.add
        result = self._binary_op(other, op)
        return tensor(result, parents=parents, op=ADD, requires_grad=requires_grad)

    def __sub__(self, other):
        """ Subtracts two tensors element-wise. """
        other = self._ensure_tensor(other)
        parents = (self, other)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.sub
        result = self._binary_op(other, op)
        return tensor(result, parents=parents, op=SUB, requires_grad=requires_grad)

    def __mul__(self, other):
        """ Multiplies two tensors element-wise. """
        other = self._ensure_tensor(other)
        parents = (self, other)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.mul
        result = self._binary_op(other, op)
        return tensor(result, parents=parents, op=MUL, requires_grad=requires_grad)

    def __truediv__(self, other):
        """ Divides two tensors element-wise. """
        other = self._ensure_tensor(other)
        parents = (self, other)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.truediv
        result = self._binary_op(other, op)
        return tensor(result, parents=parents, op=DIV, requires_grad=requires_grad)

    def __matmul__(self, other):
        """ Performs matrix multiplication between two tensors. """
        other = self._ensure_tensor(other)
        parents = (self, other)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.matmul
        result = self._binary_op(other, op)
        return tensor(result, parents=parents, op=MATMUL, requires_grad=requires_grad)

    def __getitem__(self, index):
        """ Returns the element at the specified index. """
        return tensor(self.data[index])
        
    def __radd__(self, other):
        """ Adds two tensors element-wise (right-hand side). """
        other = self._ensure_tensor(other)
        parents = (other, self)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.add
        result = other._binary_op(self, op)
        return tensor(result, parents=parents, op=op, requires_grad=requires_grad)

    def __rsub__(self, other):
        """ Subtracts two tensors element-wise (right-hand side). """
        other = self._ensure_tensor(other)
        parents = (other, self)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.sub
        result = other._binary_op(self, op)
        return tensor(result, parents=parents, op=op, requires_grad=requires_grad)

    def __rmul__(self, other):
        """ Multiplies two tensors element-wise (right-hand side). """
        other = self._ensure_tensor(other)
        parents = (other, self)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.mul
        result = other._binary_op(self, op)
        return tensor(result, parents=parents, op=op, requires_grad=requires_grad)

    def __rtruediv__(self, other):
        """ Divides two tensors element-wise (right-hand side). """
        other = self._ensure_tensor(other)
        parents = (other, self)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.truediv
        result = other._binary_op(self, op)
        return tensor(result, parents=parents, op=op, requires_grad=requires_grad)

    def __rmatmul__(self, other):
        """ Performs matrix multiplication between two tensors (right-hand side). """
        other = self._ensure_tensor(other)
        parents = (other, self)
        requires_grad = self.requires_grad or other.requires_grad
        op = operator.matmul
        result = other._binary_op(self, op)
        return tensor(result, parents=parents, op=MATMUL, requires_grad=requires_grad)

    def transpose(self, *axes):
        """
        Return a transposed view of the tensor.

        If no axes are provided, all axes are reversed.
        Otherwise, the provided axis order is used.
        """

        # No axes supplied -> reverse all axes
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))

        elif len(axes) == 1:
            axes = axes[0]
        result = np.transpose(self.data, axes)

        out = tensor(result, parents=(self,), op=TRANSPOSE, requires_grad=self.requires_grad)
        out.axes = axes
        return out

    @property
    def T(self):
        """Return the transpose of the tensor."""
        return self.transpose()
    
    def reshape(self, *shape):
        """Return a view of the tensor with a new shape."""
        result = self.data.reshape(*shape)
        return tensor(result, parents=(self,), op=RESHAPE, requires_grad=self.requires_grad)
    
    def flatten(self):
        """Return a flattened view of the tensor."""
        return self.reshape(-1)
    
    def squeeze(self, axis=None):
        """Return a view of the tensor with single-dimensional entries removed."""
        result = np.squeeze(self.data, axis=axis)
        out = tensor(result, parents=(self,), op=SQUEEZE, requires_grad=self.requires_grad)
        out.axis = axis
        return out

    def unsqueeze(self, axis):
        """Return a view of the tensor with a new axis inserted."""
        result = np.expand_dims(self.data, axis=axis)
        out = tensor(result, parents=(self,), op=UNSQUEEZE, requires_grad=self.requires_grad)
        out.axis = axis
        return out

    def sum(self, axis=None, keepdims=False):
        """Return the sum of the tensor elements over a given axis."""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return tensor(result,parents=(self,), op=SUM, requires_grad=self.requires_grad)
    
    def mean(self, axis=None, keepdims=False):
        """Return the mean of the tensor elements over a given axis."""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return tensor(result, parents=(self,), op=MEAN, requires_grad=self.requires_grad)
    
    def max(self, axis=None, keepdims=False):
        """Return the maximum of the tensor elements over a given axis."""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return tensor(result)
    
    def argmax(self, axis=None):
        """Return the indices of the maximum values along an axis."""
        result = np.argmax(self.data, axis=axis)
        return tensor(result)
    
    def __neg__(self):
        """Return the negation of the tensor."""
        result = -self.data
        return tensor(result, parents=(self,), op=NEG, requires_grad=self.requires_grad)
    
    def __exp__(self):
        """Return the exponential of the tensor elements."""
        result = np.exp(self.data)
        return tensor(result, parents=(self,), op=EXP, requires_grad=self.requires_grad)
    
    def __log__(self):
        """Return the natural logarithm of the tensor elements."""
        result = np.log(self.data)
        return tensor(result, parents=(self,), op=LOG, requires_grad=self.requires_grad)
    
    def __pow__(self,exponent):
        """Return the tensor elements raised to the power of exponent."""
        result = np.power(self.data, exponent)
        out = tensor(result, parents=(self,), op=POW, requires_grad=self.requires_grad)
        out.exponent = exponent
        return out

    def relu(self):
        """Return the ReLU activation of the tensor elements."""
        result = np.maximum(self.data, 0)

        return tensor(result,parents=(self,),op=RELU,requires_grad=self.requires_grad,)

    def sigmoid(self):
        return self._unary_op(
            lambda x: 1 / (1 + np.exp(-x)),
            SIGMOID,
        )