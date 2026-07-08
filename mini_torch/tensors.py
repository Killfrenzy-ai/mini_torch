from numpy import typing
import numpy as np

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
    CLIP,
    MAX,
    INDEX,
)

# ==========================================================
# NumPy Helper Functions
# ==========================================================

def _relu(x):
    return np.maximum(x, 0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class tensor:
    """Multi-dimensional array used throughout the framework."""

    _next_id = 0

    # ==========================================================
    # Construction
    # ==========================================================

    def __init__(
        self,
        data: typing.ArrayLike,
        *,
        parents=(),
        op=None,
        requires_grad=False,
    ):
        """Creates a tensor from array-like numeric values."""

        array = np.asarray(data)

        self._id = tensor._next_id
        tensor._next_id += 1

        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise TypeError(
                f"Unsupported dtype '{array.dtype}'. "
                "Tensor only supports integer and floating-point data."
            )

        self.data = array
        self.parents = parents
        self.op = op
        self.requires_grad = requires_grad
        self.grad = None

    # ==========================================================
    # Tensor Properties
    # ==========================================================

    @property
    def is_leaf(self):
        return len(self.parents) == 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        """Matrix transpose."""
        return self.transpose()

    # ==========================================================
    # Representation & Conversion
    # ==========================================================

    def numpy(self):
        return self.data

    def __repr__(self):
        matrix = str(self.data)
        return f"tensor({matrix}), dtype={self.dtype}, id={self._id}"

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    @staticmethod
    def _ensure_tensor(other):
        if isinstance(other, tensor):
            return other
        return tensor(other)

    def _attach_metadata(self, out, **metadata):
        """
        Attach operation-specific metadata to a tensor.
        """

        for key, value in metadata.items():
            setattr(out, key, value)

        return out

    def _create_tensor(self, data, *, parents, op):
        """
        Construct a tensor resulting from an operation and
        automatically infer requires_grad from its parents.
        """

        return tensor(
            data,
            parents=parents,
            op=op,
            requires_grad=any(
                parent.requires_grad
                for parent in parents
            ),
        )

    def _binary_op(self, other, numpy_op, graph_op):
        """
        Execute a binary operation and construct the computation graph.
        """

        other = self._ensure_tensor(other)

        try:
            result = numpy_op(self.data, other.data)

        except ValueError as e:
            raise ValueError(
                f"TensorShapeError: Cannot perform operation on tensors "
                f"with shapes {self.shape} and {other.shape}."
            ) from e

        return self._create_tensor(
            result,
            parents=(self, other),
            op=graph_op,
        )

    def _unary_op(self, numpy_op, graph_op):
        """
        Execute a unary operation and construct the computation graph.
        """

        result = numpy_op(self.data)

        return self._create_tensor(
            result,
            parents=(self,),
            op=graph_op,
        )

    def _shape_op(self, data, op, **metadata):
        """
        Construct a tensor for shape-changing operations.
        """

        out = self._create_tensor(
            data,
            parents=(self,),
            op=op,
        )

        return self._attach_metadata(
            out,
            **metadata,
        )

    def clip(self, minimum, maximum):
        """
        Clip tensor values into the interval [minimum, maximum].
        """

        result = np.clip(
            self.data,
            minimum,
            maximum,
        )

        return self._attach_metadata(
            self._create_tensor(
                result,
                parents=(self,),
                op=CLIP,
            ),
            minimum=minimum,
            maximum=maximum,
        )
    # ==========================================================
    # Binary Arithmetic Operations
    # ==========================================================

    def __add__(self, other):
        return self._binary_op(other, np.add, ADD)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract, SUB)

    def __mul__(self, other):
        return self._binary_op(other, np.multiply, MUL)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide, DIV)

    def __matmul__(self, other):
        return self._binary_op(other, np.matmul, MATMUL)

    def __pow__(self, exponent):
        """
        Raise every tensor element to the specified exponent.
        """

        result = np.power(self.data, exponent)

        return self._attach_metadata(
            self._create_tensor(
                result,
                parents=(self,),
                op=POW,
            ),
            exponent=exponent,
        )

    # ==========================================================
    # Reverse Binary Operations
    # ==========================================================

    def __radd__(self, other):
        other = self._ensure_tensor(other)
        return other.__add__(self)

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __rmul__(self, other):
        other = self._ensure_tensor(other)
        return other.__mul__(self)

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return other.__matmul__(self)

    # ==========================================================
    # Unary Operations
    # ==========================================================

    def __neg__(self):
        return self._unary_op(np.negative, NEG)

    def exp(self):
        return self._unary_op(np.exp, EXP)

    def log(self):
        return self._unary_op(np.log, LOG)
    
    # ==========================================================
    # Activation Functions
    # ==========================================================

    def relu(self):
        return self._unary_op(_relu, RELU)

    def sigmoid(self):
        return self._unary_op(_sigmoid, SIGMOID)
    
    def softmax(self, axis=-1):
        """
        Compute the Softmax over the specified axis.
        """

        shifted = self - self.max(
            axis=axis,
            keepdims=True,
        )

        exponentials = shifted.exp()

        return exponentials / exponentials.sum(
            axis=axis,
            keepdims=True,
        )

    # ==========================================================
    # Shape Operations
    # ==========================================================

    def transpose(self, *axes):
        """
        Return a transposed view of the tensor.
        """

        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1:
            axes = axes[0]
        
        elif len(axes) == 2:
            perm = list(range(self.ndim))
            i,j = axes
            perm[i], perm[j] = perm[j], perm[i]
            axes = tuple(perm)

        return self._shape_op(
            np.transpose(self.data, axes,),
            TRANSPOSE,
            axes=axes,
        )

    def reshape(self, *shape):
        """
        Return a reshaped view of the tensor.
        """

        return self._shape_op(
            self.data.reshape(*shape),
            RESHAPE,
            original_shape=self.shape,
            new_shape=shape,
        )

    def flatten(self):
        """
        Return a flattened view of the tensor.
        """

        return self.reshape(-1)

    def squeeze(self, axis=None):
        """
        Remove singleton dimensions.
        """

        return self._shape_op(
            np.squeeze(self.data, axis=axis),
            SQUEEZE,
            axis=axis,
        )

    def unsqueeze(self, axis):
        """
        Insert a singleton dimension.
        """

        return self._shape_op(
            np.expand_dims(self.data, axis=axis),
            UNSQUEEZE,
            axis=axis,
        )

    # ==========================================================
    # Reduction Operations
    # ==========================================================

    def sum(self, axis=None, keepdims=False):
        return self._shape_op(
            np.sum(
                self.data,
                axis=axis,
                keepdims=keepdims,
            ),
            SUM,
            axis=axis,
            keepdims=keepdims,
            original_shape=self.shape,
        )

    def mean(self, axis=None, keepdims=False):
        return self._shape_op(
            np.mean(
                self.data,
                axis=axis,
                keepdims=keepdims,
            ),
            MEAN,
            axis=axis,
            keepdims=keepdims,
            original_shape=self.shape,
        )

    def max(self, axis=None, keepdims=False):
        """
        Return the maximum values of the tensor.
        """

        return self._shape_op(
            np.max(
                self.data,
                axis=axis,
                keepdims=keepdims,
            ),
            MAX,
            axis=axis,
            keepdims=keepdims,
            original_shape=self.shape,
        )

    def argmax(self, axis=None):
        return tensor(
            np.argmax(
                self.data,
                axis=axis,
            )
        )

    # ==========================================================
    # Indexing
    # ==========================================================

    def __getitem__(self, index):
        """
        Return the tensor element(s) at the specified index.
        """

        if isinstance(index, tensor):
            index = index.data.astype(np.int64, copy= False)

        elif isinstance(index, tuple):

            normalized = []

            for item in index:

                if isinstance(item, tensor):
                    normalized.append(
                        item.data.astype(np.int64)
                    )
                else:
                    normalized.append(item)

            index = tuple(normalized)

        return self._attach_metadata(self._create_tensor(self.data[index], parents=(self,), op=INDEX), index=index)
    
    def backward(self):

        from mini_torch.autograd.engine import backward

        backward(self)