from numpy import typing
import numpy as np

from mini_torch.backend import (xp,asarray,asnumpy,is_gpu,to_cpu,to_gpu,cp)

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
    STACK,
    ROTARY_EMBEDDING,
    CAST,
)

# ==========================================================
# NumPy Helper Functions
# ==========================================================

def _relu(x):
    return xp().maximum(x, 0)


def _sigmoid(x):
    return 1.0 / (1.0 + xp().exp(-x))

def stack(tensors, axis=0):

    if len(tensors) == 0:
        raise ValueError(
            "stack expects at least one tensor."
        )

    tensors = tuple(
        tensor._ensure_tensor(t)
        for t in tensors
    )

    result = xp().stack(
        [t.data for t in tensors],
        axis=axis,
    )

    out = tensor(
        result,
        parents=tensors,
        op=STACK,
        requires_grad=any(
            t.requires_grad
            for t in tensors
        ),
    )

    out.axis = axis

    return out


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

        array = asarray(data)

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
    
    @property
    def device(self):

        if isinstance(self.data,np.ndarray):
            return "cpu"

        return "cuda"

    # ==========================================================
    # Representation & Conversion
    # ==========================================================
    def cpu(self):
        """
        Move tensor to CPU.
        """

        self.data = to_cpu(self.data)

        return self
    
    def cuda(self):
        """
        Move tensor to GPU.
        """

        self.data = to_gpu(self.data)

        return self

    def numpy(self):
        return to_cpu(self.data)

    def __repr__(self):
        matrix = str(asnumpy(self.data))
        return f"tensor({matrix}), dtype={self.dtype}, id={self._id}"

    def astype(self, dtype):

        if self.data.dtype == dtype:
            return self

        result = self.data.astype(
            dtype,
            copy=False,
        )

        return self._create_tensor(
            result,
            parents=(self,),
            op=CAST,
        )


    def half(self):
        return self.astype(xp().float16)


    def float(self):
        return self.astype(xp().float32)

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

        result = xp().clip(
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
        return self._binary_op(other, xp().add, ADD)

    def __sub__(self, other):
        return self._binary_op(other, xp().subtract, SUB)

    def __mul__(self, other):
        return self._binary_op(other, xp().multiply, MUL)

    def __truediv__(self, other):
        return self._binary_op(other, xp().divide, DIV)

    def __matmul__(self, other):
        return self._binary_op(other, xp().matmul, MATMUL)

    def __pow__(self, exponent):
        """
        Raise every tensor element to the specified exponent.
        """

        result = xp().power(self.data, exponent)

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
        return self._unary_op(xp().negative, NEG)

    def exp(self):
        return self._unary_op(xp().exp, EXP)

    def log(self):
        return self._unary_op(xp().log, LOG)
    
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
            xp().transpose(self.data, axes,),
            TRANSPOSE,
            axes=axes,
        )

    def reshape(self, *shape):
        """
        Return a reshaped view of the tensor.
        """

        return self._shape_op(
            xp().reshape(self.data,shape),
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
            xp().squeeze(self.data, axis=axis),
            SQUEEZE,
            axis=axis,
        )

    def unsqueeze(self, axis):
        """
        Insert a singleton dimension.
        """

        return self._shape_op(
            xp().expand_dims(self.data, axis=axis),
            UNSQUEEZE,
            axis=axis,
        )


    # ==========================================================
    # Specialized Operations
    # ==========================================================

    def rotary_embedding(
        self,
        cos,
        sin,
    ):
        """
        Apply fused Rotary Positional Embeddings (RoPE).

        Expected input shape:
            (batch_size, num_heads, seq_len, head_dim)

        cos/sin shape:
            (1, 1, seq_len, head_dim // 2)
        """

        if self.ndim != 4:
            raise ValueError(
                "rotary_embedding expects a 4D tensor "
                "with shape (B, H, T, D)."
            )

        if self.shape[-1] % 2 != 0:
            raise ValueError(
                "The last dimension must be even "
                "for rotary embeddings."
            )

        expected_shape = (
            1,
            1,
            self.shape[-2],
            self.shape[-1] // 2,
        )

        if cos.shape != expected_shape:
            raise ValueError(
                f"Expected cos shape {expected_shape}, "
                f"got {cos.shape}."
            )

        if sin.shape != expected_shape:
            raise ValueError(
                f"Expected sin shape {expected_shape}, "
                f"got {sin.shape}."
            )

        # ------------------------------------------
        # Split even and odd dimensions
        #
        # Important:
        # These operate directly on backend arrays,
        # so no autograd nodes are created.
        # ------------------------------------------

        x_even = self.data[..., 0::2]
        x_odd = self.data[..., 1::2]

        # ------------------------------------------
        # Apply rotation
        # ------------------------------------------

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        # ------------------------------------------
        # Interleave results
        # ------------------------------------------

        result = xp().empty_like(
            self.data
        )

        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd

        # ------------------------------------------
        # Create ONE autograd node
        # ------------------------------------------

        return self._attach_metadata(
            self._create_tensor(
                result,
                parents=(self,),
                op=ROTARY_EMBEDDING,
            ),
            cos=cos,
            sin=sin,
        )

    # ==========================================================
    # Reduction Operations
    # ==========================================================

    def sum(self, axis=None, keepdims=False):
        return self._shape_op(
            xp().sum(
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
            xp().mean(
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
            xp().max(
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
            xp().argmax(
                self.data,
                axis=axis,
            )
        )

    # ==========================================================
    # Indexing
    # ==========================================================

    def __getitem__(self, index):

        original_index = index
        # Tensor index
        if isinstance(index, tensor):

            index = index.data.astype(
                xp().int64,
                copy=False,
            )

        # List of indices
        elif isinstance(index, list):

            index = xp().asarray(
                index,
                dtype=np.int64,
            )

        # NumPy array
        elif isinstance(index, np.ndarray):

            index = xp().asarray(index)

        # Tuple indexing
        elif isinstance(index, tuple):

            normalized = []

            for item in index:

                if isinstance(item, tensor):

                    normalized.append(
                        item.data.astype(
                            np.int64,
                            copy=False,
                        )
                    )

                elif isinstance(item, np.ndarray):

                    normalized.append(
                        xp().asarray(item)
                    )

                elif isinstance(item, range):

                    normalized.append(
                        xp().arange(
                            item.start,
                            item.stop,
                            item.step,
                        )
                    )

                else:

                    normalized.append(item)

            index = tuple(normalized)

        return self._attach_metadata(
            self._create_tensor(
                self.data[index],
                parents=(self,),
                op=INDEX,
            ),
            index=index,
            original_index = original_index
        )
    
    def backward(self):

        from mini_torch.autograd.engine import backward

        backward(self)

    def item(self):

        return to_cpu(self.data).item()