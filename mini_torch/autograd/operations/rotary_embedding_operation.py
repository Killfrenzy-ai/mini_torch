from mini_torch.autograd.operation import Operation
from mini_torch.backend import xp

class RotaryEmbeddingOperation(Operation):
    """
    Fused Rotary Positional Embedding operation.

    Forward:
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos

    Backward:
        dx_even = dy_even * cos + dy_odd * sin
        dx_odd  = -dy_even * sin + dy_odd * cos
    """

    def backward(
        self,
        node,
        grad_output,
    ):

        cos = node.cos
        sin = node.sin

        # ------------------------------------------
        # Split output gradient into even/odd parts
        # ------------------------------------------

        grad_even = grad_output[..., 0::2]
        grad_odd = grad_output[..., 1::2]

        # ------------------------------------------
        # Apply transpose/inverse rotation
        # ------------------------------------------

        dx_even = (
            grad_even * cos
            + grad_odd * sin
        )

        dx_odd = (
            -grad_even * sin
            + grad_odd * cos
        )

        # ------------------------------------------
        # Reconstruct original gradient layout
        # ------------------------------------------

        grad_input = xp().empty_like(
            grad_output
        )

        grad_input[..., 0::2] = dx_even
        grad_input[..., 1::2] = dx_odd

        return (grad_input,)