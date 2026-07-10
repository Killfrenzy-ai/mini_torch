import numpy as np

from mini_torch.autograd.engine import backward
from mini_torch.backend import xp


def gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-5):
    """
    Numerically verify gradients computed by autograd.

    Parameters
    ----------
    fn : callable
        Function mapping input tensors to a scalar tensor.

    inputs : list[tensor]
        Input tensors requiring gradients.

    eps : float
        Perturbation size.

    atol : float
        Absolute tolerance.

    rtol : float
        Relative tolerance.
    """
    for inp in inputs:
        inp.grad = None  # Reset gradients before backward pass

    # Forward pass
    output = fn(*inputs)

    # Backward pass
    backward(output)

    analytical = [xp().array(inp.grad, copy=True)for inp in inputs]

    for input_idx, inp in enumerate(inputs):
        
        for index in np.ndindex(inp.shape):
            original = inp.data[index]
            inp.data[index] = original + eps
            plus = fn(*inputs).data.copy()
            inp.data[index] = original - eps
            minus = fn(*inputs).data.copy()
            inp.data[index] = original  # Restore original value
            numerical = (plus - minus) / (2 * eps)
            analytical_grad = analytical[input_idx][index]
            if not np.allclose(numerical, analytical_grad, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Gradient check failed for input {inp} at index {index}.\n"
                    f"Numerical: {numerical}, Analytical: {analytical_grad}\n"
                    f"Absolute error: {abs(numerical - analytical_grad)}\n"
                )
            if output.data.size != 1:
                raise ValueError("gradcheck requires the function to return a scalar tensor.")
    return True