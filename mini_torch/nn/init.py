import numpy as np
from mini_torch.backend import xp

def calculate_fan_in_out(shape):
    """
    Compute the fan-in and fan-out of a weight tensor.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    (fan_in, fan_out)
    """

    if len(shape) < 2:
        raise ValueError(
            "Weight tensor must have at least two dimensions."
        )

    fan_in = shape[0]
    fan_out = shape[1]

    return fan_in, fan_out

def xavier_uniform(shape):
    """
    Xavier (Glorot) Uniform initialization.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    numpy.ndarray
    """

    fan_in, fan_out = calculate_fan_in_out(shape)

    limit = xp().sqrt(
        6.0 / (fan_in + fan_out)
    )

    return xp().random.uniform(
        -limit,
        limit,
        size=shape,
    )

def xavier_normal(shape):
    """
    Xavier (Glorot) Normal initialization.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    numpy.ndarray
    """

    fan_in, fan_out = calculate_fan_in_out(shape)

    std = xp().sqrt(
        2.0 / (fan_in + fan_out)
    )

    return xp().random.normal(
        loc=0.0,
        scale=std,
        size=shape,
    )

def kaiming_uniform(shape):
    """
    Kaiming (He) Uniform initialization.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    numpy.ndarray
    """

    fan_in, _ = calculate_fan_in_out(shape)

    limit = xp().sqrt(
        6.0 / fan_in
    )

    return xp().random.uniform(
        -limit,
        limit,
        size=shape,
    )

def kaiming_normal(shape):
    """
    Kaiming (He) Normal initialization.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    numpy.ndarray
    """

    fan_in, _ = calculate_fan_in_out(shape)

    std = xp().sqrt(
        2.0 / fan_in
    )

    return xp().random.normal(
        loc=0.0,
        scale=std,
        size=shape,
    )

INITIALIZERS = {
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "kaiming_uniform": kaiming_uniform,
    "kaiming_normal": kaiming_normal
}

def get_initializer(name):
    """
    Return an initializer function by name.
    """

    try:
        return INITIALIZERS[name]

    except KeyError as e:
        raise ValueError(
            f"Unknown initializer '{name}'."
        ) from e