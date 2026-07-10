import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


_backend = np


def xp():
    """
    Return the currently active array backend.
    """
    return _backend


def use_cpu():
    """
    Switch the global backend to NumPy.
    """
    global _backend
    _backend = np


def use_gpu():
    """
    Switch the global backend to CuPy.
    """
    global _backend

    if cp is None:
        raise RuntimeError(
            "CuPy is not installed."
        )

    _backend = cp


def is_cpu():
    return _backend is np


def is_gpu():
    return cp is not None and _backend is cp


def asarray(array):
    """
    Convert input into an array using the active backend.
    """
    return xp().asarray(array)


def asnumpy(array):
    """
    Convert an array to a NumPy array.
    """

    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    return array


def to_gpu(array):
    """
    Move an array to GPU memory.

    Returns a CuPy array.
    """

    if cp is None:
        raise RuntimeError(
            "CuPy is not installed."
        )

    if isinstance(array, cp.ndarray):
        return array

    return cp.asarray(array)


def to_cpu(array):
    """
    Move an array to CPU memory.

    Returns a NumPy array.
    """

    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    return np.asarray(array)