import numpy as np

def casual_mask(seq_len):
    """
    Returns an upper-triangular boolean mask.

    Shape:
        (seq_len, seq_len)
    """

    return np.triu(
        np.ones(
            (seq_len, seq_len),
            dtype=bool,
        ),
        k=1,
    )