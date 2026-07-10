from mini_torch.backend import xp

def casual_mask(seq_len):
    """
    Returns an upper-triangular boolean mask.

    Shape:
        (seq_len, seq_len)
    """

    return xp().triu(
        xp().ones(
            (seq_len, seq_len),
            dtype=bool,
        ),
        k=1,
    )