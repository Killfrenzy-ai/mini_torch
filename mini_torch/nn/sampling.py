from mini_torch.backend import xp


def top_k_logits(logits, k):

    if k is None or k <= 0:
        return logits

    logits = logits.copy()

    indices = xp().argsort(logits)[-k:]

    mask = xp().ones_like(
        logits,
        dtype=bool,
    )

    mask[indices] = False

    logits[mask] = -xp().inf

    return logits

def top_p_logits(logits, p=0.9):
    """
    Apply nucleus (Top-p) filtering to logits.

    Parameters
    ----------
    logits : ndarray
        Logits of shape (vocab_size,)

    p : float
        Cumulative probability threshold.

    Returns
    -------
    ndarray
        Filtered logits with removed tokens set to -inf.
    """

    if p is None or p >= 1.0:
        return logits

    if p <= 0.0:
        raise ValueError("top_p must be in the range (0, 1].")

    logits = logits.copy()

    # -------------------------------------------------
    # Sort logits from highest to lowest
    # -------------------------------------------------

    sorted_indices = xp().argsort(logits)[::-1]

    sorted_logits = logits[sorted_indices]

    # -------------------------------------------------
    # Convert logits -> probabilities
    # (Numerically stable softmax)
    # -------------------------------------------------

    shifted = sorted_logits - xp().max(sorted_logits)

    probs = xp().exp(shifted)

    probs /= xp().sum(probs)

    # -------------------------------------------------
    # Cumulative probability
    # -------------------------------------------------

    cumulative_probs = xp().cumsum(probs)

    # -------------------------------------------------
    # Remove tokens whose cumulative probability exceeds p
    # -------------------------------------------------

    remove = cumulative_probs > p

    # Keep the first token that exceeds p
    remove[1:] = remove[:-1]
    remove[0] = False

    # -------------------------------------------------
    # Map mask back to original indices
    # -------------------------------------------------

    logits[sorted_indices[remove]] = -xp().inf

    return logits