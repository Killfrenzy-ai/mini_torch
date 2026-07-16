from mini_torch.backend import xp


def clip_grad_norm(parameters, max_norm=1.0, eps=1e-6):
    """
    Clip gradients so that their global L2 norm does not exceed max_norm.

    Parameters
    ----------
    parameters : iterable
        Model parameters.

    max_norm : float
        Maximum allowed gradient norm.

    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Gradient norm before clipping.
    """

    total_norm_sq = 0.0

    params = list(parameters)

    for parameter in params:

        if parameter.grad is None:
            continue

        total_norm_sq += float(
            xp().sum(parameter.grad ** 2)
        )

    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:

        scale = max_norm / (total_norm + eps)

        for parameter in params:

            if parameter.grad is None:
                continue

            parameter.grad *= scale

    return total_norm