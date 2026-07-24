from contextlib import contextmanager


_AMP_ENABLED = False


def is_autocast_enabled():
    return _AMP_ENABLED


@contextmanager
def autocast(enabled=True):

    global _AMP_ENABLED

    previous_state = _AMP_ENABLED

    _AMP_ENABLED = enabled

    try:
        yield

    finally:
        _AMP_ENABLED = previous_state