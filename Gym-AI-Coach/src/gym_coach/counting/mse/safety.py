"""Read-only commands/tests cannot enter any old or new learning path."""
from contextlib import contextmanager
from unittest.mock import patch
from ..velocity.safety import no_learning as previous_guard


@contextmanager
def no_learning():
    from ..velocity import runtime as velocity
    from . import runtime
    def blocked(*args, **kwargs):
        raise AssertionError('MSE preparation/evaluation forbids learning')
    with previous_guard(), patch.object(velocity, 'train', blocked), patch.object(runtime, 'train', blocked):
        yield
