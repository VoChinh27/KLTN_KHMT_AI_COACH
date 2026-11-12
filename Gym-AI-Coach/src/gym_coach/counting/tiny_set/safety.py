"""Preparation/reporting forbid learning, even through earlier entry points."""
from contextlib import contextmanager
from unittest.mock import patch
from ..mse.safety import no_learning as previous_guard


@contextmanager
def no_learning():
    from . import runtime
    def blocked(*args,**kwargs):raise AssertionError('Tiny preparation/reporting forbids learning')
    with previous_guard(),patch.object(runtime,'train',blocked):yield
