"""Preparation/report commands prohibit learning through all current entry points."""
from contextlib import contextmanager
from unittest.mock import patch
from ..tiny_set.safety import no_learning as old_guard


@contextmanager
def no_learning():
    from . import runtime
    def blocked(*args,**kwargs):raise AssertionError('Tiny1000 preparation/reporting forbids learning')
    with old_guard(),patch.object(runtime,'train',blocked):yield
