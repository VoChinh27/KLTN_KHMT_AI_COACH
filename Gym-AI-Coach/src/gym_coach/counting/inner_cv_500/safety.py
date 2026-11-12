from contextlib import contextmanager
from unittest.mock import patch
from ..tiny_set_1000.safety import no_learning as previous_guard


@contextmanager
def no_learning():
    from . import runtime
    def blocked(*args,**kwargs):raise AssertionError('CV500 preparation/evaluation forbids learning')
    with previous_guard(),patch.object(runtime,'train',blocked):yield
