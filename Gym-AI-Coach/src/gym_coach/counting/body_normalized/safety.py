from contextlib import contextmanager
from unittest.mock import patch
from ..inner_cv_500_seed.safety import no_learning as previous_guard


@contextmanager
def no_learning():
    # Import before patching frozen parent training functions.
    from . import runtime
    def blocked(*args,**kwargs):
        raise AssertionError('Body normalization preparation forbids learning')
    with previous_guard(),patch.object(runtime,'train',blocked):
        yield
