from contextlib import contextmanager
from unittest.mock import patch
from ..body_normalized.safety import no_learning as previous_guard


@contextmanager
def no_learning():
    # Import before patching frozen parent training functions.
    from . import runtime
    def blocked(*args,**kwargs):
        raise AssertionError('Fixed origin preparation forbids learning')
    with previous_guard(),patch.object(runtime,'train',blocked):
        yield
