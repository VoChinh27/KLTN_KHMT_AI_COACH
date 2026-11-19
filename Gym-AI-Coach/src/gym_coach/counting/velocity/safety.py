"""Preparation may neither fit models nor accidentally dispatch old training."""
from contextlib import contextmanager
from unittest.mock import patch

from ..xy_ablation.safety import no_learning as frozen_no_learning


@contextmanager
def no_learning():
    from ..xy_ablation import runtime
    def blocked(*args, **kwargs):
        raise AssertionError('Velocity preparation forbids zero-xy training')
    with frozen_no_learning(), patch.object(runtime, 'train', blocked):
        yield
