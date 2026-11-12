from contextlib import contextmanager
from unittest.mock import patch
from ..inner_cv_500.safety import no_learning as parent_guard


@contextmanager
def no_learning():
    from . import runtime
    def blocked(*args,**kwargs):
        raise AssertionError('Seed preparation/evaluation forbids learning')
    with parent_guard(), patch.object(runtime,'train',blocked), patch.object(runtime.engine,'train',blocked), \
            patch.dict(runtime.namespace,train=blocked):
        yield
