"""Runtime guards for all preparation, forward-only checks and aggregation."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def no_learning():
    import torch
    from ..research import baselines, experiment
    from ..inner_cv import runtime as original_runtime
    from .. import cv_baseline_comparison
    def guard():
        def blocked(*args, **kwargs):
            raise AssertionError('Ablation preparation forbids fitting, gradients and optimizer updates')
        return blocked
    with ExitStack() as stack:
        for owner, name in ((torch.Tensor, 'backward'), (torch.autograd, 'backward'),
            (torch.autograd, 'grad'), (torch.optim.Optimizer, 'step'), (torch.optim.AdamW, 'step'),
            (baselines, 'fit_baselines'), (experiment, 'train'), (experiment, 'run_baselines'),
            (original_runtime, 'train'), (cv_baseline_comparison, 'fit_inner_baselines'),
            (cv_baseline_comparison, 'run_comparison')):
            stack.enter_context(patch.object(owner, name, guard()))
        yield
