"""Approved tiny diagnostic budget/criteria; original SmoothL1 config otherwise."""
from typing import Literal
from ..research.contract import Config as OriginalConfig

VERSION='squat-count-tiny-set-v1'
CHECKPOINT_VERSION='squat-count-tiny-set-checkpoint-v1'
FOLD=1
BUDGET=500
COUNTS=tuple(range(6))
CRITERION=dict(MAE_lte=0.10,max_absolute_error_lte=0.25,consecutive_final_epochs=20,
    final_epoch_index=499,selection='all_last20_at_fixed_budget_no_best_or_early_stop',
    meaning='diagnostic_fit_convention_not_accuracy_or_form_standard')


class Config(OriginalConfig):
    epochs: Literal[500]=500
