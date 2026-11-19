from typing import Literal
from ..research.contract import Config as OriginalConfig

VERSION='squat-count-inner-cv-500-v1'
CHECKPOINT_VERSION='squat-count-inner-cv-500-checkpoint-v1'
BUDGET=500
ENDPOINTS=(49,499)
ENDPOINT=dict(epoch_index=499,epoch_one_based=500,comparison_epoch_index=49,selection='fixed_endpoints_no_minimum_search')
REPRO_FIELDS=('epoch','fold','model_sha256','records','metric','training_steps','training_loss')


class Config(OriginalConfig):
    epochs: Literal[500]=500
