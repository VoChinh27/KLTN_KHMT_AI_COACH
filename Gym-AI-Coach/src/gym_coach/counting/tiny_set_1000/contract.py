"""Budget-only extension, fresh initialization; no migration from500epoch weights."""
from typing import Literal
from ..tiny_set.contract import Config as OldConfig,CRITERION as OLD_CRITERION

VERSION='squat-count-tiny-set-1000-v1'
CHECKPOINT_VERSION='squat-count-tiny-set-1000-checkpoint-v1'
BUDGET=1000
ENDPOINTS=(49,499,999)
CRITERION=dict(OLD_CRITERION,final_epoch_index=999)
REPRO_FIELDS=('epoch','training_steps','initial_model_sha256','model_sha256','batch_size_actual',
    'batch_order_sha256','training_loss','records','evaluation','step_observation')


class Config(OldConfig):
    epochs: Literal[1000]=1000
