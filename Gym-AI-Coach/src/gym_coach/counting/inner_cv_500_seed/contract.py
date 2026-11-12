from typing import Literal
from ..inner_cv_500.contract import Config as ParentConfig, ENDPOINT, ENDPOINTS

VERSION = 'squat-count-inner-cv-500-seed20261027-v1'
CHECKPOINT_VERSION = VERSION + '-checkpoint'
SEED = 20261027
OLD_SEED = 20261026
OBSERVATIONS = (
    'fold0_held_MAE_minus_train_MAE_positive',
    'fold0_g11_mean_signed_error_positive',
    'fold0_g20_mean_signed_error_negative',
)
LIMITATION = ('AI weak reference; two seeds on the same33 videos, not66 independent videos. '
    'Adaptive development on previously inspected OOF; not independent final evaluation, '
    'significance, established stability, causality, generalization or squat form validation. '
    'No seed winner, ensemble, refit or automatic model promotion.')


class Config(ParentConfig):
    seed: Literal[20261027] = SEED
