from typing import Literal
from ..body_normalized.contract import SEEDS, EPSILON, TOLERANCE, ENDPOINT
from ..inner_cv_500.contract import Config as Raw26
from ..inner_cv_500_seed.contract import Config as Raw27

VERSION = 'squat-count-fixed-video-hip-origin-gru500-v1'
CHECKPOINT_VERSION = VERSION + '-checkpoint'
NORMALIZATION = 'video_median_hip_locked_torso_v1'


class Config26(Raw26):
    normalization: Literal['video_median_hip_locked_torso_v1'] = NORMALIZATION


class Config27(Raw27):
    normalization: Literal['video_median_hip_locked_torso_v1'] = NORMALIZATION


def config_type(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Require locked seed20261026/20261027')
    return Config26 if seed == SEEDS[0] else Config27
