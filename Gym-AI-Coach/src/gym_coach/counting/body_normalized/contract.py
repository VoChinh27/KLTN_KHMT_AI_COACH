from typing import Literal
from ..inner_cv_500.contract import Config as Raw26
from ..inner_cv_500_seed.contract import Config as Raw27

VERSION = 'squat-count-body-normalized-gru500-v1'
CHECKPOINT_VERSION = VERSION + '-checkpoint'
SEEDS = (20261026, 20261027)
EPSILON = 1e-8  # lengths are expressed in image-height units, not pixels or metres
TOLERANCE = 1e-10  # absolute normalized-coordinate tolerance for centroid checks
NORMALIZATION = 'midhip_video_median_torso_no_rotation_v1'
ENDPOINT = dict(epoch_index=499,epoch_one_based=500,selection='fixed_no_best_or_seed_selection')


class Config26(Raw26):
    normalization: Literal['midhip_video_median_torso_no_rotation_v1'] = NORMALIZATION


class Config27(Raw27):
    normalization: Literal['midhip_video_median_torso_no_rotation_v1'] = NORMALIZATION


def config_type(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Require seed20261026 or seed20261027')
    return Config26 if seed == SEEDS[0] else Config27
