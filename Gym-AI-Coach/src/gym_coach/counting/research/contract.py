"""Frozen, non-learned input/experiment policy. Import does not load torch."""
from typing import Literal
from ...counting.models import Model, canonical_hash

JOINTS = ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
CHANNELS = ('x_image_height', 'y_image_height', 'visibility', 'usable',
            'observed', 'low_visibility', 'missing')
INPUT_SIZE = len(JOINTS) * len(CHANNELS) + 2  # elapsed seconds, dt seconds
VERSION = 'squat-raw-pose-count-research-v1'
ROLES = ('train', 'validation', 'development_test')


class Config(Model):
    version: Literal['squat-raw-pose-count-research-v1'] = VERSION
    seed: Literal[20261026] = 20261026
    min_frames: Literal[5] = 5
    max_frames: Literal[1500] = 1500
    visibility_threshold: Literal[0.5] = 0.5
    min_complete_frame_fraction: Literal[0.7] = 0.7
    max_incomplete_span_frames: Literal[10] = 10
    hidden_size: Literal[16] = 16
    batch_size: Literal[8] = 8
    epochs: Literal[50] = 50
    learning_rate: Literal[0.001] = 0.001
    weight_decay: Literal[0.0001] = 0.0001
    device: Literal['cpu'] = 'cpu'
    torch_threads: Literal[2] = 2
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    assessment: Literal[False] = False
    input_policy: Literal['observed_in_frame_all8_reset_at_any_incomplete_frame'] = (
        'observed_in_frame_all8_reset_at_any_incomplete_frame')
    normalization: Literal['image_center_height_units_no_fit'] = 'image_center_height_units_no_fit'
    loss: Literal['smooth_l1_beta1_video_mean'] = 'smooth_l1_beta1_video_mean'
    selection: Literal['min_validation_continuous_MAE_earliest_tie'] = 'min_validation_continuous_MAE_earliest_tie'
    rounding: Literal['clamp_nonnegative_then_floor_plus_half'] = 'clamp_nonnegative_then_floor_plus_half'

    def content_hash(self):
        return canonical_hash(self)


def flag(code, detail, category='insufficient_data'):
    return dict(code=code, detail=detail, category=category,
                stage='count_research_input', source_tier='S3')
