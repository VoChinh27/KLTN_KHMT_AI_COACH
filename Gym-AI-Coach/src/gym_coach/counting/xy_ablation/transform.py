"""Research representation transform; zero coordinates are not observed pose."""
import copy
import math

from ..research.contract import CHANNELS, INPUT_SIZE, JOINTS

VERSION = 'squat-count-zero-xy-v1'
XY_INDICES = tuple(j * len(CHANNELS) + k for j in range(len(JOINTS)) for k in (0, 1))


def descriptor():
    if INPUT_SIZE != 58 or len(XY_INDICES) != 16 or CHANNELS[:2] != ('x_image_height', 'y_image_height'):
        raise ValueError('Frozen input layout changed')
    return dict(version=VERSION, input_size=58, indices=list(XY_INDICES),
        joints=list(JOINTS), operation='copy_then_set_exactly_xy_to_float_zero',
        application='identical_train_and_inner_evaluation_before_frozen_collate',
        mask='preserve_original_complete_mask_without_recomputation',
        retained='visibility, usability, provenance flags, elapsed, dt, duration, targets, timeline',
        semantics='research_representation_not_physical_pose_observation')


def zero_xy(sample):
    """Return an independent copy; reject malformed input rather than repairing it."""
    descriptor()
    values, complete = sample['values'], sample['complete']
    if (not isinstance(values, list) or not values or not isinstance(complete, list)
            or len(values) != len(complete) or any(type(v) is not bool for v in complete)):
        raise ValueError('Invalid ablation input shape/mask')
    for vector in values:
        if (not isinstance(vector, list) or len(vector) != INPUT_SIZE
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in vector)):
            raise ValueError('Invalid ablation input channels')
    result = copy.deepcopy(sample)
    for vector in result['values']:
        for index in XY_INDICES:
            vector[index] = 0.0
    return result


class AblatedDataset:
    """Same wrapper for both sides; original rows remain the pairing evidence."""
    def __init__(self, source):
        self.source = source
        self.rows = source.rows

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return zero_xy(self.source[index])
