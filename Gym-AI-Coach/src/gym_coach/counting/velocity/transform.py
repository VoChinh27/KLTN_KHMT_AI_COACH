"""Past-only, complete-support image-height velocity; no physical calibration."""
import copy
import math

from ..research.contract import CHANNELS, INPUT_SIZE, JOINTS

VERSION = 'squat-count-xy-velocity-v1'
XY_INDICES = tuple(j*7+k for j in range(8) for k in (0, 1))
UNIT = 'image_height_per_second'


def descriptor():
    if INPUT_SIZE != 58 or len(JOINTS) != 8 or len(CHANNELS) != 7 or CHANNELS[:2] != ('x_image_height', 'y_image_height'):
        raise ValueError('Frozen input layout changed')
    return dict(version=VERSION, input_size=58, indices=list(XY_INDICES), joints=list(JOINTS),
        formula='v[t]=(coordinate[t]-coordinate[t-1])/dt[t]', unit=UNIT,
        input_coordinates='frozen image-center/image-height units; image x right, y down',
        support='complete[t] and complete[t-1] and frame_index[t]=frame_index[t-1]+1',
        timeline='frozen full-frame cache: elapsed=index/fps, dt[0]=0, later dt=1/fps; exact stored-value checks',
        boundary='zero16 for first sample, incomplete current, first after incomplete, or nonadjacent frames',
        invalid='reject invalid shape/index/fps/dt/timeline or nonfinite result; no repair/drop/clipping',
        zero_semantics='boundary placeholders do not imply stationary subject; mask retained unchanged',
        application='same copy-based wrapper for train and inner evaluation; no future frame',
        retained='58 channels; visibility/provenance/complete/elapsed/dt/duration/target/video identity',
        smoothing=False, clipping=False, learned_scaler=False,
        semantics='research representation, not physical velocity in metres/second')


def support_reason(index, complete, frame_indices):
    if index == 0:
        return 'first_sample'
    if not complete[index]:
        return 'incomplete_current'
    if not complete[index-1]:
        return 'first_after_incomplete'
    if frame_indices[index] != frame_indices[index-1]+1:
        return 'nonadjacent_frames'
    return 'computed_backward_difference'


def velocity_xy(sample, *, frame_indices, fps):
    """Exact frozen timestamp convention, explicit adjacency; never infer a missing pose."""
    descriptor()
    values, complete = sample['values'], sample['complete']
    if (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or not isinstance(values, list) or not values or not isinstance(complete, list)
            or len(complete) != len(values) or any(type(v) is not bool for v in complete)
            or not isinstance(frame_indices, (list, tuple))
            or len(frame_indices) != len(values)):
        raise ValueError('Invalid velocity input shape/mask/fps')
    if (any(type(i) is not int or i < 0 for i in frame_indices)
            or any(a >= b for a, b in zip(frame_indices, frame_indices[1:]))):
        raise ValueError('Invalid velocity frame indices')
    for t, (frame_index, vector) in enumerate(zip(frame_indices, values)):
        if (not isinstance(vector, list) or len(vector) != INPUT_SIZE
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in vector)):
            raise ValueError(f'Invalid velocity channels at sample {t}')
        dt = vector[-1]
        if (vector[-2] != frame_index/fps or dt != (0.0 if frame_index == 0 else 1/fps)
                or (frame_index > 0 and dt <= 0)):
            raise ValueError(f'Invalid velocity dt/timeline at sample {t}; no implicit repair')
    result = copy.deepcopy(sample)
    for t, vector in enumerate(result['values']):
        supported = support_reason(t, complete, frame_indices) == 'computed_backward_difference'
        for c in XY_INDICES:
            value = (values[t][c]-values[t-1][c])/values[t][-1] if supported else 0.0
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite velocity at sample {t}, channel {c}; no clipping')
            vector[c] = value
    return result


class VelocityDataset:
    """Only full frozen clips supply implicit range(n); verify source/cache timeline first."""
    def __init__(self, source):
        self.source = source
        self.rows = source.rows

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        source = self.rows[index]['source']
        n, fps = source['frame_count'], source['fps']
        if (type(n) is not int or n <= 0 or type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0
                or sample['video_id'] != source['video_id'] or len(sample['values']) != n
                or sample['duration_seconds'] != n/fps):
            raise ValueError('Velocity full-clip source/timeline linkage mismatch')
        return velocity_xy(sample, frame_indices=list(range(n)), fps=fps)
