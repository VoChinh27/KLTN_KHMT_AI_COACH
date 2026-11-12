"""Descriptive velocity distribution; no fitted scaler, tuning, or admission change."""
from collections import Counter
import math
import statistics

from ..inner_cv.runtime import subset
from ..models import canonical_hash
from .transform import JOINTS, UNIT, XY_INDICES, VelocityDataset, support_reason


def describe(values):
    if not values:
        return dict(n=0, minimum=None, maximum=None, mean=None, population_sd=None,
                    median=None, abs_p95=None, abs_p99=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite velocity distribution')
    absolute = sorted(abs(v) for v in values)
    return dict(n=len(values), minimum=min(values), maximum=max(values), mean=statistics.mean(values),
        population_sd=statistics.pstdev(values), median=statistics.median(values),
        abs_p95=absolute[math.ceil(.95*len(values))-1], abs_p99=absolute[math.ceil(.99*len(values))-1])


def audit_inputs(root, data, base):
    """Visit exactly 33 unique admitted train clips, including every difficult frame."""
    from ..research.network import collate
    import torch
    ids = [r['source']['video_id'] for r in base['rows']]
    if len(ids) != 33 or len(set(ids)) != 33:
        raise ValueError('Velocity audit requires all 33 unique clips')
    dataset = VelocityDataset(subset(root, data, ids))
    records, counts = [], Counter()
    channel_values = {c: [] for c in XY_INDICES}
    all_values, computed_values = [], []
    for i, row in enumerate(dataset.rows):
        original = dataset.source[i]; original_hash = canonical_hash(original)
        transformed = dataset[i]
        n = len(original['values']); indices = list(range(n))
        reasons = [support_reason(t, original['complete'], indices) for t in range(n)]
        local_counts = Counter(reasons); counts.update(local_counts)
        if ({k: v for k, v in original.items() if k != 'values'} !=
                {k: v for k, v in transformed.items() if k != 'values'}):
            raise ValueError('Velocity changed mask/metadata')
        local_values = []
        for t, (a, b) in enumerate(zip(original['values'], transformed['values'])):
            if any(a[c] != b[c] for c in range(58) if c not in XY_INDICES):
                raise ValueError('Velocity changed retained channel')
            for c in XY_INDICES:
                all_values.append(b[c])
                if reasons[t] == 'computed_backward_difference':
                    channel_values[c].append(b[c]); computed_values.append(b[c]); local_values.append(b[c])
                elif b[c] != 0.0:
                    raise ValueError('Velocity boundary must use zero convention')
        tensor = collate([transformed])['x']
        if not torch.isfinite(tensor).all() or canonical_hash(dataset.source[i]) != original_hash:
            raise ValueError('Velocity float32 overflow or original cache changed')
        records.append(dict(video_id=original['video_id'], source=row['source'], input_artifact=row['input_artifact'],
            original_sample_sha256=original_hash, transformed_sample_sha256=canonical_hash(transformed),
            frames=n, complete_frames=sum(original['complete']), support_counts=dict(local_counts),
            computed_component_distribution=describe(local_values), finite_float64=True, finite_float32=True,
            retained_channels_and_metadata_unchanged=True))
    return dict(version='squat-count-velocity-input-audit-v1', clips=33, groups=12,
        frames=sum(r['frames'] for r in records), support_counts=dict(counts), unit=UNIT,
        computed_components=describe(computed_values), including_boundary_placeholders=describe(all_values),
        per_channel={JOINTS[c//7]+('_vx' if c%7 == 0 else '_vy'): describe(channel_values[c]) for c in XY_INDICES},
        records=records, original_cache_unchanged=True, all_finite_float64_and_float32=True,
        percentile='nearest rank on abs(computed velocity); descriptive, not a clipping/quality threshold',
        training_steps=0, baseline_fits=0, model_forward_calls=0,
        limitation='Differencing can amplify pose/camera noise. No uncertainty ground truth or learned scale. '
                   'Boundary zeros do not prove a stationary body.')
