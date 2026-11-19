"""Whole-video coordinate median origin; frozen parent scale, no temporal bridging."""
import copy
import math
import statistics
from ..body_normalized import transform as parent
from ..models import canonical_hash
from .contract import NORMALIZATION, EPSILON, TOLERANCE


def descriptor():
    return dict(parent.descriptor(), version=NORMALIZATION,
        origin='H0=(median complete-frame Hx, median complete-frame Hy)',
        median='coordinate-wise sorted middle value; even count arithmetic mean of two middle values',
        scale='EXACT parent manifest video scale; recompute for verification only',
        formula='(raw_joint_xy - H0) / locked_parent_scale',
        removes='static hip position; retains dynamic hip displacement including camera drift',
        gap='no interpolation, velocities or bridging; incomplete values and resets unchanged')


def normalize(sample, locked):
    if canonical_hash(sample) != locked['original_sample_sha256']:
        raise ValueError('Raw input identity differs; no mutation or double-normalization')
    old, detail = parent.normalize(sample)
    if detail != locked:
        raise ValueError('Parent scale/input audit mismatch; never replace locked scale')
    scale = locked['scale']
    centers = {i: parent.midpoint(row, 2, 3) for i, row in enumerate(sample['values']) if sample['complete'][i]}
    h0 = [statistics.median(h[k] for h in centers.values()) for k in (0, 1)]
    result = copy.deepcopy(sample)
    coordinates = []
    for i in centers:
        for j in range(8):
            for k in (0, 1):
                col = 7*j+k
                v = (sample['values'][i][col]-h0[k])/scale
                if not math.isfinite(v) or abs(v) > parent.MAX_FLOAT32:
                    raise ValueError('Nonfinite/overflow fixed-origin coordinate')
                result['values'][i][col] = v
                coordinates.append(v)
    median_residual = max(abs(statistics.median(parent.midpoint(result['values'][i], 2, 3)[k]
                                              for i in centers)) for k in (0, 1))
    if median_residual > TOLERANCE:
        raise ValueError('Median hip origin residual exceeds frozen tolerance')
    audit = dict(detail, origin=h0, normalized_sample_sha256=canonical_hash(result),
                 parent_normalized_sample_sha256=canonical_hash(old),
                 xy_min=min(coordinates), xy_max=max(coordinates),
                 max_abs_hip_midpoint=max(abs(v) for i in centers for v in parent.midpoint(result['values'][i],2,3)),
                 max_abs_median_hip=median_residual)
    return result, audit


def independent_check(raw, new, locked):
    """Independent indexing/sorted medians, no call to normalize or midpoint."""
    indices = [i for i, ok in enumerate(raw['complete']) if ok]
    def median(xs):
        a=sorted(xs); n=len(a)
        return a[n//2] if n%2 else (a[n//2-1]+a[n//2])/2
    h={i:((raw['values'][i][14]+raw['values'][i][21])/2,
          (raw['values'][i][15]+raw['values'][i][22])/2) for i in indices}
    origin=[median(v[k] for v in h.values()) for k in (0,1)]
    lengths=[math.sqrt(((raw['values'][i][0]+raw['values'][i][7])/2-h[i][0])**2+
                       ((raw['values'][i][1]+raw['values'][i][8])/2-h[i][1])**2) for i in indices]
    independent_scale=median(v for v in lengths if v>EPSILON)
    if abs(independent_scale-locked['scale'])>TOLERANCE:
        raise ValueError('Independent torso scale differs')
    scale=locked['scale']; coord_error=0.; relation_error=0.; motion_error=0.; edges=[]
    for k in raw:
        if k!='values' and raw[k]!=new[k]: raise ValueError('Non-coordinate metadata changed')
    if len(raw['values'])!=len(new['values']):raise ValueError('Frame length changed')
    for i,(a,b) in enumerate(zip(raw['values'],new['values'])):
        if len(b)!=58 or not all(math.isfinite(x) for x in b):raise ValueError('Invalid transformed row')
        for col in range(58):
            if i not in h or col>=56 or col%7>1:
                if a[col]!=b[col]:raise ValueError('Preserved value changed')
            else:
                k=col%7
                coord_error=max(coord_error,abs(b[col]-(a[col]-origin[k])/scale))
                old=(a[col]-h[i][k])/scale
                relation_error=max(relation_error,abs((b[col]-old)-(h[i][k]-origin[k])/scale))
        if i in h and i-1 in h:
            edges.append([i-1,i])
            for k in (0,1):
                get=lambda t:(new['values'][t][14+k]+new['values'][t][21+k])/2
                motion_error=max(motion_error,abs(get(i)-get(i-1)-(h[i][k]-h[i-1][k])/scale))
    residual=max(abs(median((new['values'][i][14+k]+new['values'][i][21+k])/2 for i in indices)) for k in (0,1))
    if max(coord_error,relation_error,motion_error,residual)>TOLERANCE:raise ValueError('Independent geometry mismatch')
    return dict(origin=origin,independent_scale=independent_scale,locked_scale=scale,
                coordinate_error=coord_error,old_new_relation_error=relation_error,
                adjacent_hip_motion_error=motion_error,median_hip_residual=residual,
                adjacent_complete_edges=edges,no_gap_bridging=True)


class FixedDataset:
    def __init__(self, source, audit, parent_audit):
        self.rows=source.rows;self.samples=[]
        for i in range(len(source)):
            sample=source[i]; v=sample['video_id']
            value,detail=normalize(sample,parent_audit[v])
            if detail!=audit[v]:raise ValueError('Fixed origin/input hash audit changed: '+v)
            self.samples.append(value)
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return copy.deepcopy(self.samples[index])
