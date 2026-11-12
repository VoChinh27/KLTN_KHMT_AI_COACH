"""Pure label-free geometry; original complete mask and all other channels retained."""
import copy
import math
import statistics
from ..models import canonical_hash
from ..research.contract import JOINTS, CHANNELS, INPUT_SIZE
from .contract import EPSILON, TOLERANCE, NORMALIZATION

XY = tuple(7*j+k for j in range(8) for k in (0,1))
MAX_FLOAT32 = 3.4028234663852886e38


class NormalizationError(ValueError):
    def __init__(self,reason,detail):
        self.reason=reason
        super().__init__(reason+': '+detail)


def descriptor():
    if INPUT_SIZE!=58 or CHANNELS[:2]!=('x_image_height','y_image_height') or JOINTS[:4]!=(
            'left_shoulder','right_shoulder','left_hip','right_hip'):
        raise ValueError('Unsupported raw58 layout')
    return dict(version=NORMALIZATION,xy_indices=list(XY),joints=list(JOINTS),
        input_units='image_height: x=(x_normalized-.5)*width/height, y=y_normalized-.5',
        output_units='dimensionless torso lengths; image x right and y down',
        origin='per-complete-frame midpoint(left_hip,right_hip)',
        torso='hypot(midpoint_shoulders.x-midpoint_hips.x, midpoint_shoulders.y-midpoint_hips.y)',
        scale='single video median of finite complete-frame torso lengths strictly above epsilon',
        epsilon=EPSILON,epsilon_units='image_height',minimum_scale_samples=1,
        numeric_tolerance=TOLERANCE,formula='(joint_xy - frame_hip_midpoint) / video_scale',
        incomplete='copy all58 values and original mask unchanged',labels_used=False,
        temporal_scope='whole clip, including future samples; not streaming',
        removes='absolute hip position and global body size in image',
        failure='reject entire experiment if any of33clips lacks valid scale; no fallback or dropped clip')


def midpoint(vector,left,right):
    return tuple(vector[7*left+k]/2+vector[7*right+k]/2 for k in (0,1))


def normalize(sample):
    descriptor()
    values,mask=sample['values'],sample['complete']
    if (not isinstance(values,list) or not values or not isinstance(mask,list)
            or len(values)!=len(mask) or any(type(x) is not bool for x in mask)):
        raise NormalizationError('invalid_input','shape or complete mask')
    lengths=[];centers={};complete_count=sum(mask)
    for i,(row,valid) in enumerate(zip(values,mask)):
        if (not isinstance(row,list) or len(row)!=58
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in row)):
            raise NormalizationError('nonfinite_or_invalid_input',f'frame {i}')
        if valid:
            hip=midpoint(row,2,3);shoulder=midpoint(row,0,1)
            length=math.hypot(shoulder[0]-hip[0],shoulder[1]-hip[1])
            centers[i]=hip
            if math.isfinite(length) and length>EPSILON:
                lengths.append(length)
    if not lengths:
        raise NormalizationError('no_valid_video_scale',f'{complete_count} complete, 0 valid torso samples')
    scale=statistics.median(lengths)
    if not math.isfinite(scale) or scale<=EPSILON:
        raise NormalizationError('invalid_video_scale','median is not finite and positive above epsilon')
    result=copy.deepcopy(sample)
    hip_residuals=[];coordinates=[]
    for i,hip in centers.items():
        for joint in range(8):
            for axis in (0,1):
                index=7*joint+axis
                value=(values[i][index]-hip[axis])/scale
                if not math.isfinite(value) or abs(value)>MAX_FLOAT32:
                    raise NormalizationError('nonfinite_result',f'frame {i}, column {index}')
                result['values'][i][index]=value
                coordinates.append(value)
        hip_residuals.extend(abs(v) for v in midpoint(result['values'][i],2,3))
    max_residual=max(hip_residuals)
    if max_residual>TOLERANCE:
        raise NormalizationError('centering_precision',f'hip midpoint residual {max_residual}')
    return result,dict(scale=scale,scale_frames=len(lengths),complete_frames=complete_count,
        excluded_scale_frames=complete_count-len(lengths),total_frames=len(mask),
        torso_min=min(lengths),torso_max=max(lengths),xy_min=min(coordinates),xy_max=max(coordinates),
        max_abs_hip_midpoint=max_residual,
        original_sample_sha256=canonical_hash(sample),normalized_sample_sha256=canonical_hash(result))


class BodyDataset:
    """Validate and normalize each full video once per subset in memory."""
    def __init__(self,source,audit):
        self.rows=source.rows
        self.samples=[]
        for i,row in enumerate(self.rows):
            sample=source[i]
            value,detail=normalize(sample)
            if detail!=audit[sample['video_id']]:
                raise ValueError('Raw cache/scale/normalized input identity changed: '+sample['video_id'])
            self.samples.append(value)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        return copy.deepcopy(self.samples[index])
