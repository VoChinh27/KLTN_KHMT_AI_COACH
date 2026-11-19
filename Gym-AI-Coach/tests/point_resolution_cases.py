"""New synthetic raw inputs, not the 30-case sensitivity matrix or GT labels."""
import hashlib
import json
import math

from gym_coach.contracts import JointProvenance
from gym_coach.flags import FlagCode
from gym_coach.preprocessing.models import InterpolationSource
from gym_coach.segmentation.models import SquatSignalSample, SquatTemporalInput


def make_source(case='smooth', fps=30.):
    samples=[]
    count=round(2*fps)+1 if case=='clip_edge' else round(6*fps)+1
    for i in range(count):
        t=i/fps
        value=140.-30.*math.cos(math.pi*t)
        if case=='plateau' and any(abs(t-b)<=.2 for b in (2.,4.)):
            value=110.
        if case=='jitter' and any(abs(t-b)<=.2 for b in (2.,4.)):
            value+=.04 if i%2 else -.04
        missing=(case=='missing_neighbor' and i==round(2*fps)-1) or (case=='gap' and i==round(2*fps))
        side='right' if case=='side_switch' and i>=round(2*fps) else 'left'
        imputed=(case=='imputed_neighbor' and i==round(2*fps)-1) or (case=='imputed_candidate' and i==round(1.5*fps))
        provenance=(JointProvenance.MISSING,)*3 if missing else (JointProvenance.IMPUTED,JointProvenance.OBSERVED,JointProvenance.OBSERVED) if imputed else (JointProvenance.OBSERVED,)*3
        references=(InterpolationSource(left_frame_index=i-1,right_frame_index=i+1,
            left_timestamp_seconds=(i-1)/fps,right_timestamp_seconds=(i+1)/fps),) if imputed else ()
        samples.append(SquatSignalSample(frame_index=i,timestamp_seconds=t,value=None if missing else value,
            side_source=side,joint_provenance=provenance,imputation_references=references,
            reasons=(FlagCode.MISSING_JOINT,) if missing else ()))
    raw=json.dumps([s.model_dump(mode='json') for s in samples],sort_keys=True).encode()
    # No case labels, latent signals, references or noise-free assertion in input.
    return SquatTemporalInput(video_id='synthetic-point-resolution',fps=fps,
        source_hash=hashlib.sha256(raw).hexdigest(),samples=tuple(samples))
