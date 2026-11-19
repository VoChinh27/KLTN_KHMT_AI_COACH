"""Pure stdlib experimental PVP segmentation; no real-video authorization."""
import math

from ..contracts import ErrorCategory,SourceTier,Stage
from ..flags import FlagCode,FlagLocation,GymError,Severity,StructuredFlag
from .models import (SquatSegmentationProfile,SquatSegmentationResult,SquatTemporalInput,
                     RepetitionCandidate,strict_json)


def temporal_flag(code,message,video_id=None,start=None,end=None,**evidence):
    return StructuredFlag(code=code,message=message,category=ErrorCategory.SEGMENTATION_ERROR,
        severity=Severity.WARNING,stage=Stage.SEGMENTATION,source_tier=SourceTier.S3,
        location=FlagLocation(video_id=video_id,frame_start=start,frame_end=end),evidence=evidence)


def _runs(samples):
    current=[]
    for sample in samples:
        if sample.value is None or (current and sample.side_source!=current[-1].side_source):
            if current:yield current
            current=[]
        if sample.value is not None:current.append(sample)
    if current:yield current


def _turns(values,excursion):
    """Amplitude hysteresis removes sub-excursion reversals; edges unconfirmed.

    This is a signal parameter, not a form threshold. Equal-value plateaus keep
    their midpoint. First/last turning points are never complete-cycle anchors.
    """
    if len(values)<2:return []
    direction=0;extreme=0;plateau_end=0;points=[]
    for i,value in enumerate(values[1:],1):
        if not direction:
            if abs(value-values[0])>=excursion:
                direction=1 if value>values[0] else -1
                points.append((0,'valley' if direction>0 else 'peak',True))
                extreme=plateau_end=i
            continue
        better=value>values[extreme] if direction>0 else value<values[extreme]
        if better:extreme=plateau_end=i
        elif abs(value-values[extreme])<=1e-10:plateau_end=i
        elif (values[extreme]-value if direction>0 else value-values[extreme])>=excursion:
            points.append(((extreme+plateau_end)//2,'peak' if direction>0 else 'valley',False))
            direction=-direction;extreme=plateau_end=i
    if direction:points.append(((extreme+plateau_end)//2,'peak' if direction>0 else 'valley',True))
    return points


def segment_squat(source:SquatTemporalInput,*,profile:SquatSegmentationProfile|None=None)->SquatSegmentationResult:
    """Accept only the separate synthetic temporal contract, never numeric0.4.

    No interpolation or side fusion. Every null or explicit side change splits
    smoothing and candidate extraction. Real temporal validation remains gated.
    """
    try:
        if not isinstance(source,SquatTemporalInput):raise ValueError('Expected the explicit synthetic SquatTemporalInput capability')
        source=SquatTemporalInput.model_validate_json(strict_json(source),strict=True)
        profile=SquatSegmentationProfile.model_validate_json(strict_json(profile or SquatSegmentationProfile()),strict=True)
    except (ValueError,TypeError,AttributeError) as exc:
        flag=temporal_flag(FlagCode.INVALID_TEMPORAL_INPUT,str(exc))
        raise GymError(flag.model_copy(update={'category':ErrorCategory.INPUT_ERROR,'severity':Severity.ERROR})) from exc
    flags=[];reps=[];smoothed=[None]*len(source.samples)
    for i,s in enumerate(source.samples):
        if s.value is None:
            for reason in s.reasons:flags.append(temporal_flag(reason,'Unresolved signal gap; no bridging',source.video_id,i,i))
        if i and s.side_source!=source.samples[i-1].side_source:
            flags.append(temporal_flag(FlagCode.SIDE_SWITCH_DISCONTINUITY,'Explicit side change splits smoothing and repetitions',source.video_id,i,i))
    if not source.samples:flags.append(temporal_flag(FlagCode.INSUFFICIENT_SEQUENCE_LENGTH,'Empty temporal input',source.video_id))
    for run in _runs(source.samples):
        first,last=run[0].frame_index,run[-1].frame_index
        product=profile.smoothing_span_seconds*source.fps/2
        radius=len(run) if not math.isfinite(product) else min(len(run),math.floor(product+1e-9))
        values=[sum(s.value for s in run[max(0,i-radius):min(len(run),i+radius+1)])/
                len(run[max(0,i-radius):min(len(run),i+radius+1)]) for i in range(len(run))]
        for s,value in zip(run,values):smoothed[s.frame_index]=value
        if len(run)<3 or run[-1].timestamp_seconds-run[0].timestamp_seconds<2*profile.min_half_cycle_seconds:
            flags.append(temporal_flag(FlagCode.INSUFFICIENT_SEQUENCE_LENGTH,'Finite run too short for a confirmed PVP candidate',source.video_id,first,last));continue
        if max(values)-min(values)<profile.min_excursion_degrees:
            flags.append(temporal_flag(FlagCode.LOW_SIGNAL_RANGE,'Signal excursion below experimental profile; no cycle inferred',source.video_id,first,last));continue
        turns=_turns(values,profile.min_excursion_degrees)
        flags.append(temporal_flag(FlagCode.PARTIAL_REP_START,'Leading boundary is unconfirmed; no full repetition across it',source.video_id,first,first))
        flags.append(temporal_flag(FlagCode.PARTIAL_REP_END,'Trailing boundary is unconfirmed; no full repetition across it',source.video_id,last,last))
        for a,b,c in zip(turns,turns[1:],turns[2:]):
            if (a[1],b[1],c[1])!=('peak','valley','peak') or any(p[2] for p in (a,b,c)):continue
            sa,sb,sc=(run[p[0]] for p in (a,b,c))
            if min(sb.timestamp_seconds-sa.timestamp_seconds,sc.timestamp_seconds-sb.timestamp_seconds)<profile.min_half_cycle_seconds:
                flags.append(temporal_flag(FlagCode.UNSTABLE_SIGNAL,'Candidate turns are too close in time under S3 profile',source.video_id,sa.frame_index,sc.frame_index));continue
            reps.append(RepetitionCandidate(start_frame=sa.frame_index,bottom_frame=sb.frame_index,end_frame=sc.frame_index,
                start_seconds=sa.timestamp_seconds,bottom_seconds=sb.timestamp_seconds,end_seconds=sc.timestamp_seconds,side_source=sa.side_source))
    if not reps and not flags:flags.append(temporal_flag(FlagCode.SEGMENTATION_FAILED,'No confirmed PVP candidate',source.video_id))
    return SquatSegmentationResult(input=source,input_hash=source.content_hash(),profile=profile,profile_hash=profile.content_hash(),
        status='experimental_candidates' if reps else 'unavailable',smoothed_values=tuple(smoothed),
        repetitions=tuple(reps),candidate_count=len(reps),flags=tuple(flags))
