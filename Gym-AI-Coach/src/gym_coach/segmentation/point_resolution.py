"""Opt-in necessary-condition audit; no noise evidence means no point-turn."""
from ..contracts import ErrorCategory, JointProvenance
from ..flags import FlagCode, GymError, Severity
from .models import SquatTemporalInput, strict_json
from .squat import temporal_flag
from .temporal_v2 import segment_squat_v2
from .temporal_v2_models import Support
from .point_resolution_models import (
    PointReason, PointResolutionPolicy, PointSupportCheck, SquatPointResolutionAudit,
)


def _checked(source):
    try:
        if not isinstance(source, SquatTemporalInput):
            raise ValueError('Expected explicit synthetic SquatTemporalInput, no fixture/reference wrapper')
        return SquatTemporalInput.model_validate_json(strict_json(source),strict=True)
    except (ValueError, TypeError, AttributeError) as exc:
        flag=temporal_flag(FlagCode.INVALID_TEMPORAL_INPUT,str(exc))
        raise GymError(flag.model_copy(update={'category':ErrorCategory.INPUT_ERROR,'severity':Severity.ERROR})) from exc


def inspect_point_support(source: SquatTemporalInput, bottom_frame: int,
                          candidate_support: Support, *, flags=()) -> PointSupportCheck:
    """Check exact raw/candidate support. This never accepts a point-turn.

    The flag argument is for explicit upstream structured operational evidence,
    never fixture references. The producer supplies flags from its fresh v1 run.
    """
    source=_checked(source)
    try:
        support=Support.model_validate_json(strict_json(candidate_support),strict=True)
        n=len(source.samples)
        if type(bottom_frame) is not int or not 0<=support.start_frame<=bottom_frame<=support.end_frame<n:
            raise ValueError('Anchor/candidate support must be within the supplied input')
        from ..flags import StructuredFlag
        flags=tuple(StructuredFlag.model_validate_json(strict_json(f),strict=True) for f in flags)
    except (ValueError, TypeError, AttributeError) as exc:
        flag=temporal_flag(FlagCode.INVALID_TEMPORAL_INPUT,str(exc),source.video_id)
        raise GymError(flag.model_copy(update={'category':ErrorCategory.INPUT_ERROR,'severity':Severity.ERROR})) from exc
    b=bottom_frame
    raw=Support(start_frame=max(0,b-1),end_frame=min(n-1,b+1))
    reasons=[]
    def add(reason,span):
        if reason not in {r.reason for r in reasons}:
            reasons.append(PointReason(reason=reason,support=span))
    if support.start_frame==0 or support.end_frame==n-1 or b in (0,n-1):
        add('candidate_at_clip_boundary',support)
    selected=source.samples[raw.start_frame:raw.end_frame+1]
    if len(selected)!=3 or any(s.value is None for s in selected):
        add('neighbor_unavailable',raw)
    union=Support(start_frame=min(raw.start_frame,support.start_frame),end_frame=max(raw.end_frame,support.end_frame))
    samples=source.samples[union.start_frame:union.end_frame+1]
    if any(s.value is None for s in samples):
        add('missing_candidate_sample',union)
    if len({s.side_source for s in samples})!=1:
        add('side_switch',union)
    if any(JointProvenance.IMPUTED in s.joint_provenance for s in samples):
        add('imputed_support',union)
    if any(any(p!=JointProvenance.OBSERVED for p in s.joint_provenance) for s in samples):
        add('nonobserved_support',union)
    strict=None
    if len(selected)==3 and all(s.value is not None for s in selected):
        strict=selected[1].value<selected[0].value and selected[1].value<selected[2].value
        if not strict:
            add('raw_minimum_not_strict',raw)
    for flag in flags:
        if flag.code!=FlagCode.UNSTABLE_SIGNAL:
            continue
        start,end=flag.location.frame_start,flag.location.frame_end
        # Missing either endpoint cannot localize instability, so reject globally.
        if start is None or end is None or max(start,union.start_frame)<=min(end,union.end_frame):
            add('unstable_signal_overlap',union)
    met=not reasons
    if met:
        add('noise_not_identifiable',union)
    return PointSupportCheck(bottom_frame=b,candidate_support=support,raw_support=raw,
        input_hash=source.content_hash(),strict_local_minimum=strict,
        necessary_conditions_met=met,reasons=tuple(reasons))


def segment_squat_point_resolution(source: SquatTemporalInput) -> SquatPointResolutionAudit:
    """New policy identity with unchanged conservative base and explicit abstention.

    No accepting capability or noise-free override exists in this version.
    """
    source=_checked(source)
    base=segment_squat_v2(source)
    checks=tuple(inspect_point_support(source,rep.bottom_frame,
        Support(start_frame=rep.start_frame,end_frame=rep.end_frame),flags=base.legacy.flags)
        for rep in base.legacy.repetitions)
    policy=PointResolutionPolicy()
    diagnostics=() if checks else (PointReason(reason='no_full_pvp_candidate'),)
    return SquatPointResolutionAudit(policy=policy,policy_hash=policy.content_hash(),
        base_result=base,base_hash=base.content_hash(),checks=checks,
        necessary_condition_count=sum(c.necessary_conditions_met for c in checks),
        diagnostics=diagnostics)
