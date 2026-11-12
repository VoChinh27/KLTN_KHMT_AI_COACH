"""Conservative synthetic v1-anchor adapter; deliberately no bounds detector."""
from dataclasses import dataclass
from typing import Literal

from ..contracts import ErrorCategory
from ..flags import FlagCode, GymError, Severity
from .models import SquatTemporalInput, strict_json
from .squat import segment_squat, temporal_flag
from .temporal_v2_models import (
    Bottom, Cycle, Event, EventComponent, Evidence, PauseComponent, PhaseComponent,
    SquatTemporalResultV2, Support, TemporalRun, V2Policy, V2Reason, event_id,
)


def _input_error(message):
    flag = temporal_flag(FlagCode.INVALID_TEMPORAL_INPUT, message)
    return GymError(flag.model_copy(update={'category': ErrorCategory.INPUT_ERROR, 'severity': Severity.ERROR}))


def segment_squat_v2(source: SquatTemporalInput) -> SquatTemporalResultV2:
    """Preserve PVP candidates; abstain on bounds/phase/pause, independently.

    Only the pre-existing synthetic input capability is accepted. Neither
    fixture definitions nor references are inputs. No angle computation,
    interpolation, flatness detection, side selection or profile tuning.
    """
    legacy = segment_squat(source)  # v1 validates/detaches source; default S3 only.
    source = legacy.input
    runs = []
    for sample in source.samples:
        if sample.value is None:
            continue
        i = sample.frame_index
        if i == 0 or source.samples[i-1].value is None or source.samples[i-1].side_source != sample.side_source:
            runs.append(TemporalRun(run_id=f'run-{i}', side=sample.side_source,
                                    support=Support(start_frame=i, end_frame=i)))
        else:
            run = runs[-1]
            runs[-1] = TemporalRun(run_id=run.run_id, side=run.side,
                                  support=Support(start_frame=run.support.start_frame, end_frame=i))
    events = {}
    cycles = []
    for index, rep in enumerate(legacy.repetitions):
        run = next(r for r in runs if r.support.start_frame <= rep.start_frame <= rep.end_frame <= r.support.end_frame)
        refs = []
        for kind, frame in (('top', rep.start_frame), ('bottom_anchor', rep.bottom_frame), ('top', rep.end_frame)):
            key = event_id(legacy.input_hash, run.run_id, kind, frame)
            events[key] = Event(event_id=key, kind=kind, run_id=run.run_id, side=run.side,
                frame_index=frame, timestamp_seconds=source.samples[frame].timestamp_seconds,
                evidence=Evidence(method='legacy_pvp_proxy', input_hash=legacy.input_hash, support=run.support))
            refs.append(key)
        support = Support(start_frame=rep.start_frame, end_frame=rep.end_frame)
        reasons = (V2Reason(reason='bottom_bounds_unresolved', support=support),)
        cycles.append(Cycle(cycle_id=f'candidate-{index}', run_id=run.run_id, side=run.side,
            start_event_id=refs[0], end_event_id=refs[2], bottom=Bottom(mode='bounds_unresolved',
                anchor_event_id=refs[1], arrival=EventComponent(status='unavailable', reasons=reasons),
                departure=EventComponent(status='unavailable', reasons=reasons),
                pause=PauseComponent(status='unavailable', reasons=reasons)),
            motion=PhaseComponent(status='unavailable', reasons=reasons)))
    diagnostics = []
    mapping = {FlagCode.PARTIAL_REP_START: 'partial_start', FlagCode.PARTIAL_REP_END: 'partial_end',
               FlagCode.SIDE_SWITCH_DISCONTINUITY: 'side_switch'}
    for flag in legacy.flags:
        if flag.code in mapping:
            start, end = flag.location.frame_start, flag.location.frame_end
            support = Support(start_frame=start, end_frame=end) if start is not None and end is not None else None
            diagnostics.append(V2Reason(reason=mapping[flag.code], support=support))
    for sample in source.samples:
        if sample.value is None:
            diagnostics.append(V2Reason(reason='missing_signal', support=Support(
                start_frame=sample.frame_index, end_frame=sample.frame_index)))
    if not cycles:
        diagnostics.append(V2Reason(reason='no_confirmed_cycle'))
    policy = V2Policy()
    return SquatTemporalResultV2(producer_version='squat-v2-conservative-adapter-v1',
        policy=policy, policy_hash=policy.content_hash(), legacy=legacy, legacy_hash=legacy.content_hash(),
        runs=tuple(runs), events=tuple(events.values()), cycles=tuple(cycles),
        candidate_count=len(cycles), diagnostics=tuple(diagnostics))


@dataclass(frozen=True)
class EventRole:
    event_id: str
    cycle_id: str
    role: Literal['top_start', 'top_end', 'bottom_anchor', 'bottom_arrival', 'bottom_departure']


@dataclass(frozen=True)
class FrameProjection:
    frame_index: int
    timestamp_seconds: float
    motion_phase: str | None
    motion_status: str
    cycle_id: str | None
    events: tuple[Event, ...]
    event_roles: tuple[EventRole, ...]
    reasons: tuple[V2Reason, ...]


def phase_at_v2(result: SquatTemporalResultV2, frame_index: int) -> FrameProjection:
    """Half-open phase ownership plus independent event badges, not v1 phase_at."""
    try:
        if not isinstance(result, SquatTemporalResultV2):
            raise ValueError('Expected an explicit v2 contract')
        result = SquatTemporalResultV2.model_validate_json(strict_json(result), strict=True)
        if type(frame_index) is not int or not 0 <= frame_index < len(result.legacy.input.samples):
            raise ValueError('Frame index must be an input frame')
    except (ValueError, TypeError, AttributeError) as exc:
        raise _input_error(str(exc)) from exc
    by_id = {event.event_id: event for event in result.events}
    events = tuple(e for e in result.events if e.frame_index == frame_index)
    ids = {e.event_id for e in events}
    roles = []
    owner = None
    phase = None
    status = 'unavailable'
    reasons = []
    for cycle in result.cycles:
        references = [(cycle.start_event_id, 'top_start'), (cycle.end_event_id, 'top_end'),
                      (cycle.bottom.anchor_event_id, 'bottom_anchor')]
        for component in (cycle.bottom.arrival, cycle.bottom.departure):
            if component.event:
                references.append((component.event.event_id, component.event.kind))
        roles.extend(EventRole(key, cycle.cycle_id, role) for key, role in references if key in ids)
        s, e = by_id[cycle.start_event_id], by_id[cycle.end_event_id]
        if s.frame_index <= frame_index < e.frame_index:
            owner = cycle.cycle_id
            status = cycle.motion.status
            reasons.extend(cycle.motion.reasons)
            for interval in cycle.motion.intervals:
                if by_id[interval.start_event_id].frame_index <= frame_index < by_id[interval.end_event_id].frame_index:
                    phase = interval.phase
    if owner is None:
        reasons = [r for r in result.diagnostics if r.support is None or r.support.start_frame <= frame_index <= r.support.end_frame]
        if not reasons:
            reasons = [V2Reason(reason='outside_candidate', support=Support(start_frame=frame_index, end_frame=frame_index))]
    return FrameProjection(frame_index=frame_index,
        timestamp_seconds=result.legacy.input.samples[frame_index].timestamp_seconds,
        motion_phase=phase, motion_status=status, cycle_id=owner, events=events,
        event_roles=tuple(roles), reasons=tuple(reasons))
