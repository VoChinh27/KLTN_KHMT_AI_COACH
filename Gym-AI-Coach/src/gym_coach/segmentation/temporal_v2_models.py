"""Opt-in v2 temporal component contracts; no producer or backend imports."""
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..config import AssessmentConfig
from ..contracts import ContractModel, ErrorCategory, JointProvenance, SourceTier, Stage
from .models import Hash, SquatSegmentationResult, content_hash, strict_json

PROFILE_HASH = '5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017'
Status = Literal['candidate', 'unavailable', 'not_run', 'blocked']
Reason = Literal['bottom_bounds_unresolved', 'pause_unresolved_at_resolution',
                 'stage_not_run', 'temporal_gate_blocked', 'no_confirmed_cycle',
                 'missing_signal', 'side_switch', 'partial_start', 'partial_end',
                 'outside_candidate', 'unsupported_bound_evidence']


class V2Policy(ContractModel):
    policy_version: Literal['squat-v2-anchor-only-v1'] = 'squat-v2-anchor-only-v1'
    bounds_detector: Literal['not_implemented_abstain'] = 'not_implemented_abstain'
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    closure: Literal['[)'] = '[)'
    component_evidence: Literal['explicit_method_observed_support_required'] = 'explicit_method_observed_support_required'

    def content_hash(self):
        return content_hash(self)


class Support(ContractModel):
    """Inclusive input support, distinct from half-open motion ownership."""
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)

    @model_validator(mode='after')
    def ordered(self):
        if self.end_frame < self.start_frame:
            raise ValueError('Support must be ordered')
        return self


class V2Reason(ContractModel):
    reason: Reason
    category: Literal[ErrorCategory.INSUFFICIENT_DATA] = ErrorCategory.INSUFFICIENT_DATA
    stage: Literal[Stage.SEGMENTATION] = Stage.SEGMENTATION
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    support: Support | None = None


class Evidence(ContractModel):
    method: Literal['legacy_pvp_proxy', 'declared_dwell_evidence', 'declared_point_evidence']
    input_hash: Hash
    support: Support


class Event(ContractModel):
    event_id: str = Field(min_length=1)
    kind: Literal['top', 'bottom_anchor', 'bottom_arrival', 'bottom_departure']
    run_id: str = Field(min_length=1)
    side: Literal['left', 'right']
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0, strict=True)
    evidence: Evidence


class EventComponent(ContractModel):
    status: Status
    event: Event | None = None
    reasons: tuple[V2Reason, ...] = ()

    @model_validator(mode='after')
    def availability(self):
        if self.status == 'candidate':
            if self.event is None or self.reasons:
                raise ValueError('Candidate event needs evidence and no abstention reason')
        elif self.event is not None or not self.reasons:
            raise ValueError('Abstained event must be null with reasons')
        return self


class PauseValue(ContractModel):
    arrival_event_id: str
    departure_event_id: str
    duration_seconds: float = Field(gt=0, strict=True)
    unit: Literal['seconds'] = 'seconds'
    closure: Literal['[)'] = '[)'


class PauseComponent(ContractModel):
    status: Status
    value: PauseValue | None = None
    reasons: tuple[V2Reason, ...] = ()

    @model_validator(mode='after')
    def availability(self):
        if self.status == 'candidate':
            if self.value is None or self.reasons:
                raise ValueError('Candidate pause needs positive duration and evidence')
        elif self.value is not None or not self.reasons:
            raise ValueError('Unknown pause must be null with reasons, never zero')
        return self


class MotionInterval(ContractModel):
    phase: Literal['Descent', 'Bottom', 'Ascent']
    start_event_id: str
    end_event_id: str
    closure: Literal['[)'] = '[)'
    basis: Literal['dwell_bounds', 'point_turn_marker_span']


class PhaseComponent(ContractModel):
    status: Status
    intervals: tuple[MotionInterval, ...] = ()
    reasons: tuple[V2Reason, ...] = ()

    @model_validator(mode='after')
    def availability(self):
        if self.status == 'candidate':
            if not self.intervals or self.reasons:
                raise ValueError('Candidate phase needs intervals')
        elif self.intervals or not self.reasons:
            raise ValueError('Abstained phase has no intervals and needs reasons')
        return self


class Bottom(ContractModel):
    mode: Literal['dwell_candidate', 'point_turn_at_resolution', 'bounds_unresolved']
    anchor_event_id: str
    arrival: EventComponent
    departure: EventComponent
    pause: PauseComponent

    @model_validator(mode='after')
    def mode_semantics(self):
        if self.mode == 'dwell_candidate':
            if any(c.status != 'candidate' for c in (self.arrival, self.departure, self.pause)):
                raise ValueError('Dwell requires independent arrival, departure and pause evidence')
        elif self.mode == 'point_turn_at_resolution':
            if any(c.status != 'unavailable' for c in (self.arrival, self.departure, self.pause)):
                raise ValueError('Point turn does not invent physical bounds or zero pause')
            if any(not any(r.reason == 'pause_unresolved_at_resolution' for r in c.reasons)
                   for c in (self.arrival, self.departure, self.pause)):
                raise ValueError('Point components need resolution reason')
        elif self.pause.status == 'candidate':
            raise ValueError('Unresolved bounds cannot produce a pause')
        return self


class Cycle(ContractModel):
    cycle_id: str
    run_id: str
    side: Literal['left', 'right']
    start_event_id: str
    end_event_id: str
    completeness: Literal['complete'] = 'complete'
    status: Literal['candidate'] = 'candidate'
    bottom: Bottom
    motion: PhaseComponent


class TemporalRun(ContractModel):
    run_id: str
    side: Literal['left', 'right']
    support: Support


def event_id(input_hash: str, run_id: str, kind: str, frame: int) -> str:
    return hashlib.sha256(f'{input_hash}:{run_id}:{kind}:{frame}'.encode()).hexdigest()


class SquatTemporalResultV2(ContractModel):
    segmentation_schema_version: Literal['0.2.0'] = '0.2.0'
    contract_version: Literal['squat-temporal-components-v2'] = 'squat-temporal-components-v2'
    producer_version: Literal['squat-v2-conservative-adapter-v1', 'contract_acceptance_only']
    experimental_label: Literal['EXPERIMENTAL / NOT VALIDATED'] = 'EXPERIMENTAL / NOT VALIDATED'
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    official_gate_status: Literal['BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT'] = 'BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT'
    assessment: AssessmentConfig = Field(default_factory=AssessmentConfig)
    policy: V2Policy = Field(default_factory=V2Policy)
    policy_hash: Hash
    legacy: SquatSegmentationResult
    legacy_hash: Hash
    runs: tuple[TemporalRun, ...]
    events: tuple[Event, ...]
    cycles: tuple[Cycle, ...]
    candidate_count: int = Field(ge=0, strict=True)
    diagnostics: tuple[V2Reason, ...] = ()

    @model_validator(mode='after')
    def linkage_and_semantics(self):
        legacy = SquatSegmentationResult.model_validate_json(strict_json(self.legacy), strict=True)
        if (self.legacy_hash != legacy.content_hash() or legacy.profile_hash != PROFILE_HASH
                or self.policy_hash != self.policy.content_hash()):
            raise ValueError('Input/profile/policy linkage mismatch; S3 profile is frozen')
        source = legacy.input
        if source is None:
            raise ValueError('This slice requires explicit synthetic input; no silent blocked migration')
        if self.candidate_count != len(self.cycles) or len(self.cycles) != legacy.candidate_count:
            raise ValueError('Only full legacy candidates count; phase completeness is independent')
        runs = {r.run_id: r for r in self.runs}
        if len(runs) != len(self.runs):
            raise ValueError('Duplicate run IDs')
        expected_runs = []
        for sample in source.samples:
            if sample.value is None:
                continue
            i = sample.frame_index
            if i == 0 or source.samples[i-1].value is None or source.samples[i-1].side_source != sample.side_source:
                expected_runs.append([i, i, sample.side_source])
            else:
                expected_runs[-1][1] = i
        if [(r.support.start_frame, r.support.end_frame, r.side) for r in self.runs] != [tuple(r) for r in expected_runs]:
            raise ValueError('Runs must preserve every finite input sample and every discontinuity')
        events = {e.event_id: e for e in self.events}
        if len(events) != len(self.events):
            raise ValueError('Duplicate events, including shared top')
        for event in self.events:
            run = runs.get(event.run_id)
            support = event.evidence.support
            if run is None or event.side != run.side or not (run.support.start_frame <= support.start_frame <= event.frame_index <= support.end_frame <= run.support.end_frame):
                raise ValueError('Event support cannot cross a run/gap/side boundary')
            sample = source.samples[event.frame_index]
            if (event.timestamp_seconds != sample.timestamp_seconds or event.evidence.input_hash != legacy.input_hash
                    or event.event_id != event_id(legacy.input_hash, event.run_id, event.kind, event.frame_index)):
                raise ValueError('Event identity/time/source mismatch')
            if event.evidence.method != 'legacy_pvp_proxy':
                if any(set(s.joint_provenance) != {JointProvenance.OBSERVED}
                       for s in source.samples[support.start_frame:support.end_frame+1]):
                    raise ValueError('Experimental bound/point evidence requires observed support')
            if event.kind in ('bottom_arrival', 'bottom_departure') and event.evidence.method != 'declared_dwell_evidence':
                raise ValueError('PVP proxy cannot establish physical bounds')
        used = set()
        for cycle, rep in zip(self.cycles, legacy.repetitions):
            s, b, e = (events.get(key) for key in (cycle.start_event_id, cycle.bottom.anchor_event_id, cycle.end_event_id))
            if any(x is None for x in (s, b, e)):
                raise ValueError('Missing referenced event')
            if (s.kind, b.kind, e.kind) != ('top', 'bottom_anchor', 'top') or (s.frame_index, b.frame_index, e.frame_index) != (rep.start_frame, rep.bottom_frame, rep.end_frame):
                raise ValueError('V2 must preserve exact legacy PVP anchors')
            if any(x.run_id != cycle.run_id or x.side != cycle.side for x in (s, b, e)) or cycle.side != rep.side_source:
                raise ValueError('Cycle cannot fuse runs or sides')
            used.update((s.event_id, b.event_id, e.event_id))
            bottom = cycle.bottom
            if bottom.mode == 'dwell_candidate':
                a, d = bottom.arrival.event, bottom.departure.event
                for ev, kind in ((a, 'bottom_arrival'), (d, 'bottom_departure')):
                    if ev.kind != kind or events.get(ev.event_id) != ev or ev.run_id != cycle.run_id or ev.side != cycle.side:
                        raise ValueError('Dwell bound must be a linked event in this run')
                    used.add(ev.event_id)
                if not (s.frame_index < a.frame_index < d.frame_index < e.frame_index and a.frame_index <= b.frame_index <= d.frame_index):
                    raise ValueError('Dwell requires S<A<D<E and anchor within [A,D]')
                pause = bottom.pause.value
                if (pause.arrival_event_id != a.event_id or pause.departure_event_id != d.event_id
                        or abs(pause.duration_seconds - (d.timestamp_seconds-a.timestamp_seconds)) > 1e-9):
                    raise ValueError('Pause duration must be departure-arrival, no frame padding')
                expected = [('Descent', s, a, 'dwell_bounds'), ('Bottom', a, d, 'dwell_bounds'), ('Ascent', d, e, 'dwell_bounds')]
            elif bottom.mode == 'point_turn_at_resolution':
                if b.evidence.method != 'declared_point_evidence':
                    raise ValueError('A legacy bottom marker alone cannot establish a point turn')
                expected = [('Descent', s, b, 'point_turn_marker_span'), ('Ascent', b, e, 'point_turn_marker_span')]
            else:
                expected = []
                if bottom.arrival.event or bottom.departure.event:
                    raise ValueError('Partial bound producer is outside this conservative slice')
                if cycle.motion.status == 'candidate':
                    raise ValueError('Unresolved cycle has no localized phase evidence')
            if expected:
                actual = [(m.phase, m.start_event_id, m.end_event_id, m.basis) for m in cycle.motion.intervals]
                if cycle.motion.status != 'candidate' or actual != [(p, x.event_id, y.event_id, basis) for p,x,y,basis in expected]:
                    raise ValueError('Motion intervals must follow exact half-open mode semantics')
            if self.producer_version != 'contract_acceptance_only' and bottom.mode != 'bounds_unresolved':
                raise ValueError('This producer has no arrival/departure/point detector')
        if used != set(events):
            raise ValueError('Orphan event outside supported full candidates')
        if len({c.cycle_id for c in self.cycles}) != len(self.cycles):
            raise ValueError('Duplicate cycle IDs')
        for reason in self.diagnostics:
            if reason.support and reason.support.end_frame >= len(source.samples):
                raise ValueError('Diagnostic support outside input')
        if not self.cycles and not self.diagnostics:
            raise ValueError('No candidate requires an explicit reason')
        return self

    def content_hash(self):
        return content_hash(self)


def read_temporal_json(payload: str) -> SquatSegmentationResult | SquatTemporalResultV2:
    """Explicit version dispatch; no v1 migration or inferred v2 components."""
    data = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))
    if not isinstance(data, dict):
        raise ValueError('Temporal JSON must be an object')
    version = data.get('segmentation_schema_version')
    if version == '0.1.0':
        return SquatSegmentationResult.model_validate_json(payload, strict=True)
    if version == '0.2.0':
        return SquatTemporalResultV2.model_validate_json(payload, strict=True)
    raise ValueError(f'Unsupported temporal schema version: {version!r}')
