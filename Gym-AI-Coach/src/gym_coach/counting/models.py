"""Additive, strict count/annotation contracts; independent of temporal v1/v2."""
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator
from ..contracts import ContractModel

PROTOCOL = 'squat-count-video-annotation-v1'
VERSION = 'squat-count-research-v1'
Hash = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Text = Annotated[str, Field(min_length=1, pattern=r'\S')]
STATUS = dict(validation_status='NOT_VALIDATED', official_eligibility=False,
              assessment=False, reference_status='REFERENCE_PENDING',
              g57='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT', g6='IN_PROGRESS')


def canonical_hash(value):
    if isinstance(value, ContractModel):
        value = value.model_dump(mode='json')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Model(ContractModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False, strict=True)


class Reason(Model):
    code: Text
    category: Literal['input_error', 'insufficient_data', 'operational_error', 'informational_observation']
    stage: Literal['segmentation', 'reporting'] = 'reporting'
    source_tier: Literal['S3'] = 'S3'
    message: Text


def reason(code, message, category='insufficient_data'):
    return Reason(code=code, message=message, category=category)


class VideoIdentity(Model):
    video_id: Text
    path: str = Field(pattern=r'^Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g\d{2}_c\d{2}\.avi$')
    sha256: Hash
    group_id: str = Field(pattern=r'^g\d{2}$')
    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    timestamp_policy: Literal['fps_derived_cfr_assumption'] = 'fps_derived_cfr_assumption'

    @model_validator(mode='after')
    def linkage(self):
        if self.path.rsplit('/', 1)[-1] != self.video_id or re.search(r'_(g\d{2})_', self.video_id)[1] != self.group_id:
            raise ValueError('Video ID/path/data group mismatch; group is not subject identity')
        values = (self.fps, self.frame_count, self.width, self.height)
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError('Metadata must be complete or unavailable as a whole')
        return self


class Moment(Model):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)


class Span(Model):
    start: Moment
    end: Moment

    @model_validator(mode='after')
    def order(self):
        if self.end.frame_index < self.start.frame_index or self.end.timestamp_seconds < self.start.timestamp_seconds:
            raise ValueError('Inclusive sample interval must be ordered')
        return self


class Bottom(Model):
    status: Literal['event', 'interval', 'unavailable']
    support: Span | None = None
    reason: Text | None = None

    @model_validator(mode='after')
    def shape(self):
        if self.status == 'unavailable':
            if self.support is not None or not self.reason:
                raise ValueError('Unavailable bottom requires null support and reason')
        elif self.support is None or self.reason is not None:
            raise ValueError('Observed bottom requires support')
        elif (self.support.start.frame_index == self.support.end.frame_index) != (self.status == 'event'):
            raise ValueError('Event is one sample; interval spans more than one sample')
        return self


class Cycle(Model):
    cycle_id: Text
    interval: Span
    bottom: Bottom

    @model_validator(mode='after')
    def shape(self):
        if self.interval.start.frame_index >= self.interval.end.frame_index:
            raise ValueError('A full cycle must have positive duration')
        if self.bottom.support and not (
                self.interval.start.frame_index < self.bottom.support.start.frame_index
                <= self.bottom.support.end.frame_index < self.interval.end.frame_index):
            raise ValueError('Bottom must lie strictly within full cycle')
        return self


class Partial(Model):
    kind: Literal['clip_start', 'clip_end', 'both_edges']
    interval: Span
    reason: Text


class Uncertain(Model):
    interval: Span
    affects_count: bool
    reason: Text


def overlap(a, b):
    return max(a.start.frame_index, b.start.frame_index) <= min(a.end.frame_index, b.end.frame_index)


def check_timeline(source, cycles, partials=(), uncertain=()):
    spans = [c.interval for c in cycles] + [c.bottom.support for c in cycles if c.bottom.support]
    spans += [x.interval for x in (*partials, *uncertain)]
    for span in spans:
        for m in (span.start, span.end):
            if (source.fps is None or m.frame_index >= source.frame_count or
                    not math.isclose(m.timestamp_seconds, m.frame_index/source.fps, rel_tol=0, abs_tol=1e-9)):
                raise ValueError('Frame/time outside exact source CFR timeline')
    if len({c.cycle_id for c in cycles}) != len(cycles):
        raise ValueError('Duplicate cycle IDs')
    for a, b in zip(cycles, cycles[1:]):
        if a.interval.end.frame_index > b.interval.start.frame_index:
            raise ValueError('Cycles must be ordered, non-overlapping; shared top allowed')
    for p in partials:
        if p.kind in ('clip_start', 'both_edges') and p.interval.start.frame_index != 0:
            raise ValueError('Leading partial must touch first sample')
        if p.kind in ('clip_end', 'both_edges') and p.interval.end.frame_index != source.frame_count-1:
            raise ValueError('Trailing partial must touch last sample')
    if any(overlap(c.interval, x.interval) for c in cycles for x in (*partials, *uncertain)):
        raise ValueError('Full reference cycle cannot cross partial/uncertain support')


class AnnotationPass(Model):
    schema_version: Literal['squat-count-labels/0.1.0'] = 'squat-count-labels/0.1.0'
    protocol_version: Literal['squat-count-video-annotation-v1'] = PROTOCOL
    protocol_sha256: Hash
    source: VideoIdentity
    status: Literal['pending', 'completed'] = 'pending'
    annotator: Text | None = None
    method: Literal['manual_video_review', 'synthetic_fixture'] | None = None
    annotated_at: datetime | None = None
    independent_of_predictions: bool | None = None
    blinded_to_other_pass: bool | None = None
    provenance_note: Text | None = None
    subject_description: Text | None = None
    count_status: Literal['pending', 'available', 'unavailable'] = 'pending'
    count: int | None = Field(default=None, ge=0)
    count_reason: Text | None = 'annotation_pending'
    boundaries_status: Literal['pending', 'exhaustive', 'partial', 'unavailable'] = 'pending'
    boundaries_reason: Text | None = 'annotation_pending'
    cycles: tuple[Cycle, ...] = ()
    partials: tuple[Partial, ...] = ()
    uncertain: tuple[Uncertain, ...] = ()

    @model_validator(mode='after')
    def label_policy(self):
        check_timeline(self.source, self.cycles, self.partials, self.uncertain)
        provenance = (self.annotator, self.method, self.annotated_at, self.independent_of_predictions,
                      self.blinded_to_other_pass, self.provenance_note, self.subject_description)
        if self.status == 'pending':
            if (any(v is not None for v in provenance) or self.count_status != 'pending' or
                    self.count is not None or self.boundaries_status != 'pending' or
                    not self.count_reason or not self.boundaries_reason or self.cycles or self.partials or self.uncertain):
                raise ValueError('Pending template must contain no invented labels/person/time')
            return self
        if (not self.annotator or not self.method or not self.annotated_at or
                self.annotated_at.utcoffset() is None or self.independent_of_predictions is not True or
                self.blinded_to_other_pass is None or not self.provenance_note):
            raise ValueError('Completed labels require independent source, person/method/time/provenance')
        if self.count_status == 'pending' or self.boundaries_status == 'pending':
            raise ValueError('Completed pass cannot retain pending components')
        if self.count_status == 'available':
            if (self.count is None or self.count_reason is not None or not self.subject_description or
                    self.source.fps is None or any(x.affects_count for x in self.uncertain) or len(self.cycles) > self.count):
                raise ValueError('Available whole-clip count requires complete count evidence')
        elif self.count is not None or not self.count_reason:
            raise ValueError('Uncertain whole-clip count is null, never zero')
        if self.boundaries_status == 'exhaustive':
            if self.count_status != 'available' or self.count != len(self.cycles) or self.boundaries_reason is not None:
                raise ValueError('Exhaustive boundaries must enumerate all counted cycles')
        elif not self.boundaries_reason or (self.boundaries_status == 'unavailable' and self.cycles):
            raise ValueError('Non-exhaustive boundary labels need reason; unavailable has no cycles')
        return self


class Decision(Model):
    field: Text
    rationale: Text
    reviewer: Text
    decided_at: datetime
    pass_a_sha256: Hash
    pass_b_sha256: Hash
    final_sha256: Hash

    @model_validator(mode='after')
    def time(self):
        if self.decided_at.utcoffset() is None:
            raise ValueError('Adjudication time requires timezone')
        return self


class AnnotationBundle(Model):
    version: Literal['squat-count-adjudication-v1'] = 'squat-count-adjudication-v1'
    data_kind: Literal['real_development', 'synthetic']
    pass_a: AnnotationPass
    pass_b: AnnotationPass
    final: AnnotationPass | None = None
    decisions: tuple[Decision, ...] = ()

    @model_validator(mode='after')
    def linkage(self):
        a, b = self.pass_a, self.pass_b
        rows = (a, b) + ((self.final,) if self.final else ())
        if any(r.source != a.source or r.protocol_sha256 != a.protocol_sha256 for r in rows):
            raise ValueError('Annotation passes must bind exactly one source/protocol')
        if self.data_kind == 'real_development' and any(r.method == 'synthetic_fixture' for r in rows):
            raise ValueError('Synthetic annotation cannot validate real video')
        if any(r.status == 'completed' and not r.blinded_to_other_pass for r in (a, b)):
            raise ValueError('Two passes must be independent and blind to each other')
        if a.status == b.status == 'completed' and a.annotator == b.annotator:
            raise ValueError('Two independent annotators required')
        if self.final is None:
            if self.decisions:
                raise ValueError('No adjudication log without final decision')
        else:
            f = self.final
            if (any(r.status != 'completed' for r in rows) or not self.decisions or
                    f.annotator in (a.annotator, b.annotator) or f.blinded_to_other_pass is not False):
                raise ValueError('Final requires separate adjudicator and complete logged passes')
            for d in self.decisions:
                if (d.reviewer != f.annotator or d.pass_a_sha256 != canonical_hash(a) or
                        d.pass_b_sha256 != canonical_hash(b) or d.final_sha256 != canonical_hash(f) or
                        d.decided_at < max(a.annotated_at, b.annotated_at, f.annotated_at)):
                    raise ValueError('Adjudication hashes/person/time do not match submitted labels')
        return self


class Prediction(Model):
    version: Literal['squat-count-prediction-v1'] = 'squat-count-prediction-v1'
    source: VideoIdentity
    status: Literal['estimated', 'candidates_only', 'abstained', 'failed', 'not_run']
    count: int | None = Field(default=None, ge=0)
    cycles: tuple[Cycle, ...] = ()
    events_exhaustive: bool = False
    algorithm_version: Text
    config_sha256: Hash
    evidence_ref: Text
    reasons: tuple[Reason, ...] = ()
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    official_eligibility: Literal[False] = False
    assessment: Literal[False] = False

    @model_validator(mode='after')
    def result(self):
        check_timeline(self.source, self.cycles)
        if self.status == 'estimated':
            if self.count is None or self.source.fps is None or len(self.cycles) > self.count:
                raise ValueError('Estimated count must be explicit; candidates are not confirmed counts')
            if self.events_exhaustive and self.count != len(self.cycles):
                raise ValueError('Exhaustive predicted events/count mismatch')
        elif (self.count is not None or self.events_exhaustive or not self.reasons or
              (self.status != 'candidates_only' and self.cycles)):
            raise ValueError('Abstention/failure/not_run/candidate-only requires null count and reasons')
        return self


class EvaluationPolicy(Model):
    version: Literal['squat-count-evaluation-v1'] = 'squat-count-evaluation-v1'
    boundary_tolerance_seconds: float = Field(default=0.20, ge=0)
    matching: Literal['monotone_max_cardinality_then_min_boundary_error'] = 'monotone_max_cardinality_then_min_boundary_error'
    mask_policy: Literal['exclude_and_report_prediction_overlap_inclusive'] = 'exclude_and_report_prediction_overlap_inclusive'
    source_tier: Literal['S3'] = 'S3'
