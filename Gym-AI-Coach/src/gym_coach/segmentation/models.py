"""Experimental squat temporal contracts, separate from framewise permission."""
import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..contracts import ContractModel, Exercise, JointProvenance, SourceTier
from ..flags import FlagCode, StructuredFlag
from ..preprocessing.models import InterpolationSource
from ..quality.models import QualityReport

Hash = Annotated[str,Field(pattern=r'^[a-f0-9]{64}$')]
EXPERIMENTAL_LABEL = 'EXPERIMENTAL / NOT VALIDATED'


def strict_json(value):
    return json.dumps(value.model_dump(mode='python',warnings='error'),allow_nan=False,ensure_ascii=False,separators=(',',':'))


def content_hash(value):
    return hashlib.sha256(json.dumps(value.model_dump(mode='json'),sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


class SquatSignalSample(ContractModel):
    frame_index: int = Field(ge=0,strict=True)
    timestamp_seconds: float = Field(ge=0,strict=True)
    value: float | None = Field(ge=0,le=180,strict=True)
    side_source: Literal['left','right']
    joint_provenance: tuple[JointProvenance,...] = (JointProvenance.OBSERVED,)*3
    imputation_references: tuple[InterpolationSource,...] = ()
    source_observation_hash: Hash | None = None
    reasons: tuple[FlagCode,...] = ()

    @model_validator(mode='after')
    def safe_value(self):
        if len(self.joint_provenance)!=3:
            raise ValueError('A single knee signal needs the provenance of hip/knee/ankle')
        if self.value is None:
            if not self.reasons:raise ValueError('Unavailable signal needs a reason, never a fake value')
        elif self.reasons or not set(self.joint_provenance)<={JointProvenance.OBSERVED,JointProvenance.IMPUTED}:
            raise ValueError('Available signal requires safe provenance and no unavailable reasons')
        expected=sum(p==JointProvenance.IMPUTED for p in self.joint_provenance)
        if len(self.imputation_references)!=expected:raise ValueError('Keep each imputed joint endpoint reference')
        for ref in self.imputation_references:
            if not (ref.left_frame_index<self.frame_index<ref.right_frame_index and
                    ref.left_timestamp_seconds<self.timestamp_seconds<ref.right_timestamp_seconds):
                raise ValueError('Imputation references must bracket the signal frame')
        return self


class SquatTemporalInput(ContractModel):
    temporal_input_schema_version: Literal['0.1.0'] = '0.1.0'
    input_version: Literal['squat-temporal-input-v1'] = 'squat-temporal-input-v1'
    exercise: Literal[Exercise.BODY_WEIGHT_SQUATS] = Exercise.BODY_WEIGHT_SQUATS
    scope: Literal['synthetic_contract_test'] = 'synthetic_contract_test'
    video_id: str = Field(pattern=r'^synthetic-.+')
    fps: float = Field(gt=0,strict=True)
    timestamp_policy: Literal['fps_derived_cfr_assumption'] = 'fps_derived_cfr_assumption'
    polarity: Literal['PEAK_VALLEY_PEAK'] = 'PEAK_VALLEY_PEAK'
    source_hash: Hash
    evidence_basis: Literal['synthetic_signal_fixture','genuine_synthetic_qualitygate'] = 'synthetic_signal_fixture'
    quality: QualityReport | None = None
    quality_hash: Hash | None = None
    samples: tuple[SquatSignalSample,...]

    @model_validator(mode='after')
    def timeline_and_authority(self):
        for i,sample in enumerate(self.samples):
            if sample.frame_index!=i or not math.isclose(sample.timestamp_seconds,i/self.fps,abs_tol=1e-9,rel_tol=0):
                raise ValueError('Temporal input must retain contiguous zero-based CFR identities')
        if self.evidence_basis=='synthetic_signal_fixture':
            if self.quality is not None or self.quality_hash is not None:
                raise ValueError('Pure signal fixtures cannot claim an upstream QualityReport')
        else:
            from ..features import GEOMETRY_REGISTRY
            if (self.quality is None or self.quality_hash!=content_hash(self.quality)
                    or self.quality.video_id!=self.video_id or self.quality.exercise!=self.exercise
                    or self.quality.total_frames!=len(self.samples) or self.quality.viewpoint.value!='side'
                    or not self.quality.eligibility.eligible or self.quality.eligibility.action!='allow'):
                raise ValueError('Separate temporal gate requires exact synthetic source and sequence allow')
            for key in ('squat_knee_angle_left_deg','squat_knee_angle_right_deg'):
                metric=self.quality.metric_eligibility.get(key)
                if (metric is None or not metric.eligible or metric.action!='allow'
                        or set(metric.required_joints)!=set(GEOMETRY_REGISTRY.get(key).required_joints)
                        or len(metric.required_joints)!=3):
                    raise ValueError('Separate temporal gate requires exact bilateral knee metrics allow')
        return self

    def content_hash(self):return content_hash(self)


class SquatSegmentationProfile(ContractModel):
    profile_schema_version: Literal['0.1.0'] = '0.1.0'
    profile_version: Literal['squat-temporal-s3-v1'] = 'squat-temporal-s3-v1'
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    validation_status: Literal['NOT_VALIDATED'] = 'NOT_VALIDATED'
    signal: Literal['squat_knee_angle'] = 'squat_knee_angle'
    side_policy: Literal['explicit_runs_no_fusion'] = 'explicit_runs_no_fusion'
    smoothing_method: Literal['centered_moving_mean'] = 'centered_moving_mean'
    smoothing_span_seconds: float = Field(default=.2,ge=0,strict=True)
    min_excursion_degrees: float = Field(default=5.,gt=0,le=180,strict=True)
    min_half_cycle_seconds: float = Field(default=5/30,gt=0,strict=True)

    def content_hash(self):return content_hash(self)


class RepetitionCandidate(ContractModel):
    start_frame: int = Field(ge=0,strict=True)
    bottom_frame: int = Field(ge=0,strict=True)
    end_frame: int = Field(ge=0,strict=True)
    start_seconds: float = Field(ge=0,strict=True)
    bottom_seconds: float = Field(ge=0,strict=True)
    end_seconds: float = Field(ge=0,strict=True)
    side_source: Literal['left','right']

    @model_validator(mode='after')
    def ordered(self):
        if not (self.start_frame<self.bottom_frame<self.end_frame and self.start_seconds<self.bottom_seconds<self.end_seconds):
            raise ValueError('Candidate must be ordered top_start < bottom < top_end')
        return self

    @property
    def descent_seconds(self):return self.bottom_seconds-self.start_seconds
    @property
    def ascent_seconds(self):return self.end_seconds-self.bottom_seconds


class SquatSegmentationResult(ContractModel):
    segmentation_schema_version: Literal['0.1.0'] = '0.1.0'
    algorithm_version: Literal['squat-pvp-hysteresis-v1'] = 'squat-pvp-hysteresis-v1'
    experimental_label: Literal['EXPERIMENTAL / NOT VALIDATED'] = EXPERIMENTAL_LABEL
    official_eligibility: Literal[False] = False
    assessment_enabled: Literal[False] = False
    input: SquatTemporalInput | None
    input_hash: Hash | None
    profile: SquatSegmentationProfile
    profile_hash: Hash
    status: Literal['experimental_candidates','unavailable','blocked']
    smoothed_values: tuple[Annotated[float,Field(ge=0,le=180,strict=True)] | None,...]
    repetitions: tuple[RepetitionCandidate,...]
    candidate_count: int = Field(ge=0,strict=True)
    flags: tuple[StructuredFlag,...]

    @model_validator(mode='after')
    def audit(self):
        if self.profile_hash!=self.profile.content_hash() or self.candidate_count!=len(self.repetitions):
            raise ValueError('Profile hash/count mismatch')
        if self.input is None:
            if self.input_hash is not None or self.status!='blocked' or self.repetitions or self.smoothed_values or not self.flags:
                raise ValueError('Blocked temporal analysis must have reasons and no fabricated signal/cycles')
            return self
        checked=SquatTemporalInput.model_validate_json(strict_json(self.input),strict=True)
        if self.input_hash!=checked.content_hash() or len(self.smoothed_values)!=len(checked.samples):
            raise ValueError('Temporal input hash/signal count mismatch')
        expected='experimental_candidates' if self.repetitions else 'unavailable'
        if self.status!=expected or (not self.repetitions and not self.flags):raise ValueError('Result status/reasons mismatch')
        for sample,value in zip(checked.samples,self.smoothed_values):
            if (sample.value is None)!=(value is None):raise ValueError('Smoothing cannot fill unresolved gaps')
        previous_end=-1
        for rep in self.repetitions:
            if rep.end_frame>=len(checked.samples) or rep.start_frame<previous_end:
                raise ValueError('Candidates must remain ordered in input bounds')
            selected=checked.samples[rep.start_frame:rep.end_frame+1]
            if any(s.value is None or s.side_source!=rep.side_source for s in selected):
                raise ValueError('Repetition cannot bridge any unresolved gap or side switch')
            if (rep.start_frame==0 or rep.end_frame==len(checked.samples)-1
                    or checked.samples[rep.start_frame-1].value is None
                    or checked.samples[rep.end_frame+1].value is None
                    or checked.samples[rep.start_frame-1].side_source!=rep.side_source
                    or checked.samples[rep.end_frame+1].side_source!=rep.side_source):
                raise ValueError('Unconfirmed boundary cycles remain partial')
            for index,t in ((rep.start_frame,rep.start_seconds),(rep.bottom_frame,rep.bottom_seconds),(rep.end_frame,rep.end_seconds)):
                if checked.samples[index].timestamp_seconds!=t:raise ValueError('Candidate timestamp changed')
            top,bottom,end=(self.smoothed_values[i] for i in (rep.start_frame,rep.bottom_frame,rep.end_frame))
            if min(top-bottom,end-bottom)<self.profile.min_excursion_degrees:
                raise ValueError('Candidate does not meet the declared signal excursion')
            if min(rep.descent_seconds,rep.ascent_seconds)<self.profile.min_half_cycle_seconds:
                raise ValueError('Candidate does not meet the declared temporal spacing')
            previous_end=rep.end_frame
        return self

    def content_hash(self):return content_hash(self)
