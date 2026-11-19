"""Versioned descriptive Gym Result Feedback; assessment is always disabled."""
import math
from typing import Literal

from pydantic import Field,model_validator

from ..config import AssessmentConfig
from ..contracts import ContractModel,Exercise
from ..features import GEOMETRY_REGISTRY
from ..features.models import content_hash,Hash
from ..flags import FlagCode,StructuredFlag
from ..pose.models import JointObservation
from ..preprocessing.models import ProcessedJoint
from ..quality.models import QualityReport
from ..segmentation import SquatSegmentationResult

SQUAT_IDS=tuple(f.feature_id for f in GEOMETRY_REGISTRY.for_exercise(Exercise.BODY_WEIGHT_SQUATS))
Phase=Literal['Start/Top','Descent','Bottom','Ascent','Return/Top','Unavailable']


class EvidenceInterval(ContractModel):
    start_frame:int=Field(ge=0,strict=True)
    end_frame:int=Field(ge=0,strict=True)
    start_seconds:float=Field(ge=0,strict=True)
    end_seconds:float=Field(ge=0,strict=True)

    @model_validator(mode='after')
    def ordered(self):
        if self.end_frame<self.start_frame or self.end_seconds<self.start_seconds:raise ValueError('Evidence interval must be ordered/inclusive')
        return self


class FeedbackSource(ContractModel):
    provenance_version:Literal['squat-feedback-source-v1']='squat-feedback-source-v1'
    kind:Literal['synthetic_contract','real_preview']
    evidence_status:Literal['synthetic_fixture','dataset_viewpoint_unknown']
    video_id:str=Field(min_length=1)
    relative_path:str|None=None
    video_byte_sha256:Hash|None=None
    metadata_byte_sha256:Hash|None=None
    fps:float=Field(gt=0,strict=True)
    frame_count:int=Field(ge=0,strict=True)
    width:int=Field(gt=0,strict=True)
    height:int=Field(gt=0,strict=True)
    timestamp_policy:Literal['fps_derived_cfr_assumption']='fps_derived_cfr_assumption'
    preview_sampling:Literal['all_synthetic_frames','three_preview_frames']
    preprocessing_hash:Hash|None=None
    raw_sequence_hash:Hash|None=None
    quality_hash:Hash|None=None
    documented:Literal[False]=False

    @model_validator(mode='after')
    def source_boundary(self):
        if self.kind=='synthetic_contract':
            if (not self.video_id.startswith('synthetic-') or self.evidence_status!='synthetic_fixture'
                    or self.video_byte_sha256 is not None or self.relative_path is not None
                    or self.preview_sampling!='all_synthetic_frames' or self.preprocessing_hash is None
                    or self.quality_hash is None or self.raw_sequence_hash is None):
                raise ValueError('Synthetic fixture needs explicit source hashes and cannot claim a real video')
        else:
            from pathlib import PurePosixPath
            p=PurePosixPath(self.relative_path or '')
            if (not str(p).startswith('Data/UCF-101_GYM/BodyWeightSquats/') or '..' in p.parts
                    or p.is_absolute() or '\\' in str(p) or p.name!=self.video_id
                    or self.evidence_status!='dataset_viewpoint_unknown' or self.video_byte_sha256 is None
                    or self.metadata_byte_sha256 is None or self.preview_sampling!='three_preview_frames'
                    or any(x is not None for x in (self.preprocessing_hash,self.raw_sequence_hash,self.quality_hash))):
                raise ValueError('Real preview must remain within the allowed dataset and cannot claim unrun stages')
        return self


class FeedbackMetric(ContractModel):
    feature_id:str
    definition_hash:Hash
    unit:Literal['degrees']='degrees'
    status:Literal['available','unavailable']
    value:float|None=Field(ge=0,le=180,strict=True)
    measurement_validation_status:Literal['unvalidated']='unvalidated'
    temporal_analysis_eligible:Literal[False]=False
    evidence:EvidenceInterval
    quality_action:Literal['allow','degrade','skip','not_run']
    observation_hash:Hash|None
    input_hash:Hash|None
    joint_inputs:tuple[ProcessedJoint,...]
    reasons:tuple[FlagCode,...]
    flags:tuple[StructuredFlag,...]

    @model_validator(mode='after')
    def canonical_and_safe(self):
        if self.feature_id not in SQUAT_IDS:raise ValueError('Only the five canonical squat angles belong in feedback')
        definition=GEOMETRY_REGISTRY.get(self.feature_id)
        if self.definition_hash!=definition.content_hash():raise ValueError('Feature definition hash mismatch')
        if self.quality_action=='not_run':
            if self.observation_hash is not None or self.input_hash is not None or self.joint_inputs:
                raise ValueError('No fabricated observation/input/joints for unrun quality')
        elif (self.observation_hash is None or self.input_hash is None
              or tuple(j.joint for j in self.joint_inputs)!=definition.required_joints):
            raise ValueError('Retain exact observation identity and ordered actual input joints')
        if self.status=='available':
            if (self.value is None or self.quality_action!='allow' or self.reasons or self.flags
                    or any(not j.usable or j.x is None or j.y is None for j in self.joint_inputs)):
                raise ValueError('Available measurement requires finite usable inputs and metric allow')
        elif self.value is not None or not self.flags or self.reasons!=tuple(dict.fromkeys(f.code for f in self.flags)):
            raise ValueError('Unavailable feedback retains null value and structured reasons')
        return self


class FeedbackFrame(ContractModel):
    frame_index:int=Field(ge=0,strict=True)
    timestamp_seconds:float=Field(ge=0,strict=True)
    phase:Phase
    phase_evidence:EvidenceInterval|None
    phase_reasons:tuple[FlagCode,...]
    source_frame_hash:Hash|None
    overlay_source:Literal['raw_pose','not_run']
    coordinate_space:Literal['image_normalized_xy']='image_normalized_xy'
    overlay_joints:tuple[JointObservation,...]
    preview_image:str|None=Field(default=None,pattern=r'^frames/frame-[0-9]+\.jpg$')
    preview_image_sha256:Hash|None=None
    metrics:tuple[FeedbackMetric,...]

    @model_validator(mode='after')
    def consistent_frame(self):
        from ..pose.mapping import JointName
        if tuple(m.feature_id for m in self.metrics)!=SQUAT_IDS:raise ValueError('Every frame reports all five canonical metrics in registry order')
        if self.overlay_source=='raw_pose':
            if self.source_frame_hash is None or len(self.overlay_joints)!=13 or {j.joint for j in self.overlay_joints}!=set(JointName):
                raise ValueError('Raw overlay must retain the complete mapped pose and source hash')
        elif self.overlay_joints or self.source_frame_hash is not None:raise ValueError('Unrun overlay cannot invent skeleton joints')
        if (self.preview_image is None)!=(self.preview_image_sha256 is None):raise ValueError('Preview image needs byte identity')
        if self.phase=='Unavailable':
            if self.phase_evidence is not None or not self.phase_reasons:raise ValueError('Unavailable phase needs reasons and no invented interval')
        elif self.phase_evidence is None or self.phase_reasons:raise ValueError('Candidate phase needs a supporting interval')
        for metric in self.metrics:
            if (metric.evidence.start_frame!=self.frame_index or metric.evidence.end_frame!=self.frame_index
                    or metric.evidence.start_seconds!=self.timestamp_seconds or metric.evidence.end_seconds!=self.timestamp_seconds):
                raise ValueError('Metric evidence must identify this exact frame/time')
        return self


class FeedbackMessage(ContractModel):
    kind:Literal['data_status','experimental_description','assessment_blocked']
    text:str=Field(min_length=1)
    evidence:EvidenceInterval|None=None


class SquatFeedbackResult(ContractModel):
    feedback_schema_version:Literal['0.1.0']='0.1.0'
    feedback_version:Literal['gym-squat-feedback-v1']='gym-squat-feedback-v1'
    exercise:Literal[Exercise.BODY_WEIGHT_SQUATS]=Exercise.BODY_WEIGHT_SQUATS
    experimental_label:Literal['EXPERIMENTAL / NOT VALIDATED']='EXPERIMENTAL / NOT VALIDATED'
    official_eligibility:Literal[False]=False
    official_gate_status:Literal['BLOCKED_PENDING_STAGE5_7']='BLOCKED_PENDING_STAGE5_7'
    status:Literal['synthetic_experiment','blocked']
    assessment:AssessmentConfig=Field(default_factory=AssessmentConfig)
    form_score:None=None
    technical_feedback:None=None
    registry_version:Literal['gym-feature-registry-v2']='gym-feature-registry-v2'
    registry_hash:Hash
    source:FeedbackSource
    source_hash:Hash
    quality:QualityReport|None
    segmentation:SquatSegmentationResult
    segmentation_hash:Hash
    frames:tuple[FeedbackFrame,...]
    messages:tuple[FeedbackMessage,...]
    flags:tuple[StructuredFlag,...]

    @model_validator(mode='after')
    def source_and_scope(self):
        if (self.registry_hash!=GEOMETRY_REGISTRY.content_hash() or self.source_hash!=content_hash(self.source)
                or self.segmentation_hash!=self.segmentation.content_hash()):raise ValueError('Feedback source/registry/segmentation hash mismatch')
        if self.assessment.enabled is not False:raise ValueError('Assessment stays disabled')
        if self.source.kind=='real_preview':
            if (self.status!='blocked' or self.quality is not None or self.segmentation.status!='blocked'
                    or any(f.phase!='Unavailable' or f.overlay_source!='not_run' or any(m.quality_action!='not_run' for m in f.metrics) for f in self.frames)):
                raise ValueError('Undocumented real preview cannot fabricate pipeline outputs')
        else:
            if (self.status!='synthetic_experiment' or self.quality is None
                    or self.source.quality_hash!=content_hash(self.quality)
                    or self.quality.video_id!=self.source.video_id
                    or self.quality.total_frames!=self.source.frame_count
                    or len(self.frames)!=self.source.frame_count):raise ValueError('Synthetic feedback requires its exact source/quality snapshots')
            if self.segmentation.input is not None and (self.segmentation.input.source_hash!=self.source.preprocessing_hash
                    or self.segmentation.input.quality!=self.quality or self.segmentation.input.fps!=self.source.fps
                    or len(self.segmentation.input.samples)!=self.source.frame_count):raise ValueError('Temporal source/gate must be independently linked')
        previous=-1
        from .squat_builder import phase_at
        for frame in self.frames:
            if not previous<frame.frame_index<self.source.frame_count or not math.isclose(frame.timestamp_seconds,frame.frame_index/self.source.fps,abs_tol=1e-9,rel_tol=0):
                raise ValueError('Feedback timeline must preserve source index/timestamps')
            phase,evidence,reasons=phase_at(self.segmentation,frame.frame_index,self.source.fps)
            if (frame.phase,frame.phase_evidence,frame.phase_reasons)!=(phase,evidence,reasons):raise ValueError('Frame phase must match candidate evidence, never an invented label')
            if self.quality is not None:
                for metric in frame.metrics:
                    gate=self.quality.metric_eligibility.get(metric.feature_id)
                    expected=gate.action if gate is not None else 'not_run'
                    if metric.quality_action!=expected:raise ValueError('Metric action cannot promote the upstream QualityReport')
                    if metric.status=='available' and (gate is None or not gate.eligible
                            or set(gate.required_joints)!=set(GEOMETRY_REGISTRY.get(metric.feature_id).required_joints)
                            or len(gate.required_joints)!=len(metric.joint_inputs)
                            or self.quality.viewpoint not in GEOMETRY_REGISTRY.get(metric.feature_id).viewpoint_policy.supported):
                        raise ValueError('Available metric requires exact canonical joints/viewpoint eligibility')
            previous=frame.frame_index
        return self

    def content_hash(self):return content_hash(self)
