"""Structured reasons shared by stages, without executing any quality rules."""

from enum import StrEnum

from pydantic import Field, NonNegativeInt, StrictInt, model_validator

from .contracts import ContractModel, ErrorCategory, SourceTier, Stage


class FlagCode(StrEnum):
    INVALID_CONFIG = "invalid_config"
    INVALID_POSE_INPUT = "invalid_pose_input"
    INVALID_POSE_SEQUENCE = "invalid_pose_sequence"
    INVALID_PREPROCESSING_INPUT = "invalid_preprocessing_input"
    QUALITY_INPUT_MISMATCH = "quality_input_mismatch"
    SMOOTHING_SHORT_RUN = "smoothing_short_run"
    PREPROCESSING_ERROR = "preprocessing_error"
    PREPROCESSING_SERIALIZATION_ERROR = "preprocessing_serialization_error"
    UNKNOWN_FEATURE_ID = "unknown_feature_id"
    UNSUPPORTED_EXERCISE = "unsupported_exercise"
    INVALID_VIEWPOINT = "invalid_viewpoint"
    INVALID_FEATURE_INPUT = "invalid_feature_input"
    INVALID_FEATURE_CONTEXT_INDEX = "invalid_feature_context_index"
    FEATURE_CONTEXT_MISMATCH = "feature_context_mismatch"
    UNSUPPORTED_FEATURE_CALCULATION = "unsupported_feature_calculation"
    INVALID_GEOMETRY_INPUT = "invalid_geometry_input"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    GEOMETRY_NUMERICAL_ERROR = "geometry_numerical_error"
    FEATURE_DEFINITION_BLOCKED = "feature_definition_blocked"
    FEATURE_JOINT_UNUSABLE = "feature_joint_unusable"
    FEATURE_PROVENANCE_REJECTED = "feature_provenance_rejected"
    FEATURE_QUALITY_BLOCKED = "feature_quality_blocked"
    FEATURE_METRIC_ELIGIBILITY_MISSING = "feature_metric_eligibility_missing"
    FEATURE_QUALITY_POLICY_MISMATCH = "feature_quality_policy_mismatch"
    POSE_BACKEND_ERROR = "pose_backend_error"
    POSE_NOT_DETECTED = "pose_not_detected"
    MISSING_JOINT = "missing_joint"
    MISSING_LANDMARK_FIELD = "missing_landmark_field"
    NON_FINITE_LANDMARK = "non_finite_landmark"
    INVALID_LANDMARK_CONFIDENCE = "invalid_landmark_confidence"
    POSE_SERIALIZATION_ERROR = "pose_serialization_error"
    UNREADABLE_VIDEO = "unreadable_video"
    VIDEO_TOO_SHORT = "video_too_short"
    INVALID_FPS = "invalid_fps"
    INVALID_VIDEO_RESOLUTION = "invalid_video_resolution"
    INVALID_FRAME_COUNT = "invalid_frame_count"
    INSUFFICIENT_SEQUENCE_LENGTH = "insufficient_sequence_length"
    EXCESSIVE_MISSINGNESS = "excessive_missingness"
    LOW_POSE_VISIBILITY = "low_pose_visibility"
    PARTIAL_JOINT_COVERAGE = "partial_joint_coverage"
    SINGLE_SIDE_ONLY = "single_side_only"
    SIDE_SWITCH_DISCONTINUITY = "side_switch_discontinuity"
    NORMALIZATION_ORIGIN_FALLBACK = "normalization_origin_fallback"
    NORMALIZATION_SCALE_FALLBACK = "normalization_scale_fallback"
    VIEWPOINT_UNKNOWN = "viewpoint_unknown"
    VIEWPOINT_MISMATCH = "viewpoint_mismatch"
    PARTIAL_REP_START = "partial_rep_start"
    PARTIAL_REP_END = "partial_rep_end"
    MULTI_PERSON = "multi_person"
    OUT_OF_FRAME = "out_of_frame"
    LOW_SIGNAL_RANGE = "low_signal_range"
    UNSTABLE_SIGNAL = "unstable_signal"
    SEGMENTATION_FAILED = "segmentation_failed"
    INVALID_TEMPORAL_INPUT = "invalid_temporal_input"
    INVALID_FEEDBACK_INPUT = "invalid_feedback_input"
    TEMPORAL_GATE_BLOCKED = "temporal_gate_blocked"
    DOCUMENTED_INPUT_MISSING = "documented_input_missing"
    FEEDBACK_SERIALIZATION_ERROR = "feedback_serialization_error"
    ASSESSMENT_UNAVAILABLE = "assessment_unavailable"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FlagLocation(ContractModel):
    video_id: str | None = None
    frame_start: NonNegativeInt | None = Field(default=None, strict=True)
    frame_end: NonNegativeInt | None = Field(default=None, strict=True)
    joint: str | None = None
    feature: str | None = None
    rep_index: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_interval(self) -> "FlagLocation":
        if (self.frame_start is None) != (self.frame_end is None):
            raise ValueError("frame_start and frame_end must be supplied together")
        if self.frame_start is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start (inclusive interval)")
        return self


class StructuredFlag(ContractModel):
    code: FlagCode
    category: ErrorCategory
    severity: Severity
    stage: Stage
    message: str = Field(min_length=1)
    source_tier: SourceTier
    location: FlagLocation = Field(default_factory=FlagLocation)
    # Human-readable value/unit evidence; a flag is not a scoring rule.
    evidence: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class GymError(Exception):
    """Operational exception carrying the same serializable reason as reports."""

    def __init__(self, flag: StructuredFlag) -> None:
        self.flag = flag
        super().__init__(f"{flag.code.value}: {flag.message}")
