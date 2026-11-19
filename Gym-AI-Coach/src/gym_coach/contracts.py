"""Shared vocabulary only; domain algorithms belong to their respective modules."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Exercise(StrEnum):
    BODY_WEIGHT_SQUATS = "BodyWeightSquats"
    HANDSTAND_PUSHUPS = "HandstandPushups"


class Viewpoint(StrEnum):
    UNKNOWN = "unknown"
    SIDE = "side"
    FRONT = "front"
    OBLIQUE = "oblique"


class SourceTier(StrEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    OBS = "OBS"
    TBD_EXPERT = "TBD-EXPERT"


class MeasurementTier(StrEnum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"


class JointProvenance(StrEnum):
    OBSERVED = "observed"
    LOW_VISIBILITY = "low_visibility"
    IMPUTED = "imputed"
    MISSING = "missing"


class Stage(StrEnum):
    CONFIGURATION = "configuration"
    VIDEO = "video"
    POSE = "pose"
    QUALITY = "quality"
    PREPROCESSING = "preprocessing"
    FEATURES = "features"
    SEGMENTATION = "segmentation"
    ASSESSMENT = "assessment"
    REPORTING = "reporting"


class StageStatus(StrEnum):
    NOT_RUN = "not_run"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class ErrorCategory(StrEnum):
    TECHNICAL_ERROR = "technical_error"
    MEASUREMENT_ERROR = "measurement_error"
    INSUFFICIENT_DATA = "insufficient_data"
    SEGMENTATION_ERROR = "segmentation_error"
    INFORMATIONAL_OBSERVATION = "informational_observation"
    # Operational failures are distinct from scientific conclusions.
    INPUT_ERROR = "input_error"
    CONFIGURATION_ERROR = "configuration_error"
    OPERATIONAL_ERROR = "operational_error"
