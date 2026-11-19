"""Versioned S3 quality profile and auditable quality report contracts."""

import hashlib
import json
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from ..contracts import ContractModel, Exercise, SourceTier, Viewpoint
from ..flags import StructuredFlag
from ..pose.mapping import JointName

QUALITY_SCHEMA_VERSION = "0.1.0"


class QualityProfile(ContractModel):
    """Engineering gates from the current specification; not form standards."""

    profile_version: str = Field(default="quality-s3-v1", min_length=1)
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    min_sequence_frames: PositiveInt = Field(default=5, strict=True)
    max_missing_ratio: float = Field(default=0.30, ge=0, le=1, strict=True)
    max_low_visibility_ratio: float = Field(default=0.30, ge=0, le=1, strict=True)
    max_consecutive_invalid_frames: NonNegativeInt = Field(default=10, strict=True)

    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JointGapSpan(ContractModel):
    start_frame: NonNegativeInt = Field(strict=True)
    end_frame: NonNegativeInt = Field(strict=True)
    length: PositiveInt = Field(strict=True)

    @model_validator(mode="after")
    def interval_matches_length(self) -> "JointGapSpan":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        if self.length != self.end_frame - self.start_frame + 1:
            raise ValueError("length must match the inclusive frame interval")
        return self


class JointQuality(ContractModel):
    total_frames: NonNegativeInt = Field(strict=True)
    observed_frames: NonNegativeInt = Field(strict=True)
    low_visibility_frames: NonNegativeInt = Field(strict=True)
    missing_frames: NonNegativeInt = Field(strict=True)
    out_of_frame_frames: NonNegativeInt = Field(default=0, strict=True)
    out_of_frame_observed_frames: NonNegativeInt = Field(default=0, strict=True)
    observed_ratio: float | None = Field(default=None, ge=0, le=1)
    invalid_ratio: float | None = Field(default=None, ge=0, le=1)
    consecutive_invalid_spans: tuple[JointGapSpan, ...] = ()
    max_consecutive_invalid_frames: NonNegativeInt = Field(default=0, strict=True)

    @model_validator(mode="after")
    def validate_counts(self) -> "JointQuality":
        if self.observed_frames + self.low_visibility_frames + self.missing_frames != self.total_frames:
            raise ValueError("joint provenance counts must sum to total_frames")
        # Low-visibility observations may also lie outside the image, so the
        # all-status out-of-frame count is independent of observed_frames.
        if self.out_of_frame_observed_frames > self.observed_frames:
            raise ValueError("out-of-frame observed joints must be a subset of observed joints")
        if self.out_of_frame_observed_frames > self.out_of_frame_frames:
            raise ValueError("out-of-frame observed joints must be a subset of all out-of-frame joints")
        expected_observed = self.observed_frames / self.total_frames if self.total_frames else None
        expected_invalid = (
            (self.low_visibility_frames + self.missing_frames + self.out_of_frame_observed_frames)
            / self.total_frames
            if self.total_frames else None
        )
        if self.observed_ratio != expected_observed or self.invalid_ratio != expected_invalid:
            raise ValueError("joint ratios must match provenance counts")
        expected_max = max((span.length for span in self.consecutive_invalid_spans), default=0)
        if self.max_consecutive_invalid_frames != expected_max:
            raise ValueError("max consecutive gap must match spans")
        return self


class MetricEligibility(ContractModel):
    eligible: bool
    action: Literal["allow", "degrade", "skip"]
    required_joints: tuple[JointName, ...]
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def action_consistent(self) -> "MetricEligibility":
        if self.eligible != (self.action == "allow"):
            raise ValueError("eligible is true exactly when action is allow")
        if (self.action == "allow") != (len(self.reasons) == 0):
            raise ValueError("allow has no reasons; degrade/skip must explain why")
        return self


class QualityReport(ContractModel):
    quality_schema_version: Literal["0.1.0"] = QUALITY_SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    exercise: Exercise
    viewpoint: Viewpoint
    profile: QualityProfile
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_frames: NonNegativeInt = Field(strict=True)
    frames_with_pose: NonNegativeInt = Field(strict=True)
    frames_without_pose: NonNegativeInt = Field(strict=True)
    sequence_status: Literal["eligible", "degraded", "skipped"]
    eligibility: MetricEligibility
    metric_eligibility: dict[str, MetricEligibility] = Field(default_factory=dict)
    by_joint: dict[JointName, JointQuality]
    flags: tuple[StructuredFlag, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> "QualityReport":
        if self.frames_with_pose + self.frames_without_pose != self.total_frames:
            raise ValueError("pose frame counts must sum to total_frames")
        if set(self.by_joint) != set(JointName):
            raise ValueError("quality report must include all mapped joints")
        if any(joint.total_frames != self.total_frames for joint in self.by_joint.values()):
            raise ValueError("joint counts must match sequence length")
        if self.profile_hash != self.profile.content_hash():
            raise ValueError("profile hash mismatch")
        if not self.metric_eligibility or "movement_analysis" not in self.metric_eligibility:
            raise ValueError("metric eligibility must include movement_analysis")
        if self.metric_eligibility["movement_analysis"] != self.eligibility:
            raise ValueError("sequence eligibility must match movement_analysis")
        expected_status = (
            "eligible" if self.eligibility.action == "allow" else
            "degraded" if self.eligibility.action == "degrade" else "skipped"
        )
        if self.sequence_status != expected_status:
            raise ValueError("sequence status must agree with eligibility action")
        return self
