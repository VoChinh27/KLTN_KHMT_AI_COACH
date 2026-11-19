"""Pose-only contracts: native image coordinates, no imputation or transforms."""

import hashlib
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from ..contracts import ContractModel, Exercise, JointProvenance, SourceTier, Viewpoint
from ..flags import FlagCode, StructuredFlag
from .mapping import JOINT_MAPPING_VERSION, JointName

POSE_SCHEMA_VERSION = "0.1.0"


class PoseExtractionConfig(ContractModel):
    config_version: str = Field(default="pose-extraction-v1", min_length=1)
    visibility_threshold: float = Field(default=0.5, ge=0, le=1, strict=True)
    min_detection_confidence: float = Field(default=0.5, ge=0, le=1, strict=True)
    min_tracking_confidence: float = Field(default=0.5, ge=0, le=1, strict=True)
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    # Full model is already shipped with the installed wheel; no downloads.
    model_complexity: Literal[1] = 1
    static_image_mode: Literal[False] = False
    smooth_landmarks: Literal[False] = False
    enable_segmentation: Literal[False] = False
    smooth_segmentation: Literal[False] = False

    def content_hash(self) -> str:
        import json
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


class PoseBackendIdentity(ContractModel):
    backend: Literal["mediapipe.solutions.pose.Pose"] = "mediapipe.solutions.pose.Pose"
    mediapipe_version: str = Field(min_length=1)
    model_asset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    joint_mapping_version: Literal["mediapipe-pose-13-v1"] = JOINT_MAPPING_VERSION
    coordinate_space: Literal["image_normalized_xy"] = "image_normalized_xy"
    units: Literal["dimensionless_image_width_height"] = "dimensionless_image_width_height"
    axes: Literal["x_right_y_down"] = "x_right_y_down"
    subject_selection_policy: Literal["backend_most_prominent_single_subject"] = (
        "backend_most_prominent_single_subject"
    )
    limitations: tuple[str, ...] = (
        "No multi-person enumeration, explicit subject selection or identity guarantee",
        "Inverted orientation/occlusion can reduce detection and visibility",
        "Native 2D estimates are not calibrated physical measurements",
    )


class JointObservation(ContractModel):
    joint: JointName
    x: float | None = Field(default=None, strict=True)
    y: float | None = Field(default=None, strict=True)
    visibility: float | None = Field(default=None, ge=0, le=1, strict=True)
    presence: float | None = Field(default=None, ge=0, le=1, strict=True)
    visibility_threshold: float = Field(default=0.5, ge=0, le=1, strict=True)
    status: Literal[JointProvenance.OBSERVED, JointProvenance.LOW_VISIBILITY,
                    JointProvenance.MISSING]
    missing_reason: Literal[
        FlagCode.POSE_NOT_DETECTED, FlagCode.MISSING_JOINT,
        FlagCode.MISSING_LANDMARK_FIELD, FlagCode.NON_FINITE_LANDMARK,
        FlagCode.INVALID_LANDMARK_CONFIDENCE,
    ] | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> "JointObservation":
        if self.status == JointProvenance.MISSING:
            if self.x is not None or self.y is not None or self.missing_reason is None:
                raise ValueError("Missing joints require null coordinates and a reason")
        else:
            if self.x is None or self.y is None or self.visibility is None:
                raise ValueError("Measured joints require coordinates and visibility")
            expected = (JointProvenance.OBSERVED if self.visibility >= self.visibility_threshold
                        else JointProvenance.LOW_VISIBILITY)
            if self.status != expected or self.missing_reason is not None:
                raise ValueError("Status must match measured visibility; no missing reason")
        return self


class PoseFrame(ContractModel):
    pose_schema_version: Literal["0.1.0"] = POSE_SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    exercise: Exercise
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0, strict=True)
    has_pose: StrictBool
    backend: PoseBackendIdentity
    config: PoseExtractionConfig
    joints: tuple[JointObservation, ...]
    flags: tuple[StructuredFlag, ...] = ()

    @model_validator(mode="after")
    def validate_joints(self) -> "PoseFrame":
        names = [joint.joint for joint in self.joints]
        if len(names) != len(JointName) or set(names) != set(JointName):
            raise ValueError("Exactly one observation for each mapped joint is required")
        for joint in self.joints:
            if joint.visibility_threshold != self.config.visibility_threshold:
                raise ValueError("Joint threshold must match frame config")
            if not self.has_pose and (joint.status != JointProvenance.MISSING or
                                     joint.missing_reason != FlagCode.POSE_NOT_DETECTED):
                raise ValueError("No-pose frame must retain all joints as pose_not_detected")
            if self.has_pose and joint.missing_reason == FlagCode.POSE_NOT_DETECTED:
                raise ValueError("Detected-pose frame cannot mark pose_not_detected joints")
        return self


class JointCoverage(ContractModel):
    observed: int = Field(ge=0, strict=True)
    low_visibility: int = Field(ge=0, strict=True)
    missing: int = Field(ge=0, strict=True)
    pose_coverage: float | None = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_coverage(self) -> "JointCoverage":
        total = self.observed + self.low_visibility + self.missing
        expected = self.observed / total if total else None
        if self.pose_coverage != expected:
            raise ValueError("Coverage must be observed / total, null for an empty sequence")
        return self


class PoseSequenceSummary(ContractModel):
    pose_schema_version: Literal["0.1.0"] = POSE_SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    exercise: Exercise
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    backend: PoseBackendIdentity
    config: PoseExtractionConfig
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_frames: int = Field(ge=0, strict=True)
    frames_with_pose: int = Field(ge=0, strict=True)
    frames_without_pose: int = Field(ge=0, strict=True)
    pose_detection_ratio: float | None = Field(ge=0, le=1)
    pose_coverage: float | None = Field(ge=0, le=1)
    required_joint_coverage: float | None = Field(ge=0, le=1)
    by_joint: dict[JointName, JointCoverage]
    flags: tuple[StructuredFlag, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "PoseSequenceSummary":
        from .mapping import REQUIRED_JOINTS
        if set(self.by_joint) != set(JointName):
            raise ValueError("Coverage must include all mapped joints")
        if self.frames_with_pose + self.frames_without_pose != self.total_frames:
            raise ValueError("Pose frame counts must sum to total_frames")
        if self.config_hash != self.config.content_hash():
            raise ValueError("Config hash mismatch")
        for counts in self.by_joint.values():
            if counts.observed + counts.low_visibility + counts.missing != self.total_frames:
                raise ValueError("Per-joint counts must sum to total_frames")
        required = REQUIRED_JOINTS[self.exercise]
        expected_detection = self.frames_with_pose / self.total_frames if self.total_frames else None
        expected_coverage = (sum(c.observed for c in self.by_joint.values()) /
                             (len(JointName) * self.total_frames) if self.total_frames else None)
        expected_required = (sum(self.by_joint[j].observed for j in required) /
                             (len(required) * self.total_frames) if self.total_frames else None)
        if (self.pose_detection_ratio, self.pose_coverage, self.required_joint_coverage) != (
            expected_detection, expected_coverage, expected_required
        ):
            raise ValueError("Summary ratios must match observed counts")
        return self
