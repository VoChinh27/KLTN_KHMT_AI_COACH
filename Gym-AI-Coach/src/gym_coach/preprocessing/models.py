"""Separate raw/processed contracts; every coordinate transform is auditable."""

import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from ..contracts import ContractModel, JointProvenance, SourceTier
from ..flags import FlagCode, StructuredFlag
from ..pose.mapping import JointName
from ..pose.models import PoseFrame
from ..quality.models import QualityReport
from ..video.models import VideoMetadata


def content_hash(value: ContractModel) -> str:
    payload = json.dumps(value.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_hash(frames: tuple[PoseFrame, ...]) -> str:
    return hashlib.sha256("".join(content_hash(f) for f in frames).encode("ascii")).hexdigest()


class ResolvedProfile(ContractModel):
    fps: float = Field(gt=0)
    max_gap_frames: int = Field(ge=0, strict=True)
    smoothing_window_frames: int = Field(ge=3, strict=True)
    actual_smoothing_span_seconds: float = Field(gt=0)


class PreprocessingProfile(ContractModel):
    profile_schema_version: Literal["0.1.0"] = "0.1.0"
    profile_version: str = Field(default="preprocessing-s3-v1", min_length=1)
    source_tier: Literal[SourceTier.S3] = SourceTier.S3
    max_gap_seconds: float = Field(default=5 / 30, ge=0, strict=True)
    interpolate_low_visibility: StrictBool = False
    smoothing_span_seconds: float = Field(default=10 / 30, gt=0, strict=True)
    smoothing_polyorder: int = Field(default=2, ge=0, le=10, strict=True)
    smoothing_method: Literal["savitzky_golay"] = "savitzky_golay"
    smoothing_mode: Literal["interp"] = "interp"
    min_torso_scale: float = Field(default=1e-8, gt=0, strict=True)

    def content_hash(self) -> str:
        return content_hash(self)

    def resolve(self, fps: float) -> ResolvedProfile:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        # Round span to the nearest even number of intervals (ties up), then
        # add one sample. Minimum odd window must exceed the polynomial order.
        intervals = 2 * math.floor(self.smoothing_span_seconds * fps / 2 + 0.5)
        minimum = self.smoothing_polyorder + 1
        if minimum % 2 == 0:
            minimum += 1
        window = max(3, minimum, intervals + 1)
        return ResolvedProfile(
            fps=fps, max_gap_frames=math.floor(self.max_gap_seconds * fps + 1e-9),
            smoothing_window_frames=window,
            actual_smoothing_span_seconds=(window - 1) / fps,
        )


def load_preprocessing_profile(path: str | Path) -> PreprocessingProfile:
    from .pipeline import failure
    try:
        with Path(path).open("rb") as stream:
            return PreprocessingProfile.model_validate(tomllib.load(stream))
    except (OSError, ValueError) as exc:
        raise failure(FlagCode.INVALID_CONFIG, str(exc)) from exc


class InterpolationSource(ContractModel):
    left_frame_index: int = Field(ge=0, strict=True)
    right_frame_index: int = Field(ge=0, strict=True)
    left_timestamp_seconds: float = Field(ge=0, strict=True)
    right_timestamp_seconds: float = Field(ge=0, strict=True)
    left_status: Literal["observed"] = "observed"
    right_status: Literal["observed"] = "observed"

    @model_validator(mode="after")
    def ordered(self) -> "InterpolationSource":
        if (self.left_frame_index >= self.right_frame_index or
                self.left_timestamp_seconds >= self.right_timestamp_seconds):
            raise ValueError("interpolation endpoints must be ordered")
        return self


class ProcessedJoint(ContractModel):
    joint: JointName
    source_status: Literal["observed", "low_visibility", "missing"]
    status: JointProvenance
    usable: StrictBool
    pre_normalization_x: float | None = Field(default=None, strict=True)
    pre_normalization_y: float | None = Field(default=None, strict=True)
    unsmoothed_x: float | None = Field(default=None, strict=True)
    unsmoothed_y: float | None = Field(default=None, strict=True)
    x: float | None = Field(default=None, strict=True)
    y: float | None = Field(default=None, strict=True)
    interpolation: InterpolationSource | None = None
    unavailable_reason: FlagCode | None = None
    smoothing_status: Literal["smoothed", "skipped_short_run", "unavailable"]

    @model_validator(mode="after")
    def provenance_and_coordinates(self) -> "ProcessedJoint":
        values = (self.pre_normalization_x, self.pre_normalization_y,
                  self.unsmoothed_x, self.unsmoothed_y, self.x, self.y)
        if self.usable:
            if (any(v is None for v in values) or self.unavailable_reason is not None or
                    self.status not in (JointProvenance.OBSERVED, JointProvenance.IMPUTED) or
                    self.smoothing_status == "unavailable"):
                raise ValueError("usable joint needs finite coordinates and measured/imputed provenance")
        elif (any(v is not None for v in values) or self.unavailable_reason is None or
              self.smoothing_status != "unavailable" or self.status == JointProvenance.IMPUTED):
            raise ValueError("unavailable joint must have null coordinates and a reason")
        if self.status == JointProvenance.IMPUTED:
            if self.interpolation is None or self.source_status == "observed":
                raise ValueError("imputed joint must reference endpoints and an unobserved source")
        elif self.interpolation is not None or self.status.value != self.source_status:
            raise ValueError("non-imputed status must preserve raw provenance")
        return self


class NormalizationTransform(ContractModel):
    translation_status: Literal["mid_hip", "fallback_origin"]
    scale_status: Literal["torso", "fallback_unit"]
    origin_x: float | None = Field(default=None, strict=True)
    origin_y: float | None = Field(default=None, strict=True)
    torso_scale: float | None = Field(default=None, gt=0, strict=True)
    distance_ratio_eligible: StrictBool

    @model_validator(mode="after")
    def explicit_fallback(self) -> "NormalizationTransform":
        if (self.origin_x is not None and self.origin_y is not None) != (self.translation_status == "mid_hip"):
            raise ValueError("origin is supplied exactly when mid-hip exists")
        if (self.origin_x is None) != (self.origin_y is None):
            raise ValueError("origin coordinates must be paired")
        if (self.torso_scale is not None) != (self.scale_status == "torso"):
            raise ValueError("fallback scale must be null, never a fabricated 1.0")
        if self.distance_ratio_eligible != (self.scale_status == "torso"):
            raise ValueError("distance/ratio eligibility must expose scale validity")
        return self


class ProcessedPoseFrame(ContractModel):
    video_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0, strict=True)
    raw_frame_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    normalization: NormalizationTransform
    joints: tuple[ProcessedJoint, ...]
    flags: tuple[StructuredFlag, ...] = ()

    @model_validator(mode="after")
    def all_joints(self) -> "ProcessedPoseFrame":
        if len(self.joints) != len(JointName) or {j.joint for j in self.joints} != set(JointName):
            raise ValueError("processed frame needs every mapped joint exactly once")
        return self


class SmoothingRun(ContractModel):
    joint: JointName
    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)
    status: Literal["smoothed", "skipped_short_run"]
    window_frames: int = Field(ge=3, strict=True)
    translation_status: Literal["mid_hip", "fallback_origin"]
    scale_status: Literal["torso", "fallback_unit"]

    @model_validator(mode="after")
    def interval(self) -> "SmoothingRun":
        if self.end_frame < self.start_frame:
            raise ValueError("run interval must be ordered")
        expected = "smoothed" if self.end_frame - self.start_frame + 1 >= self.window_frames else "skipped_short_run"
        if self.status != expected:
            raise ValueError("run status must match its length")
        return self


class PreprocessingResult(ContractModel):
    preprocessing_schema_version: Literal["0.1.0"] = "0.1.0"
    profile: PreprocessingProfile
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    resolved_profile: ResolvedProfile
    metadata: VideoMetadata
    quality: QualityReport
    quality_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_sequence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_frames: tuple[PoseFrame, ...]
    frames: tuple[ProcessedPoseFrame, ...]
    smoothing_runs: tuple[SmoothingRun, ...]
    working_coordinate_space: Literal["image_width_units_xy"] = "image_width_units_xy"
    output_units: Literal["torso_units_when_valid_else_working_units"] = "torso_units_when_valid_else_working_units"

    @model_validator(mode="after")
    def audit_links(self) -> "PreprocessingResult":
        from .pipeline import validate_quality, validate_sequence
        validate_sequence(self.raw_frames, self.metadata)
        validate_quality(self.raw_frames, self.quality)
        if self.profile_hash != content_hash(self.profile) or self.quality_hash != content_hash(self.quality):
            raise ValueError("profile/quality hash mismatch")
        if self.raw_sequence_hash != sequence_hash(self.raw_frames):
            raise ValueError("raw sequence hash mismatch")
        if self.resolved_profile != self.profile.resolve(self.metadata.fps):
            raise ValueError("resolved profile mismatch")
        if len(self.frames) != len(self.raw_frames) or len(self.frames) != self.metadata.frame_count:
            raise ValueError("raw/processed/metadata counts differ")
        if self.quality.total_frames != len(self.frames):
            raise ValueError("quality count mismatch")
        for raw, frame in zip(self.raw_frames, self.frames):
            if (raw.video_id, raw.frame_index, raw.timestamp_seconds) != (frame.video_id, frame.frame_index, frame.timestamp_seconds):
                raise ValueError("raw identity changed")
            if frame.raw_frame_hash != content_hash(raw):
                raise ValueError("raw frame hash mismatch")
            source = {j.joint: j for j in raw.joints}
            for joint in frame.joints:
                if joint.source_status != source[joint.joint].status:
                    raise ValueError("raw provenance changed")
                if joint.status == JointProvenance.IMPUTED:
                    refs = joint.interpolation
                    left, right = refs.left_frame_index, refs.right_frame_index
                    if not (0 <= left < frame.frame_index < right < len(self.frames)):
                        raise ValueError("interpolation endpoints must bracket the frame")
                    if right - left - 1 > self.resolved_profile.max_gap_frames:
                        raise ValueError("interpolation gap exceeds profile")
                    if joint.source_status == "low_visibility" and not self.profile.interpolate_low_visibility:
                        raise ValueError("profile forbids low-visibility interpolation")
                    for index, timestamp in ((left, refs.left_timestamp_seconds), (right, refs.right_timestamp_seconds)):
                        endpoint = self.raw_frames[index]
                        observation = next(j for j in endpoint.joints if j.joint == joint.joint)
                        if (endpoint.timestamp_seconds != timestamp or observation.status != JointProvenance.OBSERVED or
                                not (0 <= observation.x <= 1 and 0 <= observation.y <= 1)):
                            raise ValueError("interpolation endpoint is not a valid original observation")
                    for index in range(left + 1, right):
                        interior = next(j for j in self.raw_frames[index].joints if j.joint == joint.joint)
                        allowed = interior.status == JointProvenance.MISSING or (
                            self.profile.interpolate_low_visibility and interior.status == JointProvenance.LOW_VISIBILITY)
                        if not allowed:
                            raise ValueError("interpolation references must bound a candidate gap")
        seen = set()
        for run in self.smoothing_runs:
            if run.end_frame >= len(self.frames) or run.window_frames != self.resolved_profile.smoothing_window_frames:
                raise ValueError("smoothing run exceeds sequence or has incorrect window")
            for index in range(run.start_frame, run.end_frame + 1):
                frame = self.frames[index]
                joint = next(j for j in frame.joints if j.joint == run.joint)
                key = (index, run.joint)
                if key in seen or not joint.usable or joint.smoothing_status != run.status:
                    raise ValueError("smoothing run overlaps, crosses a gap or disagrees with status")
                if (frame.normalization.translation_status, frame.normalization.scale_status) != (
                        run.translation_status, run.scale_status):
                    raise ValueError("smoothing run crosses a normalization unit transition")
                seen.add(key)
        expected = {(f.frame_index, j.joint) for f in self.frames for j in f.joints if j.usable}
        if seen != expected:
            raise ValueError("smoothing audit must cover each usable point exactly once")
        return self
