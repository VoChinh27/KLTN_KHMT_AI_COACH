"""Feature metadata, readiness and future observation contracts, without geometry."""

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..contracts import ContractModel, Exercise, JointProvenance, MeasurementTier, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, StructuredFlag
from ..pose.mapping import JointName
from ..preprocessing.models import ProcessedPoseFrame
from ..quality.models import QualityReport

FEATURE_SCHEMA_VERSION = "0.1.0"
Text = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def content_hash(value: ContractModel) -> str:
    canonical = json.dumps(value.model_dump(mode="json"), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QuantityKind(StrEnum):
    ANGLE = "angle"
    DISTANCE = "distance"
    RATIO = "ratio"


class FeatureUnit(StrEnum):
    DEGREES = "degrees"
    TORSO_LENGTHS = "torso_lengths"
    DIMENSIONLESS_RATIO = "dimensionless_ratio"


class UnsupportedViewpoint(ContractModel):
    viewpoint: Viewpoint
    reason: Text


class ViewpointPolicy(ContractModel):
    supported: tuple[Viewpoint, ...] = Field(min_length=1)
    unsupported: tuple[UnsupportedViewpoint, ...] = Field(min_length=1)
    policy_reason: Text
    source_tier: Literal[SourceTier.S3]

    @model_validator(mode="after")
    def exhaustive(self) -> "ViewpointPolicy":
        denied = [v.viewpoint for v in self.unsupported]
        if (len(set(self.supported)) != len(self.supported) or len(set(denied)) != len(denied) or
                set(self.supported) & set(denied) or set(self.supported) | set(denied) != set(Viewpoint)):
            raise ValueError("viewpoint policy must partition every viewpoint exactly once")
        if Viewpoint.UNKNOWN in self.supported:
            raise ValueError("unknown viewpoint cannot establish geometric measurement support")
        return self


class JointInputPolicy(ContractModel):
    coordinate_source: Literal["stage4_processed_xy"]
    allowed_provenance: tuple[JointProvenance, ...] = Field(min_length=1)
    require_usable: Literal[True]
    require_finite_xy: Literal[True]
    imputation_policy_reason: Text

    @model_validator(mode="after")
    def safe_provenance(self) -> "JointInputPolicy":
        if (len(set(self.allowed_provenance)) != len(self.allowed_provenance) or
                not set(self.allowed_provenance) <= {JointProvenance.OBSERVED, JointProvenance.IMPUTED}):
            raise ValueError("only observed or explicitly imputed usable joints can be measurement inputs")
        return self


class FeatureQualityPolicy(ContractModel):
    metric_key: Text
    require_sequence_allow: Literal[True]
    require_metric_allow: Literal[True]
    require_exact_metric_joints: Literal[True]
    requires_valid_scale: bool = Field(strict=True)
    requires_mid_hip_origin: bool = Field(strict=True)
    reason: Text


class FeatureValidation(ContractModel):
    measurement_status: Literal["unvalidated", "blocked"]
    reason: Text
    calculation_status: Literal["not_implemented"]
    assessment_status: Literal["blocked_pending_expert_validation"]


class FeatureDefinition(ContractModel):
    feature_schema_version: Literal["0.1.0"] = FEATURE_SCHEMA_VERSION
    definition_version: Text
    feature_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", min_length=1)
    name: Text
    exercises: tuple[Exercise, ...] = Field(min_length=1)
    measurement_definition: Text
    measurement_method: Literal["joint_angle_abc", "midpoint_axis_angle_vertical",
                                "horizontal_offset_torso_ratio", "normalized_euclidean_distance"]
    formula: Text
    required_joints: tuple[JointName, ...] = Field(min_length=1)
    joint_policy: JointInputPolicy
    quantity_kind: QuantityKind
    unit: FeatureUnit
    measurement_tier: MeasurementTier
    source_tiers: tuple[SourceTier, ...] = Field(min_length=1)
    source_references: tuple[Text, ...] = Field(min_length=1)
    validation: FeatureValidation
    viewpoint_policy: ViewpointPolicy
    quality_policy: FeatureQualityPolicy
    roles: tuple[Literal["descriptive", "informational", "segmentation_candidate"], ...] = Field(min_length=1)
    unavailable_reasons: tuple[FlagCode, ...] = Field(min_length=1)
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_metadata(self) -> "FeatureDefinition":
        for name in ("exercises", "required_joints", "source_tiers", "roles", "unavailable_reasons"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        expected_unit = {QuantityKind.ANGLE: FeatureUnit.DEGREES,
                         QuantityKind.DISTANCE: FeatureUnit.TORSO_LENGTHS,
                         QuantityKind.RATIO: FeatureUnit.DIMENSIONLESS_RATIO}[self.quantity_kind]
        if self.unit != expected_unit:
            raise ValueError("unit must match angle/distance/ratio quantity kind")
        method_kind, count = {
            "joint_angle_abc": (QuantityKind.ANGLE, 3),
            "midpoint_axis_angle_vertical": (QuantityKind.ANGLE, 4),
            "horizontal_offset_torso_ratio": (QuantityKind.RATIO, 2),
            "normalized_euclidean_distance": (QuantityKind.DISTANCE, 2),
        }[self.measurement_method]
        if self.quantity_kind != method_kind or len(self.required_joints) != count:
            raise ValueError("measurement method must match quantity kind and ordered joint arity")
        if self.measurement_tier == MeasurementTier.M0 and self.validation.measurement_status != "blocked":
            raise ValueError("M0 measurements must be blocked")
        if self.quantity_kind in (QuantityKind.DISTANCE, QuantityKind.RATIO) and not (
                self.quality_policy.requires_valid_scale and self.quality_policy.requires_mid_hip_origin):
            raise ValueError("distance/ratio policy must forbid fallback scale/origin")
        needed = {
            FlagCode.UNSUPPORTED_EXERCISE, FlagCode.INVALID_FEATURE_INPUT,
            FlagCode.FEATURE_DEFINITION_BLOCKED, FlagCode.MISSING_JOINT,
            FlagCode.FEATURE_JOINT_UNUSABLE, FlagCode.FEATURE_PROVENANCE_REJECTED,
            FlagCode.NON_FINITE_LANDMARK, FlagCode.VIEWPOINT_UNKNOWN, FlagCode.VIEWPOINT_MISMATCH,
            FlagCode.FEATURE_QUALITY_BLOCKED, FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING,
            FlagCode.FEATURE_QUALITY_POLICY_MISMATCH,
        }
        if self.quality_policy.requires_valid_scale:
            needed.add(FlagCode.NORMALIZATION_SCALE_FALLBACK)
        if self.quality_policy.requires_mid_hip_origin:
            needed.add(FlagCode.NORMALIZATION_ORIGIN_FALLBACK)
        if not needed <= set(self.unavailable_reasons):
            raise ValueError("unavailable_reasons must cover every declared policy gate")
        return self

    def content_hash(self) -> str:
        return content_hash(self)


class FeatureInputContext(ContractModel):
    """Snapshots of existing contracts; no new coordinates or quality decisions."""
    frame: ProcessedPoseFrame
    quality: QualityReport

    @model_validator(mode="after")
    def same_input_identity(self) -> "FeatureInputContext":
        if self.frame.video_id != self.quality.video_id:
            raise ValueError("frame and quality video IDs differ")
        if self.frame.frame_index >= self.quality.total_frames:
            raise ValueError("frame index exceeds full Stage 4 sequence length")
        return self


class FeatureAvailability(ContractModel):
    """Ready means prerequisites pass; it never means geometry has been computed."""
    feature_schema_version: Literal["0.1.0"] = FEATURE_SCHEMA_VERSION
    registry_version: Text
    registry_hash: Hash
    definition: FeatureDefinition
    definition_hash: Hash
    input: FeatureInputContext
    input_hash: Hash
    status: Literal["ready_for_calculation", "unavailable"]
    reasons: tuple[FlagCode, ...]
    flags: tuple[StructuredFlag, ...]

    @model_validator(mode="after")
    def gate_consistency(self) -> "FeatureAvailability":
        from .policy import policy_flags
        if self.definition_hash != self.definition.content_hash() or self.input_hash != content_hash(self.input):
            raise ValueError("definition/input hash mismatch")
        expected = policy_flags(self.definition, self.input)
        codes = tuple(dict.fromkeys(flag.code for flag in expected))
        if self.flags != expected or self.reasons != codes:
            raise ValueError("availability reasons/flags must match declared input policies")
        if self.status != ("unavailable" if expected else "ready_for_calculation"):
            raise ValueError("availability status must agree with policy gates")
        if not set(self.reasons) <= set(self.definition.unavailable_reasons):
            raise ValueError("unavailable reason not declared in metadata")
        return self


class FeatureObservation(ContractModel):
    """Stage 5.1 stores no measured numbers; adding calculation requires a new schema."""
    feature_schema_version: Literal["0.1.0"] = FEATURE_SCHEMA_VERSION
    availability: FeatureAvailability
    status: Literal["not_computed", "unavailable"]
    value: None = None
    computation_status: Literal["not_implemented"] = "not_implemented"

    @model_validator(mode="after")
    def no_fake_measurement(self) -> "FeatureObservation":
        expected = "unavailable" if self.availability.status == "unavailable" else "not_computed"
        if self.status != expected:
            raise ValueError("observation must reflect availability without claiming a measured value")
        return self

    @classmethod
    def from_availability(cls, availability: FeatureAvailability) -> "FeatureObservation":
        return cls(availability=availability, status=(
            "unavailable" if availability.status == "unavailable" else "not_computed"))
