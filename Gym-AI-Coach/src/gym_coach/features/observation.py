"""Versioned numeric observations for one adapter; Stage 5.1 schema stays intact."""

from typing import Literal

from pydantic import Field, model_validator

from ..contracts import ContractModel, ErrorCategory, Exercise, JointProvenance, MeasurementTier, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, FlagLocation, Severity, StructuredFlag
from ..geometry import JOINT_ANGLE_PRIMITIVE_VERSION
from ..preprocessing.models import ProcessedJoint
from .models import FeatureAvailability, FeatureInputContext, FeatureUnit, Hash, QuantityKind, Text, content_hash
from .registry import DEFAULT_REGISTRY

NUMERIC_FEATURE_SCHEMA_VERSION = "0.2.0"
SUPPORTED_NUMERIC_FEATURE_ID = "squat_knee_angle_left_deg"
FEATURE_ADAPTER_VERSION = "squat-left-knee-v1"
MEASUREMENT_FAILURE_CODES = {FlagCode.DEGENERATE_GEOMETRY, FlagCode.GEOMETRY_NUMERICAL_ERROR}


def validate_context_references(context: FeatureInputContext) -> None:
    """Check references available in a detached frame, without inventing raw/FPS data."""
    frame = context.frame
    for joint in frame.joints:
        if joint.status == JointProvenance.IMPUTED:
            refs = joint.interpolation
            if not (0 <= refs.left_frame_index < frame.frame_index < refs.right_frame_index < context.quality.total_frames
                    and refs.left_timestamp_seconds < frame.timestamp_seconds < refs.right_timestamp_seconds):
                raise ValueError(f"Imputation references for {joint.joint} must bracket this frame identity")


def observation_fields(availability: FeatureAvailability) -> dict:
    """Derive audit fields from the validated snapshot, never from caller overrides."""
    definition, frame, quality = availability.definition, availability.input.frame, availability.input.quality
    points = {j.joint: j for j in frame.joints}
    return {
        "feature_id": definition.feature_id, "exercise": quality.exercise, "viewpoint": quality.viewpoint,
        "video_id": frame.video_id, "frame_index": frame.frame_index, "timestamp_seconds": frame.timestamp_seconds,
        "measurement_method": definition.measurement_method, "quantity_kind": definition.quantity_kind,
        "unit": definition.unit, "measurement_tier": definition.measurement_tier, "source_tiers": definition.source_tiers,
        "measurement_validation_status": definition.validation.measurement_status,
        "assessment_status": definition.validation.assessment_status,
        "joint_inputs": tuple(points[name] for name in definition.required_joints),
        "quality_report_hash": content_hash(quality), "quality_policy_hash": content_hash(definition.quality_policy),
    }


def measurement_location(availability: FeatureAvailability) -> FlagLocation:
    frame = availability.input.frame
    return FlagLocation(video_id=frame.video_id, frame_start=frame.frame_index, frame_end=frame.frame_index,
                        feature=availability.definition.feature_id)


def measurement_flag(flag: StructuredFlag, availability: FeatureAvailability) -> StructuredFlag:
    names = availability.definition.required_joints
    return flag.model_copy(update={
        "location": measurement_location(availability),
        "evidence": {**flag.evidence, "input_joint_a": names[0].value,
                     "input_joint_b": names[1].value, "input_joint_c": names[2].value},
    })


class NumericFeatureObservation(ContractModel):
    """Descriptive, unvalidated angle or explicit unavailable, with full audit input.

    This schema is intentionally restricted to one registry feature. Expanding
    supported methods/features requires a deliberate versioned implementation.
    """
    numeric_feature_schema_version: Literal["0.2.0"] = NUMERIC_FEATURE_SCHEMA_VERSION
    adapter_version: Literal["squat-left-knee-v1"] = FEATURE_ADAPTER_VERSION
    primitive_name: Literal["joint_angle"] = "joint_angle"
    primitive_version: Literal["joint-angle-2d-v1"] = JOINT_ANGLE_PRIMITIVE_VERSION
    feature_id: Literal["squat_knee_angle_left_deg"]
    exercise: Exercise
    viewpoint: Viewpoint
    video_id: Text
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0, strict=True)
    measurement_method: Literal["joint_angle_abc"]
    quantity_kind: QuantityKind
    unit: FeatureUnit
    measurement_tier: MeasurementTier
    source_tiers: tuple[SourceTier, ...]
    measurement_validation_status: Literal["unvalidated", "blocked"]
    assessment_status: Literal["blocked_pending_expert_validation"]
    availability: FeatureAvailability
    joint_inputs: tuple[ProcessedJoint, ...] = Field(min_length=3, max_length=3)
    quality_report_hash: Hash
    quality_policy_hash: Hash
    status: Literal["available", "unavailable"]
    computation_status: Literal["computed", "not_computed", "failed_measurement"]
    value: float | None = Field(ge=0, le=180, strict=True)
    reasons: tuple[FlagCode, ...]
    flags: tuple[StructuredFlag, ...]

    @model_validator(mode="after")
    def validate_audit_and_outcome(self) -> "NumericFeatureObservation":
        # Revalidate nested instances as well as JSON: frozen models can still
        # have mutable dictionaries or unsafe model_copy/model_construct input.
        checked = FeatureAvailability.model_validate_json(self.availability.model_dump_json())
        definition = DEFAULT_REGISTRY.get(SUPPORTED_NUMERIC_FEATURE_ID)
        if (checked.definition != definition or checked.registry_version != DEFAULT_REGISTRY.registry_version
                or checked.registry_hash != DEFAULT_REGISTRY.content_hash()):
            raise ValueError("Numeric observation requires the canonical MVP registry definition and identity")
        validate_context_references(checked.input)
        for field, expected in observation_fields(checked).items():
            if getattr(self, field) != expected:
                raise ValueError(f"Observation {field} must match its definition/input audit snapshot")
        if self.quantity_kind != QuantityKind.ANGLE or self.unit != FeatureUnit.DEGREES:
            raise ValueError("Joint-angle observation must be angle/degrees")
        if checked.status == "unavailable":
            if (self.status != "unavailable" or self.computation_status != "not_computed" or self.value is not None
                    or self.reasons != checked.reasons or self.flags != checked.flags):
                raise ValueError("Prerequisite failure must preserve policy reasons/flags without a value")
            return self
        a, b, c = ((j.x, j.y) for j in self.joint_inputs)
        zero_vector = "BA" if a == b else "BC" if c == b else None
        if self.status == "available":
            if (self.value is None or self.computation_status != "computed" or self.reasons or self.flags or zero_vector):
                raise ValueError("Available angle needs a finite value, valid vectors and no unavailable reasons")
        else:
            if (self.value is not None or self.computation_status != "failed_measurement" or len(self.flags) != 1
                    or self.reasons != (self.flags[0].code,)):
                raise ValueError("Measurement failure needs one structured reason/flag and no value")
            flag = self.flags[0]
            if (flag.code not in MEASUREMENT_FAILURE_CODES or flag.category != ErrorCategory.MEASUREMENT_ERROR
                    or flag.stage != Stage.FEATURES or flag.source_tier != SourceTier.S3
                    or flag.severity != Severity.ERROR or flag.location != measurement_location(checked)):
                raise ValueError("Measurement failure flag has invalid code/category/source/location")
            evidence = flag.evidence
            expected_names = {f"input_joint_{label}": name.value for label, name in
                              zip(("a", "b", "c"), definition.required_joints)}
            if (evidence.get("primitive") != "joint_angle"
                    or any(evidence.get(key) != name for key, name in expected_names.items())):
                raise ValueError("Measurement flag must retain primitive and actual input joint identities")
            if flag.code == FlagCode.DEGENERATE_GEOMETRY:
                if evidence.get("reason") != "zero_length_vector" or zero_vector is None or evidence.get("vector") != zero_vector:
                    raise ValueError("Degenerate reason must match the actual zero vector")
                allowed_keys = {"primitive", "reason", "vector", *expected_names}
            else:
                if evidence.get("reason") not in {"invalid_denominator", "non_finite_ratio", "invalid_angle_result"}:
                    raise ValueError("Numerical failure needs a known arithmetic reason")
                allowed_keys = {"primitive", "reason", *expected_names}
            if set(evidence) != allowed_keys:
                raise ValueError("Measurement error evidence cannot contain fabricated measurement values")
        return self

    def content_hash(self) -> str:
        return content_hash(self)
