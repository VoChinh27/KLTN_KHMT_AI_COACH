"""Schema 0.4: canonical framewise angles, explicit unavailable and audit gates."""

import math
from typing import Literal

from pydantic import Field, model_validator

from ..axis_geometry import AXIS_ANGLE_PRIMITIVE_VERSION, axis_vector
from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, Severity
from ..geometry import JOINT_ANGLE_PRIMITIVE_VERSION
from ..preprocessing.models import ProcessedJoint
from .context import _strict_json
from .geometry_policy import GeometryFeatureAvailability
from .geometry_registry import GEOMETRY_REGISTRY
from .models import QuantityKind
from .observation import (MEASUREMENT_FAILURE_CODES, NumericFeatureObservation,
                          measurement_location, observation_fields, validate_context_references)

GEOMETRY_OBSERVATION_SCHEMA_VERSION = "0.4.0"
GEOMETRY_ADAPTER_VERSION = "geometry-features-v1"


def primitive_fields(definition):
    if definition.measurement_method == "joint_angle_abc":
        return dict(primitive_name="joint_angle", primitive_version=JOINT_ANGLE_PRIMITIVE_VERSION)
    if definition.measurement_method == "midpoint_axis_angle_vertical":
        return dict(primitive_name="midpoint_axis_angle_vertical", primitive_version=AXIS_ANGLE_PRIMITIVE_VERSION)
    return dict(primitive_name=None, primitive_version=None)


def zero_vector(definition, points):
    if definition.measurement_method == "joint_angle_abc":
        a, b, c = points
        return "BA" if a == b else "BC" if c == b else None
    dx, dy = axis_vector(points)
    return "hip_to_shoulder" if dx == dy == 0 else None


def input_names(definition):
    return {f"input_joint_{i}": name.value for i, name in enumerate(definition.required_joints)}


class GeometryFeatureObservation(NumericFeatureObservation):
    numeric_feature_schema_version: Literal["0.4.0"] = GEOMETRY_OBSERVATION_SCHEMA_VERSION
    adapter_version: Literal["geometry-features-v1"] = GEOMETRY_ADAPTER_VERSION
    primitive_name: Literal["joint_angle", "midpoint_axis_angle_vertical"] | None
    primitive_version: Literal["joint-angle-2d-v1", "midpoint-axis-angle-vertical-2d-v1"] | None
    feature_id: str
    measurement_method: Literal["joint_angle_abc", "midpoint_axis_angle_vertical",
                                "horizontal_offset_torso_ratio", "normalized_euclidean_distance"]
    availability: GeometryFeatureAvailability
    joint_inputs: tuple[ProcessedJoint, ...] = Field(min_length=2, max_length=4)
    measurement_scope: Literal["framewise_projected_2d"] = "framewise_projected_2d"
    temporal_analysis_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_audit_and_outcome(self):
        checked = GeometryFeatureAvailability.model_validate_json(_strict_json(self.availability), strict=True)
        definition = GEOMETRY_REGISTRY.get(self.feature_id)
        validate_context_references(checked.input)
        for name, expected in {**observation_fields(checked), **primitive_fields(definition)}.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Observation {name} differs from canonical input/definition")
        if self.temporal_analysis_eligible is not False:
            raise ValueError("Framewise measurement does not authorize temporal analysis")
        # Defend unsafe nested model_copy instances as well as serialized data.
        for joint in self.joint_inputs:
            ProcessedJoint.model_validate_json(_strict_json(joint), strict=True)
        if checked.status == "unavailable":
            if (self.status != "unavailable" or self.computation_status != "not_computed" or self.value is not None
                    or self.reasons != checked.reasons or self.flags != checked.flags):
                raise ValueError("Prerequisite failures must retain reasons/flags and null value")
            return self
        if definition.quantity_kind != QuantityKind.ANGLE:
            raise ValueError("Only unblocked angle measurements are implemented")
        points = tuple((j.x, j.y) for j in self.joint_inputs)
        zero = zero_vector(definition, points)
        if self.status == "available":
            if (type(self.value) is not float or not math.isfinite(self.value) or not 0 <= self.value <= 180
                    or self.computation_status != "computed" or self.reasons or self.flags or zero):
                raise ValueError("Available angle requires finite degrees and nondegenerate geometry")
            return self
        if (self.value is not None or self.computation_status != "failed_measurement" or len(self.flags) != 1
                or self.reasons != (self.flags[0].code,)):
            raise ValueError("Measurement failure requires one structured reason and null value")
        flag = self.flags[0]
        if (flag.code not in MEASUREMENT_FAILURE_CODES or flag.category != ErrorCategory.MEASUREMENT_ERROR
                or flag.stage != Stage.FEATURES or flag.source_tier != SourceTier.S3
                or flag.severity != Severity.ERROR or flag.location != measurement_location(checked)):
            raise ValueError("Measurement error category/stage/source/location mismatch")
        names = input_names(definition)
        evidence = flag.evidence
        if (evidence.get("primitive") != self.primitive_name
                or any(evidence.get(k) != v for k, v in names.items())):
            raise ValueError("Measurement error must identify the primitive and every ordered joint")
        keys = {"primitive", "reason", *names}
        if flag.code == FlagCode.DEGENERATE_GEOMETRY:
            if evidence.get("reason") != "zero_length_vector" or zero is None or evidence.get("vector") != zero:
                raise ValueError("Degenerate reason must match actual input vector")
            keys.add("vector")
        elif evidence.get("reason") not in {"invalid_denominator", "non_finite_ratio", "invalid_angle_result"}:
            raise ValueError("Unknown numerical measurement reason")
        if set(evidence) != keys:
            raise ValueError("Measurement error cannot contain fabricated numeric results")
        return self
