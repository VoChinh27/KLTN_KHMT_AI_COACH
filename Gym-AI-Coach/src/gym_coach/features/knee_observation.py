"""Numeric schema 0.3.0 for two individual knee IDs; legacy 0.2.0 is frozen."""

import json
import math
from typing import Literal

from pydantic import model_validator

from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, Severity
from ..preprocessing.models import ProcessedJoint
from .models import FeatureAvailability, FeatureUnit, QuantityKind
from .observation import (
    FEATURE_ADAPTER_VERSION, MEASUREMENT_FAILURE_CODES, SUPPORTED_NUMERIC_FEATURE_ID,
    NumericFeatureObservation, measurement_location, observation_fields, validate_context_references,
)
from .registry import DEFAULT_REGISTRY

KNEE_ANGLE_SCHEMA_VERSION = "0.3.0"
RIGHT_KNEE_ADAPTER_VERSION = "squat-right-knee-v1"
RIGHT_NUMERIC_FEATURE_ID = "squat_knee_angle_right_deg"
SUPPORTED_KNEE_FEATURE_IDS = (SUPPORTED_NUMERIC_FEATURE_ID, RIGHT_NUMERIC_FEATURE_ID)


def _encoded(value) -> str:
    # The observation's frame/quality payload has no Path or backend objects.
    # Preserve bool and reject nonfinite numbers before strict JSON validation.
    return json.dumps(value.model_dump(mode="python", warnings="error"),
                      ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class KneeAngleFeatureObservation(NumericFeatureObservation):
    """One projected knee angle, selected by ID; never a bilateral measurement.

    Inherit the field contract, override the version and canonical-ID validator.
    The Stage 5.3 class/source/schema and its left-side JSON stay unchanged.
    """

    numeric_feature_schema_version: Literal["0.3.0"] = KNEE_ANGLE_SCHEMA_VERSION
    adapter_version: Literal["squat-left-knee-v1", "squat-right-knee-v1"]
    feature_id: Literal["squat_knee_angle_left_deg", "squat_knee_angle_right_deg"]

    @model_validator(mode="after")
    def validate_audit_and_outcome(self) -> "KneeAngleFeatureObservation":
        # Override, rather than run, the legacy left-only validator. Share audit
        # derivation/flag helpers; do not change its compatibility boundary.
        if self.feature_id not in SUPPORTED_KNEE_FEATURE_IDS:
            raise ValueError("Schema 0.3.0 supports only the two canonical squat knee IDs")
        adapter = FEATURE_ADAPTER_VERSION if self.feature_id == SUPPORTED_NUMERIC_FEATURE_ID else RIGHT_KNEE_ADAPTER_VERSION
        if self.adapter_version != adapter:
            raise ValueError("Adapter version must match the selected knee feature ID")
        checked = FeatureAvailability.model_validate_json(_encoded(self.availability), strict=True)
        definition = DEFAULT_REGISTRY.get(self.feature_id)
        if (checked.definition != definition or checked.registry_version != DEFAULT_REGISTRY.registry_version
                or checked.registry_hash != DEFAULT_REGISTRY.content_hash()):
            raise ValueError("Knee observation requires the canonical registry definition and identity")
        validate_context_references(checked.input)
        for joint in self.joint_inputs:
            ProcessedJoint.model_validate_json(_encoded(joint), strict=True)
        for field, expected in observation_fields(checked).items():
            if getattr(self, field) != expected:
                raise ValueError(f"Observation {field} must match its definition/input audit snapshot")
        if self.quantity_kind != QuantityKind.ANGLE or self.unit != FeatureUnit.DEGREES:
            raise ValueError("Knee observation must be angle/degrees")
        if self.value is not None and (type(self.value) is not float or not math.isfinite(self.value)
                                       or not 0 <= self.value <= 180):
            raise ValueError("Measured value must be a finite float in [0,180] degrees")
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
            names = {f"input_joint_{label}": name.value for label, name in
                     zip(("a", "b", "c"), definition.required_joints)}
            if evidence.get("primitive") != "joint_angle" or any(evidence.get(key) != name for key, name in names.items()):
                raise ValueError("Measurement flag must retain primitive and actual input joint identities")
            if flag.code == FlagCode.DEGENERATE_GEOMETRY:
                if evidence.get("reason") != "zero_length_vector" or zero_vector is None or evidence.get("vector") != zero_vector:
                    raise ValueError("Degenerate reason must match the actual zero vector")
                allowed_keys = {"primitive", "reason", "vector", *names}
            else:
                if evidence.get("reason") not in {"invalid_denominator", "non_finite_ratio", "invalid_angle_result"}:
                    raise ValueError("Numerical failure needs a known arithmetic reason")
                allowed_keys = {"primitive", "reason", *names}
            if set(evidence) != allowed_keys:
                raise ValueError("Measurement error evidence cannot contain fabricated measurement values")
        return self

    @classmethod
    def from_legacy(cls, observation: NumericFeatureObservation) -> "KneeAngleFeatureObservation":
        """Explicitly migrate a validated left 0.2.0 result without recalculating it."""
        if type(observation) is not NumericFeatureObservation:
            raise TypeError("Expected a legacy NumericFeatureObservation 0.2.0")
        checked = NumericFeatureObservation.model_validate_json(_encoded(observation), strict=True)
        payload = checked.model_dump(mode="python")
        payload["numeric_feature_schema_version"] = KNEE_ANGLE_SCHEMA_VERSION
        return cls.model_validate_json(json.dumps(payload, ensure_ascii=False, allow_nan=False), strict=True)
