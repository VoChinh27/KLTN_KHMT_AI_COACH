"""Additive canonical registry v2; registry v1 and legacy knee gates stay intact."""

from typing import Literal

from pydantic import Field, model_validator

from ..flags import FlagCode
from .models import FeatureDefinition, FeatureQualityPolicy, FeatureValidation, Hash, QuantityKind
from .registry import DEFAULT_REGISTRY, FeatureRegistry

GEOMETRY_REGISTRY_VERSION = "gym-feature-registry-v2"
FRAMEWISE_POLICY_VERSION = "framewise-metric-only-v1"


class GeometryQualityPolicy(FeatureQualityPolicy):
    require_sequence_allow: bool = Field(strict=True)
    policy_version: Literal["framewise-metric-only-v1", "sequence-and-metric-v1"]


class GeometryValidation(FeatureValidation):
    calculation_status: Literal["implemented", "not_implemented"]


class GeometryFeatureDefinition(FeatureDefinition):
    feature_schema_version: Literal["0.2.0"] = "0.2.0"
    definition_version: Literal["0.2.0"] = "0.2.0"
    quality_policy: GeometryQualityPolicy
    validation: GeometryValidation

    @model_validator(mode="after")
    def local_policy_scope(self):
        angle = self.quantity_kind == QuantityKind.ANGLE
        if angle:
            if (self.quality_policy.require_sequence_allow
                    or self.quality_policy.policy_version != FRAMEWISE_POLICY_VERSION
                    or self.validation.calculation_status != "implemented"):
                raise ValueError("v2 angles require the explicit metric-only framewise policy")
        elif (not self.quality_policy.require_sequence_allow
              or self.quality_policy.policy_version != "sequence-and-metric-v1"
              or self.validation.calculation_status != "not_implemented"
              or self.validation.measurement_status != "blocked"):
            raise ValueError("Non-angle candidates remain blocked and unimplemented")
        return self


def _definitions():
    definitions = []
    for old in DEFAULT_REGISTRY.features:
        data = old.model_dump(mode="json")
        angle = old.quantity_kind == QuantityKind.ANGLE
        data.update(feature_schema_version="0.2.0", definition_version="0.2.0")
        data["validation"]["calculation_status"] = "implemented" if angle else "not_implemented"
        data["quality_policy"].update(
            require_sequence_allow=not angle,
            policy_version=FRAMEWISE_POLICY_VERSION if angle else "sequence-and-metric-v1")
        if angle:
            data["quality_policy"]["reason"] = (
                "S3 framewise description requires exact metric allow and usable provenance-safe inputs; "
                "global movement_analysis remains unchanged and does not authorize temporal analysis or assessment")
            data["source_references"].append("GYM_TECHNICAL_GROUND_TRUTH_SPEC.md sections 2.1, 2.2: sequence versus feature gates")
            data["unavailable_reasons"].extend((FlagCode.DEGENERATE_GEOMETRY.value, FlagCode.GEOMETRY_NUMERICAL_ERROR.value))
        definitions.append(GeometryFeatureDefinition.model_validate(data))
    return tuple(definitions)


class GeometryFeatureRegistry(FeatureRegistry):
    registry_schema_version: Literal["0.2.0"] = "0.2.0"
    registry_version: Literal["gym-feature-registry-v2"] = GEOMETRY_REGISTRY_VERSION
    source_registry_version: Literal["gym-feature-registry-v1"] = "gym-feature-registry-v1"
    source_registry_hash: Hash
    features: tuple[GeometryFeatureDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_derivation(self):
        if self.source_registry_hash != DEFAULT_REGISTRY.content_hash() or self.features != _definitions():
            raise ValueError("Registry v2 must preserve the canonical v1 measurements and explicit v2 policy changes")
        return self


GEOMETRY_REGISTRY = GeometryFeatureRegistry(source_registry_hash=DEFAULT_REGISTRY.content_hash(), features=_definitions())
ANGLE_FEATURE_IDS = tuple(f.feature_id for f in GEOMETRY_REGISTRY.features if f.quantity_kind == QuantityKind.ANGLE)
