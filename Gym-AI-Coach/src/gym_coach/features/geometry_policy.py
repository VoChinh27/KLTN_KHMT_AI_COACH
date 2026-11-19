"""Versioned framewise prerequisite evaluation; never changes QualityReport."""

from typing import Literal

from pydantic import model_validator

from .context import _strict_json
from .geometry_registry import GEOMETRY_REGISTRY, GeometryFeatureDefinition
from .models import FeatureAvailability, FeatureInputContext, content_hash
from .policy import policy_flags


def framewise_policy_flags(definition, context):
    # Reuse every v1 safeguard. Only the explicit sequence-scope blocker is
    # omitted for the canonical v2 angle policy; metric/joint gates stay intact.
    flags = policy_flags(definition, context)
    if definition.quality_policy.require_sequence_allow:
        return flags
    return tuple(f for f in flags if not (
        f.code.value == "feature_quality_blocked" and f.evidence.get("scope") == "movement_analysis"))


class GeometryFeatureAvailability(FeatureAvailability):
    feature_schema_version: Literal["0.2.0"] = "0.2.0"
    definition: GeometryFeatureDefinition

    @model_validator(mode="after")
    def gate_consistency(self):
        checked = FeatureInputContext.model_validate_json(_strict_json(self.input), strict=True)
        definition = GEOMETRY_REGISTRY.get(self.definition.feature_id)
        if (self.definition != definition or self.definition_hash != definition.content_hash()
                or self.registry_version != GEOMETRY_REGISTRY.registry_version
                or self.registry_hash != GEOMETRY_REGISTRY.content_hash()
                or self.input_hash != content_hash(checked)):
            raise ValueError("Geometry availability requires canonical v2 definition/registry/input hashes")
        expected = framewise_policy_flags(definition, checked)
        if (self.flags != expected or self.reasons != tuple(dict.fromkeys(f.code for f in expected))
                or self.status != ("unavailable" if expected else "ready_for_calculation")):
            raise ValueError("Geometry availability must match exact versioned framewise policy gates")
        return self


def geometry_availability(definition, context):
    flags = framewise_policy_flags(definition, context)
    return GeometryFeatureAvailability(
        registry_version=GEOMETRY_REGISTRY.registry_version, registry_hash=GEOMETRY_REGISTRY.content_hash(),
        definition=definition, definition_hash=definition.content_hash(), input=context, input_hash=content_hash(context),
        status="unavailable" if flags else "ready_for_calculation",
        reasons=tuple(dict.fromkeys(f.code for f in flags)), flags=flags)
