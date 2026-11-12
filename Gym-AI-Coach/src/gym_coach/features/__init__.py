"""Metadata, linked contexts and versioned individual squat knee observations."""

from .models import (
    FEATURE_SCHEMA_VERSION, FeatureAvailability, FeatureDefinition, FeatureInputContext,
    FeatureObservation, FeatureQualityPolicy, FeatureUnit, FeatureValidation,
    JointInputPolicy, QuantityKind, UnsupportedViewpoint, ViewpointPolicy,
)
from .registry import DEFAULT_REGISTRY, FEATURE_REGISTRY_VERSION, FeatureRegistry
from .policy import check_feature_availability
from .observation import (
    FEATURE_ADAPTER_VERSION, NUMERIC_FEATURE_SCHEMA_VERSION, SUPPORTED_NUMERIC_FEATURE_ID,
    NumericFeatureObservation,
)
from .adapter import observe_feature
from .knee_observation import (
    KNEE_ANGLE_SCHEMA_VERSION, RIGHT_KNEE_ADAPTER_VERSION, RIGHT_NUMERIC_FEATURE_ID,
    SUPPORTED_KNEE_FEATURE_IDS, KneeAngleFeatureObservation,
)
from .context import (
    FEATURE_CONTEXT_FACTORY_VERSION, FEATURE_CONTEXT_SCHEMA_VERSION,
    LinkedFeatureInputContext, build_feature_input_context,
)
from .geometry_registry import (
    ANGLE_FEATURE_IDS, FRAMEWISE_POLICY_VERSION, GEOMETRY_REGISTRY_VERSION,
    GEOMETRY_REGISTRY, GeometryFeatureDefinition, GeometryFeatureRegistry,
)
from .geometry_observation import (
    GEOMETRY_ADAPTER_VERSION, GEOMETRY_OBSERVATION_SCHEMA_VERSION, GeometryFeatureObservation,
)
from .geometry_adapter import observe_geometry_feature
from .session import FEATURE_CONTEXT_SESSION_VERSION, FeatureContextSession

__all__ = [
    "FEATURE_SCHEMA_VERSION", "FeatureAvailability", "FeatureDefinition", "FeatureInputContext",
    "FeatureObservation", "FeatureQualityPolicy", "FeatureUnit", "FeatureValidation",
    "JointInputPolicy", "QuantityKind", "UnsupportedViewpoint", "ViewpointPolicy",
    "DEFAULT_REGISTRY", "FEATURE_REGISTRY_VERSION", "FeatureRegistry", "check_feature_availability",
    "FEATURE_ADAPTER_VERSION", "NUMERIC_FEATURE_SCHEMA_VERSION", "SUPPORTED_NUMERIC_FEATURE_ID",
    "NumericFeatureObservation", "observe_feature",
    "FEATURE_CONTEXT_FACTORY_VERSION", "FEATURE_CONTEXT_SCHEMA_VERSION",
    "LinkedFeatureInputContext", "build_feature_input_context",
    "KNEE_ANGLE_SCHEMA_VERSION", "RIGHT_KNEE_ADAPTER_VERSION", "RIGHT_NUMERIC_FEATURE_ID",
    "SUPPORTED_KNEE_FEATURE_IDS", "KneeAngleFeatureObservation",
    "ANGLE_FEATURE_IDS", "FRAMEWISE_POLICY_VERSION", "GEOMETRY_REGISTRY_VERSION", "GEOMETRY_REGISTRY",
    "GeometryFeatureDefinition", "GeometryFeatureRegistry", "GEOMETRY_ADAPTER_VERSION",
    "GEOMETRY_OBSERVATION_SCHEMA_VERSION", "GeometryFeatureObservation", "observe_geometry_feature",
    "FEATURE_CONTEXT_SESSION_VERSION", "FeatureContextSession",
]
