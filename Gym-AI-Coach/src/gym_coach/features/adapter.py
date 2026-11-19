"""Single-frame dispatch for the two canonical squat knee angles."""

from pydantic import ValidationError

from .. import geometry
from ..contracts import ErrorCategory
from ..flags import FlagCode, GymError
from .context import LinkedFeatureInputContext, _validated_linked_context
from .models import FeatureDefinition, FeatureInputContext
from .knee_observation import (
    RIGHT_KNEE_ADAPTER_VERSION, RIGHT_NUMERIC_FEATURE_ID, SUPPORTED_KNEE_FEATURE_IDS,
    KneeAngleFeatureObservation,
)
from .observation import (
    MEASUREMENT_FAILURE_CODES, SUPPORTED_NUMERIC_FEATURE_ID, NumericFeatureObservation,
    measurement_flag, observation_fields, validate_context_references,
)
from .policy import check_feature_availability
from .registry import DEFAULT_REGISTRY, input_error


def observe_feature(feature: str | FeatureDefinition, context: FeatureInputContext) -> NumericFeatureObservation | KneeAngleFeatureObservation:
    """Measure one canonical knee ID, preserving policies and input provenance.

    Left dispatch returns the unchanged legacy schema 0.2.0. Right dispatch
    returns KneeAngleFeatureObservation 0.3.0 with its own adapter version.

    Unknown IDs, unsupported calculators or malformed/mismatched input raise
    structured GymError. Valid prerequisite failures and primitive measurement
    failures return unavailable with null value. No quality recomputation,
    promotion, transformation, batch extraction or form interpretation occurs.
    """
    if not isinstance(context, FeatureInputContext):
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Expected FeatureInputContext")
    if isinstance(context, LinkedFeatureInputContext):
        context = _validated_linked_context(context)
    if isinstance(feature, FeatureDefinition):
        try:
            supplied = FeatureDefinition.model_validate_json(feature.model_dump_json(warnings="error"))
        except (ValidationError, ValueError, TypeError) as exc:
            raise input_error(FlagCode.INVALID_FEATURE_INPUT, str(exc)) from exc
        definition = DEFAULT_REGISTRY.get(supplied.feature_id)
        if supplied != definition:
            raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Supplied definition differs from the canonical MVP registry")
    elif isinstance(feature, str):
        definition = DEFAULT_REGISTRY.get(feature)
    else:
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Expected feature ID or canonical FeatureDefinition")
    if definition.feature_id not in SUPPORTED_KNEE_FEATURE_IDS:
        raise input_error(FlagCode.UNSUPPORTED_FEATURE_CALCULATION,
                          f"Only {SUPPORTED_KNEE_FEATURE_IDS} have numeric adapters; requested {definition.feature_id}")
    observation_type = KneeAngleFeatureObservation if definition.feature_id == RIGHT_NUMERIC_FEATURE_ID else NumericFeatureObservation
    version_fields = {"adapter_version": RIGHT_KNEE_ADAPTER_VERSION} if definition.feature_id == RIGHT_NUMERIC_FEATURE_ID else {}
    availability = check_feature_availability(definition.feature_id, context.frame, context.quality)
    try:
        validate_context_references(availability.input)
    except ValueError as exc:
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, str(exc)) from exc
    fields = observation_fields(availability)
    if availability.status == "unavailable":
        return observation_type(**fields, **version_fields, availability=availability, status="unavailable",
                                         computation_status="not_computed", value=None,
                                         reasons=availability.reasons, flags=availability.flags)
    points = tuple((joint.x, joint.y) for joint in fields["joint_inputs"])
    try:
        value = geometry.joint_angle(*points)
    except GymError as exc:
        if exc.flag.category != ErrorCategory.MEASUREMENT_ERROR or exc.flag.code not in MEASUREMENT_FAILURE_CODES:
            raise
        flag = measurement_flag(exc.flag, availability)
        return observation_type(**fields, **version_fields, availability=availability, status="unavailable",
                                         computation_status="failed_measurement", value=None,
                                         reasons=(flag.code,), flags=(flag,))
    return observation_type(**fields, **version_fields, availability=availability, status="available",
                                     computation_status="computed", value=value, reasons=(), flags=())
