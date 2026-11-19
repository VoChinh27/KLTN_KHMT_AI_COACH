"""Uniform one-frame/one-feature measurement API using canonical registry v2."""

from pydantic import ValidationError

from .. import axis_geometry, geometry
from ..contracts import ErrorCategory
from ..flags import FlagCode, GymError
from ..preprocessing.models import ProcessedPoseFrame
from ..quality.models import QualityReport
from .context import LinkedFeatureInputContext, _strict_json, _validated_linked_context
from .geometry_observation import GeometryFeatureObservation, input_names, primitive_fields
from .geometry_policy import geometry_availability
from .geometry_registry import GEOMETRY_REGISTRY, GeometryFeatureDefinition
from .models import FeatureInputContext
from .observation import MEASUREMENT_FAILURE_CODES, measurement_location, observation_fields, validate_context_references
from .registry import input_error, parse_exercise, parse_viewpoint


def observe_geometry_feature(feature: str | GeometryFeatureDefinition,
                             context: FeatureInputContext) -> GeometryFeatureObservation:
    """Measure a canonical angle or return explicit unavailable, without promotion.

    Registry v1/legacy knee API remains separate. Input viewpoint is a caller
    assertion, not proof of annotation. Exploratory assumptions must be audited
    separately; no observation validates camera protocol, accuracy or form.
    """
    if (not isinstance(context, FeatureInputContext)
            or not isinstance(context.frame, ProcessedPoseFrame)
            or not isinstance(context.quality, QualityReport)):
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Expected FeatureInputContext")
    if isinstance(feature, str):
        definition = GEOMETRY_REGISTRY.get(feature)
    elif isinstance(feature, GeometryFeatureDefinition):
        definition = GEOMETRY_REGISTRY.get(feature.feature_id)
        try:
            supplied = GeometryFeatureDefinition.model_validate_json(_strict_json(feature), strict=True)
            if supplied != definition:
                raise ValueError("Supplied definition differs from canonical registry v2")
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise input_error(FlagCode.INVALID_FEATURE_INPUT, str(exc)) from exc
    else:
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Expected canonical v2 definition or feature ID")
    parse_exercise(context.quality.exercise)
    parse_viewpoint(context.quality.viewpoint)
    try:
        if isinstance(context, LinkedFeatureInputContext):
            context = _validated_linked_context(context)
        context = FeatureInputContext.model_validate_json(_strict_json(
            FeatureInputContext(frame=context.frame, quality=context.quality)), strict=True)
        validate_context_references(context)
    except (ValidationError, ValueError, TypeError, AttributeError) as exc:
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, str(exc)) from exc
    availability = geometry_availability(definition, context)
    fields = {**observation_fields(availability), **primitive_fields(definition), "availability": availability}
    if availability.status == "unavailable":
        return GeometryFeatureObservation(**fields, status="unavailable", computation_status="not_computed",
                                          value=None, reasons=availability.reasons, flags=availability.flags)
    points = tuple((j.x, j.y) for j in fields["joint_inputs"])
    try:
        if definition.measurement_method == "joint_angle_abc":
            value = geometry.joint_angle(*points)
        elif definition.measurement_method == "midpoint_axis_angle_vertical":
            value = axis_geometry.midpoint_axis_angle_vertical(*points)
        else:
            raise input_error(FlagCode.UNSUPPORTED_FEATURE_CALCULATION, "No calculator for blocked method")
    except GymError as exc:
        if exc.flag.category != ErrorCategory.MEASUREMENT_ERROR or exc.flag.code not in MEASUREMENT_FAILURE_CODES:
            raise
        flag = exc.flag.model_copy(update={"location": measurement_location(availability),
                                          "evidence": {**exc.flag.evidence, **input_names(definition)}})
        return GeometryFeatureObservation(**fields, status="unavailable", computation_status="failed_measurement",
                                          value=None, reasons=(flag.code,), flags=(flag,))
    return GeometryFeatureObservation(**fields, status="available", computation_status="computed",
                                      value=value, reasons=(), flags=())
