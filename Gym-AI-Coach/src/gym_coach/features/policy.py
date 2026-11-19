"""Read-only feature prerequisite checks; never calculate geometry or change quality."""

import math

from pydantic import ValidationError

from ..contracts import ErrorCategory, JointProvenance, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..preprocessing.models import ProcessedPoseFrame
from ..quality.models import QualityReport
from .models import FeatureAvailability, FeatureDefinition, FeatureInputContext, content_hash
from .registry import DEFAULT_REGISTRY, FeatureRegistry, input_error, parse_exercise, parse_viewpoint


def policy_flags(definition: FeatureDefinition, context: FeatureInputContext) -> tuple[StructuredFlag, ...]:
    frame, report = context.frame, context.quality
    flags = []

    def add(code, message, *, joint=None, category=ErrorCategory.MEASUREMENT_ERROR, **evidence):
        flags.append(StructuredFlag(
            code=code, message=message, category=category, severity=Severity.WARNING,
            stage=Stage.FEATURES, source_tier=SourceTier.S3,
            location=FlagLocation(video_id=frame.video_id, frame_start=frame.frame_index,
                                  frame_end=frame.frame_index, joint=joint, feature=definition.feature_id),
            evidence=evidence,
        ))

    if report.exercise not in definition.exercises:
        add(FlagCode.UNSUPPORTED_EXERCISE, "Feature does not apply to this MVP exercise", category=ErrorCategory.INPUT_ERROR,
            exercise=report.exercise.value)
    if definition.validation.measurement_status == "blocked":
        add(FlagCode.FEATURE_DEFINITION_BLOCKED, definition.validation.reason,
            category=ErrorCategory.INSUFFICIENT_DATA)
    if report.viewpoint not in definition.viewpoint_policy.supported:
        reason = next(v.reason for v in definition.viewpoint_policy.unsupported if v.viewpoint == report.viewpoint)
        add(FlagCode.VIEWPOINT_UNKNOWN if report.viewpoint == Viewpoint.UNKNOWN else FlagCode.VIEWPOINT_MISMATCH,
            reason, viewpoint=report.viewpoint.value)
    if not report.eligibility.eligible or report.eligibility.action != "allow":
        add(FlagCode.FEATURE_QUALITY_BLOCKED, "Sequence quality does not meet allow-only feature policy",
            category=ErrorCategory.INSUFFICIENT_DATA, scope="movement_analysis", action=report.eligibility.action,
            upstream_reasons=",".join(report.eligibility.reasons))
    policy = definition.quality_policy
    metric = report.metric_eligibility.get(policy.metric_key)
    if metric is None:
        add(FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING, "Required per-feature metric eligibility is absent; no fallback decision inferred",
            category=ErrorCategory.INSUFFICIENT_DATA, metric_key=policy.metric_key)
    else:
        if set(metric.required_joints) != set(definition.required_joints) or len(metric.required_joints) != len(definition.required_joints):
            add(FlagCode.FEATURE_QUALITY_POLICY_MISMATCH, "Metric eligibility joint declaration differs from feature inputs",
                category=ErrorCategory.INPUT_ERROR, metric_key=policy.metric_key)
        if not metric.eligible or metric.action != "allow":
            add(FlagCode.FEATURE_QUALITY_BLOCKED, "Metric quality does not meet allow-only feature policy",
                category=ErrorCategory.INSUFFICIENT_DATA, scope=policy.metric_key, action=metric.action,
                upstream_reasons=",".join(metric.reasons))
    points = {j.joint: j for j in frame.joints}
    for name in definition.required_joints:
        joint = points.get(name)
        if joint is None or joint.status == JointProvenance.MISSING:
            add(FlagCode.MISSING_JOINT, "Required joint is missing", joint=name.value,
                category=ErrorCategory.INSUFFICIENT_DATA)
        if joint is None:
            continue
        if joint.status not in definition.joint_policy.allowed_provenance:
            add(FlagCode.FEATURE_PROVENANCE_REJECTED, "Joint provenance is not permitted by feature policy",
                joint=name.value, status=joint.status.value, source_status=joint.source_status)
        if not joint.usable or joint.x is None or joint.y is None:
            add(FlagCode.FEATURE_JOINT_UNUSABLE, "Required joint has no usable processed coordinates",
                joint=name.value, source_reason=joint.unavailable_reason.value if joint.unavailable_reason else None)
        elif not (math.isfinite(joint.x) and math.isfinite(joint.y)):
            add(FlagCode.NON_FINITE_LANDMARK, "Required joint coordinates must be finite", joint=name.value)
    if policy.requires_mid_hip_origin and frame.normalization.translation_status != "mid_hip":
        add(FlagCode.NORMALIZATION_ORIGIN_FALLBACK, "Feature requires valid mid-hip translation; origin fallback is not permitted")
    if policy.requires_valid_scale and (frame.normalization.scale_status != "torso" or
            frame.normalization.torso_scale is None or not frame.normalization.distance_ratio_eligible):
        add(FlagCode.NORMALIZATION_SCALE_FALLBACK, "Distance/ratio requires valid torso scale; no unit scale is substituted")
    return tuple(flags)


def check_feature_availability(feature_id: str, frame: ProcessedPoseFrame, quality: QualityReport,
                               *, registry: FeatureRegistry = DEFAULT_REGISTRY) -> FeatureAvailability:
    """Return readiness/unavailable with full input snapshots, no numerical feature."""
    if not isinstance(frame, ProcessedPoseFrame) or not isinstance(quality, QualityReport) or not isinstance(registry, FeatureRegistry):
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, "Expected ProcessedPoseFrame, QualityReport and FeatureRegistry")
    definition = registry.get(feature_id)
    # Preserve explicit error codes for unsafe/unknown enum input as well as
    # UNKNOWN (a valid enum with a viewpoint-dependent unavailable outcome).
    parse_exercise(quality.exercise)
    parse_viewpoint(quality.viewpoint)
    try:
        checked_registry = FeatureRegistry.model_validate_json(registry.model_dump_json())
        definition = checked_registry.get(feature_id)
        context = FeatureInputContext.model_validate_json(FeatureInputContext(
            frame=frame, quality=quality).model_dump_json())
    except (ValidationError, ValueError, TypeError, AttributeError) as exc:
        raise input_error(FlagCode.INVALID_FEATURE_INPUT, str(exc)) from exc
    flags = policy_flags(definition, context)
    return FeatureAvailability(
        registry_version=checked_registry.registry_version, registry_hash=checked_registry.content_hash(),
        definition=definition, definition_hash=definition.content_hash(), input=context,
        input_hash=content_hash(context), status="unavailable" if flags else "ready_for_calculation",
        reasons=tuple(dict.fromkeys(f.code for f in flags)), flags=flags,
    )
