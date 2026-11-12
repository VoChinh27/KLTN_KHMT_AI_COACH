"""Immutable MVP metadata registry, with no feature calculation or form rules."""

from pydantic import Field, model_validator
from typing import Literal

from ..contracts import ContractModel, ErrorCategory, Exercise, JointProvenance, MeasurementTier, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, GymError, Severity, StructuredFlag
from ..pose.mapping import JointName as J
from .models import (
    FeatureDefinition, FeatureQualityPolicy, FeatureUnit, FeatureValidation,
    JointInputPolicy, QuantityKind, UnsupportedViewpoint, ViewpointPolicy, content_hash,
)

FEATURE_REGISTRY_VERSION = "gym-feature-registry-v1"


def input_error(code: FlagCode, message: str) -> GymError:
    return GymError(StructuredFlag(
        code=code, category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
        stage=Stage.FEATURES, source_tier=SourceTier.S3, message=message,
    ))


def parse_exercise(exercise: Exercise | str) -> Exercise:
    try:
        return Exercise(exercise)
    except (ValueError, TypeError) as exc:
        raise input_error(FlagCode.UNSUPPORTED_EXERCISE, f"Unsupported exercise: {exercise!r}") from exc


def parse_viewpoint(viewpoint: Viewpoint | str) -> Viewpoint:
    try:
        return Viewpoint(viewpoint)
    except (ValueError, TypeError) as exc:
        raise input_error(FlagCode.INVALID_VIEWPOINT, f"Invalid viewpoint: {viewpoint!r}") from exc


class FeatureRegistry(ContractModel):
    registry_schema_version: Literal["0.1.0"] = "0.1.0"
    registry_version: str = Field(min_length=1)
    features: tuple[FeatureDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "FeatureRegistry":
        ids = [f.feature_id for f in self.features]
        if len(set(ids)) != len(ids):
            raise ValueError("feature IDs must be unique")
        if any(f.quality_policy.metric_key != f.feature_id for f in self.features):
            raise ValueError("registry metric key must equal the stable feature ID")
        return self

    def get(self, feature_id: str) -> FeatureDefinition:
        for feature in self.features:
            if feature.feature_id == feature_id:
                return feature
        raise input_error(FlagCode.UNKNOWN_FEATURE_ID, f"Unknown feature ID: {feature_id!r}")

    def for_exercise(self, exercise: Exercise | str, *, include_blocked: bool = False) -> tuple[FeatureDefinition, ...]:
        selected = parse_exercise(exercise)
        return tuple(f for f in self.features if selected in f.exercises and (
            include_blocked or f.validation.measurement_status != "blocked"))

    def quality_requirements(self, exercise: Exercise | str) -> dict[str, tuple[J, ...]]:
        """New mapping for the Stage 3 caller; never updates an existing report."""
        return {f.quality_policy.metric_key: f.required_joints for f in self.for_exercise(exercise)}

    def quality_viewpoints(self, exercise: Exercise | str) -> dict[str, tuple[Viewpoint, ...]]:
        return {f.quality_policy.metric_key: f.viewpoint_policy.supported for f in self.for_exercise(exercise)}

    def content_hash(self) -> str:
        return content_hash(self)


_REASONS = (
    FlagCode.UNSUPPORTED_EXERCISE, FlagCode.INVALID_FEATURE_INPUT,
    FlagCode.FEATURE_DEFINITION_BLOCKED, FlagCode.MISSING_JOINT,
    FlagCode.FEATURE_JOINT_UNUSABLE, FlagCode.FEATURE_PROVENANCE_REJECTED,
    FlagCode.NON_FINITE_LANDMARK, FlagCode.VIEWPOINT_UNKNOWN, FlagCode.VIEWPOINT_MISMATCH,
    FlagCode.FEATURE_QUALITY_BLOCKED, FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING,
    FlagCode.FEATURE_QUALITY_POLICY_MISMATCH,
)

_VIEW = ViewpointPolicy(
    supported=(Viewpoint.SIDE,),
    unsupported=(
        UnsupportedViewpoint(viewpoint=Viewpoint.UNKNOWN, reason="No annotated viewpoint; image plane applicability cannot be established"),
        UnsupportedViewpoint(viewpoint=Viewpoint.FRONT, reason="Front projection is outside this sagittal candidate policy; no symmetry/form inference"),
        UnsupportedViewpoint(viewpoint=Viewpoint.OBLIQUE, reason="Oblique projection/foreshortening protocol has not been validated"),
    ),
    policy_reason="Conservative S3 side-view candidate policy, not a validated camera protocol; HSPU also needs visible joints",
    source_tier=SourceTier.S3,
)
_JOINTS = JointInputPolicy(
    coordinate_source="stage4_processed_xy",
    allowed_provenance=(JointProvenance.OBSERVED, JointProvenance.IMPUTED),
    require_usable=True, require_finite_xy=True,
    imputation_policy_reason="Explicit Stage 4 imputation with endpoints may support description; retain source status and never promote quality",
)


def _definition(feature_id, name, exercise, joints, formula, *,
                method="joint_angle_abc", tier=MeasurementTier.M1, blocked=False):
    ratio = method == "horizontal_offset_torso_ratio"
    return FeatureDefinition(
        definition_version="0.1.0", feature_id=feature_id, name=name, exercises=(exercise,),
        measurement_definition=("Projected horizontal knee-to-ankle offset relative to torso length; no toe or depth assertion" if ratio else
                                "Directed projected mid-hip to mid-shoulder axis angle to image-up vertical (0,-1); not full-body alignment" if method == "midpoint_axis_angle_vertical" else
                                "Projected internal angle at ordered joint B, vectors B->A and B->C, mathematical range [0,180] degrees"),
        measurement_method=method, formula=formula, required_joints=joints, joint_policy=_JOINTS,
        quantity_kind=QuantityKind.RATIO if ratio else QuantityKind.ANGLE,
        unit=FeatureUnit.DIMENSIONLESS_RATIO if ratio else FeatureUnit.DEGREES,
        measurement_tier=tier,
        source_tiers=(SourceTier.S3, SourceTier.TBD_EXPERT) if ratio or method == "midpoint_axis_angle_vertical" else (SourceTier.S3,),
        source_references=("docs/GYM_TECHNICAL_GROUND_TRUTH_SPEC.md §§0.2,2.2," + ("3.2-3.3" if exercise == Exercise.BODY_WEIGHT_SQUATS else "4.2-4.3"),),
        validation=FeatureValidation(
            measurement_status="blocked" if blocked else "unvalidated",
            reason=("Viewpoint/scale measurement protocol and interpretation require development/expert validation; metadata retained only" if blocked else
                    "Mathematical candidate definition only; no geometry implementation, pose accuracy or form validation"),
            calculation_status="not_implemented", assessment_status="blocked_pending_expert_validation",
        ),
        viewpoint_policy=_VIEW,
        quality_policy=FeatureQualityPolicy(
            metric_key=feature_id, require_sequence_allow=True, require_metric_allow=True,
            require_exact_metric_joints=True, requires_valid_scale=ratio,
            requires_mid_hip_origin=ratio,
            reason="Conservative S3 allow-only quality policy; no degrade/skip promotion; distance/ratio forbid normalization fallbacks",
        ),
        roles=("descriptive", "informational") if ratio or method == "midpoint_axis_angle_vertical" else ("descriptive", "segmentation_candidate"),
        unavailable_reasons=_REASONS + ((FlagCode.NORMALIZATION_ORIGIN_FALLBACK, FlagCode.NORMALIZATION_SCALE_FALLBACK) if ratio else ()),
        limitations=("Single-camera 2D image projection, no physical calibration or expert form ground truth",
                     "Side viewpoint is supplied metadata, not classified or empirically validated",
                     "Degenerate vectors and arithmetic safety remain calculation-stage checks; readiness is not a numeric result",
                     "No bilateral fusion, side switching, repetition or phase logic"),
    )


def _mvp() -> tuple[FeatureDefinition, ...]:
    entries = []
    for side in ("left", "right"):
        shoulder, hip, knee, ankle, elbow, wrist = (J(f"{side}_{part}") for part in (
            "shoulder", "hip", "knee", "ankle", "elbow", "wrist"))
        entries.extend((
            _definition(f"squat_knee_angle_{side}_deg", f"Squat {side} projected knee angle", Exercise.BODY_WEIGHT_SQUATS,
                        (hip, knee, ankle), f"angle({hip.value}, {knee.value}, {ankle.value})"),
            _definition(f"squat_hip_angle_{side}_deg", f"Squat {side} projected hip angle", Exercise.BODY_WEIGHT_SQUATS,
                        (shoulder, hip, knee), f"angle({shoulder.value}, {hip.value}, {knee.value})"),
            _definition(f"hspu_elbow_angle_{side}_deg", f"HSPU {side} projected elbow angle", Exercise.HANDSTAND_PUSHUPS,
                        (shoulder, elbow, wrist), f"angle({shoulder.value}, {elbow.value}, {wrist.value})"),
            _definition(f"hspu_shoulder_angle_{side}_deg", f"HSPU {side} projected shoulder angle", Exercise.HANDSTAND_PUSHUPS,
                        (hip, shoulder, elbow), f"angle({hip.value}, {shoulder.value}, {elbow.value})", tier=MeasurementTier.M2),
            _definition(f"squat_knee_to_ankle_horizontal_{side}_ratio", f"Squat {side} projected knee-to-ankle offset candidate", Exercise.BODY_WEIGHT_SQUATS,
                        (knee, ankle), f"abs({knee.value}.x_processed - {ankle.value}.x_processed); equivalent to abs(delta_x_working)/valid_torso_scale, no second division",
                        method="horizontal_offset_torso_ratio", tier=MeasurementTier.M2, blocked=True),
        ))
    for feature_id, name, exercise in (
        ("squat_torso_angle_vertical_deg", "Squat projected torso axis to image vertical", Exercise.BODY_WEIGHT_SQUATS),
        ("hspu_body_line_angle_deg", "HSPU projected shoulder-hip axis to image vertical", Exercise.HANDSTAND_PUSHUPS),
    ):
        entries.append(_definition(feature_id, name, exercise,
            (J.LEFT_HIP, J.RIGHT_HIP, J.LEFT_SHOULDER, J.RIGHT_SHOULDER),
            "angle_between(midpoint(shoulders) - midpoint(hips), image_up=(0,-1))",
            method="midpoint_axis_angle_vertical", tier=MeasurementTier.M2))
    return tuple(sorted(entries, key=lambda f: f.feature_id))


DEFAULT_REGISTRY = FeatureRegistry(registry_version=FEATURE_REGISTRY_VERSION, features=_mvp())
