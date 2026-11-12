"""One Stage 5.3 feature: synthetic measurement, gating and strict audit tests."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import geometry
from gym_coach.contracts import ErrorCategory, Exercise, JointProvenance, SourceTier, Stage, Viewpoint
from gym_coach.features import (
    DEFAULT_REGISTRY, FeatureDefinition, FeatureInputContext, FeatureObservation,
    NumericFeatureObservation, observe_feature,
)
from gym_coach.features.models import content_hash
from gym_coach.flags import FlagCode, GymError, Severity, StructuredFlag
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import PreprocessingProfile, preprocess_pose
from gym_coach.quality import MetricEligibility, QualityReport, evaluate_quality

from test_preprocessing import make_frames, metadata, missing

ID = "squat_knee_angle_left_deg"
DEFINITION = DEFAULT_REGISTRY.get(ID)
ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_DEGREES = 1e-10


def fixture_context(*, ankle=(0.7, 0.4), changes=None, viewpoint=Viewpoint.SIDE,
                    exercise=Exercise.BODY_WEIGHT_SQUATS, profile=None):
    coords = {(i, J.LEFT_KNEE): ("observed", 0.4, 0.4) for i in range(31)}
    coords.update({(i, J.LEFT_ANKLE): ("observed", *ankle) for i in range(31)})
    coords.update(changes or {})
    raw = tuple(f.model_copy(update={"viewpoint": viewpoint, "exercise": exercise}) for f in make_frames(changes=coords))
    quality = evaluate_quality(raw, metric_requirements={ID: DEFINITION.required_joints},
                              metric_viewpoints={ID: DEFINITION.viewpoint_policy.supported})
    result = preprocess_pose(raw, quality, metadata(raw, width=320, height=320), profile=profile)
    return FeatureInputContext(frame=result.frames[5], quality=result.quality), result


def block_primitive(monkeypatch):
    def forbidden(*args):
        pytest.fail("Primitive called before prerequisites passed")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)


def assert_unavailable(observation, reason):
    assert observation.status == "unavailable" and observation.value is None
    assert reason in observation.reasons and any(f.code == reason for f in observation.flags)
    assert all(f.category != ErrorCategory.TECHNICAL_ERROR for f in observation.flags)
    assert NumericFeatureObservation.model_validate_json(observation.model_dump_json()) == observation


@pytest.mark.parametrize("ankle,expected", [((0.7, 0.4), 90.0), ((0.4, 0.1), 180.0),
                                           ((0.4, 0.55), 0.0), ((0.7, 0.7), 45.0)])
def test_real_registry_feature_has_correct_synthetic_measurement_and_identity(monkeypatch, ankle, expected):
    context, result = fixture_context(ankle=ankle)
    original = geometry.joint_angle
    calls = []
    def recording(*points):
        calls.append(points)
        return original(*points)
    monkeypatch.setattr(geometry, "joint_angle", recording)
    observation = observe_feature(ID, context)
    assert len(calls) == 1
    assert observation.value == pytest.approx(expected, abs=TOLERANCE_DEGREES, rel=0)
    assert observation.value == original(*calls[0])
    assert observation.status == "available" and observation.computation_status == "computed"
    assert observation.unit == "degrees" and observation.quantity_kind == "angle"
    assert observation.feature_id == ID and observation.measurement_method == "joint_angle_abc"
    assert observation.measurement_validation_status == "unvalidated"
    assert observation.assessment_status == "blocked_pending_expert_validation"
    assert observation.measurement_tier == "M1" and observation.source_tiers == (SourceTier.S3,)
    assert (observation.video_id, observation.frame_index, observation.timestamp_seconds) == (
        context.frame.video_id, context.frame.frame_index, context.frame.timestamp_seconds)
    assert observation.exercise == context.quality.exercise and observation.viewpoint == context.quality.viewpoint
    assert tuple(j.joint for j in observation.joint_inputs) == DEFINITION.required_joints
    assert observation.quality_report_hash == content_hash(context.quality)
    assert observation.availability.input.frame.raw_frame_hash == result.frames[5].raw_frame_hash
    assert NumericFeatureObservation.model_validate_json(observation.model_dump_json()) == observation


def test_definition_and_id_apis_agree_and_old_observation_stays_null_only():
    context, _ = fixture_context()
    measured = observe_feature(DEFINITION, context)
    assert measured == observe_feature(ID, context)
    assert NumericFeatureObservation.model_validate_json(measured.model_dump_json()).content_hash() == measured.content_hash()
    old = FeatureObservation.from_availability(measured.availability)
    assert old.feature_schema_version == "0.1.0" and old.value is None
    assert measured.numeric_feature_schema_version == "0.2.0"
    with pytest.raises(ValidationError):
        FeatureObservation.model_validate_json(measured.model_dump_json())
    with pytest.raises(ValidationError):
        FeatureObservation.model_validate({**old.model_dump(), "value": measured.value})


@pytest.mark.parametrize("viewpoint,reason", [(Viewpoint.UNKNOWN, FlagCode.VIEWPOINT_UNKNOWN),
    (Viewpoint.FRONT, FlagCode.VIEWPOINT_MISMATCH), (Viewpoint.OBLIQUE, FlagCode.VIEWPOINT_MISMATCH)])
def test_viewpoint_policy_stops_primitive(monkeypatch, viewpoint, reason):
    context, _ = fixture_context(viewpoint=viewpoint)
    block_primitive(monkeypatch)
    assert_unavailable(observe_feature(ID, context), reason)


def test_wrong_mvp_exercise_is_unavailable_without_primitive(monkeypatch):
    context, _ = fixture_context(exercise=Exercise.HANDSTAND_PUSHUPS)
    block_primitive(monkeypatch)
    assert_unavailable(observe_feature(ID, context), FlagCode.UNSUPPORTED_EXERCISE)


@pytest.mark.parametrize("scope", ["sequence", "metric"])
@pytest.mark.parametrize("action", ["degrade", "skip"])
def test_quality_actions_never_promoted(monkeypatch, scope, action):
    context, _ = fixture_context()
    before = context.model_dump_json()
    metrics = dict(context.quality.metric_eligibility)
    key = "movement_analysis" if scope == "sequence" else ID
    denied = MetricEligibility(eligible=False, action=action, required_joints=metrics[key].required_joints,
                               reasons=("synthetic-quality-failure",))
    metrics[key] = denied
    updates = {"metric_eligibility": metrics}
    if scope == "sequence":
        updates.update(eligibility=denied, sequence_status="degraded" if action == "degrade" else "skipped")
    quality = QualityReport.model_validate({**context.quality.model_dump(), **updates})
    changed = FeatureInputContext(frame=context.frame, quality=quality)
    block_primitive(monkeypatch)
    observation = observe_feature(ID, changed)
    assert_unavailable(observation, FlagCode.FEATURE_QUALITY_BLOCKED)
    assert observation.availability.input.quality == quality
    assert quality.metric_eligibility[key].action == action
    assert context.model_dump_json() == before


@pytest.mark.parametrize("cause,reason", [("missing_key", FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING),
    ("wrong_key", FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING), ("wrong_joints", FlagCode.FEATURE_QUALITY_POLICY_MISMATCH)])
def test_metric_key_and_joint_requirements_are_exact(monkeypatch, cause, reason):
    context, _ = fixture_context()
    metrics = dict(context.quality.metric_eligibility)
    eligible = metrics.pop(ID)
    if cause == "wrong_key":
        metrics["squat_knee_angle_right_deg"] = eligible
    elif cause == "wrong_joints":
        metrics[ID] = MetricEligibility(eligible=True, action="allow", required_joints=(J.NOSE,))
    quality = QualityReport.model_validate({**context.quality.model_dump(), "metric_eligibility": metrics})
    block_primitive(monkeypatch)
    assert_unavailable(observe_feature(ID, FeatureInputContext(frame=context.frame, quality=quality)), reason)


@pytest.mark.parametrize("state", ["missing", "low_visibility", "out_of_frame"])
def test_unusable_or_disallowed_joint_provenance_stops_primitive(monkeypatch, state):
    changes = missing([5], J.LEFT_KNEE) if state == "missing" else {
        (5, J.LEFT_KNEE): ("low_visibility", 0.4, 0.4) if state == "low_visibility" else ("observed", 1.2, 0.4)}
    context, _ = fixture_context(changes=changes, profile=PreprocessingProfile(max_gap_seconds=0.0))
    block_primitive(monkeypatch)
    observation = observe_feature(ID, context)
    assert_unavailable(observation, FlagCode.FEATURE_JOINT_UNUSABLE)
    if state != "out_of_frame":
        assert FlagCode.FEATURE_PROVENANCE_REJECTED in observation.reasons


def test_scale_fallback_does_not_add_an_angle_gate():
    changes = {(i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6, 0.7)
               for i in range(31) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)}
    context, _ = fixture_context(changes=changes)
    assert context.frame.normalization.scale_status == "fallback_unit"
    assert context.frame.normalization.torso_scale is None
    assert DEFINITION.quality_policy.requires_valid_scale is False
    observation = observe_feature(ID, context)
    assert observation.status == "available" and observation.value == 90.0
    assert FlagCode.NORMALIZATION_SCALE_FALLBACK not in observation.reasons
    assert observation.availability.input.frame.normalization == context.frame.normalization


def test_imputed_input_keeps_provenance_and_real_quality_degrade(monkeypatch):
    context, result = fixture_context(changes=missing([5], J.LEFT_KNEE))
    before = result.model_dump_json()
    block_primitive(monkeypatch)
    observation = observe_feature(ID, context)
    assert_unavailable(observation, FlagCode.FEATURE_QUALITY_BLOCKED)
    joint = observation.joint_inputs[1]
    assert joint.status == JointProvenance.IMPUTED and joint.source_status == "missing"
    assert joint.interpolation.left_frame_index == 4 and joint.interpolation.right_frame_index == 6
    assert FlagCode.FEATURE_PROVENANCE_REJECTED not in observation.reasons
    assert result.model_dump_json() == before


def test_available_imputed_contract_fixture_preserves_references():
    context, result = fixture_context(changes=missing([5], J.LEFT_KNEE))
    before = result.model_dump_json()
    # A NEW, explicitly synthetic caller-supplied contract fixture, NOT output
    # of QualityGate. Never change the original report or any real report.
    data = context.quality.model_dump(mode="json")
    allow = {"eligible": True, "action": "allow", "required_joints": [j.value for j in DEFINITION.required_joints], "reasons": []}
    sequence_allow = {**allow, "required_joints": [j.value for j in context.quality.eligibility.required_joints]}
    data.update(sequence_status="eligible", eligibility=sequence_allow,
                metric_eligibility={"movement_analysis": sequence_allow, ID: allow}, flags=[])
    data["profile"]["profile_version"] = "synthetic-imputed-allow-contract-v1"
    from gym_coach.quality import QualityProfile
    data["profile_hash"] = QualityProfile.model_validate(data["profile"]).content_hash()
    synthetic_report = QualityReport.model_validate(data)
    synthetic = FeatureInputContext(frame=context.frame, quality=synthetic_report)
    observation = observe_feature(ID, synthetic)
    assert observation.status == "available"
    assert observation.joint_inputs[1].status == "imputed" and observation.joint_inputs[1].source_status == "missing"
    assert observation.joint_inputs[1].interpolation == context.frame.joints[[j.joint for j in context.frame.joints].index(J.LEFT_KNEE)].interpolation
    assert observation.value == pytest.approx(90.0, abs=TOLERANCE_DEGREES, rel=0)
    assert NumericFeatureObservation.model_validate_json(observation.model_dump_json()) == observation
    assert result.model_dump_json() == before and result.quality.eligibility.action == "degrade"


@pytest.mark.parametrize("vertex", ["BA", "BC"])
def test_degenerate_geometry_returns_unavailable_with_real_identity(vertex):
    if vertex == "BA":
        changes = {(i, J.LEFT_KNEE): ("observed", 0.4, 0.7) for i in range(31)}
        context, _ = fixture_context(changes=changes)
    else:
        context, _ = fixture_context(ankle=(0.4, 0.4))
    observation = observe_feature(ID, context)
    assert_unavailable(observation, FlagCode.DEGENERATE_GEOMETRY)
    assert observation.computation_status == "failed_measurement"
    flag = observation.flags[0]
    assert flag.category == ErrorCategory.MEASUREMENT_ERROR and flag.stage == Stage.FEATURES and flag.source_tier == SourceTier.S3
    assert flag.location.video_id == context.frame.video_id and flag.location.feature == ID
    assert flag.location.frame_start == flag.location.frame_end == context.frame.frame_index
    assert flag.evidence["vector"] == vertex and flag.evidence["input_joint_b"] == "left_knee"
    assert "value" not in flag.evidence


def test_numeric_measurement_failure_is_auditable_without_fake_value(monkeypatch):
    context, _ = fixture_context()
    def fail(*points):
        raise GymError(StructuredFlag(code=FlagCode.GEOMETRY_NUMERICAL_ERROR,
            category=ErrorCategory.MEASUREMENT_ERROR, severity=Severity.ERROR, stage=Stage.FEATURES,
            source_tier=SourceTier.S3, message="Injected numerical guard", evidence={"primitive": "joint_angle", "reason": "invalid_denominator"}))
    monkeypatch.setattr(geometry, "joint_angle", fail)
    assert_unavailable(observe_feature(ID, context), FlagCode.GEOMETRY_NUMERICAL_ERROR)


@pytest.mark.parametrize("kind", ["input_error", "programming_error"])
def test_adapter_does_not_hide_primitive_input_or_programming_error(monkeypatch, kind):
    context, _ = fixture_context()
    error = ValueError("programming failure") if kind == "programming_error" else GymError(StructuredFlag(
        code=FlagCode.INVALID_GEOMETRY_INPUT, category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
        stage=Stage.FEATURES, source_tier=SourceTier.S3, message="Injected input failure"))
    def fail(*points):
        raise error
    monkeypatch.setattr(geometry, "joint_angle", fail)
    with pytest.raises(type(error)) as caught:
        observe_feature(ID, context)
    assert caught.value is error


# Stage 5.5 implements right knee; all remaining calculators stay unsupported.
@pytest.mark.parametrize("feature", ["unknown", "", None, 42] + [f.feature_id for f in DEFAULT_REGISTRY.features
    if f.feature_id not in (ID, "squat_knee_angle_right_deg")])
def test_registry_features_without_calculators_cannot_call_primitive(monkeypatch, feature):
    context, _ = fixture_context()
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(feature, context)
    expected = FlagCode.INVALID_FEATURE_INPUT if not isinstance(feature, str) else (
        FlagCode.UNKNOWN_FEATURE_ID if feature in ("unknown", "") else FlagCode.UNSUPPORTED_FEATURE_CALCULATION)
    assert caught.value.flag.code == expected


@pytest.mark.parametrize("change", ["feature_id", "joint_order", "unit", "provenance", "scale_policy"])
def test_noncanonical_definition_is_rejected_before_primitive(monkeypatch, change):
    context, _ = fixture_context()
    data = DEFINITION.model_dump(mode="json")
    if change == "feature_id":
        data["feature_id"] = "squat_knee_angle_right_deg"
    elif change == "joint_order":
        data["required_joints"] = list(reversed(data["required_joints"]))
    elif change == "unit":
        data["unit"] = "dimensionless_ratio"
    elif change == "provenance":
        data["joint_policy"]["allowed_provenance"] = ["observed"]
    else:
        data["quality_policy"]["requires_valid_scale"] = True
        data["unavailable_reasons"].append("normalization_scale_fallback")
    supplied = DEFINITION.model_copy(update={"unit": "bad"}) if change == "unit" else FeatureDefinition.model_validate(data)
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(supplied, context)
    assert caught.value.flag.code == FlagCode.INVALID_FEATURE_INPUT


@pytest.mark.parametrize("cause", ["context_type", "video", "index", "timestamp", "joint_null", "joint_nan", "joint_inf", "joint_missing", "report_identity", "profile_hash"])
def test_unsafe_context_is_rejected_before_measurement(monkeypatch, cause):
    context, _ = fixture_context()
    frame, report = context.frame, context.quality
    if cause == "context_type":
        context = None
    else:
        if cause == "video":
            frame = frame.model_copy(update={"video_id": "other.avi"})
        elif cause == "index":
            frame = frame.model_copy(update={"frame_index": report.total_frames})
        elif cause == "timestamp":
            frame = frame.model_copy(update={"timestamp_seconds": float("nan")})
        elif cause.startswith("joint_"):
            if cause == "joint_missing":
                frame = frame.model_copy(update={"joints": tuple(j for j in frame.joints if j.joint != J.LEFT_KNEE)})
            else:
                value = {"joint_null": None, "joint_nan": float("nan"), "joint_inf": float("inf")}[cause]
                frame = frame.model_copy(update={"joints": tuple(j.model_copy(update={"x": value}) if j.joint == J.LEFT_KNEE else j for j in frame.joints)})
        elif cause == "report_identity":
            report = report.model_copy(update={"video_id": "other.avi"})
        else:
            report = report.model_copy(update={"profile_hash": "0" * 64})
        context = context.model_copy(update={"frame": frame, "quality": report})
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(ID, context)
    assert caught.value.flag.code == FlagCode.INVALID_FEATURE_INPUT


@pytest.mark.parametrize("field,value,code", [("exercise", "BenchPress", FlagCode.UNSUPPORTED_EXERCISE),
    ("viewpoint", "diagonal", FlagCode.INVALID_VIEWPOINT)])
def test_unknown_input_enums_are_explicit(monkeypatch, field, value, code):
    context, _ = fixture_context()
    context = context.model_copy(update={"quality": context.quality.model_copy(update={field: value})})
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(ID, context)
    assert caught.value.flag.code == code


def test_imputation_references_must_bracket_frame_identity(monkeypatch):
    context, _ = fixture_context(changes=missing([5], J.LEFT_KNEE))
    context = context.model_copy(update={"frame": context.frame.model_copy(update={"timestamp_seconds": 1.0})})
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(ID, context)
    assert caught.value.flag.code == FlagCode.INVALID_FEATURE_INPUT


@pytest.mark.parametrize("field,value", [
    ("value", None), ("value", float("nan")), ("value", float("inf")), ("value", float("-inf")),
    ("value", -1.0), ("value", 181.0), ("value", True), ("value", "90"),
    ("numeric_feature_schema_version", "0.1.0"), ("adapter_version", "unknown"),
    ("primitive_version", "unknown"), ("primitive_name", "other"), ("measurement_method", "other"),
    ("feature_id", "squat_knee_angle_right_deg"), ("exercise", "BenchPress"), ("exercise", "HandstandPushups"),
    ("viewpoint", "unknown"), ("viewpoint", "diagonal"), ("video_id", "other.avi"),
    ("frame_index", 6), ("frame_index", True), ("frame_index", -1),
    ("timestamp_seconds", 0.5), ("timestamp_seconds", float("nan")), ("timestamp_seconds", -1),
    ("unit", "dimensionless_ratio"), ("quantity_kind", "distance"), ("measurement_tier", "M2"),
    ("source_tiers", ["S1"]), ("measurement_validation_status", "validated"), ("measurement_validation_status", "blocked"),
    ("assessment_status", "validated"), ("computation_status", "not_computed"), ("status", "unavailable"),
    ("quality_report_hash", "0" * 64), ("quality_policy_hash", "0" * 64), ("joint_inputs", []),
    ("reasons", ["missing_joint"]), ("score", 100),
])
def test_numeric_schema_rejects_invalid_values_metadata_and_identity(field, value):
    context, _ = fixture_context()
    data = observe_feature(ID, context).model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        NumericFeatureObservation.model_validate(data)


@pytest.mark.parametrize("cause", ["registry_hash", "registry_version", "definition_hash", "definition", "input_hash", "quality", "joint_provenance", "joint_order", "raw_hash", "lost_policy_flags"])
def test_audit_hashes_and_snapshots_are_strict(cause):
    context, _ = fixture_context(viewpoint=Viewpoint.UNKNOWN if cause == "lost_policy_flags" else Viewpoint.SIDE)
    data = observe_feature(ID, context).model_dump(mode="json")
    availability = data["availability"]
    if cause in ("registry_hash", "definition_hash", "input_hash"):
        availability[cause] = "0" * 64
    elif cause == "registry_version":
        availability[cause] = "other"
    elif cause == "definition":
        availability["definition"]["name"] = "Changed definition"
        availability["definition_hash"] = FeatureDefinition.model_validate(availability["definition"]).content_hash()
    elif cause == "quality":
        availability["input"]["quality"]["video_id"] = "other.avi"
    elif cause == "joint_provenance":
        data["joint_inputs"][1]["source_status"] = "missing"
    elif cause == "joint_order":
        data["joint_inputs"].reverse()
    elif cause == "raw_hash":
        availability["input"]["frame"]["raw_frame_hash"] = "0" * 64
    else:
        availability["flags"] = []
    with pytest.raises(ValidationError):
        NumericFeatureObservation.model_validate(data)


@pytest.mark.parametrize("cause", ["value", "status", "reason", "flag", "category", "location", "evidence"])
def test_unavailable_cannot_be_forged_as_measurement(cause):
    context, _ = fixture_context(ankle=(0.4, 0.4))
    data = observe_feature(ID, context).model_dump(mode="json")
    if cause == "value":
        data["value"] = 0.0
    elif cause == "status":
        data.update(status="available", computation_status="computed", value=0.0, reasons=[], flags=[])
    elif cause == "reason":
        data["reasons"] = []
    elif cause == "flag":
        data["flags"] = []
    elif cause == "category":
        data["flags"][0]["category"] = "technical_error"
    elif cause == "location":
        data["flags"][0]["location"]["frame_start"] = 4
    else:
        data["flags"][0]["evidence"]["value"] = 0.0
    with pytest.raises(ValidationError):
        NumericFeatureObservation.model_validate(data)


def test_inputs_and_registry_are_detached_and_unchanged():
    context, result = fixture_context()
    before = (context.model_dump_json(), result.model_dump_json(), DEFAULT_REGISTRY.model_dump_json())
    observation = observe_feature(ID, context)
    assert (context.model_dump_json(), result.model_dump_json(), DEFAULT_REGISTRY.model_dump_json()) == before
    assert observation.availability.input.quality.by_joint is not context.quality.by_joint
    original = observation.model_dump_json()
    context.quality.metric_eligibility.clear()
    assert observation.model_dump_json() == original


def test_numeric_public_import_schema_and_serialization_have_no_heavy_backends(tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'cv2', 'mediapipe', 'torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from gym_coach.features import NumericFeatureObservation, observe_feature, DEFAULT_REGISTRY, FeatureObservation
from gym_coach.geometry import JOINT_ANGLE_PRIMITIVE_VERSION
assert NumericFeatureObservation.model_json_schema() and FeatureObservation.model_json_schema()
assert JOINT_ANGLE_PRIMITIVE_VERSION == 'joint-angle-2d-v1'
assert DEFAULT_REGISTRY.get('squat_knee_angle_left_deg')
print('numeric-feature-import-ok')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "numeric-feature-import-ok"
