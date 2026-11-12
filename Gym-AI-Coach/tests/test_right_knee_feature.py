"""Stage 5.5 right-only calculation, two-ID schema and frozen left compatibility."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import geometry
from gym_coach.contracts import ErrorCategory, Exercise, JointProvenance, SourceTier, Stage, Viewpoint
from gym_coach.features import (
    DEFAULT_REGISTRY, FEATURE_ADAPTER_VERSION, KNEE_ANGLE_SCHEMA_VERSION,
    RIGHT_KNEE_ADAPTER_VERSION, SUPPORTED_KNEE_FEATURE_IDS, FeatureDefinition,
    KneeAngleFeatureObservation, NumericFeatureObservation, build_feature_input_context, observe_feature,
)
from gym_coach.features.models import content_hash
from gym_coach.flags import FlagCode, GymError, Severity, StructuredFlag
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import PreprocessingProfile, PreprocessingResult, preprocess_pose
from gym_coach.preprocessing.models import content_hash as preprocessing_hash
from gym_coach.quality import QualityReport, evaluate_quality

from test_feature_observation import ID as LEFT_ID, block_primitive, fixture_context
from test_preprocessing import make_frames, metadata, missing

ID = "squat_knee_angle_right_deg"
DEFINITION = DEFAULT_REGISTRY.get(ID)
ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_DEGREES = 1e-10


def right_result(*, ankle=(0.9, 0.4), knee=(0.6, 0.4), changes=None,
                 viewpoint=Viewpoint.SIDE, exercise=Exercise.BODY_WEIGHT_SQUATS,
                 profile=None, metric=True, metric_joints=None):
    coords = {(i, J.RIGHT_KNEE): ("observed", *knee) for i in range(31)}
    coords.update({(i, J.RIGHT_ANKLE): ("observed", *ankle) for i in range(31)})
    coords.update(changes or {})
    raw = tuple(f.model_copy(update={"viewpoint": viewpoint, "exercise": exercise})
                for f in make_frames(changes=coords))
    quality = evaluate_quality(raw,
        metric_requirements={ID: metric_joints or DEFINITION.required_joints} if metric else None,
        metric_viewpoints={ID: DEFINITION.viewpoint_policy.supported} if metric else None)
    return preprocess_pose(raw, quality, metadata(raw, width=320, height=320), profile=profile)


def replace_quality(source, update):
    """Fresh synthetic contract report/result; never modify the source snapshot."""
    data = source.quality.model_dump(mode="json")
    update(data)
    quality = QualityReport.model_validate_json(json.dumps(data), strict=True)
    return PreprocessingResult.model_validate_json(source.model_copy(update={
        "quality": quality, "quality_hash": preprocessing_hash(quality)}).model_dump_json(), strict=True)


def measure(source):
    return observe_feature(ID, build_feature_input_context(source, frame_index=5))


def assert_unavailable(observation, reason):
    assert observation.status == "unavailable" and observation.value is None
    assert reason in observation.reasons and any(f.code == reason for f in observation.flags)
    assert all(f.category != ErrorCategory.TECHNICAL_ERROR for f in observation.flags)
    assert KneeAngleFeatureObservation.model_validate_json(observation.model_dump_json(), strict=True) == observation


@pytest.mark.parametrize("ankle,expected", [((0.9, 0.4), 90.0), ((0.6, 0.1), 180.0),
    ((0.6, 0.55), 0.0), ((0.9, 0.7), 45.0)])
@pytest.mark.parametrize("index", [0, 5, 30])
def test_right_angle_factory_ready_allow_correct_order_primitive_and_identity(monkeypatch, ankle, expected, index):
    source = right_result(ankle=ankle)
    before = source.model_dump_json()
    context = build_feature_input_context(source, frame_index=index)
    calls = []
    original = geometry.joint_angle
    def record(*points):
        calls.append(points)
        return original(*points)
    monkeypatch.setattr(geometry, "joint_angle", record)
    measured = observe_feature(ID, context)
    assert type(measured) is KneeAngleFeatureObservation
    assert len(calls) == 1 and measured.value == original(*calls[0])
    assert measured.value == pytest.approx(expected, abs=TOLERANCE_DEGREES, rel=0)
    assert measured.status == "available" and measured.computation_status == "computed"
    assert measured.availability.status == "ready_for_calculation"
    assert context.quality.eligibility.action == context.quality.metric_eligibility[ID].action == "allow"
    assert measured.numeric_feature_schema_version == KNEE_ANGLE_SCHEMA_VERSION == "0.3.0"
    assert measured.adapter_version == RIGHT_KNEE_ADAPTER_VERSION == "squat-right-knee-v1"
    assert measured.primitive_name == "joint_angle" and measured.primitive_version == "joint-angle-2d-v1"
    assert tuple(j.joint for j in measured.joint_inputs) == (J.RIGHT_HIP, J.RIGHT_KNEE, J.RIGHT_ANKLE)
    assert calls[0] == tuple((j.x, j.y) for j in measured.joint_inputs)
    assert (measured.video_id, measured.frame_index, measured.timestamp_seconds) == (
        context.frame.video_id, index, context.frame.timestamp_seconds)
    assert measured.exercise == context.quality.exercise and measured.viewpoint == context.quality.viewpoint
    assert measured.unit == "degrees" and measured.quantity_kind == "angle"
    assert measured.measurement_tier == "M1" and measured.source_tiers == (SourceTier.S3,)
    assert measured.measurement_validation_status == "unvalidated"
    assert measured.assessment_status == "blocked_pending_expert_validation"
    assert measured.quality_report_hash == content_hash(context.quality)
    assert measured.quality_policy_hash == content_hash(DEFINITION.quality_policy)
    assert measured.availability.input_hash == context.measurement_input_hash
    assert not measured.reasons and not measured.flags
    assert source.model_dump_json() == before
    block_primitive(monkeypatch)
    assert KneeAngleFeatureObservation.model_validate_json(measured.model_dump_json(), strict=True) == measured


def test_canonical_definition_dispatch_and_left_schema_compatibility(monkeypatch):
    source = right_result()
    context = build_feature_input_context(source, frame_index=5)
    assert observe_feature(DEFINITION, context) == observe_feature(ID, context)
    old_context, _ = fixture_context()
    left = observe_feature(LEFT_ID, old_context)
    assert type(left) is NumericFeatureObservation
    assert left.numeric_feature_schema_version == "0.2.0" and left.adapter_version == FEATURE_ADAPTER_VERSION
    assert left.value == 90.0
    assert NumericFeatureObservation.model_validate_json(left.model_dump_json()) == left
    block_primitive(monkeypatch)
    upgraded = KneeAngleFeatureObservation.from_legacy(left)
    assert upgraded.numeric_feature_schema_version == "0.3.0" and upgraded.feature_id == LEFT_ID
    assert upgraded.adapter_version == "squat-left-knee-v1"
    assert {**upgraded.model_dump(), "numeric_feature_schema_version": "0.2.0"} == left.model_dump()
    assert KneeAngleFeatureObservation.model_validate_json(upgraded.model_dump_json(), strict=True) == upgraded
    with pytest.raises(ValidationError):
        KneeAngleFeatureObservation.model_validate_json(left.model_dump_json(), strict=True)
    with pytest.raises(ValidationError):
        NumericFeatureObservation.model_validate_json(upgraded.model_dump_json())
    with pytest.raises(TypeError):
        KneeAngleFeatureObservation.from_legacy(upgraded)


def test_legacy_left_available_golden_payload_is_unchanged():
    golden = ROOT/"outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/synthetic-available.observation.json"
    old = NumericFeatureObservation.model_validate_json(golden.read_text(encoding="utf-8"))
    measured = observe_feature(LEFT_ID, old.availability.input)
    assert measured == old and measured.model_dump_json() == old.model_dump_json()


@pytest.mark.parametrize("viewpoint,reason", [(Viewpoint.UNKNOWN, FlagCode.VIEWPOINT_UNKNOWN),
    (Viewpoint.FRONT, FlagCode.VIEWPOINT_MISMATCH), (Viewpoint.OBLIQUE, FlagCode.VIEWPOINT_MISMATCH)])
def test_viewpoint_prevents_primitive(monkeypatch, viewpoint, reason):
    source = right_result(viewpoint=viewpoint)
    block_primitive(monkeypatch)
    assert_unavailable(measure(source), reason)


def test_wrong_exercise_prevents_primitive(monkeypatch):
    source = right_result(exercise=Exercise.HANDSTAND_PUSHUPS)
    block_primitive(monkeypatch)
    assert_unavailable(measure(source), FlagCode.UNSUPPORTED_EXERCISE)


@pytest.mark.parametrize("scope", ["sequence", "metric"])
@pytest.mark.parametrize("action", ["skip", "degrade"])
def test_quality_decisions_preserved_without_recompute_or_promotion(monkeypatch, scope, action):
    source = right_result()
    key = "movement_analysis" if scope == "sequence" else ID
    def change(data):
        denied = {**data["metric_eligibility"][key], "eligible": False, "action": action, "reasons": ["synthetic-denied"]}
        data["metric_eligibility"][key] = denied
        if scope == "sequence":
            data.update(eligibility=denied, sequence_status="skipped" if action == "skip" else "degraded")
    fresh = replace_quality(source, change)
    before = source.model_dump_json(), fresh.model_dump_json()
    from gym_coach.quality import gate
    def forbidden(*args, **kwargs):
        pytest.fail("Adapter/factory called quality eligibility")
    monkeypatch.setattr(gate, "_eligibility", forbidden)
    monkeypatch.setattr(gate.QualityGate, "evaluate", forbidden)
    block_primitive(monkeypatch)
    measured = measure(fresh)
    assert_unavailable(measured, FlagCode.FEATURE_QUALITY_BLOCKED)
    assert measured.availability.input.quality == fresh.quality
    assert measured.availability.input.quality.metric_eligibility[key].action == action
    assert (source.model_dump_json(), fresh.model_dump_json()) == before


@pytest.mark.parametrize("cause,reason", [("absent", FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING),
    ("wrong_key", FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING), ("wrong_joints", FlagCode.FEATURE_QUALITY_POLICY_MISMATCH)])
def test_right_metric_is_exact_and_not_inferred(monkeypatch, cause, reason):
    source = right_result(metric=cause != "absent", metric_joints=(J.LEFT_HIP, J.LEFT_KNEE, J.LEFT_ANKLE) if cause == "wrong_joints" else None)
    if cause == "wrong_key":
        def change(data):
            data["metric_eligibility"][LEFT_ID] = data["metric_eligibility"].pop(ID)
        source = replace_quality(source, change)
    block_primitive(monkeypatch)
    assert_unavailable(measure(source), reason)


@pytest.mark.parametrize("state", ["missing", "low_visibility", "out_of_frame"])
def test_unusable_right_joint_prevents_primitive(monkeypatch, state):
    changes = missing([5], J.RIGHT_KNEE) if state == "missing" else {
        (5, J.RIGHT_KNEE): ("low_visibility", 0.6, 0.4) if state == "low_visibility" else ("observed", 1.2, 0.4)}
    source = right_result(changes=changes, profile=PreprocessingProfile(max_gap_seconds=0.0))
    block_primitive(monkeypatch)
    measured = measure(source)
    assert_unavailable(measured, FlagCode.FEATURE_JOINT_UNUSABLE)
    if state != "out_of_frame":
        assert FlagCode.FEATURE_PROVENANCE_REJECTED in measured.reasons
    assert measured.joint_inputs[1].x is measured.joint_inputs[1].y is None


def test_scale_fallback_does_not_introduce_angle_gate():
    changes = {(i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6, 0.7)
               for i in range(31) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)}
    source = right_result(changes=changes)
    measured = measure(source)
    assert source.frames[5].normalization.scale_status == "fallback_unit"
    assert source.frames[5].normalization.torso_scale is None
    assert not DEFINITION.quality_policy.requires_valid_scale and not DEFINITION.quality_policy.requires_mid_hip_origin
    assert measured.status == "available" and measured.value == 90.0
    assert FlagCode.NORMALIZATION_SCALE_FALLBACK not in measured.reasons
    assert measured.availability.input.frame.normalization == source.frames[5].normalization


def test_imputed_right_joint_keeps_endpoints_and_actual_quality_degrade(monkeypatch):
    source = right_result(changes=missing([5], J.RIGHT_KNEE))
    block_primitive(monkeypatch)
    measured = measure(source)
    assert_unavailable(measured, FlagCode.FEATURE_QUALITY_BLOCKED)
    knee = measured.joint_inputs[1]
    assert knee.status == JointProvenance.IMPUTED and knee.source_status == "missing"
    assert (knee.interpolation.left_frame_index, knee.interpolation.right_frame_index) == (4, 6)
    assert knee.interpolation.left_timestamp_seconds == source.raw_frames[4].timestamp_seconds
    assert knee.interpolation.right_timestamp_seconds == source.raw_frames[6].timestamp_seconds
    assert measured.frame_index == 5 and measured.timestamp_seconds == source.frames[5].timestamp_seconds
    assert FlagCode.FEATURE_PROVENANCE_REJECTED not in measured.reasons


def test_fresh_synthetic_imputed_allow_contract_is_explicit_and_does_not_promote_original():
    source = right_result(changes=missing([5], J.RIGHT_KNEE))
    before = source.model_dump_json()
    def synthetic_allow(data):
        # NEW caller-supplied contract fixture, NOT QualityGate evidence or ground truth.
        data["profile"]["profile_version"] = "synthetic-right-imputed-allow-contract-v1"
        from gym_coach.quality import QualityProfile
        data["profile_hash"] = QualityProfile.model_validate(data["profile"]).content_hash()
        for key in ("movement_analysis", ID):
            data["metric_eligibility"][key].update(eligible=True, action="allow", reasons=[])
        data.update(eligibility=data["metric_eligibility"]["movement_analysis"], sequence_status="eligible", flags=[])
    fresh = replace_quality(source, synthetic_allow)
    measured = measure(fresh)
    assert measured.status == "available" and measured.value == 90.0
    assert measured.joint_inputs[1].status == "imputed" and measured.joint_inputs[1].source_status == "missing"
    assert measured.joint_inputs[1].interpolation == next(j for j in source.frames[5].joints if j.joint == J.RIGHT_KNEE).interpolation
    assert source.model_dump_json() == before and source.quality.eligibility.action == "degrade"


@pytest.mark.parametrize("vertex", ["BA", "BC"])
def test_degenerate_right_vector_returns_null_with_actual_joint_names(vertex):
    source = right_result(knee=(0.6, 0.7)) if vertex == "BA" else right_result(ankle=(0.6, 0.4))
    measured = measure(source)
    assert_unavailable(measured, FlagCode.DEGENERATE_GEOMETRY)
    assert measured.computation_status == "failed_measurement"
    flag = measured.flags[0]
    assert flag.category == ErrorCategory.MEASUREMENT_ERROR and flag.stage == Stage.FEATURES and flag.source_tier == SourceTier.S3
    assert flag.location.video_id == source.metadata.video_id and flag.location.feature == ID
    assert flag.location.frame_start == flag.location.frame_end == 5
    assert flag.evidence == {"primitive":"joint_angle", "reason":"zero_length_vector", "vector":vertex,
        "input_joint_a":"right_hip", "input_joint_b":"right_knee", "input_joint_c":"right_ankle"}


def test_numerical_measurement_failure_remains_structured(monkeypatch):
    source = right_result()
    def fail(*points):
        raise GymError(StructuredFlag(code=FlagCode.GEOMETRY_NUMERICAL_ERROR, category=ErrorCategory.MEASUREMENT_ERROR,
            severity=Severity.ERROR, stage=Stage.FEATURES, source_tier=SourceTier.S3, message="Injected guard",
            evidence={"primitive":"joint_angle", "reason":"invalid_denominator"}))
    monkeypatch.setattr(geometry, "joint_angle", fail)
    measured = measure(source)
    assert_unavailable(measured, FlagCode.GEOMETRY_NUMERICAL_ERROR)
    assert measured.flags[0].evidence["input_joint_b"] == "right_knee"


@pytest.mark.parametrize("kind", ["input", "programming"])
def test_right_adapter_propagates_input_and_programming_errors(monkeypatch, kind):
    source = right_result()
    error = RuntimeError("programming failure") if kind == "programming" else GymError(StructuredFlag(
        code=FlagCode.INVALID_GEOMETRY_INPUT, category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
        stage=Stage.FEATURES, source_tier=SourceTier.S3, message="Injected input failure"))
    def fail(*points):
        raise error
    monkeypatch.setattr(geometry, "joint_angle", fail)
    with pytest.raises(type(error)) as caught:
        measure(source)
    assert caught.value is error


@pytest.mark.parametrize("feature", ["unknown", "", None, 42] + [f.feature_id for f in DEFAULT_REGISTRY.features
    if f.feature_id not in SUPPORTED_KNEE_FEATURE_IDS])
def test_dispatch_unsupported_ids_are_structured_and_do_not_call_primitive(monkeypatch, feature):
    source = right_result()
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(feature, build_feature_input_context(source, frame_index=5))
    expected = FlagCode.INVALID_FEATURE_INPUT if not isinstance(feature, str) else (
        FlagCode.UNKNOWN_FEATURE_ID if feature in ("unknown", "") else FlagCode.UNSUPPORTED_FEATURE_CALCULATION)
    assert caught.value.flag.code == expected
    assert caught.value.flag.category == ErrorCategory.INPUT_ERROR


@pytest.mark.parametrize("change", ["joint_order", "provenance", "scale_policy", "metric_key", "name"])
def test_noncanonical_right_definition_cannot_bypass_policy(monkeypatch, change):
    source = right_result()
    data = DEFINITION.model_dump(mode="json")
    if change == "joint_order":
        data["required_joints"].reverse()
    elif change == "provenance":
        data["joint_policy"]["allowed_provenance"] = ["observed"]
    elif change == "scale_policy":
        data["quality_policy"]["requires_valid_scale"] = True
        data["unavailable_reasons"].append("normalization_scale_fallback")
    elif change == "metric_key":
        data["quality_policy"]["metric_key"] = LEFT_ID
    else:
        data["name"] = "Changed"
    supplied = FeatureDefinition.model_validate_json(json.dumps(data))
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(supplied, build_feature_input_context(source, frame_index=5))
    assert caught.value.flag.code == FlagCode.INVALID_FEATURE_INPUT


@pytest.mark.parametrize("field,value", [
    ("value", None), ("value", float("nan")), ("value", float("inf")), ("value", float("-inf")),
    ("value", -1.0), ("value", 181.0), ("value", True), ("value", "90"),
    ("numeric_feature_schema_version", "0.2.0"), ("adapter_version", "squat-left-knee-v1"),
    ("primitive_name", "other"), ("primitive_version", "unknown"), ("measurement_method", "other"),
    ("feature_id", LEFT_ID), ("feature_id", "hspu_elbow_angle_right_deg"), ("feature_id", "unknown"),
    ("exercise", "BenchPress"), ("exercise", "HandstandPushups"), ("viewpoint", "unknown"),
    ("viewpoint", "diagonal"), ("video_id", "other.avi"), ("frame_index", 6), ("frame_index", True),
    ("timestamp_seconds", 0.5), ("timestamp_seconds", float("nan")), ("unit", "dimensionless_ratio"),
    ("quantity_kind", "distance"), ("measurement_tier", "M2"), ("source_tiers", ["S1"]),
    ("measurement_validation_status", "validated"), ("measurement_validation_status", "blocked"),
    ("assessment_status", "validated"), ("status", "unavailable"), ("computation_status", "not_computed"),
    ("quality_report_hash", "0"*64), ("quality_policy_hash", "0"*64), ("joint_inputs", []),
    ("reasons", ["missing_joint"]), ("score", 100)])
def test_new_schema_rejects_invalid_value_metadata_status_and_identity(field, value):
    data = measure(right_result()).model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        KneeAngleFeatureObservation.model_validate(data)


@pytest.mark.parametrize("cause", ["registry_hash", "registry_version", "definition_hash", "definition",
    "input_hash", "quality", "raw_hash", "joint_order", "joint_side", "joint_provenance", "lost_policy_flags"])
def test_new_schema_rejects_corrupt_audit_and_joint_order(cause):
    data = measure(right_result(viewpoint=Viewpoint.UNKNOWN if cause == "lost_policy_flags" else Viewpoint.SIDE)).model_dump(mode="json")
    availability = data["availability"]
    if cause in ("registry_hash", "definition_hash", "input_hash"):
        availability[cause] = "0"*64
    elif cause == "registry_version":
        availability[cause] = "unknown"
    elif cause == "definition":
        availability["definition"]["name"] = "Changed"
        availability["definition_hash"] = FeatureDefinition.model_validate(availability["definition"]).content_hash()
    elif cause == "quality":
        availability["input"]["quality"]["video_id"] = "other"
    elif cause == "raw_hash":
        availability["input"]["frame"]["raw_frame_hash"] = "0"*64
    elif cause == "joint_order":
        data["joint_inputs"].reverse()
    elif cause == "joint_side":
        data["joint_inputs"][1]["joint"] = "left_knee"
    elif cause == "joint_provenance":
        data["joint_inputs"][1]["source_status"] = "missing"
    else:
        availability["flags"] = []
    with pytest.raises(ValidationError):
        KneeAngleFeatureObservation.model_validate_json(json.dumps(data), strict=True)


@pytest.mark.parametrize("cause", ["value", "forged_available", "lost_reason", "lost_flag", "category", "location", "joint_name", "evidence"])
def test_measurement_unavailable_cannot_be_forged(cause):
    data = measure(right_result(ankle=(0.6, 0.4))).model_dump(mode="json")
    if cause == "value":
        data["value"] = 0.0
    elif cause == "forged_available":
        data.update(status="available", computation_status="computed", value=0.0, reasons=[], flags=[])
    elif cause == "lost_reason":
        data["reasons"] = []
    elif cause == "lost_flag":
        data["flags"] = []
    elif cause == "category":
        data["flags"][0]["category"] = "technical_error"
    elif cause == "location":
        data["flags"][0]["location"]["feature"] = LEFT_ID
    elif cause == "joint_name":
        data["flags"][0]["evidence"]["input_joint_b"] = "left_knee"
    else:
        data["flags"][0]["evidence"]["value"] = 0.0
    with pytest.raises(ValidationError):
        KneeAngleFeatureObservation.model_validate_json(json.dumps(data), strict=True)


def test_context_and_registry_immutability_and_detached_result():
    source = right_result()
    context = build_feature_input_context(source, frame_index=5)
    before = source.model_dump_json(), context.model_dump_json(), DEFAULT_REGISTRY.model_dump_json()
    measured = observe_feature(ID, context)
    assert (source.model_dump_json(), context.model_dump_json(), DEFAULT_REGISTRY.model_dump_json()) == before
    original = measured.model_dump_json()
    context.quality.metric_eligibility.clear()
    source.quality.by_joint.clear()
    assert measured.model_dump_json() == original
    assert KneeAngleFeatureObservation.model_validate_json(original, strict=True) == measured


def test_corrupt_linked_context_is_rejected_before_right_primitive(monkeypatch):
    context = build_feature_input_context(right_result(), frame_index=5)
    context = context.model_copy(update={"preprocessing_result_hash":"0"*64})
    block_primitive(monkeypatch)
    with pytest.raises(GymError) as caught:
        observe_feature(ID, context)
    assert caught.value.flag.code == FlagCode.FEATURE_CONTEXT_MISMATCH


def test_right_public_import_factory_execution_and_schema_are_lightweight(tmp_path):
    path = tmp_path/"source.json"
    path.write_text(right_result().model_dump_json(), encoding="utf-8")
    code = r'''
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from gym_coach.features import KneeAngleFeatureObservation, NumericFeatureObservation, build_feature_input_context, observe_feature
from gym_coach.preprocessing import PreprocessingResult
assert KneeAngleFeatureObservation.model_json_schema() and NumericFeatureObservation.model_json_schema()
source = PreprocessingResult.model_validate_json(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),strict=True)
context = build_feature_input_context(source,frame_index=5)
measured = observe_feature('squat_knee_angle_right_deg',context)
assert measured.value == 90.0 and measured.numeric_feature_schema_version == '0.3.0'
assert KneeAngleFeatureObservation.model_validate_json(measured.model_dump_json(),strict=True) == measured
assert not {'numpy','scipy','cv2','mediapipe','torch'}.intersection(sys.modules)
print('right-knee-import-execution-ok')
'''
    done = subprocess.run([sys.executable,"-I","-c",code,str(path)],cwd=tmp_path,capture_output=True,text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "right-knee-import-execution-ok"
