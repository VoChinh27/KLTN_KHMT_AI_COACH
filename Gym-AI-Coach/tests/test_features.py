"""Stage 5.1 schema/registry/prerequisite checks, never geometry calculations."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.contracts import ErrorCategory, Exercise, JointProvenance, Stage, Viewpoint
from gym_coach.features import (
    DEFAULT_REGISTRY, FeatureAvailability, FeatureDefinition, FeatureObservation,
    FeatureRegistry, check_feature_availability,
)
from gym_coach.flags import FlagCode, GymError
from gym_coach.preprocessing import PreprocessingProfile, preprocess_pose
from gym_coach.quality import evaluate_quality

from test_preprocessing import make_frames, metadata, missing
from gym_coach.pose.mapping import JointName as J

ROOT = Path(__file__).resolve().parents[1]
ID = "squat_knee_angle_left_deg"


def context(*, changes=None, viewpoint=Viewpoint.SIDE, exercise=Exercise.BODY_WEIGHT_SQUATS,
            registry=DEFAULT_REGISTRY, include_metrics=True, profile=None):
    raw = tuple(f.model_copy(update={"viewpoint": viewpoint, "exercise": exercise})
                for f in make_frames(changes=changes))
    quality = evaluate_quality(raw,
        metric_requirements=registry.quality_requirements(exercise) if include_metrics else None,
        metric_viewpoints=registry.quality_viewpoints(exercise) if include_metrics else None)
    return preprocess_pose(raw, quality, metadata(raw), profile=profile)


def available(result, frame_index=5, feature_id=ID, registry=DEFAULT_REGISTRY):
    return check_feature_availability(feature_id, result.frames[frame_index], result.quality, registry=registry)


def changed_registry(feature_id, **updates):
    data = DEFAULT_REGISTRY.get(feature_id).model_dump(mode="json")
    data.update(updates)
    definition = FeatureDefinition.model_validate(data)
    return FeatureRegistry(registry_version="test-only-v1", features=(definition,))


def ratio_registry(*, distance=False):
    feature_id = "squat_knee_to_ankle_horizontal_left_ratio"
    definition = DEFAULT_REGISTRY.get(feature_id)
    updates = {"validation": {**definition.validation.model_dump(), "measurement_status": "unvalidated"}}
    if distance:
        updates.update(quantity_kind="distance", unit="torso_lengths", measurement_method="normalized_euclidean_distance",
                       formula="norm(A_working-B_working)/valid_torso_scale")
    return feature_id, changed_registry(feature_id, **updates)


def assert_unavailable(decision, reason):
    assert decision.status == "unavailable"
    assert reason in decision.reasons
    assert decision.flags and any(f.code == reason for f in decision.flags)
    assert all(f.category != ErrorCategory.TECHNICAL_ERROR and f.stage == Stage.FEATURES for f in decision.flags)
    observation = FeatureObservation.from_availability(decision)
    assert observation.value is None and observation.status == "unavailable"


def test_registry_stable_ids_joint_order_and_metadata():
    expected = {f"{exercise}_{part}_angle_{side}_deg"
                for exercise, parts in (("squat", ("knee", "hip")), ("hspu", ("elbow", "shoulder")))
                for part in parts for side in ("left", "right")}
    expected |= {"squat_torso_angle_vertical_deg", "hspu_body_line_angle_deg",
                 "squat_knee_to_ankle_horizontal_left_ratio", "squat_knee_to_ankle_horizontal_right_ratio"}
    assert {f.feature_id for f in DEFAULT_REGISTRY.features} == expected
    assert len(DEFAULT_REGISTRY.features) == 12
    assert DEFAULT_REGISTRY.get(ID).required_joints == (J.LEFT_HIP, J.LEFT_KNEE, J.LEFT_ANKLE)
    assert DEFAULT_REGISTRY.get("hspu_elbow_angle_right_deg").required_joints == (J.RIGHT_SHOULDER, J.RIGHT_ELBOW, J.RIGHT_WRIST)
    assert len(DEFAULT_REGISTRY.for_exercise("BodyWeightSquats")) == 5
    assert len(DEFAULT_REGISTRY.for_exercise("HandstandPushups")) == 5
    assert len(DEFAULT_REGISTRY.for_exercise("BodyWeightSquats", include_blocked=True)) == 7
    for f in DEFAULT_REGISTRY.features:
        assert f.validation.calculation_status == "not_implemented"
        assert f.validation.assessment_status == "blocked_pending_expert_validation"
        assert f.viewpoint_policy.supported == (Viewpoint.SIDE,)
        assert f.quality_policy.metric_key == f.feature_id
        assert f.source_references and f.source_tiers and f.roles


@pytest.mark.parametrize("field", ["feature_id", "name", "exercises", "formula", "measurement_definition", "measurement_method",
    "required_joints", "unit", "quantity_kind", "measurement_tier", "source_tiers", "source_references", "validation",
    "viewpoint_policy", "quality_policy", "joint_policy", "roles", "unavailable_reasons", "limitations"])
def test_missing_required_metadata_is_rejected(field):
    data = DEFAULT_REGISTRY.get(ID).model_dump(mode="json")
    del data[field]
    with pytest.raises(ValidationError):
        FeatureDefinition.model_validate(data)


@pytest.mark.parametrize("updates", [
    {"required_joints": []}, {"required_joints": ["left_hip", "left_knee"]},
    {"required_joints": ["left_hip", "left_knee", "left_knee"]},
    {"unit": "centimeters"}, {"unit": "dimensionless_ratio"}, {"measurement_tier": "M99"},
    {"measurement_tier": "M0"}, {"exercises": ["BenchPress"]}, {"source_tiers": []},
    {"source_tiers": ["invented"]}, {"unavailable_reasons": ["missing_joint"]},
    {"formula": " "}, {"roles": ["form_assessment"]}, {"feature_schema_version": "9.0.0"},
    {"score_threshold": 90},
])
def test_invalid_or_unsafe_metadata_is_rejected(updates):
    data = DEFAULT_REGISTRY.get(ID).model_dump(mode="json")
    with pytest.raises(ValidationError):
        FeatureDefinition.model_validate({**data, **updates})


def test_registry_duplicates_and_metric_key_collision_are_rejected():
    f = DEFAULT_REGISTRY.get(ID)
    with pytest.raises(ValidationError):
        FeatureRegistry(registry_version="bad", features=(f, f))
    with pytest.raises(ValidationError):
        changed_registry(ID, quality_policy={**f.quality_policy.model_dump(), "metric_key": "wrong"})


@pytest.mark.parametrize("bad", ["low_visibility", "missing"])
def test_joint_policy_cannot_accept_unmeasured_provenance(bad):
    data = DEFAULT_REGISTRY.get(ID).model_dump(mode="json")
    data["joint_policy"]["allowed_provenance"] = ["observed", bad]
    with pytest.raises(ValidationError):
        FeatureDefinition.model_validate(data)


@pytest.mark.parametrize("kind", ["unknown_allowed", "missing_policy", "missing_reason", "duplicate", "empty"])
def test_viewpoint_policy_must_be_exhaustive_and_explained(kind):
    data = DEFAULT_REGISTRY.get(ID).model_dump(mode="json")
    policy = data["viewpoint_policy"]
    if kind == "unknown_allowed":
        policy["supported"] = ["side", "unknown"]
        policy["unsupported"] = policy["unsupported"][1:]
    elif kind == "missing_policy":
        del data["viewpoint_policy"]
    elif kind == "missing_reason":
        del policy["unsupported"][0]["reason"]
    elif kind == "duplicate":
        policy["supported"] = ["side", "side"]
    else:
        policy["supported"] = []
    with pytest.raises(ValidationError):
        FeatureDefinition.model_validate(data)


@pytest.mark.parametrize("quantity", ["ratio", "distance"])
def test_distance_ratio_contract_requires_scale_and_origin_policy(quantity):
    feature_id, registry = ratio_registry(distance=quantity == "distance")
    data = registry.get(feature_id).model_dump(mode="json")
    for field in ("requires_valid_scale", "requires_mid_hip_origin"):
        altered = json.loads(json.dumps(data))
        altered["quality_policy"][field] = False
        with pytest.raises(ValidationError):
            FeatureDefinition.model_validate(altered)


def test_ready_means_not_computed_and_inputs_are_unchanged():
    result = context()
    before = result.model_dump_json()
    decision = available(result)
    assert decision.status == "ready_for_calculation" and not decision.reasons and not decision.flags
    observation = FeatureObservation.from_availability(decision)
    assert observation.status == "not_computed" and observation.value is None
    assert result.model_dump_json() == before
    assert decision.input.quality == result.quality
    assert decision.input.quality.by_joint is not result.quality.by_joint
    assert decision.input.frame.raw_frame_hash == result.frames[5].raw_frame_hash


@pytest.mark.parametrize("status", ["missing", "low_visibility", "out_of_frame"])
def test_bad_joint_blocks_feature(status):
    changes = missing([5], J.LEFT_KNEE) if status == "missing" else {
        (5, J.LEFT_KNEE): ("low_visibility", 0.5, 0.4) if status == "low_visibility" else ("observed", 1.2, 0.4)}
    result = context(changes=changes, profile=PreprocessingProfile(max_gap_seconds=0.0))
    decision = available(result)
    assert_unavailable(decision, FlagCode.FEATURE_JOINT_UNUSABLE)
    if status == "missing":
        assert FlagCode.MISSING_JOINT in decision.reasons
    if status != "out_of_frame":
        assert FlagCode.FEATURE_PROVENANCE_REJECTED in decision.reasons
    assert result.frames[5].joints == decision.input.frame.joints


def test_imputed_source_is_preserved_and_observed_only_policy_rejects_it():
    result = context(changes=missing([5], J.LEFT_KNEE))
    # Stage 3 remains degrade despite successful imputation; no quality promotion.
    decision = available(result)
    point = next(j for j in decision.input.frame.joints if j.joint == J.LEFT_KNEE)
    assert point.status == "imputed" and point.source_status == "missing" and point.interpolation
    assert_unavailable(decision, FlagCode.FEATURE_QUALITY_BLOCKED)
    assert FlagCode.FEATURE_PROVENANCE_REJECTED not in decision.reasons
    definition = DEFAULT_REGISTRY.get(ID)
    registry = changed_registry(ID, joint_policy={**definition.joint_policy.model_dump(), "allowed_provenance": ["observed"]})
    assert_unavailable(available(result, registry=registry), FlagCode.FEATURE_PROVENANCE_REJECTED)


@pytest.mark.parametrize("viewpoint,reason", [(Viewpoint.UNKNOWN, FlagCode.VIEWPOINT_UNKNOWN),
    (Viewpoint.FRONT, FlagCode.VIEWPOINT_MISMATCH), (Viewpoint.OBLIQUE, FlagCode.VIEWPOINT_MISMATCH)])
def test_unsupported_viewpoint_is_explicit(viewpoint, reason):
    assert_unavailable(available(context(viewpoint=viewpoint)), reason)


def test_wrong_mvp_exercise_blocks_feature():
    assert_unavailable(available(context(exercise=Exercise.HANDSTAND_PUSHUPS)), FlagCode.UNSUPPORTED_EXERCISE)


def test_missing_metric_eligibility_is_never_inferred_from_global_allow():
    result = context(include_metrics=False)
    assert result.quality.eligibility.action == "allow"
    assert_unavailable(available(result), FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING)


def test_allow_for_different_joints_cannot_unlock_feature():
    result = context()
    raw = result.raw_frames
    wrong = evaluate_quality(raw, metric_requirements={ID: (J.NOSE,)})
    assert wrong.metric_eligibility[ID].action == "allow"
    assert_unavailable(check_feature_availability(ID, result.frames[5], wrong), FlagCode.FEATURE_QUALITY_POLICY_MISMATCH)


@pytest.mark.parametrize("action", ["degrade", "skip"])
def test_metric_quality_fail_blocks_even_if_sequence_is_allow(action):
    from gym_coach.quality import MetricEligibility
    result = context()
    metrics = dict(result.quality.metric_eligibility)
    metrics[ID] = MetricEligibility(eligible=False, action=action, required_joints=DEFAULT_REGISTRY.get(ID).required_joints,
                                    reasons=("synthetic-quality-failure",))
    quality = result.quality.model_copy(update={"metric_eligibility": metrics})
    assert_unavailable(check_feature_availability(ID, result.frames[5], quality), FlagCode.FEATURE_QUALITY_BLOCKED)
    assert quality.metric_eligibility[ID].action == action


def test_global_fail_cannot_be_overridden_by_clean_metric():
    result = context(changes=missing(range(31), J.RIGHT_KNEE))
    assert result.quality.eligibility.action == "skip"
    assert result.quality.metric_eligibility[ID].action == "allow"
    assert_unavailable(available(result), FlagCode.FEATURE_QUALITY_BLOCKED)


@pytest.mark.parametrize("distance", [False, True])
@pytest.mark.parametrize("cause", ["missing_origin", "missing_scale", "zero_scale"])
def test_distance_ratio_unavailable_on_normalization_fallback(distance, cause):
    feature_id, registry = ratio_registry(distance=distance)
    changes = missing(range(31), J.RIGHT_HIP if cause == "missing_origin" else J.RIGHT_SHOULDER)
    if cause == "zero_scale":
        changes = {(i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6, 0.7)
                   for i in range(31) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)}
    result = context(changes=changes, registry=registry)
    decision = available(result, feature_id=feature_id, registry=registry)
    assert_unavailable(decision, FlagCode.NORMALIZATION_SCALE_FALLBACK)
    assert decision.input.frame.normalization.torso_scale is None
    if cause == "missing_origin":
        assert FlagCode.NORMALIZATION_ORIGIN_FALLBACK in decision.reasons
        assert decision.input.frame.normalization.origin_x is None


def test_angles_can_be_ready_with_fallback_scale_when_quality_and_inputs_pass():
    changes = {(i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6, 0.7)
               for i in range(31) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)}
    result = context(changes=changes)
    assert result.frames[5].normalization.scale_status == "fallback_unit"
    assert available(result).status == "ready_for_calculation"
    # Degenerate vectors must still be rejected by the future calculator;
    # contract readiness does not claim a computed angle.
    assert FeatureObservation.from_availability(available(result)).value is None


def test_blocked_registry_candidate_is_explicitly_unavailable():
    result = context()
    assert_unavailable(available(result, feature_id="squat_knee_to_ankle_horizontal_left_ratio"), FlagCode.FEATURE_DEFINITION_BLOCKED)


@pytest.mark.parametrize("feature_id", ["unknown", "", "squat_knee_angle_Left_deg"])
def test_unknown_feature_id_has_structured_reason(feature_id):
    with pytest.raises(GymError) as caught:
        DEFAULT_REGISTRY.get(feature_id)
    assert caught.value.flag.code == FlagCode.UNKNOWN_FEATURE_ID


def test_unknown_exercise_and_invalid_viewpoint_are_structured():
    with pytest.raises(GymError) as caught:
        DEFAULT_REGISTRY.for_exercise("BenchPress")
    assert caught.value.flag.code == FlagCode.UNSUPPORTED_EXERCISE
    result = context()
    for field, value, reason in (("exercise", "BenchPress", FlagCode.UNSUPPORTED_EXERCISE),
                                  ("viewpoint", "upside_down", FlagCode.INVALID_VIEWPOINT)):
        quality = result.quality.model_copy(update={field: value})
        with pytest.raises(GymError) as caught:
            check_feature_availability(ID, result.frames[5], quality)
        assert caught.value.flag.code == reason


@pytest.mark.parametrize("cause", ["missing_joint", "null", "nan", "inf", "video_id", "frame_index", "wrong_type"])
def test_unsafe_input_is_revalidated_and_returns_structured_error(cause):
    result = context()
    frame = result.frames[5]
    point = next(j for j in frame.joints if j.joint == J.LEFT_KNEE)
    if cause == "missing_joint":
        frame = frame.model_copy(update={"joints": tuple(j for j in frame.joints if j.joint != J.LEFT_KNEE)})
    elif cause in ("null", "nan", "inf"):
        altered = point.model_copy(update={"x": {"null": None, "nan": float("nan"), "inf": float("inf")}[cause]})
        frame = frame.model_copy(update={"joints": tuple(altered if j.joint == J.LEFT_KNEE else j for j in frame.joints)})
    elif cause == "video_id":
        frame = frame.model_copy(update={"video_id": "another"})
    elif cause == "frame_index":
        frame = frame.model_copy(update={"frame_index": 1000})
    else:
        frame = None
    with pytest.raises(GymError) as caught:
        check_feature_availability(ID, frame, result.quality)
    assert caught.value.flag.code == FlagCode.INVALID_FEATURE_INPUT


def test_serialization_preserves_metadata_and_unavailable_reason_flags():
    restored_registry = FeatureRegistry.model_validate_json(DEFAULT_REGISTRY.model_dump_json())
    assert restored_registry == DEFAULT_REGISTRY
    assert restored_registry.content_hash() == DEFAULT_REGISTRY.content_hash()
    decision = available(context(viewpoint=Viewpoint.UNKNOWN, changes=missing([5], J.LEFT_KNEE)))
    restored = FeatureAvailability.model_validate_json(decision.model_dump_json())
    assert restored == decision
    observation = FeatureObservation.from_availability(restored)
    assert FeatureObservation.model_validate_json(observation.model_dump_json()) == observation
    assert observation.availability.definition.validation.measurement_status == "unvalidated"
    assert observation.availability.flags and observation.value is None


@pytest.mark.parametrize("kind", ["fake_value", "available_status", "no_reason", "no_flag", "false_ready", "hash"])
def test_cannot_forge_unavailable_as_numeric_or_ready(kind):
    observation = FeatureObservation.from_availability(available(context(viewpoint=Viewpoint.UNKNOWN)))
    data = observation.model_dump(mode="json")
    if kind == "fake_value":
        data["value"] = 0.0
    elif kind == "available_status":
        data["status"] = "available"
    elif kind == "no_reason":
        data["availability"]["reasons"] = []
    elif kind == "no_flag":
        data["availability"]["flags"] = []
    elif kind == "false_ready":
        data["availability"]["status"] = "ready_for_calculation"
    else:
        data["availability"]["input_hash"] = "a" * 64
    with pytest.raises(ValidationError):
        FeatureObservation.model_validate(data)


def test_registry_returned_mappings_do_not_mutate_registry_and_hash_detects_metadata_changes():
    before = DEFAULT_REGISTRY.content_hash()
    mapping = DEFAULT_REGISTRY.quality_requirements("BodyWeightSquats")
    mapping.clear()
    assert DEFAULT_REGISTRY.content_hash() == before
    changed = changed_registry(ID, name="Another descriptive name")
    assert changed.get(ID).content_hash() != DEFAULT_REGISTRY.get(ID).content_hash()


def test_public_import_lookup_and_schema_have_no_heavy_backends(tmp_path):
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from gym_coach.features import DEFAULT_REGISTRY, FeatureDefinition, FeatureObservation
from gym_coach.features.registry import FeatureRegistry
assert len(DEFAULT_REGISTRY.quality_requirements('BodyWeightSquats'))==5
assert FeatureDefinition.model_json_schema() and FeatureObservation.model_json_schema()
print('feature-contract-import-ok')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "feature-contract-import-ok"
