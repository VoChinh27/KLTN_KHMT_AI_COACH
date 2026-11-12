"""Stage 5.4 full-source linkage, strict intake, compatibility and isolation."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import geometry
from gym_coach.contracts import ErrorCategory, JointProvenance, SourceTier, Stage, Viewpoint
from gym_coach.features import (
    DEFAULT_REGISTRY, FEATURE_CONTEXT_FACTORY_VERSION, FEATURE_CONTEXT_SCHEMA_VERSION,
    FeatureInputContext, LinkedFeatureInputContext, NumericFeatureObservation,
    build_feature_input_context, observe_feature,
)
from gym_coach.flags import FlagCode, GymError, Severity, StructuredFlag
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import PreprocessingProfile, PreprocessingResult, preprocess_pose
from gym_coach.preprocessing.models import content_hash as preprocessing_hash
from gym_coach.quality import MetricEligibility, QualityReport, evaluate_quality

from test_feature_observation import ID, block_primitive, fixture_context
from test_preprocessing import make_frames, metadata, missing


@pytest.fixture
def result():
    return fixture_context()[1]


def assert_error(call, code):
    with pytest.raises(GymError) as caught:
        call()
    flag = caught.value.flag
    assert flag.code == code
    assert flag.category == ErrorCategory.INPUT_ERROR
    assert flag.stage == Stage.FEATURES and flag.source_tier == SourceTier.S3
    assert flag.severity == Severity.ERROR and flag.evidence["reason"]
    assert "value" not in flag.evidence
    assert StructuredFlag.model_validate_json(flag.model_dump_json()) == flag
    return caught.value


@pytest.mark.parametrize("index", [0, 5, 12, 30])
def test_factory_selects_exact_frame_full_audit_identity_and_snapshots(result, index):
    before = result.model_dump_json()
    context = build_feature_input_context(result, frame_index=index)
    assert isinstance(context, FeatureInputContext) and isinstance(context, LinkedFeatureInputContext)
    assert context.context_schema_version == FEATURE_CONTEXT_SCHEMA_VERSION == "0.2.0"
    assert context.context_factory_version == FEATURE_CONTEXT_FACTORY_VERSION == "preprocessing-context-v1"
    assert context.frame == result.frames[index] and context.quality == result.quality
    assert context.raw_frame == result.raw_frames[index]
    assert (context.frame.video_id, context.frame.frame_index, context.frame.timestamp_seconds) == (
        result.metadata.video_id, index, result.raw_frames[index].timestamp_seconds)
    assert context.raw_frame_hash == result.frames[index].raw_frame_hash == preprocessing_hash(context.raw_frame)
    assert context.raw_sequence_hash == result.raw_sequence_hash
    assert context.processed_frame_hash == preprocessing_hash(result.frames[index])
    assert context.preprocessing_result_hash == preprocessing_hash(result)
    assert context.metadata_hash == preprocessing_hash(result.metadata)
    assert (context.profile_version, context.profile_hash) == (result.profile.profile_version, result.profile_hash)
    assert context.quality_hash == result.quality_hash
    assert context.preprocessing_result == result and context.preprocessing_result is not result
    assert context.quality.by_joint is not result.quality.by_joint
    assert context.quality.metric_eligibility is not result.quality.metric_eligibility
    assert result.model_dump_json() == before
    restored = LinkedFeatureInputContext.model_validate_json(context.model_dump_json(), strict=True)
    assert restored == context and restored.content_hash() == context.content_hash()


@pytest.mark.parametrize("index", [-1, -100, True, False, 1.0, "1", None, [], float("nan"), float("inf"), 31, 100])
def test_invalid_index_has_structured_error(result, index):
    assert_error(lambda: build_feature_input_context(result, frame_index=index), FlagCode.INVALID_FEATURE_CONTEXT_INDEX)


@pytest.mark.parametrize("value", [None, {}, (), "result"])
def test_invalid_result_type_has_structured_error(value):
    assert_error(lambda: build_feature_input_context(value, frame_index=0), FlagCode.INVALID_FEATURE_INPUT)


def test_empty_complete_result_has_no_selectable_frame():
    raw = make_frames(n=0)
    empty = preprocess_pose(raw, evaluate_quality(raw), metadata(raw))
    assert_error(lambda: build_feature_input_context(empty, frame_index=0), FlagCode.INVALID_FEATURE_CONTEXT_INDEX)


@pytest.mark.parametrize("kind", ["imputed", "low_visibility", "missing", "out_of_frame", "fallback"])
def test_provenance_endpoints_normalization_and_flags_preserved(kind):
    changes = missing([5], J.LEFT_KNEE) if kind in ("imputed", "missing") else {}
    if kind == "low_visibility":
        changes[(5, J.LEFT_KNEE)] = ("low_visibility", 0.4, 0.4)
    elif kind == "out_of_frame":
        changes[(5, J.LEFT_KNEE)] = ("observed", 1.2, 0.4)
    elif kind == "fallback":
        changes = missing(range(31), J.RIGHT_HIP)
    profile = PreprocessingProfile(max_gap_seconds=0.0) if kind == "missing" else None
    _, source = fixture_context(changes=changes, profile=profile)
    before = source.model_dump_json()
    context = build_feature_input_context(source, frame_index=5)
    assert context.frame.joints == source.frames[5].joints
    assert context.frame.flags == source.frames[5].flags
    assert context.frame.normalization == source.frames[5].normalization
    assert context.quality == source.quality
    if kind == "imputed":
        knee = next(j for j in context.frame.joints if j.joint == J.LEFT_KNEE)
        assert knee.source_status == "missing" and knee.status == JointProvenance.IMPUTED
        assert (knee.interpolation.left_frame_index, knee.interpolation.right_frame_index) == (4, 6)
        assert knee.interpolation.left_timestamp_seconds == source.raw_frames[4].timestamp_seconds
        assert knee.interpolation.right_timestamp_seconds == source.raw_frames[6].timestamp_seconds
    elif kind == "fallback":
        assert context.frame.normalization.origin_x is None
        assert context.frame.normalization.torso_scale is None
    assert source.model_dump_json() == before


@pytest.mark.parametrize("kind", ["raw_length", "processed_length", "metadata_count", "quality_count",
    "raw_video", "raw_index", "raw_timestamp", "processed_video", "processed_index", "processed_timestamp",
    "raw_frame_hash", "raw_sequence_hash", "quality_hash", "quality_report", "quality_core",
    "profile_hash", "profile", "resolved_profile", "endpoint", "source_status", "mixed_frame"])
def test_unsafe_preprocessing_result_audit_corruption_rejected(result, kind):
    updates = {}
    if kind in ("raw_length", "processed_length"):
        key = "raw_frames" if kind == "raw_length" else "frames"
        updates[key] = getattr(result, key)[:-1]
    elif kind == "metadata_count":
        updates["metadata"] = result.metadata.model_copy(update={"frame_count": 30})
    elif kind == "quality_count":
        updates["quality"] = result.quality.model_copy(update={"total_frames": 30})
    elif kind in ("raw_frame_hash", "processed_video", "processed_index", "processed_timestamp", "source_status", "mixed_frame"):
        frames = list(result.frames)
        change = {"raw_frame_hash": "0" * 64} if kind == "raw_frame_hash" else {
            "processed_video": {"video_id": "other.avi"}, "processed_index": {"frame_index": 6},
            "processed_timestamp": {"timestamp_seconds": 0.6}}.get(kind, {})
        if kind == "source_status":
            change["joints"] = tuple(j.model_copy(update={"source_status": "missing"}) if j.joint == J.LEFT_KNEE else j for j in frames[5].joints)
        if kind == "mixed_frame":
            frames[5] = fixture_context(ankle=(0.7, 0.7))[1].frames[5]
        else:
            frames[5] = frames[5].model_copy(update=change)
        updates["frames"] = tuple(frames)
    elif kind in ("raw_video", "raw_index", "raw_timestamp"):
        raw = list(result.raw_frames)
        field, value = {"raw_video": ("video_id", "other.avi"), "raw_index": ("frame_index", 6),
                        "raw_timestamp": ("timestamp_seconds", 0.6)}[kind]
        raw[5] = raw[5].model_copy(update={field: value})
        updates["raw_frames"] = tuple(raw)
    elif kind in ("raw_sequence_hash", "quality_hash", "profile_hash"):
        updates[kind] = "0" * 64
    elif kind == "quality_report":
        updates["quality"] = fixture_context(viewpoint=Viewpoint.UNKNOWN)[1].quality
    elif kind == "quality_core":
        other = fixture_context(changes=missing([5], J.LEFT_KNEE))[1].quality
        updates.update(quality=other, quality_hash=preprocessing_hash(other))
    elif kind == "profile":
        updates["profile"] = result.profile.model_copy(update={"profile_version": "other-run"})
    elif kind == "resolved_profile":
        updates["resolved_profile"] = result.resolved_profile.model_copy(update={"fps": 60.0})
    else:
        result = fixture_context(changes=missing([5], J.LEFT_KNEE))[1]
        frames = list(result.frames)
        frames[5] = frames[5].model_copy(update={"joints": tuple(j.model_copy(update={
            "interpolation": j.interpolation.model_copy(update={"left_timestamp_seconds": 0.12})})
            if j.joint == J.LEFT_KNEE else j for j in frames[5].joints)})
        updates["frames"] = tuple(frames)
    unsafe = result.model_copy(update=updates)
    error = assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.FEATURE_CONTEXT_MISMATCH)
    assert error.__cause__ is not None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.5", True])
def test_strict_serialization_does_not_hide_nonfinite_or_coerce_coordinates(result, value):
    # Corrupt a nullable coordinate that JSON NaN -> null could otherwise hide.
    source = fixture_context(changes={(5, J.LEFT_KNEE): ("low_visibility", 0.4, 0.4)})[1]
    frames = list(source.frames)
    frames[5] = frames[5].model_copy(update={"joints": tuple(
        j.model_copy(update={"x": value}) if j.joint == J.LEFT_KNEE else j for j in frames[5].joints)})
    unsafe = source.model_copy(update={"frames": tuple(frames)})
    assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.INVALID_FEATURE_INPUT)


def test_model_construct_is_revalidated_and_coercible_metadata_rejected(result):
    unsafe = PreprocessingResult.model_construct(**{**result.__dict__, "raw_sequence_hash": "0" * 64})
    assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.FEATURE_CONTEXT_MISMATCH)
    unsafe = result.model_copy(update={"metadata": result.metadata.model_copy(update={"fps": "30.0"})})
    assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.INVALID_FEATURE_INPUT)


@pytest.mark.parametrize("value", [True, "0.5", float("nan"), float("inf"), float("-inf")])
def test_usable_coordinate_unsafe_type_is_rejected_before_measurement(result, value):
    frames = list(result.frames)
    frames[5] = frames[5].model_copy(update={"joints": tuple(
        j.model_copy(update={"x": value}) if j.joint == J.LEFT_KNEE else j for j in frames[5].joints)})
    unsafe = result.model_copy(update={"frames": tuple(frames)})
    assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.INVALID_FEATURE_INPUT)


def test_factory_does_not_catch_programming_exceptions(result, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("programming failure")
    monkeypatch.setattr(PreprocessingResult, "model_validate_json", broken)
    with pytest.raises(RuntimeError, match="programming failure"):
        build_feature_input_context(result, frame_index=5)


def test_factory_and_adapter_never_rerun_quality_eligibility(result, monkeypatch):
    from gym_coach.quality import gate
    def forbidden(*args, **kwargs):
        pytest.fail("Factory/adapter recomputed quality eligibility")
    monkeypatch.setattr(gate, "_eligibility", forbidden)
    monkeypatch.setattr(gate, "evaluate_quality", forbidden)
    monkeypatch.setattr(gate.QualityGate, "evaluate", forbidden)
    context = build_feature_input_context(result, frame_index=5)
    assert observe_feature(ID, context).value == 90.0
    assert LinkedFeatureInputContext.model_validate_json(context.model_dump_json(), strict=True) == context


@pytest.mark.parametrize("kind", ["empty", "clean", "missing", "low_visibility", "out_of_frame"])
def test_stage4_core_only_audit_matches_existing_quality_statistics(kind, monkeypatch):
    from gym_coach.preprocessing.pipeline import validate_quality
    from gym_coach.quality import gate
    changes = missing([3, 4, 6], J.LEFT_KNEE) if kind == "missing" else {}
    if kind in ("low_visibility", "out_of_frame"):
        changes[(5, J.LEFT_KNEE)] = ("low_visibility", 0.4, 0.4) if kind == "low_visibility" else ("observed", 1.2, 0.4)
    raw = make_frames(n=0 if kind == "empty" else 31, changes=changes)
    report = evaluate_quality(raw)
    before = report.model_dump_json()
    def forbidden(*args, **kwargs):
        pytest.fail("Audit called eligibility")
    monkeypatch.setattr(gate, "_eligibility", forbidden)
    validate_quality(raw, report)
    assert report.model_dump_json() == before


@pytest.mark.parametrize("kind", ["whitespace_profile_version", "overflowing_profile"])
def test_profile_intake_failures_are_structured(result, kind):
    profile = result.profile.model_copy(update={"profile_version": " "} if kind == "whitespace_profile_version" else {"max_gap_seconds": 1e308})
    unsafe = result.model_copy(update={"profile": profile, "profile_hash": preprocessing_hash(profile)})
    assert_error(lambda: build_feature_input_context(unsafe, frame_index=5), FlagCode.INVALID_FEATURE_INPUT)


@pytest.mark.parametrize("kind", ["source_raw_hash", "source_quality_hash", "source_profile_hash", "frame", "quality", "result"])
def test_linked_json_corruption_rejects_mixed_snapshots(result, kind):
    context = build_feature_input_context(result, frame_index=5)
    payload = context.model_dump(mode="json")
    if kind.startswith("source_"):
        key = {"source_raw_hash": "raw_sequence_hash", "source_quality_hash": "quality_hash",
               "source_profile_hash": "profile_hash"}[kind]
        payload["preprocessing_result"][key] = "0" * 64
    elif kind == "frame":
        payload["frame"]["timestamp_seconds"] = 0.6
    elif kind == "quality":
        payload["quality"]["metric_eligibility"].pop(ID)
    else:
        payload["preprocessing_result"] = fixture_context(ankle=(0.7, 0.7))[1].model_dump(mode="json")
    with pytest.raises(ValidationError):
        LinkedFeatureInputContext.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("field", ["preprocessing_result_hash", "raw_sequence_hash", "raw_frame_hash",
    "processed_frame_hash", "quality_hash", "profile_hash", "metadata_hash", "profile_version"])
def test_linkage_json_roundtrip_rejects_corrupt_identities(result, field):
    context = build_feature_input_context(result, frame_index=5)
    payload = context.model_dump(mode="json")
    payload[field] = "wrong-version" if field == "profile_version" else "0" * 64
    with pytest.raises(ValidationError):
        LinkedFeatureInputContext.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("field", ["context_schema_version", "context_factory_version", "preprocessing_result_hash", "preprocessing_result"])
def test_linked_and_legacy_schema_boundaries_are_explicit(result, field):
    context = build_feature_input_context(result, frame_index=5)
    payload = context.model_dump(mode="json")
    if field in ("context_schema_version", "context_factory_version"):
        payload[field] = "legacy"
    else:
        del payload[field]
    with pytest.raises(ValidationError):
        LinkedFeatureInputContext.model_validate_json(json.dumps(payload), strict=True)
    with pytest.raises(ValidationError):
        FeatureInputContext.model_validate_json(context.model_dump_json())
    old = FeatureInputContext(frame=result.frames[5], quality=result.quality)
    assert set(old.model_dump()) == {"frame", "quality"}
    assert FeatureInputContext.model_validate_json(old.model_dump_json()) == old


@pytest.mark.parametrize("kind", ["frame", "quality", "result", "index", "timestamp", "processed_hash", "raw_hash", "result_hash", "quality_dictionary"])
def test_adapter_revalidates_linked_context_before_primitive(result, monkeypatch, kind):
    context = build_feature_input_context(result, frame_index=5)
    other = fixture_context(ankle=(0.7, 0.7), viewpoint=Viewpoint.UNKNOWN)[1]
    changes = {"frame": {"frame": other.frames[5]}, "quality": {"quality": other.quality},
        "result": {"preprocessing_result": other},
        "index": {"frame": context.frame.model_copy(update={"frame_index": 6})},
        "timestamp": {"frame": context.frame.model_copy(update={"timestamp_seconds": 0.6})},
        "processed_hash": {"processed_frame_hash": "0" * 64},
        "raw_hash": {"raw_frame_hash": "0" * 64}, "result_hash": {"preprocessing_result_hash": "0" * 64}}
    if kind == "quality_dictionary":
        context.quality.metric_eligibility.pop(ID)
    else:
        context = context.model_copy(update=changes[kind])
    block_primitive(monkeypatch)
    assert_error(lambda: observe_feature(ID, context), FlagCode.FEATURE_CONTEXT_MISMATCH)


def test_factory_snapshot_survives_original_nested_mutation(result):
    context = build_feature_input_context(result, frame_index=0)
    before = context.model_dump_json()
    result.quality.metric_eligibility.pop(ID)
    result.quality.by_joint.clear()
    assert context.model_dump_json() == before
    assert LinkedFeatureInputContext.model_validate_json(before, strict=True) == context


def test_flags_evidence_is_detached_including_raw_snapshot():
    raw = make_frames(n=5)
    flag = StructuredFlag(code=FlagCode.VIEWPOINT_UNKNOWN, category=ErrorCategory.MEASUREMENT_ERROR,
        stage=Stage.POSE, source_tier=SourceTier.S3, severity=Severity.INFO,
        message="Synthetic audit", evidence={"marker": "original"})
    raw = (raw[0].model_copy(update={"flags": (flag,)}), *raw[1:])
    source = preprocess_pose(raw, evaluate_quality(raw), metadata(raw))
    context = build_feature_input_context(source, frame_index=0)
    before = context.model_dump_json()
    source.raw_frames[0].flags[0].evidence["marker"] = "changed"
    source.frames[0].flags[0].evidence["marker"] = "changed"
    source.quality.flags[0].evidence["marker"] = "changed"
    assert context.model_dump_json() == before


@pytest.mark.parametrize("scope", ["sequence", "metric"])
@pytest.mark.parametrize("action", ["degrade", "skip"])
def test_custom_quality_decisions_are_preserved_without_promotion(result, scope, action, monkeypatch):
    # Stage 4 explicitly permits caller-defined metric decisions. Preserve them.
    data = result.quality.model_dump(mode="json")
    key = "movement_analysis" if scope == "sequence" else ID
    denied = MetricEligibility(eligible=False, action=action,
        required_joints=result.quality.metric_eligibility[key].required_joints, reasons=("synthetic-denied",))
    data["metric_eligibility"][key] = denied.model_dump(mode="json")
    if scope == "sequence":
        data.update(eligibility=denied.model_dump(mode="json"), sequence_status="degraded" if action == "degrade" else "skipped")
    quality = QualityReport.model_validate_json(json.dumps(data), strict=True)
    source = PreprocessingResult.model_validate_json(result.model_copy(update={
        "quality": quality, "quality_hash": preprocessing_hash(quality)}).model_dump_json(), strict=True)
    before = source.model_dump_json()
    context = build_feature_input_context(source, frame_index=5)
    block_primitive(monkeypatch)
    measured = observe_feature(ID, context)
    assert measured.value is None and FlagCode.FEATURE_QUALITY_BLOCKED in measured.reasons
    assert measured.availability.input.quality == quality and context.quality == quality
    assert context.quality.metric_eligibility[key].action == action
    assert source.model_dump_json() == before


@pytest.mark.parametrize("kind", ["available", "imputed_degrade", "missing", "viewpoint", "fallback", "degenerate"])
def test_factory_adapter_outcome_matches_legacy_and_preserves_numeric_schema(kind, monkeypatch):
    changes = missing([5], J.LEFT_KNEE) if kind in ("imputed_degrade", "missing") else {}
    if kind == "fallback":
        changes = {(i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6, 0.7)
                   for i in range(31) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)}
    old, source = fixture_context(changes=changes,
        ankle=(0.4, 0.4) if kind == "degenerate" else (0.7, 0.4),
        viewpoint=Viewpoint.UNKNOWN if kind == "viewpoint" else Viewpoint.SIDE,
        profile=PreprocessingProfile(max_gap_seconds=0.0) if kind == "missing" else None)
    before = source.model_dump_json()
    if kind in ("imputed_degrade", "missing", "viewpoint"):
        block_primitive(monkeypatch)
    context = build_feature_input_context(source, frame_index=5)
    measured = observe_feature(ID, context)
    assert measured == observe_feature(ID, old)
    assert measured.availability.input_hash == context.measurement_input_hash
    assert measured.numeric_feature_schema_version == "0.2.0"
    assert NumericFeatureObservation.model_validate_json(measured.model_dump_json()) == measured
    if kind in ("available", "fallback"):
        assert measured.status == "available" and measured.value == pytest.approx(90.0, abs=1e-10, rel=0)
    else:
        assert measured.status == "unavailable" and measured.value is None
    assert source.model_dump_json() == before


def test_public_factory_import_and_execution_remain_lightweight(result, tmp_path):
    source = tmp_path/"preprocessing.json"
    source.write_text(result.model_dump_json(), encoding="utf-8")
    code = r'''
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'cv2', 'mediapipe', 'torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from gym_coach.features import LinkedFeatureInputContext, build_feature_input_context, observe_feature
from gym_coach.preprocessing import PreprocessingResult
result = PreprocessingResult.model_validate_json(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), strict=True)
context = build_feature_input_context(result, frame_index=5)
assert LinkedFeatureInputContext.model_json_schema()
assert LinkedFeatureInputContext.model_validate_json(context.model_dump_json(), strict=True) == context
assert observe_feature('squat_knee_angle_left_deg', context).value == 90.0
assert not {'numpy', 'scipy', 'cv2', 'mediapipe', 'torch'}.intersection(sys.modules)
print('factory-import-execution-ok')
'''
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(source)], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "factory-import-execution-ok"
    assert DEFAULT_REGISTRY.registry_version == "gym-feature-registry-v1"
