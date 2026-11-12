"""Stage 5.6: actual QualityGate evidence, not manually promoted eligibility."""
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from gym_coach import geometry
from gym_coach.features import (
    DEFAULT_REGISTRY, SUPPORTED_KNEE_FEATURE_IDS, build_feature_input_context,
    check_feature_availability, observe_feature,
)
from gym_coach.flags import FlagCode
from gym_coach.pose.mapping import JointName as J
from gym_coach.quality import QualityGate, QualityProfile, QualityReport

from stage5_6_fixture import ROOT, VIDEO_ID, fixture_input, fixture_result, knee_requirements, knee_viewpoints


@pytest.fixture(scope="module")
def audit_directory():
    directory = ROOT/"outputs/stage5_6"/("synthetic-"+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def forbidden(*args, **kwargs):
    pytest.fail("Blocked input reached primitive or recomputed QualityGate")


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("index", [0, 15, 30])
def test_actual_qualitygate_sequence_and_both_metric_allow_before_adapters(monkeypatch, fps, index):
    source = fixture_result(fps=fps)
    report = source.quality
    assert source.metadata.video_id == report.video_id == VIDEO_ID
    assert len(source.raw_frames) == source.metadata.frame_count == report.total_frames == 31
    assert source.metadata.fps == fps
    assert report.profile == QualityProfile() and report.profile_hash == QualityProfile().content_hash()
    assert report.viewpoint == "side" and report.sequence_status == "eligible"
    assert report.eligibility.eligible and report.eligibility.action == "allow" and not report.eligibility.reasons
    for key in SUPPORTED_KNEE_FEATURE_IDS:
        metric = report.metric_eligibility[key]
        assert metric.eligible and metric.action == "allow" and not metric.reasons
        assert metric.required_joints == DEFAULT_REGISTRY.get(key).required_joints
    assert all(f.timestamp_seconds == f.frame_index/fps for f in source.raw_frames)
    assert all(j.status == "observed" and 0 <= j.x <= 1 and 0 <= j.y <= 1
               for frame in source.raw_frames for j in frame.joints)
    before = source.model_dump_json()
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    original = geometry.joint_angle
    calls = []
    def spy(*points):
        calls.append(points)
        return original(*points)
    monkeypatch.setattr(geometry, "joint_angle", spy)
    context = build_feature_input_context(source, frame_index=index)
    for key in SUPPORTED_KNEE_FEATURE_IDS:
        ready = check_feature_availability(key, context.frame, context.quality)
        assert ready.status == "ready_for_calculation" and not ready.reasons
        measured = observe_feature(key, context)
        assert measured.status == "available" and measured.value == pytest.approx(90, abs=1e-10, rel=0)
        assert measured.value == original(*calls[-1])
        assert tuple(j.joint for j in measured.joint_inputs) == knee_requirements()[key]
        assert measured.measurement_validation_status == "unvalidated"
        assert measured.assessment_status == "blocked_pending_expert_validation"
        assert measured.availability.input.quality == report
        assert measured.availability.input_hash == context.measurement_input_hash
        assert type(measured).model_validate_json(measured.model_dump_json(), strict=True) == measured
    assert len(calls) == 2 and source.model_dump_json() == before


@pytest.mark.parametrize("key", SUPPORTED_KNEE_FEATURE_IDS)
@pytest.mark.parametrize("mode", ["missing", "wrong_key", "opposite_only", "wrong_joints"])
def test_true_global_allow_never_infers_missing_or_mismatched_metric(monkeypatch, key, mode):
    opposite = next(k for k in SUPPORTED_KNEE_FEATURE_IDS if k != key)
    requirements = {"wrong_knee_key": knee_requirements()[key]} if mode == "wrong_key" else (
        {opposite: knee_requirements()[opposite]} if mode == "opposite_only" else (
            {key: knee_requirements()[opposite]} if mode == "wrong_joints" else {}))
    source = fixture_result(requirements=requirements)
    assert source.quality.eligibility.action == "allow"
    # QualityGate evaluates exactly the joints the caller declares, not registry
    # semantics. A clean wrong joint set may be allow; feature readiness rejects it.
    if mode == "wrong_joints":
        assert source.quality.metric_eligibility[key].action == "allow"
    before = source.quality.model_dump_json()
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    context = build_feature_input_context(source, frame_index=15)
    measured = observe_feature(key, context)
    reason = FlagCode.FEATURE_QUALITY_POLICY_MISMATCH if mode == "wrong_joints" else FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING
    assert measured.status == "unavailable" and measured.value is None and reason in measured.reasons
    assert measured.availability.input.quality.model_dump_json() == before
    assert source.quality.model_dump_json() == before


@pytest.mark.parametrize("key", SUPPORTED_KNEE_FEATURE_IDS)
@pytest.mark.parametrize("viewpoint", ["unknown", "front", "oblique"])
def test_qualitygate_specific_viewpoint_skip_is_never_promoted(monkeypatch, key, viewpoint):
    source = fixture_result(viewpoint=viewpoint)
    assert source.quality.eligibility.action == "allow"  # Generic movement has no viewpoint policy.
    metric = source.quality.metric_eligibility[key]
    assert metric.action == "skip" and not metric.eligible
    assert ("viewpoint_unknown" if viewpoint == "unknown" else "viewpoint_mismatch") in metric.reasons
    before = source.model_dump_json()
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    measured = observe_feature(key, build_feature_input_context(source, frame_index=15))
    assert measured.status == "unavailable" and measured.value is None
    assert FlagCode.FEATURE_QUALITY_BLOCKED in measured.reasons
    assert (FlagCode.VIEWPOINT_UNKNOWN if viewpoint == "unknown" else FlagCode.VIEWPOINT_MISMATCH) in measured.reasons
    assert source.model_dump_json() == before


@pytest.mark.parametrize("key", SUPPORTED_KNEE_FEATURE_IDS)
@pytest.mark.parametrize("case", ["short", "sequence_degrade", "sequence_skip", "low", "low_skip",
                                 "imputed_gap", "missing_endpoint", "out_of_frame"])
def test_real_gate_skip_degrade_or_unusable_joint_cannot_reach_primitive(monkeypatch, key, case):
    target_joint = knee_requirements()[key][1]
    changes = {}
    n = 4 if case == "short" else 31
    index = 0 if case in {"short", "missing_endpoint"} else 15
    if case.startswith("sequence_"):
        for i in ([15] if case == "sequence_degrade" else range(n)):
            changes[i, J.LEFT_SHOULDER] = ("low_visibility", 0.4, 0.3)
    elif case in {"low", "low_skip"}:
        for i in ([15] if case == "low" else range(n)):
            changes[i, target_joint] = ("low_visibility", 0.5, 0.4)
    elif case in {"imputed_gap", "missing_endpoint"}:
        changes[index, target_joint] = ("missing", None, None)
    elif case == "out_of_frame":
        changes[index, target_joint] = ("observed", 1.2, 0.4)
    source = fixture_result(n=n, changes=changes)
    expected = "skip" if case in {"short", "sequence_skip", "low_skip"} else "degrade"
    assert source.quality.eligibility.action == expected and not source.quality.eligibility.eligible
    for feature in SUPPORTED_KNEE_FEATURE_IDS:
        assert source.quality.metric_eligibility[feature].required_joints == knee_requirements()[feature]
    if case.startswith("sequence_"):
        assert source.quality.metric_eligibility[key].action == "allow"
    elif case == "imputed_gap":
        selected = next(j for j in source.frames[index].joints if j.joint == target_joint)
        assert selected.usable and selected.status == "imputed" and selected.source_status == "missing"
        assert source.quality.metric_eligibility[key].action == "degrade"
    elif case != "short":
        assert not next(j for j in source.frames[index].joints if j.joint == target_joint).usable
    before = source.model_dump_json()
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    measured = observe_feature(key, build_feature_input_context(source, frame_index=index))
    assert measured.status == "unavailable" and measured.value is None and FlagCode.FEATURE_QUALITY_BLOCKED in measured.reasons
    assert measured.availability.input.quality == source.quality
    assert source.model_dump_json() == before


def test_fixture_is_deterministic_immutable_and_saved_evidence_is_explicitly_synthetic(audit_directory):
    profile_path = ROOT/"configs/preprocessing_s3_v1.toml"
    config_before = profile_path.read_bytes()
    registry_before = DEFAULT_REGISTRY.model_dump_json()
    caller_changes = {(15, J.NOSE): ("observed", 0.5, 0.4)}
    options_before = dict(caller_changes)
    raw, info, report = fixture_input(changes=caller_changes)
    raw_before = tuple(f.model_dump_json() for f in raw)
    report_before = report.model_dump_json()
    source = fixture_result(changes=caller_changes)
    assert source.raw_frames == raw and source.quality == report
    assert fixture_result(changes=caller_changes) == source
    assert caller_changes == options_before and DEFAULT_REGISTRY.model_dump_json() == registry_before
    context = build_feature_input_context(source, frame_index=15)
    observations = {key: observe_feature(key, context) for key in SUPPORTED_KNEE_FEATURE_IDS}
    assert raw_before == tuple(f.model_dump_json() for f in raw) and report_before == report.model_dump_json()
    assert profile_path.read_bytes() == config_before
    assert QualityReport.model_validate_json(report.model_dump_json(), strict=True) == report
    payloads = {"synthetic.raw.jsonl": "\n".join(raw_before)+"\n", "synthetic.quality.json": report.model_dump_json(indent=2),
                "synthetic.processed.json": source.model_dump_json(indent=2), "synthetic.context.json": context.model_dump_json(indent=2)}
    payloads.update({key+".synthetic-observation.json": obs.model_dump_json(indent=2) for key, obs in observations.items()})
    records = {}
    for name, text in payloads.items():
        with (audit_directory/name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        records[name] = hashlib.sha256((audit_directory/name).read_bytes()).hexdigest()
    summary = {"evidence_type": "synthetic_contract_test", "real_video": False, "viewpoint_source": "Explicit deterministic fixture construction",
        "video_id": info.video_id, "fps": info.fps, "total_frames": len(raw), "profile": report.profile.model_dump(mode="json"),
        "profile_hash": report.profile_hash, "sequence": report.eligibility.model_dump(mode="json"),
        "knee_metrics": {key: report.metric_eligibility[key].model_dump(mode="json") for key in SUPPORTED_KNEE_FEATURE_IDS},
        "observations": {key: {"status": obs.status, "value": obs.value, "validation": obs.measurement_validation_status,
                              "numeric_schema": obs.numeric_feature_schema_version} for key, obs in observations.items()},
        "files_sha256": records, "accuracy_form_validation": "NOT_PERFORMED"}
    with (audit_directory/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
