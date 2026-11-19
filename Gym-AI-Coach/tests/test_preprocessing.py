"""Stage 4 policy and numerical tests; no movement/technical conclusions."""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from gym_coach.contracts import Exercise, JointProvenance, Stage
from gym_coach.flags import FlagCode, GymError
from gym_coach.pose.mapping import JointName
from gym_coach.pose.models import (
    JointObservation, PoseBackendIdentity, PoseExtractionConfig, PoseFrame,
)
from gym_coach.preprocessing import (
    PreprocessingProfile, PreprocessingResult, load_preprocessing_profile,
    preprocess_pose, read_preprocessing_result, summarize_preprocessing,
    write_preprocessing_result,
)
from gym_coach.preprocessing.pipeline import _torso_scale
from gym_coach.quality import evaluate_quality
from gym_coach.video.models import VideoMetadata

ROOT = Path(__file__).resolve().parents[1]
J = JointName


def make_frames(n=31, fps=30.0, changes=None, trajectory=None):
    frames = []
    changes = changes or {}
    for i in range(n):
        joints = []
        for name in J:
            x, y = 0.5, 0.4
            if name in (J.LEFT_HIP, J.RIGHT_HIP):
                x, y = (0.4 if name == J.LEFT_HIP else 0.6), 0.7
            if name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER):
                x, y = (0.4 if name == J.LEFT_SHOULDER else 0.6), 0.3
            if name == J.NOSE and trajectory:
                x = trajectory(i / fps)
            status = changes.get((i, name), ("observed", x, y))
            kind, x, y = status
            joints.append(JointObservation(
                joint=name, status=kind, x=x, y=y,
                visibility=None if kind == "missing" else 0.2 if kind == "low_visibility" else 0.9,
                missing_reason=FlagCode.MISSING_JOINT if kind == "missing" else None,
            ))
        frames.append(PoseFrame(
            video_id="synthetic.avi", exercise=Exercise.BODY_WEIGHT_SQUATS,
            frame_index=i, timestamp_seconds=i / fps, has_pose=True,
            backend=PoseBackendIdentity(mediapipe_version="test"),
            config=PoseExtractionConfig(), joints=tuple(joints),
        ))
    return tuple(frames)


def metadata(frames, fps=30.0, width=320, height=240):
    return VideoMetadata(video_id="synthetic.avi", source_path=Path("synthetic.avi"),
                         fps=fps, frame_count=len(frames), width=width, height=height,
                         decoder="synthetic")


def process(frames, fps=30.0, profile=None):
    return preprocess_pose(frames, evaluate_quality(frames), metadata(frames, fps), profile=profile)


def joint(result, i, name=J.NOSE):
    return next(j for j in result.frames[i].joints if j.joint == name)


def missing(indices, name=J.NOSE):
    return {(i, name): ("missing", None, None) for i in indices}


def test_raw_and_quality_are_unchanged_and_detached():
    raw = make_frames(changes=missing([5, 6]))
    quality = evaluate_quality(raw)
    before = [f.model_dump_json() for f in raw], quality.model_dump_json()
    result = preprocess_pose(raw, quality, metadata(raw))
    assert ([f.model_dump_json() for f in raw], quality.model_dump_json()) == before
    assert result.raw_frames == raw and result.quality == quality
    assert result.raw_frames[0] is not raw[0]
    assert result.quality.by_joint is not quality.by_joint
    assert [(f.video_id, f.frame_index, f.timestamp_seconds) for f in result.frames] == [
        (f.video_id, f.frame_index, f.timestamp_seconds) for f in raw]
    assert result.quality.sequence_status == quality.sequence_status


def test_short_gap_interpolation_coordinates_and_exact_endpoint_audit():
    raw = make_frames(changes=missing([5, 6]), trajectory=lambda t: 0.2 + 0.2 * t)
    result = process(raw)
    for i in (5, 6):
        point = joint(result, i)
        assert point.source_status == "missing" and point.status == "imputed"
        assert point.pre_normalization_x == pytest.approx(0.2 + 0.2 * i / 30)
        assert point.interpolation.left_frame_index == 4
        assert point.interpolation.right_frame_index == 7
        assert point.interpolation.left_timestamp_seconds == raw[4].timestamp_seconds
        assert point.interpolation.right_timestamp_seconds == raw[7].timestamp_seconds
        assert point.interpolation.left_status == point.interpolation.right_status == "observed"
    assert summarize_preprocessing(result)["imputed_points"] == 2


@pytest.mark.parametrize("indices", [range(5, 11), range(0, 3), range(28, 31)])
def test_long_head_and_tail_gaps_remain_null(indices):
    result = process(make_frames(changes=missing(indices)))
    for i in indices:
        point = joint(result, i)
        assert point.status == "missing" and point.interpolation is None
        assert point.x is point.y is point.pre_normalization_x is None


def test_exact_gap_limit_and_zero_gap_profile():
    raw = make_frames(changes=missing(range(5, 10)))
    assert all(joint(process(raw), i).status == "imputed" for i in range(5, 10))
    result = process(raw, profile=PreprocessingProfile(max_gap_seconds=0.0))
    assert all(joint(result, i).status == "missing" for i in range(5, 10))


def test_low_visibility_barrier_and_explicit_policy():
    changes = {**missing([6]), (5, J.NOSE): ("low_visibility", 0.5, 0.4)}
    raw = make_frames(changes=changes)
    result = process(raw)
    assert joint(result, 5).status == "low_visibility"
    assert not joint(result, 5).usable and joint(result, 5).x is None
    assert joint(result, 6).status == "missing"  # left endpoint is invalid
    permitted = process(raw, profile=PreprocessingProfile(interpolate_low_visibility=True))
    assert joint(permitted, 5).status == "imputed"
    assert joint(permitted, 5).source_status == "low_visibility"
    assert joint(permitted, 5).interpolation.left_frame_index == 4
    assert joint(permitted, 6).interpolation.right_frame_index == 7
    assert joint(permitted, 5).status != "observed"


def test_low_visibility_is_never_an_endpoint_even_when_enabled():
    raw = make_frames(changes={(0, J.NOSE): ("low_visibility", 0.5, 0.4), **missing([1])})
    result = process(raw, profile=PreprocessingProfile(interpolate_low_visibility=True))
    assert joint(result, 0).status == "low_visibility"
    assert joint(result, 1).status == "missing"


def test_out_of_frame_observation_is_unavailable_not_an_endpoint():
    raw = make_frames(changes={(5, J.NOSE): ("observed", 1.2, 0.4), **missing([6])})
    result = process(raw)
    assert joint(result, 5).status == "observed"  # native raw provenance remains
    assert not joint(result, 5).usable
    assert joint(result, 5).unavailable_reason == FlagCode.OUT_OF_FRAME
    assert joint(result, 6).status == "missing"


def test_midhip_and_isotropic_torso_normalization():
    result = process(make_frames(n=3))
    transform = result.frames[0].normalization
    assert transform.translation_status == "mid_hip" and transform.scale_status == "torso"
    assert transform.origin_x == pytest.approx(0.5)
    assert transform.origin_y == pytest.approx(0.7 * 240 / 320)
    assert transform.torso_scale == pytest.approx(0.4 * 240 / 320)
    assert transform.distance_ratio_eligible
    assert joint(result, 0).unsmoothed_y == pytest.approx(-0.75)
    left, right = joint(result, 0, J.LEFT_HIP), joint(result, 0, J.RIGHT_HIP)
    assert (left.unsmoothed_x + right.unsmoothed_x) / 2 == pytest.approx(0)
    assert left.unsmoothed_y == pytest.approx(0)


def test_missing_hip_fallback_has_null_origin_safe_coordinates_and_flags():
    result = process(make_frames(n=3, changes=missing(range(3), J.LEFT_HIP)))
    transform = result.frames[0].normalization
    assert transform.translation_status == "fallback_origin"
    assert transform.origin_x is transform.origin_y is transform.torso_scale is None
    assert transform.scale_status == "fallback_unit" and not transform.distance_ratio_eligible
    point = joint(result, 0)
    assert point.unsmoothed_x == point.pre_normalization_x
    assert point.unsmoothed_y == point.pre_normalization_y
    codes = {f.code for f in result.frames[0].flags}
    assert {FlagCode.NORMALIZATION_ORIGIN_FALLBACK, FlagCode.NORMALIZATION_SCALE_FALLBACK} <= codes


@pytest.mark.parametrize("cause", ["missing", "zero", "near_zero"])
def test_torso_fallback_retains_translation_and_disallows_distance(cause):
    changes = missing(range(3), J.LEFT_SHOULDER) if cause == "missing" else {
        (i, name): ("observed", 0.4 if name == J.LEFT_SHOULDER else 0.6,
                    0.7 if cause == "zero" else 0.7 - 1e-10)
        for i in range(3) for name in (J.LEFT_SHOULDER, J.RIGHT_SHOULDER)
    }
    result = process(make_frames(n=3, changes=changes))
    transform = result.frames[0].normalization
    assert transform.translation_status == "mid_hip"
    assert transform.scale_status == "fallback_unit" and transform.torso_scale is None
    assert not transform.distance_ratio_eligible
    assert FlagCode.NORMALIZATION_SCALE_FALLBACK in {f.code for f in result.frames[0].flags}
    assert joint(result, 0).unsmoothed_y == pytest.approx((0.4 - 0.7) * 0.75)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scale_guard(value):
    assert _torso_scale((0.5, 0.5), (value, 0.3), 1e-8) is None


def test_imputed_hips_can_supply_normalization_but_keep_source_audit():
    result = process(make_frames(changes=missing([5], J.LEFT_HIP)))
    assert joint(result, 5, J.LEFT_HIP).status == "imputed"
    assert result.frames[5].normalization.translation_status == "mid_hip"
    assert result.frames[5].normalization.distance_ratio_eligible


def test_smoothing_does_not_cross_long_gap_or_unit_transition():
    gap = range(15, 21)
    first = process(make_frames(n=41, changes=missing(gap), trajectory=lambda t: 0.4 + 0.01 * math.sin(70 * t)))
    second = process(make_frames(n=41, changes=missing(gap), trajectory=lambda t: (0.9 if t >= 21 / 30 else 0.4 + 0.01 * math.sin(70 * t))))
    assert [joint(first, i).x for i in range(15)] == [joint(second, i).x for i in range(15)]
    assert all(joint(first, i).x is None for i in gap)
    runs = [r for r in first.smoothing_runs if r.joint == J.NOSE]
    assert [(r.start_frame, r.end_frame) for r in runs] == [(0, 14), (21, 40)]
    assert all(r.status == "smoothed" for r in runs)
    assert any(abs(joint(first, i).x - joint(first, i).unsmoothed_x) > 1e-5 for i in range(15))
    transition = process(make_frames(n=31, changes=missing(range(12, 18), J.LEFT_HIP)))
    nose_runs = [r for r in transition.smoothing_runs if r.joint == J.NOSE]
    assert [(r.start_frame, r.end_frame) for r in nose_runs] == [(0, 11), (12, 17), (18, 30)]


def test_short_run_is_preserved_and_status_is_explicit():
    result = process(make_frames(n=4))
    assert all(j.smoothing_status == "skipped_short_run" and j.x == j.unsmoothed_x
               for f in result.frames for j in f.joints)
    assert any(f.code == FlagCode.SMOOTHING_SHORT_RUN for f in result.frames[0].flags)


def test_profile_time_conversion_and_synthetic_30_60_consistency():
    profile = load_preprocessing_profile(ROOT / "configs/preprocessing_s3_v1.toml")
    assert profile == PreprocessingProfile()
    assert profile.resolve(30).max_gap_frames == 5
    assert profile.resolve(60).max_gap_frames == 10
    assert profile.resolve(30).smoothing_window_frames == 11
    assert profile.resolve(60).smoothing_window_frames == 21
    assert profile.resolve(30).actual_smoothing_span_seconds == profile.resolve(60).actual_smoothing_span_seconds
    # SG reproduces a quadratic signal, including edge polynomial fits. Sample
    # the same physical instants; this does not assert universal resampling equivalence.
    trajectory = lambda t: 0.2 + 0.04 * t + 0.03 * t * t
    a = process(make_frames(n=61, fps=30.0, trajectory=trajectory))
    b = process(make_frames(n=121, fps=60.0, trajectory=trajectory), fps=60.0)
    assert [joint(a, i).x for i in range(61)] == pytest.approx([joint(b, 2 * i).x for i in range(61)], abs=1e-12)
    # Same 1/6 s missing interval is accepted at either sampling rate.
    a = process(make_frames(changes=missing(range(6, 11))))
    b = process(make_frames(n=61, fps=60.0, changes=missing(range(12, 22))), fps=60.0)
    assert sum(j.status == "imputed" for f in a.frames for j in f.joints) == 5
    assert sum(j.status == "imputed" for f in b.frames for j in f.joints) == 10


def test_json_roundtrip_preserves_raw_flags_provenance_and_timestamps(tmp_path):
    raw = make_frames(changes={**missing([5]), **missing(range(10, 20), J.LEFT_HIP)})
    result = process(raw)
    target = tmp_path / "processed.json"
    write_preprocessing_result(result, target)
    restored = read_preprocessing_result(target)
    assert restored == result
    assert json.loads(target.read_text())["frames"][5]["joints"][0]["status"] == "imputed"
    with pytest.raises(GymError) as caught:
        write_preprocessing_result(result, target)
    assert caught.value.flag.code == FlagCode.PREPROCESSING_SERIALIZATION_ERROR


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_unsafe_nonfinite_input_is_structured(value):
    raw = make_frames(n=3)
    bad = raw[0].joints[0].model_copy(update={"x": value})
    unsafe = (raw[0].model_copy(update={"joints": (bad, *raw[0].joints[1:])}), *raw[1:])
    with pytest.raises(GymError) as caught:
        preprocess_pose(unsafe, evaluate_quality(raw), metadata(raw))
    assert caught.value.flag.code == FlagCode.INVALID_PREPROCESSING_INPUT
    assert caught.value.flag.stage == Stage.PREPROCESSING


@pytest.mark.parametrize("kind", ["reverse", "duplicate", "timestamp", "video_id", "fps", "count"])
def test_invalid_sequence_is_structured(kind):
    raw = make_frames(n=3)
    meta = metadata(raw)
    quality = evaluate_quality(raw)
    if kind == "reverse":
        raw = tuple(reversed(raw))
    elif kind == "duplicate":
        raw = (raw[0], raw[0], raw[2])
    elif kind == "timestamp":
        raw = (raw[0], raw[1].model_copy(update={"timestamp_seconds": 0.07}), raw[2])
    elif kind == "video_id":
        raw = (raw[0].model_copy(update={"video_id": "another"}), *raw[1:])
    elif kind == "fps":
        meta = meta.model_copy(update={"fps": 60.0})
    else:
        meta = meta.model_copy(update={"frame_count": 4})
    with pytest.raises(GymError) as caught:
        preprocess_pose(raw, quality, meta)
    assert caught.value.flag.code == FlagCode.INVALID_PREPROCESSING_INPUT


def test_quality_from_different_pose_sequence_is_rejected():
    raw = make_frames()
    another = make_frames(changes=missing([5]))
    with pytest.raises(GymError) as caught:
        preprocess_pose(raw, evaluate_quality(another), metadata(raw))
    assert caught.value.flag.code == FlagCode.QUALITY_INPUT_MISMATCH


def test_custom_metric_quality_decisions_are_preserved():
    raw = make_frames()
    quality = evaluate_quality(raw, metric_requirements={"custom": (J.NOSE,)},
                               metric_viewpoints={"custom": ()})
    result = preprocess_pose(raw, quality, metadata(raw))
    assert result.quality == quality


def test_empty_and_no_pose_sequences_do_not_crash():
    result = process(())
    assert result.frames == () and summarize_preprocessing(result)["usable_coverage_after"] is None
    raw = make_frames(n=3)
    no_pose = tuple(f.model_copy(update={"has_pose": False, "joints": tuple(
        JointObservation(joint=j.joint, status="missing", missing_reason=FlagCode.POSE_NOT_DETECTED)
        for j in f.joints)}) for f in raw)
    result = process(no_pose)
    assert all(j.status == "missing" and j.x is None for f in result.frames for j in f.joints)
    assert all(f.normalization.scale_status == "fallback_unit" for f in result.frames)


def test_smoothing_failure_is_structured(monkeypatch):
    import scipy.signal
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic SG failure")
    monkeypatch.setattr(scipy.signal, "savgol_filter", fail)
    with pytest.raises(GymError) as caught:
        process(make_frames())
    assert caught.value.flag.code == FlagCode.PREPROCESSING_ERROR


@pytest.mark.parametrize("text", ['max_gap_seconds = -1.0', 'smoothing_polyorder = -1',
                                   'unknown_key = 1', 'profile_schema_version = "9.0.0"', 'bad toml !'])
def test_profile_invalid_is_structured(tmp_path, text):
    path = tmp_path / "profile.toml"
    path.write_text(text)
    with pytest.raises(GymError) as caught:
        load_preprocessing_profile(path)
    assert caught.value.flag.code == FlagCode.INVALID_CONFIG


def test_artifact_tampering_and_bad_json_are_rejected(tmp_path):
    result = process(make_frames(n=3))
    data = result.model_dump(mode="json")
    data["profile_hash"] = "a" * 64
    with pytest.raises(ValidationError):
        PreprocessingResult.model_validate(data)
    path = tmp_path / "bad.json"
    path.write_text("{bad}")
    with pytest.raises(GymError) as caught:
        read_preprocessing_result(path)
    assert caught.value.flag.code == FlagCode.PREPROCESSING_SERIALIZATION_ERROR


@pytest.mark.parametrize("kind", ["endpoint_time", "endpoint_index", "provenance", "scale_validity", "run_gap", "missing_run", "raw_order"])
def test_deserialization_rejects_broken_audit_links(kind):
    result = process(make_frames(changes=missing([5, 6])))
    data = result.model_dump(mode="json")
    if kind == "endpoint_time":
        data["frames"][5]["joints"][0]["interpolation"]["left_timestamp_seconds"] = 0.0
    elif kind == "endpoint_index":
        data["frames"][5]["joints"][0]["interpolation"]["right_frame_index"] = 500
    elif kind == "provenance":
        data["frames"][5]["joints"][0]["source_status"] = "low_visibility"
    elif kind == "scale_validity":
        data["frames"][0]["normalization"]["distance_ratio_eligible"] = False
    elif kind == "run_gap":
        data["smoothing_runs"][0]["end_frame"] = 1000
    elif kind == "missing_run":
        data["smoothing_runs"] = []
    else:
        data["raw_frames"].reverse()
    with pytest.raises(ValidationError):
        PreprocessingResult.model_validate(data)


@pytest.mark.parametrize("field", ["x", "pre_normalization_y", "unsmoothed_x"])
def test_processed_schema_rejects_nonfinite_and_fake_coordinates(field):
    result = process(make_frames(n=3, changes=missing(range(3))))
    data = joint(result, 0).model_dump(mode="json")
    from gym_coach.preprocessing import ProcessedJoint
    for value in (float("nan"), float("inf"), 0.0):
        with pytest.raises(ValidationError):
            ProcessedJoint.model_validate({**data, field: value})


def test_smoothing_nonfinite_output_is_structured(monkeypatch):
    import scipy.signal
    monkeypatch.setattr(scipy.signal, "savgol_filter", lambda points, *a, **k: [(float("nan"), 0.0)] * len(points))
    with pytest.raises(GymError) as caught:
        process(make_frames())
    assert caught.value.flag.code == FlagCode.PREPROCESSING_ERROR


def test_sinusoidal_signal_30_60fps_is_close_at_common_instants():
    trajectory = lambda t: 0.5 + 0.02 * math.sin(2 * math.pi * t)
    a = process(make_frames(n=61, fps=30.0, trajectory=trajectory))
    b = process(make_frames(n=121, fps=60.0, trajectory=trajectory), fps=60.0)
    assert [joint(a, i).x for i in range(61)] == pytest.approx([joint(b, 2 * i).x for i in range(61)], abs=0.0005)


def test_cli_bad_profile_returns_structured_json(tmp_path, capsys):
    from gym_coach.preprocessing.cli import main
    code = main(["unopened.avi", "--exercise", "BodyWeightSquats", "--profile",
                 str(tmp_path / "absent.toml"), "--output", str(tmp_path / "unused.json")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["flag"]["code"] == "invalid_config" and payload["ok"] is False


def test_missing_artifact_read_is_structured(tmp_path):
    with pytest.raises(GymError) as caught:
        read_preprocessing_result(tmp_path / "absent.json")
    assert caught.value.flag.code == FlagCode.PREPROCESSING_SERIALIZATION_ERROR
