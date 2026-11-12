"""Full real-video integration. Saves auditable outputs; no accuracy claim."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from gym_coach.contracts import Exercise
from gym_coach.pose import (JOINT_INDEX, JointName, PoseCoverageAccumulator,
                            PoseSequenceSummary, read_pose_frames)
from gym_coach.pose.run import extract_video, file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def integration_artifacts():
    destination = ROOT / "outputs/stage2" / ("integration-" + uuid4().hex)
    destination.mkdir(parents=True)
    return destination


@pytest.mark.integration
@pytest.mark.parametrize("exercise,relative", [
    (Exercise.BODY_WEIGHT_SQUATS, "Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi"),
    (Exercise.HANDSTAND_PUSHUPS, "Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi"),
])
def test_complete_real_video_pose_and_saved_provenance(exercise, relative, integration_artifacts, monkeypatch):
    import mediapipe as mp
    # Verify all mapping entries against the actual installed backend enum.
    for joint, index in JOINT_INDEX.items():
        assert getattr(mp.solutions.pose.PoseLandmark, joint.name).value == index
    monkeypatch.chdir(ROOT)
    assert Path(relative).is_file(), f"Required development fixture missing: {relative}"
    destination = integration_artifacts / exercise.value
    summary = extract_video(relative, exercise, destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_path"] == relative
    assert manifest["completed"] is True
    assert summary.total_frames == manifest["metadata"]["frame_count"] > 0
    assert summary.frames_with_pose + summary.frames_without_pose == summary.total_frames
    assert summary.backend.mediapipe_version == "0.10.14"
    assert summary.backend.subject_selection_policy == "backend_most_prominent_single_subject"
    assert summary.backend.model_asset_sha256 is not None
    assert summary.config.smooth_landmarks is False
    assert summary.config.enable_segmentation is False
    restored = PoseSequenceSummary.model_validate_json((destination / "summary.json").read_text(encoding="utf-8"))
    assert restored == summary
    accumulator = PoseCoverageAccumulator(summary.video_id, exercise, summary.backend, summary.config)
    low = {joint.value: [] for joint in JointName}
    missing = {joint.value: [] for joint in JointName}
    observed_count = 0
    for index, frame in enumerate(read_pose_frames(destination / "frames.jsonl")):
        assert frame.video_id == manifest["metadata"]["video_id"]
        assert frame.frame_index == index
        assert frame.timestamp_seconds == index / manifest["metadata"]["fps"]
        assert len(frame.joints) == 13
        assert all(j.presence is None for j in frame.joints)  # This legacy backend clears presence.
        for joint in frame.joints:
            if joint.status.value == "observed":
                assert joint.visibility >= frame.config.visibility_threshold
                observed_count += 1
            elif joint.status.value == "low_visibility":
                assert joint.visibility < frame.config.visibility_threshold
                low[joint.joint.value].append(index)
            else:
                assert joint.x is None and joint.y is None
                missing[joint.joint.value].append(index)
        accumulator.add(frame)
    assert accumulator.summary() == summary
    assert summary.pose_coverage == observed_count / (13 * summary.total_frames)
    for joint in JointName:
        assert manifest["joint_issues"][joint.value]["low_visibility_frames"] == low[joint.value]
        assert manifest["joint_issues"][joint.value]["missing_frames"] == missing[joint.value]
    assert file_sha256(destination / "frames.jsonl") == manifest["pose_frames_sha256"]
    print(f"artifact={destination.relative_to(ROOT).as_posix()} frames={summary.total_frames} coverage={summary.pose_coverage}")
