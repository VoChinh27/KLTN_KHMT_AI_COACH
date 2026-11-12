"""Phase 3 quality gate tests using controlled pose contracts."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.contracts import (
    ErrorCategory,
    Exercise,
    JointProvenance,
    SourceTier,
    Viewpoint,
)
from gym_coach.flags import FlagCode, GymError
from gym_coach.pose import (
    JointObservation,
    JointName,
    PoseBackendIdentity,
    PoseExtractionConfig,
    PoseFrame,
    REQUIRED_JOINTS,
    read_pose_frames,
)
from gym_coach.quality import JointGapSpan, QualityProfile, evaluate_quality

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56"


def make_frame(
    index: int,
    *,
    exercise: Exercise = Exercise.BODY_WEIGHT_SQUATS,
    viewpoint: Viewpoint = Viewpoint.SIDE,
    invalid: dict[JointName, JointProvenance] | None = None,
    out_of_frame: tuple[JointName, ...] = (),
) -> PoseFrame:
    config = PoseExtractionConfig()
    backend = PoseBackendIdentity(mediapipe_version="synthetic")
    invalid = invalid or {}
    joints = []
    for joint in JointName:
        status = invalid.get(joint, JointProvenance.OBSERVED)
        if status is JointProvenance.MISSING:
            joints.append(JointObservation(
                joint=joint,
                status=status,
                missing_reason=FlagCode.MISSING_JOINT,
                visibility_threshold=config.visibility_threshold,
            ))
        else:
            visibility = 0.2 if status is JointProvenance.LOW_VISIBILITY else 0.9
            x = 1.2 if joint in out_of_frame else 0.5
            joints.append(JointObservation(
                joint=joint, x=x, y=0.5, visibility=visibility,
                status=status, visibility_threshold=config.visibility_threshold,
            ))
    return PoseFrame(
        video_id="synthetic", exercise=exercise, viewpoint=viewpoint,
        frame_index=index, timestamp_seconds=index / 30.0, has_pose=True,
        backend=backend, config=config, joints=tuple(joints),
    )


def test_gap_span_inclusive_validation():
    assert JointGapSpan(start_frame=4, end_frame=8, length=5).length == 5
    with pytest.raises(ValidationError):
        JointGapSpan(start_frame=4, end_frame=8, length=4)


def test_empty_sequence_skips_with_structured_reason():
    report = evaluate_quality([])
    assert report.total_frames == 0
    assert report.sequence_status == "skipped"
    assert report.eligibility.action == "skip"
    assert FlagCode.INSUFFICIENT_SEQUENCE_LENGTH in {flag.code for flag in report.flags}
    assert all(item.observed_ratio is None for item in report.by_joint.values())
    assert json.loads(report.model_dump_json())["quality_schema_version"] == "0.1.0"
    assert type(report).model_validate_json(report.model_dump_json()) == report


def test_missing_gaps_at_both_edges_and_middle_are_reported():
    joint = JointName.RIGHT_KNEE
    frames = [
        make_frame(i, invalid={joint: JointProvenance.MISSING}
                   if i in {0, 1, 5, 6, 7, 9} else {})
        for i in range(10)
    ]
    report = evaluate_quality(frames)
    quality = report.by_joint[joint]
    assert [(gap.start_frame, gap.end_frame, gap.length)
            for gap in quality.consecutive_invalid_spans] == [(0, 1, 2), (5, 7, 3), (9, 9, 1)]
    assert quality.max_consecutive_invalid_frames == 3


def test_long_gap_skips_metric_and_records_reason():
    required = JointName.RIGHT_KNEE
    frames = [make_frame(i, invalid={required: JointProvenance.MISSING} if 3 <= i <= 13 else {})
              for i in range(20)]
    report = evaluate_quality(frames, metric_requirements={"right_knee_metric": [required]})
    eligibility = report.metric_eligibility["right_knee_metric"]
    assert eligibility.action == "skip"
    assert FlagCode.EXCESSIVE_MISSINGNESS.value in eligibility.reasons
    assert any(flag.code is FlagCode.EXCESSIVE_MISSINGNESS and
               flag.location.joint == required.value for flag in report.flags)


def test_low_visibility_gate_is_per_joint_metric_and_not_technical_error():
    joint = JointName.RIGHT_WRIST
    frames = [make_frame(i, exercise=Exercise.HANDSTAND_PUSHUPS,
                         invalid={joint: JointProvenance.LOW_VISIBILITY} if i < 4 else {})
              for i in range(10)]
    report = evaluate_quality(
        frames,
        metric_requirements={"wrist_metric": [joint], "shoulder_metric": [JointName.LEFT_SHOULDER]},
    )
    assert report.metric_eligibility["wrist_metric"].action == "skip"
    assert FlagCode.LOW_POSE_VISIBILITY.value in report.metric_eligibility["wrist_metric"].reasons
    assert report.metric_eligibility["shoulder_metric"].action == "allow"
    assert all(flag.category is not ErrorCategory.TECHNICAL_ERROR for flag in report.flags)


def test_small_invalid_ratio_degrades_only_affected_metric():
    joint = JointName.RIGHT_KNEE
    frames = [make_frame(i, invalid={joint: JointProvenance.MISSING} if i == 4 else {})
              for i in range(20)]
    report = evaluate_quality(
        frames,
        metric_requirements={"bilateral_knee": [JointName.LEFT_KNEE, joint],
                             "left_knee": [JointName.LEFT_KNEE]},
    )
    assert report.metric_eligibility["bilateral_knee"].action == "degrade"
    assert report.metric_eligibility["left_knee"].action == "allow"
    assert report.eligibility.action == "degrade"
    assert report.by_joint[joint].missing_frames == 1


def test_short_sequence_and_unknown_viewpoint_are_explicit():
    short = [make_frame(i) for i in range(4)]
    report = evaluate_quality(
        short,
        metric_requirements={"side_metric": [JointName.LEFT_KNEE]},
        metric_viewpoints={"side_metric": [Viewpoint.SIDE]},
    )
    assert report.metric_eligibility["side_metric"].action == "skip"
    assert FlagCode.INSUFFICIENT_SEQUENCE_LENGTH.value in report.metric_eligibility["side_metric"].reasons

    unknown = [make_frame(i, viewpoint=Viewpoint.UNKNOWN) for i in range(6)]
    report_unknown = evaluate_quality(
        unknown,
        metric_requirements={"side_metric": [JointName.LEFT_KNEE]},
        metric_viewpoints={"side_metric": [Viewpoint.SIDE]},
    )
    assert FlagCode.VIEWPOINT_UNKNOWN.value in report_unknown.metric_eligibility["side_metric"].reasons
    assert FlagCode.VIEWPOINT_UNKNOWN in {flag.code for flag in report_unknown.flags}

    wrong_view = [make_frame(i, viewpoint=Viewpoint.FRONT) for i in range(6)]
    mismatch = evaluate_quality(
        wrong_view,
        metric_requirements={"side_metric": [JointName.LEFT_KNEE]},
        metric_viewpoints={"side_metric": [Viewpoint.SIDE]},
    )
    assert FlagCode.VIEWPOINT_MISMATCH.value in mismatch.metric_eligibility["side_metric"].reasons


def test_out_of_frame_is_reported_and_gates_metric_when_prevalent():
    frames = [make_frame(i, out_of_frame=(JointName.NOSE,) if i == 2 else ()) for i in range(6)]
    report = evaluate_quality(frames, metric_requirements={"knee_metric": [JointName.LEFT_KNEE]})
    assert report.by_joint[JointName.NOSE].out_of_frame_frames == 1
    assert report.metric_eligibility["knee_metric"].action == "allow"
    assert any(flag.code is FlagCode.OUT_OF_FRAME and flag.location.joint == JointName.NOSE.value
               for flag in report.flags)

    required_joint = JointName.LEFT_WRIST
    mostly_oof = [make_frame(i, exercise=Exercise.HANDSTAND_PUSHUPS,
                             out_of_frame=(required_joint,) if i < 5 else ())
                  for i in range(6)]
    gated = evaluate_quality(
        mostly_oof,
        metric_requirements={"wrist_metric": [required_joint]},
    )
    assert gated.metric_eligibility["wrist_metric"].action == "skip"
    assert FlagCode.OUT_OF_FRAME.value in gated.metric_eligibility["wrist_metric"].reasons


def test_non_contiguous_sequence_is_rejected_with_structured_reason():
    frames = [make_frame(0), make_frame(1), make_frame(3)]
    with pytest.raises(GymError) as caught:
        evaluate_quality(frames)
    assert caught.value.flag.code is FlagCode.INVALID_POSE_SEQUENCE


@pytest.mark.parametrize("exercise,directory", [
    (Exercise.BODY_WEIGHT_SQUATS, "BodyWeightSquats"),
    (Exercise.HANDSTAND_PUSHUPS, "HandstandPushups"),
])
@pytest.mark.integration
def test_quality_gate_on_saved_real_pose_sequence(exercise: Exercise, directory: str):
    frames_path = STAGE2 / directory / "frames.jsonl"
    assert frames_path.is_file(), f"Stage 2 pose artifact missing: {frames_path}"
    frames = list(read_pose_frames(frames_path))
    report = evaluate_quality(frames)
    assert report.total_frames == len(frames)
    assert report.exercise is exercise
    assert sum(j.total_frames for j in report.by_joint.values()) == len(JointName) * len(frames)
    assert report.profile_hash == report.profile.content_hash()
    assert report.sequence_status in {"eligible", "degraded", "skipped"}
    assert report.eligibility.required_joints == tuple(REQUIRED_JOINTS[exercise])


@pytest.mark.integration
def test_quality_cli_emits_json_for_saved_pose_sequence():
    frames_path = STAGE2 / "BodyWeightSquats" / "frames.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "gym_coach.quality", str(frames_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["quality_report"]["total_frames"] == 267
    assert payload["quality_report"]["profile"]["source_tier"] == "S3"
