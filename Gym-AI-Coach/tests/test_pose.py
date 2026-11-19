"""Mapping, provenance, adapter operational boundaries and coverage contracts."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from gym_coach.contracts import ErrorCategory, Exercise, JointProvenance
from gym_coach.flags import FlagCode, GymError
from gym_coach.pose import (
    JOINT_INDEX, JOINT_MAPPING_VERSION, JointName, JointObservation,
    MediaPipePoseExtractor, PoseBackendIdentity, PoseCoverageAccumulator,
    PoseExtractionConfig, PoseFrame, PoseSequenceSummary, read_pose_frames, write_pose_frames,
)
from gym_coach.pose import extractor as adapter_module
from gym_coach.video import VideoFrame


class FakeBackend:
    def __init__(self, results):
        self.results = iter(results)
        self.inputs = []
        self.closed = False

    def process(self, image):
        self.inputs.append(image.copy())
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def landmarks():
    return [SimpleNamespace(x=0.25, y=0.75, visibility=0.9) for _ in range(33)]


def result(points):
    return SimpleNamespace(pose_landmarks=None if points is None else SimpleNamespace(landmark=points))


def fake_extractor(results, **kwargs):
    backend = FakeBackend(results)
    identity = PoseBackendIdentity(mediapipe_version="fake-unit-backend")
    instance = MediaPipePoseExtractor(
        "synthetic.avi", Exercise.BODY_WEIGHT_SQUATS,
        backend_factory=lambda config: (identity, backend), **kwargs,
    )
    return instance, backend


def video_frame(index=0, timestamp=0.0):
    return VideoFrame(index, timestamp, np.full((8, 12, 3), [1, 2, 3], dtype=np.uint8))


def test_mapping_known_indices_unique_and_immutable():
    assert JOINT_MAPPING_VERSION == "mediapipe-pose-13-v1"
    assert len(JOINT_INDEX) == len(set(JOINT_INDEX.values())) == 13
    assert tuple(JOINT_INDEX.values()) == (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    assert JOINT_INDEX[JointName.LEFT_HIP] == 23
    with pytest.raises(TypeError):
        JOINT_INDEX[JointName.NOSE] = 1


@pytest.mark.parametrize("field,value", [
    ("x", math.nan), ("x", math.inf), ("y", -math.inf),
    ("visibility", math.nan), ("visibility", -0.1), ("visibility", 1.1),
    ("presence", math.inf), ("presence", -0.1), ("presence", 1.1),
])
def test_schema_rejects_invalid_numbers(field, value):
    data = dict(joint="nose", x=0.2, y=0.3, visibility=0.9, status="observed")
    data[field] = value
    with pytest.raises(ValidationError):
        JointObservation(**data)


@pytest.mark.parametrize("changes", [
    {"visibility": 0.2, "status": "observed"},
    {"visibility": 0.9, "status": "low_visibility"},
    {"status": "missing", "missing_reason": "missing_joint"},
    {"status": "imputed"},
])
def test_schema_rejects_false_provenance(changes):
    base = dict(joint="nose", x=0.2, y=0.3, visibility=0.9, status="observed")
    with pytest.raises(ValidationError):
        JointObservation(**(base | changes))


def test_low_visibility_keeps_native_coords_presence_and_exact_frame_identity():
    points = landmarks()
    points[0] = SimpleNamespace(x=0.123456789, y=0.4, visibility=0.49, presence=0.8)
    extractor, backend = fake_extractor([result(points)])
    source = video_frame(17, 17 / 29.97)
    before = source.image.copy()
    with extractor:
        frame = extractor.extract(source)
    nose = frame.joints[0]
    assert (frame.video_id, frame.frame_index, frame.timestamp_seconds) == (
        "synthetic.avi", source.frame_index, source.timestamp_seconds
    )
    assert nose.status == JointProvenance.LOW_VISIBILITY
    assert (nose.x, nose.y, nose.visibility, nose.presence) == (0.123456789, 0.4, 0.49, 0.8)
    assert all(j.presence is None for j in frame.joints[1:])
    assert frame.flags[0].code == FlagCode.LOW_POSE_VISIBILITY
    np.testing.assert_array_equal(source.image, before)
    np.testing.assert_array_equal(backend.inputs[0], before[..., ::-1])
    assert backend.closed


def test_threshold_boundary_and_custom_config_are_preserved():
    points = landmarks()
    points[0].visibility = 0.5
    extractor, _ = fake_extractor([result(points)])
    with extractor:
        assert extractor.extract(video_frame()).joints[0].status == JointProvenance.OBSERVED
    config = PoseExtractionConfig(visibility_threshold=0.8)
    points[0].visibility = 0.7
    extractor, _ = fake_extractor([result(points)], config=config)
    with extractor:
        pose = extractor.extract(video_frame())
    assert pose.joints[0].status == JointProvenance.LOW_VISIBILITY
    assert pose.joints[0].visibility_threshold == 0.8
    assert pose.config == config


@pytest.mark.parametrize("change,reason", [
    ({"x": math.nan}, FlagCode.NON_FINITE_LANDMARK),
    ({"y": math.inf}, FlagCode.NON_FINITE_LANDMARK),
    ({"visibility": 2.0}, FlagCode.INVALID_LANDMARK_CONFIDENCE),
    ({"presence": -0.1}, FlagCode.INVALID_LANDMARK_CONFIDENCE),
    ({"visibility": None}, FlagCode.MISSING_LANDMARK_FIELD),
])
def test_invalid_backend_joint_is_missing_and_flagged(change, reason):
    points = landmarks()
    points[0] = SimpleNamespace(**(vars(points[0]) | change))
    extractor, _ = fake_extractor([result(points)])
    with extractor:
        frame = extractor.extract(video_frame())
    assert frame.has_pose
    assert frame.joints[0].status == JointProvenance.MISSING
    assert frame.joints[0].x is None and frame.joints[0].y is None
    assert frame.joints[0].missing_reason == reason
    assert frame.flags[0].code == reason
    assert frame.flags[0].location.joint == "nose"


def test_short_landmark_list_preserves_missing_joints_without_zeros():
    extractor, _ = fake_extractor([result(landmarks()[:12])])
    with extractor:
        frame = extractor.extract(video_frame())
    assert len(frame.joints) == 13
    right_shoulder = next(j for j in frame.joints if j.joint == JointName.RIGHT_SHOULDER)
    assert right_shoulder.status == JointProvenance.MISSING
    assert right_shoulder.x is right_shoulder.y is None
    assert right_shoulder.missing_reason == FlagCode.MISSING_JOINT


def test_protobuf_unset_presence_and_coordinates_are_not_fake_zero():
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
    points = landmarks()
    points[0] = NormalizedLandmark(x=0.1, y=0.2, visibility=0.9)
    points[11] = NormalizedLandmark(y=0.2, visibility=0.9)
    extractor, _ = fake_extractor([result(points)])
    with extractor:
        frame = extractor.extract(video_frame())
    assert frame.joints[0].presence is None
    shoulder = next(j for j in frame.joints if j.joint == JointName.LEFT_SHOULDER)
    assert shoulder.status == JointProvenance.MISSING
    assert shoulder.x is None


def test_no_pose_and_jsonl_roundtrip(tmp_path):
    points = landmarks()
    points[0].visibility = 0.2
    extractor, _ = fake_extractor([result(None), result(points)])
    with extractor:
        frames = list(extractor.extract_frames([video_frame(), video_frame(1, 0.1)]))
    empty = frames[0]
    assert not empty.has_pose
    assert all(j.status == JointProvenance.MISSING and j.x is None and j.y is None for j in empty.joints)
    assert empty.flags[0].code == FlagCode.POSE_NOT_DETECTED
    path = tmp_path / "frames.jsonl"
    assert write_pose_frames(path, frames) == 2
    assert list(read_pose_frames(path)) == frames
    assert list(read_pose_frames(path))[1].joints[0].status == JointProvenance.LOW_VISIBILITY
    assert PoseFrame.model_validate_json(empty.model_dump_json()) == empty
    with pytest.raises(GymError) as duplicate:
        write_pose_frames(path, frames)
    assert duplicate.value.flag.code == FlagCode.POSE_SERIALIZATION_ERROR


def test_malformed_json_is_structured(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"pose_schema_version":"99.0.0"}\n', encoding="utf-8")
    with pytest.raises(GymError) as error:
        list(read_pose_frames(path))
    assert error.value.flag.code == FlagCode.POSE_SERIALIZATION_ERROR
    assert error.value.flag.evidence["line_number"] == 1


def test_coverage_on_synthetic_and_empty_sequence():
    points = landmarks()
    points[0].visibility = 0.1
    extractor, _ = fake_extractor([result(landmarks()), result(points), result(None)])
    counts = PoseCoverageAccumulator(extractor.video_id, extractor.exercise, extractor.identity, extractor.config)
    empty = counts.summary()
    assert empty.total_frames == 0 and empty.pose_coverage is None
    assert empty.flags[0].code == FlagCode.INSUFFICIENT_SEQUENCE_LENGTH
    with extractor:
        for pose in extractor.extract_frames([video_frame(i, i / 10) for i in range(3)]):
            counts.add(pose)
    summary = counts.summary()
    assert (summary.total_frames, summary.frames_with_pose, summary.frames_without_pose) == (3, 2, 1)
    assert summary.pose_detection_ratio == 2 / 3
    assert summary.by_joint[JointName.NOSE].model_dump() == dict(
        observed=1, low_visibility=1, missing=1, pose_coverage=1 / 3,
    )
    assert summary.pose_coverage == 25 / 39
    assert summary.required_joint_coverage == 2 / 3
    assert PoseSequenceSummary.model_validate_json(summary.model_dump_json()) == summary


@pytest.mark.parametrize("error", [RuntimeError("graph failed"), ValueError("backend failed"), OSError("model failure")])
def test_backend_exception_is_operational_and_keeps_cause(error):
    extractor, backend = fake_extractor([error])
    with pytest.raises(GymError) as caught, extractor:
        extractor.extract(video_frame(9, 0.9))
    assert caught.value.flag.code == FlagCode.POSE_BACKEND_ERROR
    assert caught.value.flag.category == ErrorCategory.OPERATIONAL_ERROR
    assert caught.value.flag.location.frame_start == 9
    assert caught.value.__cause__ is error
    assert backend.closed


def test_backend_initialization_exception_is_structured():
    def fail(config):
        raise ImportError("dependency missing")
    with pytest.raises(GymError) as error:
        MediaPipePoseExtractor("x", "HandstandPushups", backend_factory=fail)
    assert error.value.flag.code == FlagCode.POSE_BACKEND_ERROR
    assert error.value.flag.evidence["operation"] == "initialize"


@pytest.mark.parametrize("image", [None, np.zeros((5, 5), dtype=np.uint8),
                                    np.zeros((5, 5, 3), dtype=np.float32),
                                    np.zeros((0, 5, 3), dtype=np.uint8)])
def test_invalid_input_image_is_structured(image):
    extractor, _ = fake_extractor([])
    # VideoFrame itself rejects None; exercise invalid image types at the adapter boundary.
    source = object() if image is None else VideoFrame(0, 0.0, image)
    with pytest.raises(GymError) as error, extractor:
        extractor.extract(source)
    assert error.value.flag.code == FlagCode.INVALID_POSE_INPUT


def test_unordered_sequence_and_mixed_coverage_identity_rejected():
    extractor, _ = fake_extractor([result(landmarks())])
    with extractor:
        frame = extractor.extract(video_frame(2, 0.2))
        with pytest.raises(GymError) as error:
            extractor.extract(video_frame(1, 0.1))
        assert error.value.flag.code == FlagCode.INVALID_POSE_SEQUENCE
    accumulator = PoseCoverageAccumulator("other", extractor.exercise, extractor.identity, extractor.config)
    with pytest.raises(GymError):
        accumulator.add(frame)
    matching = PoseCoverageAccumulator(extractor.video_id, extractor.exercise, extractor.identity, extractor.config)
    matching.add(frame)
    with pytest.raises(GymError) as duplicate:
        matching.add(frame)
    assert duplicate.value.flag.code == FlagCode.INVALID_POSE_SEQUENCE


def test_out_of_frame_is_flagged_without_clipping_coordinates():
    points = landmarks()
    points[0].x = -0.2
    extractor, _ = fake_extractor([result(points)])
    with extractor:
        frame = extractor.extract(video_frame())
    assert frame.joints[0].x == -0.2
    assert frame.flags[0].code == FlagCode.OUT_OF_FRAME


def test_production_backend_passes_disabled_smoothing_and_uses_bundled_model(monkeypatch, tmp_path):
    asset = tmp_path / "modules/pose_landmark/pose_landmark_full.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"unit-only-model-asset")
    passed = {}
    def create(**kwargs):
        passed.update(kwargs)
        return FakeBackend([])
    import sys
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        __version__="0.10.14", __file__=str(tmp_path / "__init__.py"),
        solutions=SimpleNamespace(pose=SimpleNamespace(Pose=create)),
    ))
    identity, backend = adapter_module._create_backend(PoseExtractionConfig())
    assert passed["smooth_landmarks"] is False
    assert passed["smooth_segmentation"] is False
    assert passed["enable_segmentation"] is False
    assert passed["model_complexity"] == 1
    assert identity.mediapipe_version == "0.10.14"
    assert identity.model_asset_sha256 is not None
    backend.close()


def test_missing_asset_never_attempts_model_download(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        __version__="0.10.14", __file__=str(tmp_path / "__init__.py"),
    ))
    with pytest.raises(GymError) as error:
        MediaPipePoseExtractor("x", "BodyWeightSquats")
    assert error.value.flag.code == FlagCode.POSE_BACKEND_ERROR
    assert "automatic download is disabled" in error.value.flag.evidence["detail"]


def test_schema_joint_duplicates_frame_numbers_and_unknown_version_rejected():
    extractor, _ = fake_extractor([result(landmarks())])
    with extractor:
        pose = extractor.extract(video_frame())
    base = pose.model_dump()
    for changes in (
        {"joints": (pose.joints[0],) * 13}, {"timestamp_seconds": math.nan},
        {"frame_index": -1}, {"frame_index": True}, {"pose_schema_version": "9.0.0"},
    ):
        with pytest.raises(ValidationError):
            PoseFrame.model_validate(base | changes)


@pytest.mark.parametrize("malformed", [object(), SimpleNamespace(pose_landmarks=object())])
def test_malformed_backend_result_is_operational(malformed):
    extractor, _ = fake_extractor([malformed])
    with pytest.raises(GymError) as error, extractor:
        extractor.extract(video_frame())
    assert error.value.flag.code == FlagCode.POSE_BACKEND_ERROR
    assert error.value.flag.category == ErrorCategory.OPERATIONAL_ERROR


def test_backend_close_exception_is_operational_and_closed_use_is_rejected():
    extractor, backend = fake_extractor([])
    def fail_close():
        raise RuntimeError("cleanup failed")
    backend.close = fail_close
    with pytest.raises(GymError) as error:
        extractor.close()
    assert error.value.flag.evidence["operation"] == "close"
    with pytest.raises(GymError) as closed:
        extractor.extract(video_frame())
    assert closed.value.flag.code == FlagCode.POSE_BACKEND_ERROR


@pytest.mark.parametrize("change", [
    {"smooth_landmarks": True}, {"enable_segmentation": True}, {"model_complexity": 0},
])
def test_config_cannot_enable_processing_outside_stage_two(change):
    with pytest.raises(ValidationError):
        PoseExtractionConfig(**change)


def test_pose_cli_unreadable_video_outputs_structured_json(tmp_path, capsys):
    from gym_coach.pose.cli import main
    import json
    assert main([str(tmp_path / "missing.avi"), "--exercise", "BodyWeightSquats",
                 "--output-dir", str(tmp_path / "new")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["error"]["code"] == "unreadable_video"
    assert not (tmp_path / "new").exists()
