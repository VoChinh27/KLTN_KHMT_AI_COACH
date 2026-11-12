"""Phase 1 unit, CLI, and real-data video ingestion checks."""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from gym_coach.flags import FlagCode, GymError
from gym_coach.video import OpenCVVideoReader, TimestampPolicy, VideoFrame, VideoMetadata, inspect_video
from gym_coach.video import reader as reader_module

ROOT = Path(__file__).resolve().parents[1]
REAL_VIDEO = ROOT / "Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi"


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for value in (0, 60, 120, 180):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    assert path.is_file()
    return path


def test_metadata_contract_rejects_invalid_values(tmp_path: Path):
    base = dict(
        video_id="x.avi", source_path=tmp_path / "x.avi", fps=30.0,
        frame_count=1, width=10, height=10, decoder="test",
    )
    for changes in (
        {"fps": 0.0}, {"fps": math.inf}, {"frame_count": -1},
        {"frame_count": 1.5}, {"width": 0}, {"height": -1},
    ):
        with pytest.raises(ValidationError):
            VideoMetadata(**(base | changes))


def test_video_frame_contract():
    marker = object()
    assert VideoFrame(0, 0.0, marker).image is marker
    for args in ((-1, 0.0, marker), (0, -0.1, marker), (0, math.nan, marker), (0, 0.0, None)):
        with pytest.raises((TypeError, ValueError)):
            VideoFrame(*args)


def test_inspect_and_stream_tiny_video(tiny_video: Path):
    metadata = inspect_video(tiny_video)
    assert metadata.video_id == "tiny.avi"
    assert metadata.frame_count == 4
    assert metadata.width == 32 and metadata.height == 24
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.duration_seconds == pytest.approx(0.4)
    assert metadata.timestamp_policy is TimestampPolicy.FPS_DERIVED_CFR_ASSUMPTION

    reader = OpenCVVideoReader(tiny_video)
    iterator = reader.frames()
    assert iter(iterator) is iterator
    frames = list(iterator)
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert all(frame.image.shape == (24, 32, 3) for frame in frames)


def test_minimum_frames_and_unreadable_are_structured(tiny_video: Path, tmp_path: Path):
    with pytest.raises(GymError) as short:
        inspect_video(tiny_video, min_frames=5)
    assert short.value.flag.code is FlagCode.VIDEO_TOO_SHORT
    assert short.value.flag.evidence["minimum_frames"] == 5

    missing = tmp_path / "missing.avi"
    with pytest.raises(GymError) as unreadable:
        inspect_video(missing)
    assert unreadable.value.flag.code is FlagCode.UNREADABLE_VIDEO
    assert unreadable.value.flag.evidence["source_path"] == str(missing.resolve())


@pytest.mark.parametrize(("properties", "expected_code"), [
    ({1: 0.0, 2: 3.0, 3: 32.0, 4: 24.0}, FlagCode.INVALID_FPS),
    ({1: 10.0, 2: -1.0, 3: 32.0, 4: 24.0}, FlagCode.INVALID_FRAME_COUNT),
    ({1: 10.0, 2: 3.0, 3: 0.0, 4: 24.0}, FlagCode.INVALID_VIDEO_RESOLUTION),
])
def test_invalid_decoder_metadata_has_specific_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, properties: dict[int, float], expected_code: FlagCode
):
    source = tmp_path / "exists.avi"
    source.write_bytes(b"placeholder")

    class FakeCapture:
        def __init__(self, _path: str):
            self.released = False

        def isOpened(self):
            return True

        def get(self, key: int):
            return properties[key]

        def getBackendName(self):
            return "FAKE"

        def release(self):
            self.released = True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
        VideoCapture=FakeCapture,
    )
    monkeypatch.setattr(reader_module, "_cv2", lambda: fake_cv2)

    with pytest.raises(GymError) as caught:
        inspect_video(source)
    assert caught.value.flag.code is expected_code
    assert caught.value.flag.stage.value == "video"


def test_cli_outputs_json(tiny_video: Path):
    result = subprocess.run(
        [sys.executable, "-m", "gym_coach.video", str(tiny_video)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["metadata"]["frame_count"] == 4
    assert payload["metadata"]["timestamp_policy"] == "fps_derived_cfr_assumption"


@pytest.mark.integration
def test_real_development_video_metadata_and_stream_prefix():
    assert REAL_VIDEO.is_file(), f"Required development fixture missing: {REAL_VIDEO}"
    reader = OpenCVVideoReader(REAL_VIDEO)
    assert reader.metadata.video_id == REAL_VIDEO.name
    assert reader.metadata.fps > 0
    assert reader.metadata.frame_count > 0
    iterator = reader.frames()
    first = next(iterator)
    second = next(iterator)
    iterator.close()
    assert first.frame_index == 0 and first.timestamp_seconds == 0.0
    assert second.frame_index == 1
    assert second.timestamp_seconds == pytest.approx(1.0 / reader.metadata.fps)
    assert first.image.shape[:2] == (reader.metadata.height, reader.metadata.width)
