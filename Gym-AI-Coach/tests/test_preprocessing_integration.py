"""Fresh full Stage 1 -> 4 runs; artifacts never substitute for pose inference."""

import json
from pathlib import Path
import uuid

import pytest

from gym_coach.contracts import Exercise
from gym_coach.pose import MediaPipePoseExtractor
from gym_coach.preprocessing import (
    load_preprocessing_profile, preprocess_pose, read_preprocessing_result,
    summarize_preprocessing, write_preprocessing_result,
)
from gym_coach.quality import evaluate_quality
from gym_coach.video import OpenCVVideoReader

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit_directory():
    directory = ROOT / "outputs" / "stage4" / f"integration-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    return directory


@pytest.mark.integration
@pytest.mark.parametrize("exercise,relative_path", [
    (Exercise.BODY_WEIGHT_SQUATS, "Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi"),
    (Exercise.HANDSTAND_PUSHUPS, "Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi"),
])
def test_fresh_video_to_preprocessing(exercise, relative_path, audit_directory):
    reader = OpenCVVideoReader(ROOT / relative_path)
    with MediaPipePoseExtractor(video_id=reader.metadata.video_id, exercise=exercise) as extractor:
        raw = tuple(extractor.extract_frames(reader.frames()))
    quality = evaluate_quality(raw)
    raw_before = tuple(f.model_dump_json() for f in raw)
    quality_before = quality.model_dump_json()
    profile = load_preprocessing_profile(ROOT / "configs/preprocessing_s3_v1.toml")
    result = preprocess_pose(raw, quality, reader.metadata, profile=profile)
    assert tuple(f.model_dump_json() for f in raw) == raw_before
    assert quality.model_dump_json() == quality_before
    assert result.quality == quality
    assert len(result.frames) == reader.metadata.frame_count
    assert [(f.video_id, f.frame_index, f.timestamp_seconds) for f in result.frames] == [
        (f.video_id, f.frame_index, f.timestamp_seconds) for f in raw]
    artifact = audit_directory / f"{exercise.value}.processed.json"
    write_preprocessing_result(result, artifact)
    assert read_preprocessing_result(artifact) == result
    summary = summarize_preprocessing(result)
    summary.update({
        "exercise": exercise.value, "source_relative_path": relative_path,
        "artifact_relative_path": artifact.relative_to(ROOT).as_posix(),
        "mediapipe_version": raw[0].backend.mediapipe_version,
        "model_asset_sha256": raw[0].backend.model_asset_sha256,
        "raw_sequence_hash": result.raw_sequence_hash, "quality_hash": result.quality_hash,
        "quality_preserved": result.quality == quality, "raw_preserved": result.raw_frames == raw,
        "identity_preserved": True,
    })
    (audit_directory / f"{exercise.value}.summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage4_measured_result": summary}, allow_nan=False))
