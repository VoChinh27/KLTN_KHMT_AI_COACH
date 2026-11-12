"""Factory on saved Stage 4 artifacts; no quality/viewpoint changes or inference."""

import hashlib
import json
from pathlib import Path
import uuid

import pytest

from gym_coach import geometry
from gym_coach.features import LinkedFeatureInputContext, NumericFeatureObservation, build_feature_input_context, observe_feature
from gym_coach.flags import FlagCode
from gym_coach.preprocessing import read_preprocessing_result

ROOT = Path(__file__).resolve().parents[1]
ID = "squat_knee_angle_left_deg"


@pytest.fixture(scope="module")
def audit_directory():
    path = ROOT/"outputs/stage5_4"/("integration-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.mark.integration
@pytest.mark.parametrize("exercise", ["BodyWeightSquats", "HandstandPushups"])
def test_real_stage4_factory_context_preserves_unavailable_quality(exercise, audit_directory, monkeypatch):
    calls = []
    def forbidden(*points):
        calls.append(points)
        pytest.fail("Unavailable real input reached primitive")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    snapshot = json.loads((ROOT/"docs/artifacts/stage4_measured_results.json").read_text(encoding="utf-8"))
    video = next(v for v in snapshot["videos"] if v["exercise"] == exercise)
    path = ROOT/video["artifact_relative_path"]
    before_file = path.read_bytes()
    result = read_preprocessing_result(path)
    before_result = result.model_dump_json()
    samples = []
    for index in (0, len(result.frames)//2, len(result.frames)-1):
        context = build_feature_input_context(result, frame_index=index)
        assert context.frame == result.frames[index] and context.quality == result.quality
        assert context.raw_frame == result.raw_frames[index]
        assert context.frame.frame_index == index and context.frame.timestamp_seconds == result.raw_frames[index].timestamp_seconds
        encoded = context.model_dump_json(indent=2)
        restored = LinkedFeatureInputContext.model_validate_json(encoded, strict=True)
        assert restored == context and restored.content_hash() == context.content_hash()
        measured = observe_feature(ID, restored)
        assert measured.status == "unavailable" and measured.value is None
        assert measured.availability.input_hash == context.measurement_input_hash
        assert measured.availability.input.quality == result.quality
        assert {FlagCode.FEATURE_QUALITY_BLOCKED, FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING,
                FlagCode.VIEWPOINT_UNKNOWN} <= set(measured.reasons)
        if exercise == "HandstandPushups":
            assert FlagCode.UNSUPPORTED_EXERCISE in measured.reasons
        assert NumericFeatureObservation.model_validate_json(measured.model_dump_json()) == measured
        entries = {}
        for kind, text in (("context", encoded), ("observation", measured.model_dump_json(indent=2))):
            out = audit_directory/f"{exercise}.frame-{index}.{kind}.json"
            with out.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
            entries[kind + "_relative_path"] = out.relative_to(ROOT).as_posix()
            entries[kind + "_file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        samples.append({"frame_index":index,"timestamp_seconds":context.frame.timestamp_seconds,
            "status":measured.status,"value":measured.value,"reasons":[r.value for r in measured.reasons],
            "context_hash":context.content_hash(),"measurement_input_hash":context.measurement_input_hash,
            "preprocessing_result_hash":context.preprocessing_result_hash,"raw_frame_hash":context.raw_frame_hash,
            "raw_sequence_hash":context.raw_sequence_hash,"quality_hash":context.quality_hash,
            "profile_version":context.profile_version,"profile_hash":context.profile_hash,**entries})
    assert not calls
    assert path.read_bytes() == before_file and result.model_dump_json() == before_result
    summary = {"exercise":exercise,"feature_id":ID,"video_id":result.metadata.video_id,
        "source_relative_path":video["source_relative_path"],"stage4_artifact_relative_path":video["artifact_relative_path"],
        "stage4_file_sha256":hashlib.sha256(before_file).hexdigest(),"input_unchanged":True,
        "total_frames":len(result.frames),"sample_count":len(samples),"available":0,"unavailable":len(samples),
        "primitive_calls":len(calls),"quality_action":result.quality.eligibility.action,
        "viewpoint":result.quality.viewpoint.value,"samples":samples,
        "scope":"saved Stage 4 factory linkage and adapter policy, no fresh/full-video feature extraction or accuracy claim"}
    with (audit_directory/f"{exercise}.summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
