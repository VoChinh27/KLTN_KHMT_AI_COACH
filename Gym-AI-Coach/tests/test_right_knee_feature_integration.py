"""Right knee through factory on real saved input; left golden results unchanged."""

import hashlib
import json
from pathlib import Path
import uuid

import pytest

from gym_coach import geometry
from gym_coach.features import KneeAngleFeatureObservation, NumericFeatureObservation, build_feature_input_context, observe_feature
from gym_coach.flags import FlagCode
from gym_coach.preprocessing import read_preprocessing_result

ROOT = Path(__file__).resolve().parents[1]
ID = "squat_knee_angle_right_deg"
LEFT_ID = "squat_knee_angle_left_deg"


@pytest.fixture(scope="module")
def audit_directory():
    out = ROOT/"outputs/stage5_5"/("integration-" + uuid.uuid4().hex)
    out.mkdir(parents=True, exist_ok=False)
    return out


@pytest.mark.integration
@pytest.mark.parametrize("exercise", ["BodyWeightSquats", "HandstandPushups"])
def test_right_on_saved_stage4_and_left_golden_compatibility(exercise, audit_directory, monkeypatch):
    def forbidden(*points):
        pytest.fail("Real unavailable input reached primitive")
    monkeypatch.setattr(geometry,"joint_angle",forbidden)
    snapshot = json.loads((ROOT/"docs/artifacts/stage4_measured_results.json").read_text(encoding="utf-8"))
    video = next(v for v in snapshot["videos"] if v["exercise"] == exercise)
    path = ROOT/video["artifact_relative_path"]
    before_file = path.read_bytes()
    source = read_preprocessing_result(path)
    before_result = source.model_dump_json()
    samples = []
    for index in (0,len(source.frames)//2,len(source.frames)-1):
        context = build_feature_input_context(source,frame_index=index)
        measured = observe_feature(ID,context)
        assert measured.value is None and measured.status == "unavailable"
        assert measured.numeric_feature_schema_version == "0.3.0" and measured.adapter_version == "squat-right-knee-v1"
        assert measured.availability.input_hash == context.measurement_input_hash
        assert measured.availability.input.quality == source.quality
        assert (measured.video_id,measured.frame_index,measured.timestamp_seconds) == (
            source.metadata.video_id,index,source.frames[index].timestamp_seconds)
        assert tuple(j.joint.value for j in measured.joint_inputs) == ("right_hip","right_knee","right_ankle")
        assert {FlagCode.VIEWPOINT_UNKNOWN,FlagCode.FEATURE_QUALITY_BLOCKED,
                FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING} <= set(measured.reasons)
        if exercise == "HandstandPushups":
            assert FlagCode.UNSUPPORTED_EXERCISE in measured.reasons
        assert KneeAngleFeatureObservation.model_validate_json(measured.model_dump_json(),strict=True) == measured
        left = observe_feature(LEFT_ID,context)
        old_path = ROOT/f"outputs/stage5_3/integration-563152cdd3aa4a3394bebd0619d16858/{exercise}.frame-{index}.observation.json"
        old_bytes = old_path.read_bytes()
        golden = NumericFeatureObservation.model_validate_json(old_bytes)
        assert type(left) is NumericFeatureObservation and left == golden
        assert left.model_dump_json() == golden.model_dump_json()
        assert old_path.read_bytes() == old_bytes
        entries = {}
        for kind,text in (("context",context.model_dump_json(indent=2)),("right-observation",measured.model_dump_json(indent=2))):
            out = audit_directory/f"{exercise}.frame-{index}.{kind}.json"
            with out.open("x",encoding="utf-8") as stream: stream.write(text+"\n")
            entries[kind+"_relative_path"] = out.relative_to(ROOT).as_posix()
            entries[kind+"_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        samples.append({"frame_index":index,"timestamp_seconds":measured.timestamp_seconds,
            "status":measured.status,"value":measured.value,"reasons":[r.value for r in measured.reasons],
            "joint_statuses":[{"joint":j.joint.value,"status":j.status.value,"source_status":j.source_status,"usable":j.usable} for j in measured.joint_inputs],
            "context_hash":context.content_hash(),"input_hash":measured.availability.input_hash,
            "observation_hash":measured.content_hash(),"preprocessing_result_hash":context.preprocessing_result_hash,
            "left_golden_equal":True,**entries})
    assert path.read_bytes() == before_file and source.model_dump_json() == before_result
    summary = {"exercise":exercise,"feature_id":ID,"video_id":source.metadata.video_id,
        "source_relative_path":video["source_relative_path"],"stage4_artifact_relative_path":video["artifact_relative_path"],
        "stage4_file_sha256":hashlib.sha256(before_file).hexdigest(),"total_frames":len(source.frames),
        "sample_count":3,"available":0,"unavailable":3,"primitive_calls":0,"left_golden_equal_count":3,
        "quality_action":source.quality.eligibility.action,"viewpoint":source.quality.viewpoint.value,
        "input_unchanged":True,"samples":samples,
        "scope":"saved Stage 4 sampled factory/right adapter, left 0.2.0 golden compatibility; no full-video extraction/accuracy claim"}
    with (audit_directory/f"{exercise}.summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
