"""Sample real Stage 4 input through one adapter; retain unavailable decisions."""

import json
from pathlib import Path
import uuid

import pytest

from gym_coach.features import FeatureInputContext, NumericFeatureObservation, observe_feature
from gym_coach.flags import FlagCode
from gym_coach.preprocessing import read_preprocessing_result

ROOT = Path(__file__).resolve().parents[1]
ID = "squat_knee_angle_left_deg"


@pytest.fixture(scope="module")
def audit_directory():
    path = ROOT/"outputs/stage5_3"/("integration-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.mark.integration
@pytest.mark.parametrize("exercise", ["BodyWeightSquats", "HandstandPushups"])
def test_one_adapter_on_real_stage4_sample_never_promotes_quality(exercise, audit_directory, monkeypatch):
    from gym_coach import geometry
    def forbidden(*points):
        pytest.fail("Real unavailable input reached primitive")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    snapshot = json.loads((ROOT/"docs/artifacts/stage4_measured_results.json").read_text(encoding="utf-8"))
    video = next(v for v in snapshot["videos"] if v["exercise"] == exercise)
    path = ROOT/video["artifact_relative_path"]
    before_file = path.read_bytes()
    result = read_preprocessing_result(path)
    before_result = result.model_dump_json()
    measurements = []
    for index in (0, len(result.frames)//2, len(result.frames)-1):
        context = FeatureInputContext(frame=result.frames[index], quality=result.quality)
        observation = observe_feature(ID, context)
        assert observation.status == "unavailable" and observation.value is None
        assert {FlagCode.FEATURE_QUALITY_BLOCKED, FlagCode.FEATURE_METRIC_ELIGIBILITY_MISSING,
                FlagCode.VIEWPOINT_UNKNOWN} <= set(observation.reasons)
        if exercise == "HandstandPushups":
            assert FlagCode.UNSUPPORTED_EXERCISE in observation.reasons
        assert (observation.video_id, observation.frame_index, observation.timestamp_seconds) == (
            result.frames[index].video_id, index, result.frames[index].timestamp_seconds)
        assert observation.availability.input.quality == result.quality
        encoded = observation.model_dump_json(indent=2)
        assert NumericFeatureObservation.model_validate_json(encoded) == observation
        out = audit_directory/f"{exercise}.frame-{index}.observation.json"
        with out.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        measurements.append({"frame_index":index,"timestamp_seconds":observation.timestamp_seconds,
            "status":observation.status,"value":observation.value,"reasons":[r.value for r in observation.reasons],
            "observation_hash":observation.content_hash(),"output_relative_path":out.relative_to(ROOT).as_posix()})
    assert path.read_bytes() == before_file and result.model_dump_json() == before_result
    summary = {"exercise":exercise,"feature_id":ID,"video_id":result.metadata.video_id,
        "source_relative_path":video["source_relative_path"],"stage4_artifact_relative_path":video["artifact_relative_path"],
        "total_frames":len(result.frames),"sample_count":3,"available":0,"unavailable":3,
        "primitive_calls":0,"quality_action":result.quality.eligibility.action,"viewpoint":result.quality.viewpoint.value,
        "input_unchanged":True,"samples":measurements,"scope":"three independent frame observations, no full-video extraction or accuracy claim"}
    with (audit_directory/f"{exercise}.summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    print(json.dumps(summary,allow_nan=False))
