"""Current real files/recorded reports: evidence audit, no viewpoint inference."""
import importlib.util
import json
from pathlib import Path
import uuid

import pytest

from gym_coach import geometry
from gym_coach.features import SUPPORTED_KNEE_FEATURE_IDS, build_feature_input_context, observe_feature
from gym_coach.preprocessing import read_preprocessing_result

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage5_6_inventory", ROOT/"scripts/audit_stage5_6_development.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.integration
def test_workspace_inventory_keeps_documentation_and_pose_eligibility_separate():
    paths = [p for p in (ROOT/"Data").rglob("*") if p.is_file() and (
        p.suffix.lower() not in audit.VIDEO_SUFFIXES or "BodyWeightSquats" in p.parts)]
    before = {p: audit.byte_hash(p) for p in paths}
    result = audit.inventory(ROOT)
    assert result["candidate_file_count"] > 0
    assert result["status"] == "BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT"
    assert result["documented_side_view_count"] == result["eligible_count"] == 0
    assert result["ineligible_count"]+result["blocked_count"] == result["candidate_file_count"]
    assert result["unique_content_count"] <= result["candidate_file_count"]
    for candidate in result["candidates"]:
        assert candidate["video_id"].startswith("v_BodyWeightSquats_")
        assert candidate["viewpoint"] == "unknown" and not candidate["documented"]
        assert candidate["source_camera_view"] == "UNKNOWN" and candidate["visual_review"] == "not_performed"
        assert not candidate["historical_image_quality_is_pose_eligibility"]
        assert "documented_side_view_evidence_missing" in candidate["reasons"]
        assert candidate["decision"] in {"blocked", "ineligible"}
        assert all(r["sequence"]["action"] == "skip" and r["viewpoint"] == "unknown" for r in candidate["existing_quality_reports"])
    directory = ROOT/"outputs/stage5_6"/("inventory-test-"+uuid.uuid4().hex)
    audit.write_inventory(result, directory)
    assert json.loads((directory/"inventory.json").read_text(encoding="utf-8")) == result
    assert (directory/"candidates.csv").exists()
    with pytest.raises(FileExistsError):
        audit.write_inventory(result, directory)
    assert {p: audit.byte_hash(p) for p in paths} == before


@pytest.mark.integration
def test_saved_real_squat_source_never_promoted_and_never_measured(monkeypatch):
    snapshot = json.loads((ROOT/"docs/artifacts/stage4_measured_results.json").read_text(encoding="utf-8"))
    entry = next(v for v in snapshot["videos"] if v["exercise"] == "BodyWeightSquats")
    path = ROOT/entry["artifact_relative_path"]
    before = path.read_bytes()
    source = read_preprocessing_result(path)
    source_before = source.model_dump_json()
    def forbidden(*points):
        pytest.fail("Undocumented, ineligible real input reached primitive")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    for index in (0, len(source.frames)//2, len(source.frames)-1):
        context = build_feature_input_context(source, frame_index=index)
        for key in SUPPORTED_KNEE_FEATURE_IDS:
            observation = observe_feature(key, context)
            assert observation.status == "unavailable" and observation.value is None
            assert observation.availability.input.quality == source.quality
            assert source.quality.viewpoint == "unknown" and source.quality.eligibility.action == "skip"
            assert key not in source.quality.metric_eligibility
    assert path.read_bytes() == before and source.model_dump_json() == source_before
