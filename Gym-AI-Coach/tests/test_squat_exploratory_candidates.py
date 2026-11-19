"""Scope/ranking/gate regressions; synthetic only, no model or video decoding."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from gym_coach.contracts import Viewpoint
from gym_coach.pose.mapping import JointName as J
from gym_coach.quality import QualityGate, QualityProfile
from stage5_6_fixture import fixture_input

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("squat_candidate_audit", ROOT / "scripts/audit_squat_exploratory_candidates.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_audit(root):
    source = root / audit.DATA_ROOT
    source.mkdir(parents=True)
    evidence = root / "outputs" / "saved-audit"
    evidence.mkdir(parents=True)
    (evidence / "sample.png").write_bytes(b"identity-only placeholder, not decoded")
    rows = []
    for i in range(3):
        video = source / f"clip-{i}.avi"
        video.write_bytes(b"same-content" if i < 2 else b"front-content")
        rows.append(dict(relative_path=video.relative_to(root).as_posix(), video_id=video.stem,
            file_name=video.name, video_sha256=audit.sha(video), original_camera_view="UNKNOWN",
            visual_viewpoint="likely_side" if i < 2 else "front", documented=False,
            visual_review_status="visually_assessed_unverified", review_id=f"T{i}", observation="saved visual opinion",
            confidence="high", reviewer="Test agent", review_date="2026-10-18",
            samples=[dict(frame_index=n, timestamp_seconds=float(n),
                image_path=(evidence / "sample.png").relative_to(root).as_posix(),
                image_sha256=audit.sha(evidence / "sample.png"), agent_viewed=True) for n in (0, 1)]))
    audit.write_json(evidence / "visual.json", dict(videos=rows))
    return evidence / "visual.json"


def test_prepare_deduplicates_and_freezes_ranking_without_decoder(tmp_path, monkeypatch):
    evidence = fixture_audit(tmp_path)
    import gym_coach.pose as pose
    import gym_coach.video as video
    def forbidden(*args, **kwargs):
        raise AssertionError("Pre-inference planning cannot start decoder/model")
    monkeypatch.setattr(pose, "MediaPipePoseExtractor", forbidden)
    monkeypatch.setattr(video, "OpenCVVideoReader", forbidden)
    plan = audit.prepare(tmp_path, evidence.relative_to(tmp_path), Path("outputs/selection"))
    assert plan["candidate_count"] == 1 and len(plan["duplicate_paths_not_reinferred"]) == 1
    assert len(plan["excluded_before_inference"]) == 1
    assert plan["inference_started"] is False and plan["criteria"] == audit.CRITERIA
    assert plan["candidates"][0]["camera_view"] == "UNKNOWN"
    assert plan["candidates"][0]["documented"] is False
    assert plan["quality_profile_hash"] == QualityProfile().content_hash()
    assert plan == json.loads((tmp_path / "outputs/selection/selection-plan.json").read_bytes())
    with pytest.raises(FileExistsError):
        audit.prepare(tmp_path, evidence.relative_to(tmp_path), Path("outputs/selection"))


@pytest.mark.parametrize("fault", ["hash", "sample", "unviewed", "outside", "documented", "camera", "status"])
def test_selection_rejects_bad_linkage_before_opening_other_sources(tmp_path, fault):
    evidence = fixture_audit(tmp_path)
    data = json.loads(evidence.read_bytes())
    row = data["videos"][0]
    if fault == "hash": row["video_sha256"] = "0" * 64
    elif fault == "sample": row["samples"][0]["image_sha256"] = "0" * 64
    elif fault == "unviewed": row["samples"][0]["agent_viewed"] = False
    elif fault == "outside": row["relative_path"] = "Data/OTHER/do-not-read.avi"
    elif fault == "documented": row["documented"] = True
    elif fault == "camera": row["original_camera_view"] = "side"
    else: row["visual_review_status"] = "documented"
    evidence.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        audit.prepare(tmp_path, evidence.relative_to(tmp_path), Path("outputs/new"))
    assert not (tmp_path / "outputs/new").exists()


def test_assumption_projection_preserves_raw_joints_flags_frame_timestamp_and_unknown():
    raw, _, _ = fixture_input(viewpoint=Viewpoint.UNKNOWN)
    before = [f.model_dump_json() for f in raw]
    projected = audit.side_projection(raw)
    for original, derived in zip(raw, projected):
        a = original.model_dump(mode="json"); b = derived.model_dump(mode="json")
        assert a.pop("viewpoint") == "unknown" and b.pop("viewpoint") == "side"
        assert a == b
    assert before == [f.model_dump_json() for f in raw]
    with pytest.raises(ValueError):
        audit.side_projection(projected)


@pytest.mark.parametrize("case", ["clean", "one_missing", "low", "outside", "long_gap", "short", "shoulder"])
def test_bilateral_mask_never_promotes_degrade_skip_or_raw_only_coverage(case):
    changes = {}
    if case == "one_missing": changes[15, J.RIGHT_KNEE] = ("missing", None, None)
    elif case == "low": changes[15, J.RIGHT_KNEE] = ("low_visibility", .6, .4)
    elif case == "outside": changes[15, J.RIGHT_KNEE] = ("observed", 1.2, .4)
    elif case == "long_gap":
        changes = {(i, J.RIGHT_KNEE): ("missing", None, None) for i in range(5, 17)}
    elif case == "shoulder": changes[15, J.LEFT_SHOULDER] = ("missing", None, None)
    raw, _, _ = fixture_input(n=4 if case == "short" else 31, changes=changes, viewpoint=Viewpoint.UNKNOWN)
    reqs, views = audit.requirements()
    quality = QualityGate().evaluate(audit.side_projection(raw), metric_requirements=reqs, metric_viewpoints=views)
    before = quality.model_dump_json()
    result = audit.decisions(raw, quality)
    assert quality.model_dump_json() == before
    assert len(result["frames"]) == len(raw)
    if case == "clean":
        assert result["bilateral_allowed_frames"] == 31 and result["exploratory_demo_eligible"] is True
    else:
        assert result["bilateral_allowed_frames"] == 0 and result["exploratory_demo_eligible"] is False
        assert audit.choose([dict(path="x", **result)]) is None
    if case == "shoulder":
        assert all(m["action"] == "allow" for m in result["knees"].values())
        assert result["raw_bilateral_valid_frames"] == 31 and result["sequence"]["action"] == "degrade"


@pytest.mark.parametrize("fault", ["missing_key", "wrong_joints", "profile"])
def test_exact_metrics_and_profile_must_match(fault):
    raw, _, _ = fixture_input(viewpoint=Viewpoint.UNKNOWN)
    reqs, views = audit.requirements()
    if fault == "missing_key": del reqs[audit.KEYS[1]]
    elif fault == "wrong_joints": reqs[audit.KEYS[1]] = reqs[audit.KEYS[0]]
    profile = QualityProfile(max_missing_ratio=.4) if fault == "profile" else QualityProfile()
    quality = QualityGate(profile).evaluate(audit.side_projection(raw), metric_requirements=reqs, metric_viewpoints=views)
    with pytest.raises(ValueError): audit.decisions(raw, quality)


def test_ranking_is_fixed_and_diagnostics_cannot_select_ineligible():
    good = dict(path="b", exploratory_demo_eligible=True, bilateral_allowed_frame_ratio=1., bilateral_allowed_frames=30)
    longer = dict(good, path="z", bilateral_allowed_frames=31)
    tie = dict(longer, path="a")
    blocked = dict(longer, path="0", exploratory_demo_eligible=False, raw_bilateral_valid_ratio=1.)
    before = copy.deepcopy([good, longer, tie, blocked])
    assert audit.choose(before)["path"] == "a"
    assert before == [good, longer, tie, blocked]
    assert audit.choose([blocked]) is None


def test_tampered_precommit_rejected_before_inference(tmp_path, monkeypatch):
    evidence = fixture_audit(tmp_path)
    audit.prepare(tmp_path, evidence.relative_to(tmp_path), Path("outputs/selection"))
    selection = Path("outputs/selection/selection-plan.json")
    import gym_coach.pose as pose
    def forbidden(*args, **kwargs): raise AssertionError("No backend before freeze checks")
    monkeypatch.setattr(pose, "MediaPipePoseExtractor", forbidden)
    with pytest.raises(ValueError, match="hash"):
        audit.run(tmp_path, selection, "0" * 64, Path("outputs/fresh"))
    assert not (tmp_path / "outputs/fresh").exists()
