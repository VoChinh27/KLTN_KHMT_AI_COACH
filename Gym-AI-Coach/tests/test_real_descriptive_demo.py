"""Fixed-side presentation/contract tests, using synthetic geometry only."""
import copy
import io
import json
from pathlib import Path

import pytest

from gym_coach import geometry
from gym_coach.contracts import Viewpoint
from gym_coach.features import observe_feature, FeatureInputContext
from gym_coach.features.models import content_hash
from gym_coach.pose.mapping import JointName as J
from gym_coach.quality import QualityGate, QualityProfile
from gym_coach.reporting import real_descriptive_demo as demo
from gym_coach.reporting import real_descriptive_render as render
from stage5_6_fixture import fixture_result


def source(case="global_skip", side="left"):
    joint = J.LEFT_KNEE if side == "left" else J.RIGHT_KNEE
    changes = {}
    if case == "global_skip": changes = {(i, J.LEFT_SHOULDER): ("missing", None, None) for i in range(31)}
    elif case == "degrade": changes = {(15, joint): ("low_visibility", .4, .4)}
    elif case == "skip": changes = {(i, joint): ("low_visibility", .4, .4) for i in range(31)}
    return fixture_result(changes=changes)


def result_from(processed, side="left"):
    key = demo.KEYS[("left", "right").index(side)]
    raw = tuple(f.model_copy(update={"viewpoint": Viewpoint.UNKNOWN}) for f in processed.raw_frames)
    selected = demo.rank_pair("Data/UCF-101_GYM/BodyWeightSquats/test.avi", side, raw, processed.quality)
    receipt = dict(**selected, video_id=processed.metadata.video_id, sha256="1" * 64,
        reused_artifacts={"raw-pose-unknown.jsonl": {"sha256": "0" * 64}})
    observations, links = demo.measure_frames(processed, key)
    return demo.make_result(receipt, raw, processed, observations, links)


@pytest.mark.parametrize("side", ["left", "right"])
def test_v2_metric_allow_preserves_skipped_sequence_and_legacy_behavior(side):
    processed = source(side=side)
    before = processed.model_dump_json()
    result = result_from(processed, side)
    assert result["coverage"] == dict(total_frames=31, available_frames=31, unavailable_frames=0, available_ratio=1.)
    assert result["selection"]["sequence"]["action"] == "skip"
    assert result["official_eligibility"] is result["assessment"] is False
    assert result["g57"] == "BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT" and result["g6"] == "IN_PROGRESS"
    assert all(r["observation"]["temporal_analysis_eligible"] is False for r in result["frames"])
    old = observe_feature(result["selection"]["feature_id"], FeatureInputContext(frame=processed.frames[0], quality=processed.quality))
    assert old.value is None and old.status == "unavailable"
    assert processed.model_dump_json() == before


@pytest.mark.parametrize("case", ["degrade", "skip"])
def test_degrade_and_skip_stay_null_with_full_denominator(case, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Blocked metric reached geometry")
    processed = source(case)
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    result = result_from(processed)
    assert result["coverage"]["total_frames"] == result["coverage"]["unavailable_frames"] == 31
    assert result["coverage"]["available_ratio"] == 0.
    for row in result["frames"]:
        assert row["quality_action"] == case
        assert row["observation"]["value"] is None and row["observation"]["reasons"]


def test_ranking_tie_is_path_then_left_right_without_length_priority():
    base = dict(candidate=True, observed_valid_ratio=1., path="b", side="left")
    a_right = dict(base, path="a", side="right", total_frames=999)
    a_left = dict(base, path="a", side="left", total_frames=2)
    blocked = dict(base, path="0", candidate=False)
    rows = [base, a_right, a_left, blocked]
    before = copy.deepcopy(rows)
    assert demo.choose_pair(rows) is a_left and rows == before
    assert demo.choose_pair([dict(base, observed_valid_ratio=.9), a_right]) is a_right
    with pytest.raises(ValueError): demo.choose_pair([blocked])


@pytest.fixture(scope="module")
def saved_result():
    return result_from(source())


@pytest.mark.parametrize("bad", ["side", "official", "assessment", "sequence", "frame", "timestamp", "value", "coverage", "temporal", "source_hash", "input_hash"])
def test_readback_rejects_mismatched_or_promoted_artifacts(saved_result, bad):
    value = copy.deepcopy(saved_result)
    row = value["frames"][0]
    if bad == "side": value["selection"]["side"] = "right"
    elif bad == "official": value["official_eligibility"] = True
    elif bad == "assessment": value["assessment"] = True
    elif bad == "sequence": value["selection"]["sequence"]["action"] = "allow"
    elif bad == "frame": row["frame_index"] = 1
    elif bad == "timestamp": row["timestamp_seconds"] = .1
    elif bad == "value": row["observation"]["value"] = None
    elif bad == "coverage": value["coverage"]["total_frames"] -= 1
    elif bad == "temporal": value["temporal"]["pause"] = 0.
    elif bad == "source_hash": row["source_video_sha256"] = "f" * 64
    else: row["source_linkage"]["measurement_input_hash"] = "f" * 64
    value.pop("result_hash");value["result_hash"] = demo.artifact_hash(value)
    with pytest.raises(ValueError): demo.validate_result(value)


def test_renderer_text_uses_saved_values_never_recalculates(saved_result, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Renderer must not compute or rerun policy")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(demo, "observe_geometry_feature", forbidden)
    monkeypatch.setattr(demo, "preprocess_pose", forbidden)
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    demo.validate_result(saved_result)
    row = saved_result["frames"][0]
    value = render.overlay_text(row, "left", saved_result["selection"]["sequence"])
    assert value["angle"] == f"{row['observation']['value']:.1f}°" and value["sequence"] == "SKIP"
    row = copy.deepcopy(row);row["observation"].update(value=None, status="unavailable", reasons=["feature_quality_blocked"])
    assert render.overlay_text(row, "left", saved_result["selection"]["sequence"])["angle"] == "Unavailable"


def test_saved_renderer_checks_raw_frame_linkage_without_recomputing(saved_result, tmp_path, monkeypatch):
    processed = source()
    raw = [f.model_copy(update={"viewpoint": Viewpoint.UNKNOWN}) for f in processed.raw_frames]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(f.model_dump_json() for f in raw), encoding="utf-8")
    value = copy.deepcopy(saved_result)
    value["selection"]["reused_artifacts"]["raw-pose-unknown.jsonl"]["sha256"] = demo.sha(path)
    value.pop("result_hash");value["result_hash"] = demo.artifact_hash(value)
    result_path = tmp_path / "result.json";demo.write_json(result_path, value)
    def forbidden(*args, **kwargs): raise AssertionError("Saved renderer recalculated a domain result")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(demo, "observe_geometry_feature", forbidden)
    monkeypatch.setattr(demo, "preprocess_pose", forbidden)
    monkeypatch.setattr(QualityGate, "evaluate", forbidden)
    assert render.load_saved(result_path, path) == value
    path.write_bytes(b"tampered raw skeleton")
    with pytest.raises(ValueError, match="byte hash"):
        render.load_saved(result_path, path)


def test_raw_missing_and_low_visibility_draw_rules(saved_result):
    import numpy as np
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    row = copy.deepcopy(saved_result["frames"][0])
    missing = dict(joint="nose", status="missing", x=None, y=None)
    low = dict(joint="nose", status="low_visibility", x=.5, y=.5)
    assert not render.drawable(missing) and render.drawable(low)
    assert not render.drawable(dict(low, x=1.1))
    painted = render.paint_frame(frame, row, "left", saved_result["selection"]["sequence"])
    assert painted.shape == (720, 1280, 3)
    assert render.OBSERVED != render.LOW


def test_html_escapes_source_and_retains_unavailable_gates(saved_result):
    from html.parser import HTMLParser
    value = copy.deepcopy(saved_result);value["selection"]["video_id"] = '</video><img src="bad"> &'
    markup = render.opening_html(value, "overlay-mpeg4.mp4")
    assert '&lt;/video&gt;&lt;img src=&quot;bad&quot;&gt; &amp;' in markup
    assert '<img src="bad">' not in markup and "<script" not in markup and "fetch(" not in markup
    for phrase in ("NOT_VALIDATED", "official_eligibility=false", "assessment=false", "G6 IN_PROGRESS", "Unavailable", "camera_view=UNKNOWN"):
        assert phrase in markup
    class Links(HTMLParser):
        def __init__(self): super().__init__();self.links=[]
        def handle_starttag(self, tag, attrs):
            for key, v in attrs:
                if key in ("href", "src"): self.links.append(v)
    p=Links();p.feed(markup)
    assert all(not Path(n).is_absolute() and ":" not in n and ".." not in n for n in p.links)


def fake_audit(root):
    processed = source()
    raw = [f.model_copy(update={"viewpoint": Viewpoint.UNKNOWN}) for f in processed.raw_frames]
    directory = root / "outputs/saved";directory.mkdir(parents=True)
    video = root / demo.DATA_SCOPE / processed.metadata.video_id;video.parent.mkdir(parents=True);video.write_bytes(b"linkage test only")
    meta = processed.metadata.model_copy(update={"source_path": video})
    candidate = dict(path=video.relative_to(root).as_posix(), video_id=video.name, sha256=demo.sha(video),
        review_id="T1", documented=False, camera_view="UNKNOWN", evidence_status="visually_assessed_unverified")
    visual = root / "outputs/visual.json"
    demo.write_json(visual, dict(videos=[dict(relative_path=candidate["path"], video_sha256=candidate["sha256"], visual_viewpoint="likely_side")]))
    plan = dict(candidates=[candidate], audit_path="outputs/visual.json", audit_sha256=demo.sha(visual),
                registry_hash=demo.GEOMETRY_REGISTRY.content_hash(), quality_profile_hash=QualityProfile().content_hash())
    (directory / "selection").mkdir();demo.write_json(directory / "selection/selection-plan.json", plan)
    folder = directory / "audit/T1";folder.mkdir(parents=True)
    unknown = QualityGate().evaluate(raw, metric_requirements=demo.GEOMETRY_REGISTRY.quality_requirements("BodyWeightSquats"),
                                    metric_viewpoints=demo.GEOMETRY_REGISTRY.quality_viewpoints("BodyWeightSquats"))
    files = {"raw-pose-unknown.jsonl":b"\n".join(f.model_dump_json().encode() for f in raw),
        "pose-assumed-side.jsonl":b"\n".join(f.model_dump_json().encode() for f in processed.raw_frames),
        "quality-unknown.json":unknown.model_dump_json().encode(),
        "quality-assumed-side.json":processed.quality.model_dump_json().encode(),
        "source.json":json.dumps(dict(decoder_metadata=meta.model_dump(mode="json"), candidate=candidate,
            camera_view="UNKNOWN", assumption_status="visually_assessed_unverified")).encode()}
    for name, data in files.items(): (folder / name).write_bytes(data)
    entries={p.name:dict(sha256=demo.sha(p),size_bytes=p.stat().st_size) for p in folder.iterdir()}
    demo.write_json(folder / "artifact-manifest.json", entries)
    demo.write_json(directory / "audit/summary.json", dict(**demo.STATUS, camera_view="UNKNOWN",
        assumption_status="visually_assessed_unverified", documented=False,
        selection_sha256=demo.sha(directory / "selection/selection-plan.json")))
    entries={p.relative_to(directory).as_posix():dict(sha256=demo.sha(p),size_bytes=p.stat().st_size) for p in directory.rglob('*') if p.is_file()}
    demo.write_json(directory / "artifact-inventory.json", dict(run_files=entries))
    return directory


def test_saved_pool_linkage_is_read_only_and_no_inference(tmp_path, monkeypatch):
    directory = fake_audit(tmp_path)
    import gym_coach.pose.extractor as pose
    def forbidden(*args, **kwargs): raise AssertionError("No inference")
    monkeypatch.setattr(pose, "_create_backend", forbidden)
    before={p:demo.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    _, pairs, _ = demo.load_pool(tmp_path, directory.relative_to(tmp_path))
    assert len(pairs)==2 and demo.choose_pair(pairs)["side"]=="left"
    assert all(demo.sha(p)==digest for p,digest in before.items())
    (directory/'audit/T1/raw-pose-unknown.jsonl').write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash/size"):
        demo.load_pool(tmp_path, directory.relative_to(tmp_path))


def test_disallowed_source_rejected_before_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Do not open out-of-scope input")
    monkeypatch.setattr(io, "open", forbidden)
    with pytest.raises(ValueError): demo.scoped(tmp_path, "Data/OTHER/clip.avi", demo.DATA_SCOPE)
