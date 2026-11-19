"""Kiểm thử core/phase_schema.py"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.phase_schema import PhaseSet, Rep, PhaseWindow, load_phases, save_phases

V1_SAMPLE = {
    "video": "coach_HLV01_running_v001.mp4", "fps": 30.0, "total_frames": 743,
    "phases": {
        "vao_cho": {"start_frame": 540, "end_frame": 600},
        "static": {"start_frame": 601, "end_frame": 685},
        "sprint_start": {"start_frame": 686, "end_frame": 708},
        "drive_phase": {"start_frame": 709, "end_frame": 734},
    },
    "marked_by": "manual",
}

def test_load_v1_and_roundtrip():
    ps = PhaseSet.from_dict(V1_SAMPLE)
    assert ps.schema_version == 1 and len(ps.reps) == 1
    assert ps.validate() == [] and ps.warnings() == []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.phases.json"
        save_phases(p, ps)
        raw = json.load(open(p, encoding="utf-8"))
        assert raw["schema_version"] == 2
        assert raw["phases"] == V1_SAMPLE["phases"]          # khối v1 giữ nguyên
        ps2 = load_phases(p)
        assert ps2.phases["static"].start_frame == 601 and ps2.labeled_at

def test_absent_phase_roundtrip():
    rep = Rep.from_windows({"vao_cho": (10, 50), "static": (51, 90), "sprint_start": None,
                            "drive_phase": (91, 120)})
    ps = PhaseSet("a.mp4", 30.0, 200, [rep], marked_by="auto")
    assert ps.validate() == []
    d = ps.to_dict()
    assert "sprint_start" not in d["phases"]                  # mirror chỉ gồm pha có mặt
    assert d["reps"][0]["phases"]["sprint_start"] == {"absent": True}
    ps2 = PhaseSet.from_dict(d)
    assert ps2.phases["sprint_start"].absent and ps2.phases["drive_phase"].start_frame == 91

def test_validation_errors():
    bad = json.loads(json.dumps(V1_SAMPLE))
    bad["phases"]["static"] = {"start_frame": 500, "end_frame": 685}    # chồng lấn vao_cho
    assert any("chồng" in e for e in PhaseSet.from_dict(bad).validate())
    bad = json.loads(json.dumps(V1_SAMPLE)); bad["total_frames"] = 700   # vượt video
    assert any("ngoài video" in e for e in PhaseSet.from_dict(bad).validate())
    ps = PhaseSet.from_dict(V1_SAMPLE); ps.reps.append(Rep.from_windows({"vao_cho": (1, 2)}))
    assert any("tối đa" in e for e in ps.validate())                     # v1: 1 rep
    gap = json.loads(json.dumps(V1_SAMPLE)); gap["phases"]["static"]["start_frame"] = 610
    assert PhaseSet.from_dict(gap).warnings()                            # khe hở -> cảnh báo mềm

def test_save_rejects_invalid():
    ps = PhaseSet.from_dict(V1_SAMPLE); ps.total_frames = 100
    try:
        save_phases(Path(tempfile.gettempdir()) / "never.json", ps)
    except ValueError:
        return
    raise AssertionError("phải từ chối ghi nhãn không hợp lệ")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK", n)
