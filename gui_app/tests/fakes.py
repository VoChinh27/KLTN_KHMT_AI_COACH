"""Môi trường GIẢ cho test/smoke GUI khi KHÔNG có code lõi (core/, stage3_features/, render_overlay).
Chỉ dùng cho test; khi chạy thật các module này được import từ dự án."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PHASE_ORDER = ("vao_cho", "static", "sprint_start", "drive_phase")


def _mod(name: str, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs)
    sys.modules[name] = m
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg); sys.modules[pkg].__path__ = []
    return m


@dataclass
class PhaseWindow:
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    absent: bool = False

    def to_dict(self): return {"start_frame": self.start_frame, "end_frame": self.end_frame, "absent": self.absent}


@dataclass
class Rep:
    phases: Dict[str, PhaseWindow] = field(default_factory=dict)

    @classmethod
    def from_windows(cls, w):
        return cls({n: (PhaseWindow(*w[n]) if w.get(n) else PhaseWindow(absent=True)) for n in PHASE_ORDER})

    def present(self): return [(n, p) for n, p in self.phases.items() if not p.absent]


@dataclass
class PhaseSet:
    video: str; fps: float; total_frames: int
    reps: List[Rep] = field(default_factory=list)
    marked_by: str = "manual"
    auto_meta: Any = None; auto_phases: Any = None; review: Any = None

    def validate(self):
        errs, prev = [], -1
        for n, p in self.reps[0].present():
            if p.start_frame <= prev: errs.append(f"{n} chồng lấn")
            if p.end_frame >= self.total_frames: errs.append(f"{n} ngoài video")
            prev = p.end_frame
        return errs

    def warnings(self): return []


def save_phases(path, ps, validate=True):
    import json, pathlib
    if validate and ps.validate(): raise ValueError(ps.validate())
    pathlib.Path(path).write_text(json.dumps({"marked_by": ps.marked_by}), encoding="utf-8")


class _Cfg:
    sport = "sprint_start"; quality_gates = {"min_fps_for_temporal": 30.0}
    technique_phases = {"san_sang": {"engine_phase": "static", "mo_ta": "Mô tả thử"}}
    feature_weights = {"static": {"knee_front": 1.5}}
    phase_weights = {"static": 1.0, "drive_phase": 0.0}
    def validate(self): return []


class SportConfig:
    @classmethod
    def from_yaml(cls, p): return _Cfg()
    @classmethod
    def _from_dict(cls, raw): return _Cfg()


@dataclass
class FakeItem:
    feature_key: str; display_name: str; status_symbol: str; student_value: float; reference_center: float
    unit: str; score_0_100: float; message: str
    error_type: Any = None; level: str = "PRIMARY"


@dataclass
class FakeSub:
    result: Any = None; skip_reason: str = ""
    def has_score(self): return self.result is not None


def install(n_frames: int = 60):
    import cv2
    _mod("core.phase_schema", PHASE_ORDER=PHASE_ORDER, PhaseWindow=PhaseWindow, Rep=Rep, PhaseSet=PhaseSet,
         save_phases=save_phases)
    _mod("core.sport_interface", SportConfig=SportConfig)
    _mod("core.phase_resolver", _pose_array=lambda frames: np.zeros((len(frames), 17, 3)))

    @dataclass
    class PoseFrame:
        frame_idx: int
    class Pipeline:
        @classmethod
        def from_config(cls, *a, **k): return cls()
        def _extract_tracked(self, v, w, h): return [PoseFrame(i) for i in range(n_frames)]
        def process_sprint_start_sequence(self, frames, fps, video_id, phases_path):
            E = types.SimpleNamespace
            item = FakeItem("knee_front", "Góc gối trước", "✗", 147.0, 90.0, "°", 30.0, "Hạ hông",
                            E(value="technique_error"))
            warn = FakeItem("ankle", "Cổ chân", "✗", 10.0, 90.0, "°", 10.0, "pose nhiễu", E(value="measurement_error"))
            fb = E(items=[item, warn], overall_score=65.0, summary_line="ok", render_text=lambda: "FB")
            bundle = E(static_summary=E(segment_start_frame=10, segment_end_frame=30))
            sp = FakeSub(E(feedback=fb, bundle=bundle))
            return E(overall_score=65.0, score_confidence="medium", n_phases_scored=1, phase_source="auto_reviewed",
                     quality_flags={"low_resolution": False}, needs_review=False, render_text=lambda: "TONG",
                     sub_phase_order=list(PHASE_ORDER),
                     sub_phase_results={"static": sp, "vao_cho": FakeSub(skip_reason="quá ngắn")})
    _mod("core.pipeline", Pipeline=Pipeline, SprintStartFullResult=object)

    def phase_at_frame(i, phases):
        return next((p for p, (s, e) in phases.items() if s <= i <= e), None)
    _mod("render_overlay", save_pose_cache=lambda p, f: open(p, "w").write("{}"),
         draw_skeleton=lambda *a, **k: None, draw_panel=lambda *a, **k: None, draw_timeline=lambda *a, **k: None,
         draw_texts_bgr=lambda img, texts: img, phase_at_frame=phase_at_frame, PHASE_FOCUS_JOINTS={},
         RenderContext=lambda **k: types.SimpleNamespace(**k))
    _mod("stage3_features.phase_model.support.phase_feedback_log", log_review=lambda *a, **k: None)
    boundaries = [0, 10, 30, 40, n_frames]
    prop = types.SimpleNamespace(boundaries=boundaries, boundary_conf=np.array([1, .9, .5, .8, 1.0]),
                                 flags=["low_conf"], model_version="fake-v1", auto_meta=lambda: {"flags": []})
    class PhaseSegmenter:
        @classmethod
        def from_registry(cls, d): return cls()
        def propose(self, pose, fps, video_id=""): return prop
    _mod("stage3_features.phase_model.inference.phase_segmenter", PhaseSegmenter=PhaseSegmenter)


def make_video(path: str, n: int = 60, w: int = 320, h: int = 240, fps: int = 30):
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8); cv2.putText(f, str(i), (20, 120), 0, 2, (255, 255, 255), 3); vw.write(f)
    vw.release()
