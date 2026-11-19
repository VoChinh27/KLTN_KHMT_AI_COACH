"""
core/phase_schema.py
======================
Mục đích : MỘT nơi duy nhất đọc / ghi / kiểm tra file `<video>.phases.json`
           cho cả công cụ đánh dấu (tools/mark_phases.py), mô hình tự động
           (stage3_features/phase_*.py) và pipeline chấm điểm.

Schema v2 (tương thích ngược với v1):
    - v1 (cũ)  : {"video","fps","total_frames","phases":{4 pha},"marked_by"}
    - v2 (mới) : thêm "schema_version", "reps" (danh sách các lần thực hiện),
                 cờ "absent" cho pha vắng mặt, "annotator_id", "labeled_at",
                 "auto_meta", "auto_phases", "review".
    - Khối "phases" ở cấp cao nhất LUÔN được ghi lại = phản chiếu reps[0]
      (chỉ gồm các pha CÓ MẶT) để code đọc theo v1 vẫn chạy được.
    - File v1 đọc vào được bình thường (coi như 1 rep, đủ 4 pha).

Phạm vi v1 của hệ thống: mỗi clip = 1 rep (xem MAX_REPS_V1). Schema vẫn
chứa được nhiều rep để sau này mở rộng mà không phải đổi định dạng file.

Quy ước frame: start_frame / end_frame đều là chỉ số frame, BAO GỒM CẢ HAI
ĐẦU (end_frame inclusive), giống file do mark_phases.py ghi ra.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PHASE_ORDER: Tuple[str, ...] = ("vao_cho", "static", "sprint_start", "drive_phase")
SCHEMA_VERSION = 2
MARKED_BY_VALUES: Tuple[str, ...] = ("manual", "auto", "auto_reviewed")
MAX_REPS_V1 = 1  # v1: mỗi clip = 1 lần thực hiện

# Các khóa cấp cao nhất do schema này quản lý. MỌI khóa khác trong file (ví dụ
# "front_leg_override" mà stage5_eval/loso_cv_eval.py đọc) được GIỮ NGUYÊN qua
# PhaseSet.extra khi đọc rồi ghi lại — không được làm mất dữ liệu của công cụ khác.
_KNOWN_KEYS = frozenset({
    "schema_version", "video", "fps", "total_frames", "phases", "reps", "marked_by",
    "annotator_id", "labeled_at", "auto_meta", "auto_phases", "review",
    "n_frames", "boundaries_frames",
})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------
# Cửa sổ 1 pha
# ---------------------------------------------------------------------
@dataclass
class PhaseWindow:
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None   # inclusive
    absent: bool = False              # pha không xảy ra trong clip này

    @classmethod
    def missing(cls) -> "PhaseWindow":
        return cls(None, None, True)

    @property
    def n_frames(self) -> int:
        if self.absent or self.start_frame is None or self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict[str, Any]:
        if self.absent:
            return {"absent": True}
        return {"start_frame": int(self.start_frame), "end_frame": int(self.end_frame)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseWindow":
        if d.get("absent"):
            return cls.missing()
        try:
            return cls(int(d["start_frame"]), int(d["end_frame"]), False)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cửa sổ pha thiếu/sai start_frame/end_frame: {d!r}") from exc


# ---------------------------------------------------------------------
# 1 lần thực hiện (rep)
# ---------------------------------------------------------------------
@dataclass
class Rep:
    phases: Dict[str, PhaseWindow] = field(
        default_factory=lambda: {p: PhaseWindow.missing() for p in PHASE_ORDER}
    )

    @classmethod
    def from_windows(cls, windows: Dict[str, Optional[Tuple[int, int]]]) -> "Rep":
        """windows: {phase: (start, end_inclusive) hoặc None nếu vắng mặt}."""
        rep = cls()
        for p in PHASE_ORDER:
            w = windows.get(p)
            rep.phases[p] = PhaseWindow.missing() if w is None else PhaseWindow(int(w[0]), int(w[1]))
        return rep

    def present(self) -> List[Tuple[str, PhaseWindow]]:
        return [(p, self.phases[p]) for p in PHASE_ORDER
                if p in self.phases and not self.phases[p].absent]

    def span(self) -> Optional[Tuple[int, int]]:
        pr = self.present()
        if not pr:
            return None
        return pr[0][1].start_frame, pr[-1][1].end_frame

    def validate(self, total_frames: int, rep_idx: int = 0) -> List[str]:
        errs: List[str] = []
        tag = f"rep[{rep_idx}]"
        unknown = [k for k in self.phases if k not in PHASE_ORDER]
        if unknown:
            errs.append(f"{tag}: pha lạ {unknown}")
        pr = self.present()
        if not pr:
            errs.append(f"{tag}: không có pha nào có mặt")
            return errs
        for name, w in pr:
            if w.start_frame is None or w.end_frame is None:
                errs.append(f"{tag}.{name}: thiếu start/end")
                continue
            if w.start_frame < 0 or w.end_frame >= total_frames:
                errs.append(f"{tag}.{name}: [{w.start_frame},{w.end_frame}] ngoài video [0,{total_frames - 1}]")
            if w.end_frame < w.start_frame:
                errs.append(f"{tag}.{name}: end_frame {w.end_frame} < start_frame {w.start_frame}")
        for (n1, w1), (n2, w2) in zip(pr, pr[1:]):
            if None in (w1.end_frame, w2.start_frame):
                continue
            if w2.start_frame <= w1.end_frame:
                errs.append(f"{tag}: '{n2}' (bắt đầu {w2.start_frame}) chồng/lùi so với '{n1}' (kết thúc {w1.end_frame})")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        return {"phases": {p: self.phases[p].to_dict() for p in PHASE_ORDER if p in self.phases}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rep":
        raw = d.get("phases", {})
        rep = cls()
        for p in PHASE_ORDER:
            rep.phases[p] = PhaseWindow.from_dict(raw[p]) if p in raw else PhaseWindow.missing()
        return rep


# ---------------------------------------------------------------------
# Toàn bộ nhãn của 1 video
# ---------------------------------------------------------------------
@dataclass
class PhaseSet:
    video: str
    fps: float
    total_frames: int
    reps: List[Rep] = field(default_factory=list)
    marked_by: str = "manual"
    annotator_id: str = ""
    labeled_at: str = ""
    auto_meta: Optional[Dict[str, Any]] = None      # model_version, boundary_confidence, flags
    auto_phases: Optional[Dict[str, Any]] = None    # đề xuất gốc của mô hình (để đo phần người sửa)
    review: Optional[Dict[str, Any]] = None         # edits, seconds_spent
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)  # khóa lạ được bảo toàn (front_leg_override...)

    # --- tiện ích ---
    @property
    def phases(self) -> Dict[str, PhaseWindow]:
        """Phản chiếu reps[0] (tương thích cách dùng v1)."""
        return self.reps[0].phases if self.reps else {}

    def validate(self, max_reps: int = MAX_REPS_V1) -> List[str]:
        errs: List[str] = []
        if not (self.fps and self.fps > 0):
            errs.append(f"fps không hợp lệ: {self.fps}")
        if self.total_frames <= 0:
            errs.append(f"total_frames không hợp lệ: {self.total_frames}")
        if self.marked_by not in MARKED_BY_VALUES:
            errs.append(f"marked_by '{self.marked_by}' không thuộc {MARKED_BY_VALUES}")
        if not self.reps:
            errs.append("không có rep nào")
            return errs
        if len(self.reps) > max_reps:
            errs.append(f"{len(self.reps)} rep > tối đa {max_reps} (v1 chỉ hỗ trợ 1 lần thực hiện/clip)")
        for i, rep in enumerate(self.reps):
            errs.extend(rep.validate(self.total_frames, i))
        spans = [r.span() for r in self.reps]
        for i in range(len(spans) - 1):
            if spans[i] and spans[i + 1] and spans[i + 1][0] <= spans[i][1]:
                errs.append(f"rep[{i + 1}] bắt đầu trước khi rep[{i}] kết thúc")
        return errs

    def warnings(self) -> List[str]:
        """Cảnh báo mềm: khe hở giữa 2 pha liền kề (không có pha vắng mặt xen giữa)."""
        out: List[str] = []
        for i, rep in enumerate(self.reps):
            pr = rep.present()
            for (n1, w1), (n2, w2) in zip(pr, pr[1:]):
                adjacent = PHASE_ORDER.index(n2) - PHASE_ORDER.index(n1) == 1
                if adjacent and w2.start_frame != w1.end_frame + 1:
                    out.append(f"rep[{i}]: khe hở giữa '{n1}' (kết thúc {w1.end_frame}) và '{n2}' (bắt đầu {w2.start_frame})")
        return out

    # --- (de)serialize ---
    def to_dict(self) -> Dict[str, Any]:
        mirror: Dict[str, Any] = {}
        if self.reps:
            mirror = {p: w.to_dict() for p, w in self.reps[0].present()}
        d: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "video": self.video,
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "phases": mirror,  # tương thích v1 (chỉ pha có mặt)
            "reps": [r.to_dict() for r in self.reps],
            "marked_by": self.marked_by,
            "annotator_id": self.annotator_id,
            "labeled_at": self.labeled_at,
        }
        for key in ("auto_meta", "auto_phases", "review"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        for key, val in self.extra.items():
            if key not in _KNOWN_KEYS:
                d[key] = val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseSet":
        if "reps" in d:
            reps = [Rep.from_dict(r) for r in d["reps"]]
        elif "phases" in d:
            reps = [Rep.from_dict({"phases": d["phases"]})]  # file v1
        else:
            raise ValueError("file nhãn không có 'reps' lẫn 'phases'")
        try:
            return cls(
                video=str(d.get("video", "")),
                fps=float(d["fps"]),
                total_frames=int(d["total_frames"]),
                reps=reps,
                marked_by=str(d.get("marked_by", "manual")),
                annotator_id=str(d.get("annotator_id", "")),
                labeled_at=str(d.get("labeled_at", "")),
                auto_meta=d.get("auto_meta"),
                auto_phases=d.get("auto_phases"),
                review=d.get("review"),
                schema_version=int(d.get("schema_version", 1)),
                extra={k: v for k, v in d.items() if k not in _KNOWN_KEYS},
            )
        except KeyError as exc:
            raise ValueError(f"file nhãn thiếu trường {exc}") from exc


def load_phases(path: str | Path) -> PhaseSet:
    with open(path, "r", encoding="utf-8") as f:
        return PhaseSet.from_dict(json.load(f))


def save_phases(path: str | Path, ps: PhaseSet, validate: bool = True,
                max_reps: int = MAX_REPS_V1) -> None:
    """Ghi nguyên tử (tmp + os.replace). Mặc định từ chối ghi nhãn không hợp lệ."""
    if validate:
        errs = ps.validate(max_reps=max_reps)
        if errs:
            raise ValueError("Không lưu được nhãn không hợp lệ:\n  - " + "\n  - ".join(errs))
    if not ps.labeled_at:
        ps.labeled_at = utc_now_iso()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ps.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)