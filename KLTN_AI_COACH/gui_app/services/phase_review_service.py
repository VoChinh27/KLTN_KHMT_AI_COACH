"""PhaseProposal/ManualProposal -> PhaseSet; chỉnh biên; lưu auto_reviewed/manual + log_review."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.phase_schema import PHASE_ORDER, PhaseSet, PhaseWindow, Rep, save_phases
from gui_app import constants as C


@dataclass
class ManualProposal:
    """Đề xuất giả khi chưa có mô hình pha: chia đều 4 phần, độ tin cậy 0 (bắt buộc người dùng chỉnh)."""
    n_frames: int
    boundaries: List[int] = field(default_factory=list)
    boundary_conf: np.ndarray = field(default_factory=lambda: np.zeros(5))
    flags: List[str] = field(default_factory=lambda: ["manual_no_model"])
    model_version: str = "manual"

    def __post_init__(self):
        if not self.boundaries:
            n = self.n_frames
            self.boundaries = [0, n // 4, n // 2, 3 * n // 4, n]

    def auto_meta(self) -> dict:
        return {"model_version": "manual", "phase_source": "manual-ui", "flags": list(self.flags),
                "boundary_conf": [0.0] * 5}


class PhaseReviewService:
    """5 biên half-open [b0..b4]; pha i = [b[i], b[i+1]-1]."""

    def __init__(self, proposal, video_name: str, fps: float, total_frames: int, manual: bool = False):
        self.proposal, self.manual = proposal, manual
        self.video_name, self.fps, self.total = video_name, fps, int(total_frames)
        self.b: List[int] = [int(x) for x in proposal.boundaries]
        self.initial = list(self.b)
        self._undo: List[List[int]] = []
        self._redo: List[List[int]] = []
        self.edit_log: List[dict] = []
        self.t0 = time.monotonic()

    @property
    def edits(self) -> int:
        return len(self.edit_log)

    # ---- chỉnh biên ----
    def _push(self):
        self._undo.append(list(self.b)); self._redo.clear()

    def move_boundary(self, idx: int, delta: int) -> None:
        self.set_boundary(idx, self.b[idx] + delta)

    def set_boundary(self, idx: int, frame: int) -> None:
        lo = self.b[idx - 1] if idx > 0 else 0
        hi = self.b[idx + 1] if idx < 4 else self.total
        frame = max(lo, min(hi, int(frame)))
        if frame != self.b[idx]:
            self._push()
            self.edit_log.append({"boundary": idx, "from": self.b[idx], "to": frame})
            self.b[idx] = frame

    def is_absent(self, i: int) -> bool:
        return self.b[i + 1] <= self.b[i]

    def set_absent(self, i: int, absent: bool) -> None:
        if absent and not self.is_absent(i):
            self.set_boundary(i + 1, self.b[i])
        elif not absent and self.is_absent(i):
            hi = self.b[i + 2] if i + 2 <= 4 else self.total
            self.set_boundary(i + 1, min(max(self.initial[i + 1], self.b[i] + 1), hi))

    def undo(self):
        if self._undo:
            self._redo.append(list(self.b)); self.b = self._undo.pop()

    def redo(self):
        if self._redo:
            self._undo.append(list(self.b)); self.b = self._redo.pop()

    def reset(self):
        self._push(); self.b = list(self.initial); self.edit_log.append({"reset": True})

    # ---- chuyển đổi ----
    def windows(self) -> Dict[str, tuple]:
        return {n: (self.b[i], self.b[i + 1] - 1) for i, n in enumerate(PHASE_ORDER)
                if self.b[i + 1] > self.b[i]}

    def to_phase_set(self, marked_by: str = "auto") -> PhaseSet:
        rep = Rep.from_windows({n: self.windows().get(n) for n in PHASE_ORDER})
        orig = {n: (PhaseWindow(self.initial[i], self.initial[i + 1] - 1).to_dict()
                    if self.initial[i + 1] > self.initial[i] else PhaseWindow(absent=True).to_dict())
                for i, n in enumerate(PHASE_ORDER)}
        return PhaseSet(video=self.video_name, fps=float(self.fps), total_frames=int(self.total),
                        reps=[rep], marked_by=marked_by, auto_meta=self.proposal.auto_meta(),
                        auto_phases=orig)

    def validate(self) -> List[str]:
        return self.to_phase_set().validate()

    def warnings(self) -> List[str]:
        return self.to_phase_set().warnings()

    def low_conf_boundaries(self) -> List[int]:
        return [i for i, c in enumerate(self.proposal.boundary_conf) if c < C.CONF_MID]

    # ---- lưu ----
    def save_auto(self, path: Path) -> None:
        save_phases(path, self.to_phase_set("auto"), validate=True)

    def save_reviewed(self, auto_path: Path, out_path: Path) -> PhaseSet:
        ps = self.to_phase_set("manual" if self.manual else "auto_reviewed")
        ps.review = {"edits": list(self.edit_log), "seconds_spent": round(time.monotonic() - self.t0, 2),
                     "accepted_all": self.edits == 0, "source_file": str(auto_path)}
        errs = ps.validate()
        if errs:
            raise ValueError("; ".join(errs))
        save_phases(out_path, ps, validate=True)
        if not self.manual:                       # chỉ ghi log học liên tục khi có đề xuất của mô hình
            from stage3_features.phase_model.support.phase_feedback_log import log_review
            C.PHASE_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            log_review(str(auto_path), str(out_path), str(C.PHASE_FEEDBACK_LOG))
        return ps
