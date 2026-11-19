"""Bọc draw_* của render_overlay. Trả khung BGR đã vẽ."""
from __future__ import annotations

from typing import Optional

import numpy as np

import render_overlay as RO


class OverlayRenderer:
    def __init__(self, frames: list, result, phases: dict, fps: float, total: int,
                 backend: str = "rtmpose", source: str = "auto_reviewed"):
        self.by_idx = {pf.frame_idx: pf for pf in frames}
        self.ctx = RO.RenderContext(fps=fps, total_frames=total, backend_name=backend,
                                    phase_source=source, phases=phases, result=result)

    def render(self, frame: np.ndarray, idx: int) -> np.ndarray:
        out = frame.copy()
        phase = RO.phase_at_frame(idx, self.ctx.phases)
        pf = self.by_idx.get(idx)
        texts: list = []
        if pf is not None:
            RO.draw_skeleton(out, pf, phase, RO.PHASE_FOCUS_JOINTS.get(phase, set()))
        RO.draw_panel(out, texts, self.ctx, idx, phase, has_pose=pf is not None)
        RO.draw_timeline(out, texts, self.ctx, idx)
        return RO.draw_texts_bgr(out, texts)
