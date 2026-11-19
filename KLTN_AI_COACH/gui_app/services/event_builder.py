"""SprintStartFullResult -> List[PauseEvent]. Quy tắc:
- chỉ dừng với technique_error (⚠/✗ tuỳ cờ); measurement_error/insufficient_data -> cảnh báo chất lượng
- mỗi pha dừng 1 lần, gom lỗi vào 1 thẻ (phân trang); khung dừng = khung giữa pha
- pha bộc phát: dùng khung launch nếu bundle còn events/launch_frame_idx
"""
from __future__ import annotations

from typing import List, Tuple

from gui_app import constants as C
from gui_app.session import PauseEvent

TECH = "technique_error"


def _etype(item) -> str:
    et = getattr(item, "error_type", None)
    return getattr(et, "value", et) or ""


def _launch_frame(sp) -> int | None:
    b = sp.result.bundle if sp.result else None
    for attr in ("sprint_start_summary", "explosive_summary"):
        s = getattr(b, attr, None)
        if s is not None and getattr(s, "launch_frame_idx", None) is not None:
            return int(s.launch_frame_idx)
    return None


def _phase_window(sp, windows) -> Tuple[int, int] | None:
    b = sp.result.bundle if sp.result else None
    for attr in ("static_summary", "sprint_start_summary"):
        s = getattr(b, attr, None)
        if s is not None and hasattr(s, "segment_start_frame"):
            return s.segment_start_frame, s.segment_end_frame
    return None


def build_events(result, windows: dict, include_warn: bool = False) -> Tuple[List[PauseEvent], List[str]]:
    events: List[PauseEvent] = []
    quality: List[str] = []
    for phase in result.sub_phase_order:
        sp = result.sub_phase_results.get(phase)
        if sp is None or not sp.has_score():
            continue
        items = []
        for it in sp.result.feedback.items:
            sym = it.status_symbol
            if sym == "✓" or it.level == "INFO":
                continue
            if _etype(it) != TECH:
                quality.append(f"[{C.PHASE_VI.get(phase, phase)}] {it.display_name}: {it.message}")
                continue
            if sym == "⚠" and not include_warn:
                continue
            items.append(dict(name=it.display_name, severity="error" if sym == "✗" else "warn",
                              message_error=f"{it.display_name}: {it.student_value:.1f}{it.unit} "
                                            f"(chuẩn {it.reference_center:.1f}{it.unit})",
                              message_correct=it.message))
        if not items:
            continue
        win = windows.get(phase) or _phase_window(sp, windows)
        if not win:
            continue
        frame = (_launch_frame(sp) if phase == "sprint_start" else None) or (win[0] + win[1]) // 2
        sev = "error" if any(i["severity"] == "error" for i in items) else "warn"
        events.append(PauseEvent(frame=frame, phase=phase, severity=sev, items=items))
    events.sort(key=lambda e: e.frame)
    return events, quality
