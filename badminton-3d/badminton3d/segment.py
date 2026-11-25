"""Tách đoạn cần chấm:  ready  = các đoạn đã TRỞ VỀ tư thế sẵn sàng (loại lúc di chuyển đỡ cầu và lúc đang ổn định lại)
                         serve  = từ BƯỚC CHUẨN BỊ phát đến khi phát xong (loại lúc nhặt cầu / đi lại), kèm 4 pha."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_closing, median_filter
from scipy.signal import find_peaks

from badminton3d.cleaning import CleanPose
from badminton3d.signals import arm_activity, body_speed

READY_STATES = {"INVALID": 0, "MOVING": 1, "SETTLING": 2, "READY": 3}


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Các đoạn [a, b] (gồm cả hai đầu) liên tiếp mask=True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _smooth(x: np.ndarray, fps: float, win_s: float) -> np.ndarray:
    k = max(1, int(round(win_s * fps)) | 1)
    y = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x)
    return median_filter(y, size=k, mode="nearest")


# ------------------------------------------------------------------ READY
@dataclass
class ReadySegmentation:
    state: np.ndarray                          # mã trạng thái theo frame (READY_STATES)
    episodes: List[Tuple[int, int]]            # các đoạn sẵn sàng được chấm
    speed: np.ndarray


def segment_ready(clean: CleanPose, cfg: dict) -> ReadySegmentation:
    c, fps = cfg["ready"], clean.fps
    sp = _smooth(body_speed(clean, c["speed_window_s"]), fps, 0.2)
    valid_body = np.all([clean.valid[j] for j in ("left_hip", "right_hip")], axis=0)
    n = clean.n
    state = np.full(n, READY_STATES["MOVING"], int)
    settle = int(round(c["settle_time_s"] * fps))
    moving, calm = True, 0
    for i in range(n):
        if not valid_body[i] or np.isnan(sp[i]):
            state[i] = READY_STATES["INVALID"]
            continue
        if sp[i] > c["v_move_mps"]:
            moving, calm = True, 0
        elif sp[i] < c["v_settle_mps"]:
            calm += 1
            if moving and calm >= settle:
                moving = False
        else:
            calm = 0 if moving else calm
        if moving:
            state[i] = READY_STATES["SETTLING"] if (calm > 0 and sp[i] < c["v_settle_mps"]) else READY_STATES["MOVING"]
        else:
            state[i] = READY_STATES["READY"]
    min_len = int(round(c["min_episode_s"] * fps))
    episodes = [(a, b) for a, b in _runs(state == READY_STATES["READY"]) if b - a + 1 >= min_len]
    keep = np.zeros(n, bool)
    for a, b in episodes:
        keep[a:b + 1] = True
    state[(state == READY_STATES["READY"]) & ~keep] = READY_STATES["SETTLING"]     # đoạn quá ngắn: không chấm
    return ReadySegmentation(state, episodes, sp)


# ------------------------------------------------------------------ SERVE
@dataclass
class ServeRep:
    prep_start: int
    swing_start: int
    turn: int
    peak: int
    contact_end: int
    end: int
    peak_activity: float
    phases: Dict[str, Tuple[int, int]]
    truncated: bool = False                    # video kết thúc khi cú phát chưa xong
    no_prep: bool = False                      # clip cắt sát: không có đoạn đứng yên trước cú vung (chuẩn bị lấy tạm 0,3 giây trước cú vung)


@dataclass
class ServeSegmentation:
    reps: List[ServeRep]
    activity: np.ndarray
    body: np.ndarray
    not_scored: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def _serve_pass(clean: CleanPose, cfg: dict, reach: Optional[np.ndarray], require_prep: bool, min_peak_rel: float = 0.0) -> ServeSegmentation:
    c, fps, n = cfg["serve"], clean.fps, clean.n
    act = _smooth(arm_activity(clean, c["speed_window_s"]), fps, 0.15)
    body = _smooth(body_speed(clean, c["speed_window_s"]), fps, 0.2)
    close_n = max(1, int(round(c["close_gap_s"] * fps)))
    peaks, _ = find_peaks(act, height=c["min_peak_mps"], prominence=c["min_prominence_mps"],
                          distance=max(1, int(round(c["peak_distance_s"] * fps))))
    if min_peak_rel > 0 and len(peaks):                                              # chế độ không cần chuẩn bị: chỉ giữ cú vung mạnh, bỏ nhịp thả tay
        peaks = np.array([p for p in peaks if act[p] >= min_peak_rel * act[peaks].max()], int)
    reps, last_end = [], -1
    for p in peaks:
        thr = max(c["low_abs_mps"], c["low_frac"] * act[p])
        active = binary_closing(act >= thr, structure=np.ones(close_n + 1, bool))     # lấp khoảng tụt ngắn (quay đầu)
        a = p
        while a - 1 >= 0 and active[a - 1]:
            a -= 1
        e = p
        while e + 1 < n and active[e + 1]:
            e += 1
        a = max(a, last_end + 1)
        if a > p:                                                                     # đỉnh nằm trong cú phát trước -> bỏ
            continue
        if (e - a + 1) > int(round(c["max_rep_s"] * fps)):                           # quá dài: giữ phần gần đỉnh
            a = max(a, p - int(round(c["max_rep_s"] * 0.5 * fps)))
            e = min(e, p + int(round(c["max_rep_s"] * 0.5 * fps)))
        # đoạn tăng tốc của tay (từ lúc rời trạng thái đứng yên đến ngưỡng 25% đỉnh) thuộc cú vung, không phải "chưa đứng yên"
        ramp_n = int(round(c.get("ramp_max_s", 0.4) * fps))
        while a - 1 >= 0 and a - 1 > last_end and (p - a) < int(round(c["max_rep_s"] * fps)) and act[a - 1] >= c["prep_arm_max_mps"] and ramp_n > 0:
            a, ramp_n = a - 1, ramp_n - 1
        # bước chuẩn bị = đoạn đứng yên liền ngay trước cú vung (bỏ nhặt cầu / đi lại trước đó)
        stable = (body < c["prep_body_max_mps"]) & (act < c["prep_arm_max_mps"])
        b = a - 1
        while b >= 0 and b > last_end and stable[b] and (a - b) <= int(round(c["prep_max_s"] * fps)):
            b -= 1
        prep_start = b + 1
        no_prep = False
        if (a - prep_start) < int(round(c["prep_min_s"] * fps)):
            if require_prep:
                continue                                                              # không có bước chuẩn bị (vd. thả lỏng tay sau cú phát) -> không phải 1 lần phát
            prep_start, no_prep = max(last_end + 1, a - int(round(c["prep_min_s"] * fps)), 0), True
        p = a + int(np.nanargmax(act[a:e + 1]))                                       # chạm cầu = đỉnh CAO NHẤT trong cửa sổ (không phải đỉnh đầu tiên)
        turn = a
        if reach is not None and np.isfinite(reach[a:p + 1]).any():
            turn = a + int(np.nanargmin(reach[a:p + 1]))                               # tay thu về sát thân nhất = hết rút vợt
        contact_end = min(p + int(round(c["contact_after_peak_s"] * fps)), e)
        end = min(e + int(round(c["end_hold_s"] * fps)), n - 1)
        ph = {"chuan_bi": (prep_start, max(prep_start, a - 1)), "rut_vot": (a, max(a, turn)),
              "danh_cau": (max(a, turn), max(turn, contact_end)), "ket_thuc": (min(contact_end + 1, end), end)}
        ph = {k: v for k, v in ph.items() if v[1] - v[0] + 1 >= 2}
        if no_prep:
            ph.pop("chuan_bi", None)                                                      # 0,3 giây giả định không phải bước chuẩn bị thật -> không chấm, không dựng chuẩn
        reps.append(ServeRep(prep_start, a, turn, int(p), contact_end, end, float(act[p]), ph, truncated=(e >= n - 2), no_prep=no_prep))
        last_end = end
    ns = np.ones(n, bool)
    for r in reps:
        ns[r.prep_start:r.end + 1] = False
    return ServeSegmentation(reps, act, body, ns)


def segment_serve(clean: CleanPose, cfg: dict, reach: Optional[np.ndarray] = None) -> ServeSegmentation:
    """Lượt 1: cú phát đầy đủ (có bước chuẩn bị đứng yên). Không tìm được cú nào và cho phép (serve.fallback_without_prep)
    thì lượt 2: clip cắt sát — lấy các cú vung mạnh (>= serve.fallback_min_rel_peak x cú mạnh nhất), đánh dấu no_prep để cảnh báo."""
    c = cfg["serve"]
    seg = _serve_pass(clean, cfg, reach, c.get("require_prep", True))
    if not seg.reps and c.get("require_prep", True) and c.get("fallback_without_prep", True):
        seg = _serve_pass(clean, cfg, reach, False, c.get("fallback_min_rel_peak", 0.6))
    return seg


def serve_from_windows(clean: CleanPose, cfg: dict, windows: List[Tuple[float, float]], reach: Optional[np.ndarray] = None) -> ServeSegmentation:
    """Đánh dấu tay: mỗi cửa sổ (giây) là một lần phát từ bước chuẩn bị tới khi phát xong. Pha vẫn được tách tự động trong cửa sổ."""
    c, fps, n = cfg["serve"], clean.fps, clean.n
    act = _smooth(arm_activity(clean, c["speed_window_s"]), fps, 0.15)
    body = _smooth(body_speed(clean, c["speed_window_s"]), fps, 0.2)
    reps, ns = [], np.ones(n, bool)
    for s_s, e_s in windows:
        a0, e = int(round(s_s * fps)), min(int(round(e_s * fps)), n - 1)
        if e - a0 < 5:
            continue
        peak = a0 + int(np.nanargmax(act[a0:e + 1])) if np.isfinite(act[a0:e + 1]).any() else (a0 + e) // 2
        sw = a0 + max(2, (peak - a0) // 2)                                            # nửa đầu = chuẩn bị, nửa sau tới đỉnh = vung
        turn = sw
        if reach is not None and np.isfinite(reach[sw:peak + 1]).any():
            turn = sw + int(np.nanargmin(reach[sw:peak + 1]))
        ce = min(peak + int(round(c["contact_after_peak_s"] * fps)), e)
        ph = {"chuan_bi": (a0, max(a0, sw - 1)), "rut_vot": (sw, max(sw, turn)), "danh_cau": (max(sw, turn), max(turn, ce)),
              "ket_thuc": (min(ce + 1, e), e)}
        ph = {k: v for k, v in ph.items() if v[1] - v[0] + 1 >= 2}
        reps.append(ServeRep(a0, sw, turn, int(peak), ce, e, float(act[peak]) if np.isfinite(act[peak]) else 0.0, ph, truncated=(e >= n - 2)))
        ns[a0:e + 1] = False
    return ServeSegmentation(reps, act, body, ns)
