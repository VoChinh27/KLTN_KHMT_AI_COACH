"""Làm sạch tín hiệu khớp: mặt nạ visibility, nội suy khoảng mất NGẮN, làm mượt Savitzky-Golay, độ tin cậy từng khớp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.signal import savgol_filter

from badminton3d.pose_io import JOINTS, PoseData


def smooth_segments(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay riêng từng đoạn liên tục (không NaN) — không 'bắc cầu' qua khoảng mất dấu. x: (N,) hoặc (N,k)."""
    if x.ndim == 2:
        return np.column_stack([smooth_segments(x[:, k], window, poly) for k in range(x.shape[1])])
    out = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        seg = x[s:e]
        w = min(window, len(seg) if len(seg) % 2 else len(seg) - 1)
        out[s:e] = savgol_filter(seg, w, min(poly, w - 1)) if w >= 3 and w > poly else seg
    return out


def fill_short_gaps(valid: np.ndarray, max_len: int):
    """Trả (mask_cần_nội_suy, chỉ_số_trái, chỉ_số_phải) cho các đoạn mất <= max_len frame có frame hợp lệ ở HAI đầu."""
    n, fill = len(valid), np.zeros(len(valid), bool)
    left, right = np.full(n, -1), np.full(n, -1)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_len:
            fill[i:j], left[i:j], right[i:j] = True, i - 1, j
        i = j
    return fill, left, right


@dataclass
class CleanPose:
    pose: PoseData
    valid: Dict[str, np.ndarray]        # khớp dùng được (đo trực tiếp hoặc nội suy ngắn)
    rel: Dict[str, np.ndarray]          # độ tin cậy 0..1 (0 = không dùng được)
    interpolated: Dict[str, np.ndarray]
    w: Dict[str, np.ndarray]            # 3D đã làm sạch (NaN nếu không dùng được)
    xy: Dict[str, np.ndarray]           # 2D đã làm sạch

    @property
    def n(self) -> int:
        return self.pose.n

    @property
    def fps(self) -> float:
        return self.pose.fps


def smooth_window_frames(q: dict, fps: float) -> int:
    """Cửa sổ Savitzky-Golay theo THỜI GIAN (smooth_window_s) đổi ra số frame lẻ, để làm mượt như nhau ở 24/30/60 FPS.
    Không khai báo smooth_window_s thì dùng smooth_window (số frame) như cũ."""
    if q.get("smooth_window_s"):
        n = max(int(q["smooth_poly"]) + 2, int(round(q["smooth_window_s"] * fps)))
    else:
        n = int(q["smooth_window"])
    return n if n % 2 == 1 else n + 1


def clean_pose(pose: PoseData, cfg: dict, interpolate: bool = True, smooth: bool = True) -> CleanPose:
    q = cfg["quality"]
    gap_max = int(round(q["gap_interp_max_s"] * pose.fps))
    win = smooth_window_frames(q, pose.fps)
    valid, rel, interp, W, XY = {}, {}, {}, {}, {}
    for j in JOINTS:
        vis = np.nan_to_num(pose.vis[j], nan=0.0)
        raw = pose.detected & (vis >= q["visibility_min"]) & ~np.isnan(pose.w[j]).any(axis=1)
        w3, x2, r = pose.w[j].copy(), pose.xy[j].copy(), np.where(raw, np.clip(vis, 0, 1), 0.0)
        w3[~raw], x2[~raw] = np.nan, np.nan
        fi = np.zeros(pose.n, bool)
        if interpolate and gap_max > 0:
            fill, L, R = fill_short_gaps(raw, gap_max)
            for i in np.flatnonzero(fill):
                t = (i - L[i]) / float(R[i] - L[i])
                w3[i] = (1 - t) * w3[L[i]] + t * w3[R[i]]
                x2[i] = (1 - t) * x2[L[i]] + t * x2[R[i]]
                r[i] = q["interp_reliability"] * min(r[L[i]], r[R[i]])
            fi = fill
        if smooth:
            w3, x2 = smooth_segments(w3, win, q["smooth_poly"]), smooth_segments(x2, win, q["smooth_poly"])
        valid[j], rel[j], interp[j], W[j], XY[j] = raw | fi, r, fi, w3, x2
    return CleanPose(pose, valid, rel, interp, W, XY)
