"""Dữ liệu giả lập: một người đứng, toạ độ world (m, gốc giữa hông; y XUỐNG) + toạ độ ảnh (px) + visibility."""
from __future__ import annotations

import numpy as np
import pandas as pd

from badminton3d.pose_io import JOINTS

BASE = {"left_shoulder": (-0.18, -0.50, 0.0), "right_shoulder": (0.18, -0.50, 0.0), "left_elbow": (-0.22, -0.25, 0.10), "right_elbow": (0.22, -0.25, 0.10),
        "left_wrist": (-0.20, -0.05, 0.30), "right_wrist": (0.20, -0.05, 0.30), "left_hip": (-0.10, 0.0, 0.0), "right_hip": (0.10, 0.0, 0.0),
        "left_knee": (-0.10, 0.45, 0.05), "right_knee": (0.10, 0.45, 0.05), "left_ankle": (-0.12, 0.90, 0.0), "right_ankle": (0.12, 0.90, 0.0)}
PX = 700.0


def make_df(n=150, fps=30.0, name="syn_ready_01", override=None, hide=None, hip_dx=None, wrist=None):
    """override: {joint: (x,y,z)} đổi vị trí gốc; hide: {joint: mảng bool (True = bị che, visibility 0.1)};
    hip_dx: mảng (n,) dịch toàn thân trong ảnh (px) -> chuyển động thân; wrist: {side: mảng (n,3)} cộng thêm vào cổ tay world."""
    pos = dict(BASE)
    pos.update(override or {})
    rows = []
    hip_dx = np.zeros(n) if hip_dx is None else np.asarray(hip_dx, float)
    for i in range(n):
        r = {"video": name, "frame": i, "fps_src": fps, "width": 1280, "height": 720, "detected": True}
        for j in JOINTS:
            w = np.array(pos[j], float)
            if wrist and j.endswith("_wrist") and j.split("_")[0] in wrist:
                w = w + np.asarray(wrist[j.split("_")[0]])[i]
            vis = 0.1 if (hide and j in hide and hide[j][i]) else 0.95
            r.update({f"{j}_x": 640 + PX * w[0] + hip_dx[i], f"{j}_y": 300 + PX * w[1], f"{j}_vis": vis, f"{j}_wx": w[0], f"{j}_wy": w[1], f"{j}_wz": w[2]})
        rows.append(r)
    return pd.DataFrame(rows)


def burst(n, start, length, amp, axis=2):
    """Cổ tay đẩy ra trước amp mét trong `length` frame (nội suy cosin) rồi giữ nguyên."""
    out = np.zeros((n, 3))
    for i in range(n):
        if i >= start + length:
            out[i, axis] = amp
        elif i >= start:
            out[i, axis] = amp * (1 - np.cos(np.pi * (i - start) / length)) / 2
    return out
