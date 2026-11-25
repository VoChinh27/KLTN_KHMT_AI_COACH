"""Tín hiệu chuyển động MÉT/GIÂY: tốc độ thân (từ ảnh, quy đổi bằng chiều dài thân 3D) và tốc độ cổ tay (3D)."""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from badminton3d.cleaning import CleanPose, smooth_segments


def body_scale(clean: CleanPose) -> float:
    """Mét trên pixel = chiều dài thân 3D (m) / chiều dài thân trong ảnh (px), lấy trung vị. Thay cho giả định 'thân = 0,5 m'."""
    def mid(src, names):
        ok = np.all([clean.valid[n] for n in names], axis=0)
        return np.mean([src[n] for n in names], axis=0), ok
    s3, o1 = mid(clean.w, ["left_shoulder", "right_shoulder"])
    h3, o2 = mid(clean.w, ["left_hip", "right_hip"])
    s2, o3 = mid(clean.xy, ["left_shoulder", "right_shoulder"])
    h2, o4 = mid(clean.xy, ["left_hip", "right_hip"])
    ok = o1 & o2 & o3 & o4
    if ok.sum() < 5:
        return float("nan")
    m = np.linalg.norm(s3[ok] - h3[ok], axis=1) / np.maximum(np.linalg.norm(s2[ok] - h2[ok], axis=1), 1e-6)
    return float(np.median(m))


def view_yaw_deg(clean: CleanPose) -> float:
    """Góc quay của người so với camera (độ): 0 = chính diện (vai song song mặt phẳng ảnh), 90 = nhìn ngang. Trung vị theo video."""
    ok = clean.valid["left_shoulder"] & clean.valid["right_shoulder"]
    if ok.sum() < 5:
        return float("nan")
    d = clean.w["right_shoulder"][ok] - clean.w["left_shoulder"][ok]
    return float(np.median(np.degrees(np.arctan2(np.abs(d[:, 2]), np.abs(d[:, 0])))))


def _speed(pos: np.ndarray, fps: float, win_s: float) -> np.ndarray:
    w = max(1, int(round(win_s * fps)))
    v = np.full(len(pos), np.nan)
    v[w:] = np.linalg.norm(pos[w:] - pos[:-w], axis=1) / (w / fps)
    return v


def body_speed(clean: CleanPose, win_s: float = 0.2) -> np.ndarray:
    """Tốc độ giữa hông (m/s). Toạ độ world gốc ở hông nên phải đo từ ảnh rồi nhân hệ số mét/pixel."""
    sc = body_scale(clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # khung không có hông nào thấy được -> NaN, không cần cảnh báo
        hip = np.nanmean(np.stack([clean.xy["left_hip"], clean.xy["right_hip"]]), axis=0)
    return _speed(hip, clean.fps, win_s) * sc


def wrist_speed(clean: CleanPose, side: str, win_s: float = 0.2) -> np.ndarray:
    """Tốc độ cổ tay (m/s) so với hông — chính toạ độ world (đã tính theo gốc giữa hông)."""
    return _speed(clean.w[f"{side}_wrist"], clean.fps, win_s)


def arm_activity(clean: CleanPose, win_s: float = 0.2) -> np.ndarray:
    """Hoạt động của tay (m/s) = max tốc độ hai cổ tay thấy được; không phụ thuộc hướng tiến hay tay nào là tay vợt."""
    sp = np.vstack([wrist_speed(clean, "left", win_s), wrist_speed(clean, "right", win_s)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmax(sp, axis=0)
    return out
