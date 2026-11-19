"""Tăng cường dữ liệu cho tập TRAIN (không dùng cho validation/test).

Mọi hàm biến đổi đồng bộ ``poses`` (N, 17, 3) và ``labels`` (N,) và trả bản sao mới.
Nhãn: 0=pre, 5=post nên pad đầu/cuối dùng đúng 0 và 5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from stage3_features.phase_model.support.phase_common import NUM_CLASSES

# Cặp trái/phải theo COCO-17
LR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
_SH, _HIP = (5, 6), (11, 12)


def _check(poses: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
        raise ValueError(f"poses phải (N,17,3), nhận {poses.shape}")
    if len(poses) != len(labels):
        raise ValueError("poses và labels lệch độ dài")
    return poses, labels


def _torso_len(poses: np.ndarray) -> float:
    sh = np.nanmean(poses[:, list(_SH), :2], axis=1)
    hip = np.nanmean(poses[:, list(_HIP), :2], axis=1)
    d = np.linalg.norm(sh - hip, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 1.0


def time_stretch(poses: np.ndarray, labels: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """factor>1: chuyển động chậm lại (clip dài hơn); factor<1: nhanh hơn. Dùng 0.8-1.2."""
    poses, labels = _check(poses, labels)
    if factor <= 0:
        raise ValueError("factor phải > 0")
    n = len(poses)
    n_new = max(2, int(round(n * factor)))
    src = np.linspace(0, n - 1, n_new)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0)[:, None, None]
    a, b = poses[i0], poses[i1]
    out = (1 - w) * a + w * b
    # nếu một đầu NaN, lấy đầu còn lại thay vì lan truyền NaN
    out = np.where(np.isnan(out), np.where(np.isnan(a), b, a), out)
    new_labels = labels[np.clip(np.rint(src).astype(int), 0, n - 1)]
    return out, new_labels


def pad_head_tail(poses: np.ndarray, labels: np.ndarray, head: int, tail: int,
                  rng: Optional[np.random.Generator] = None,
                  jitter_rel: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    """Thêm ``head``/``tail`` frame lặp lại frame đầu/cuối (nhãn 0 / 5)."""
    poses, labels = _check(poses, labels)
    head, tail = max(0, int(head)), max(0, int(tail))
    rng = rng or np.random.default_rng()
    sigma = jitter_rel * _torso_len(poses)

    def block(frame: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return np.empty((0,) + frame.shape)
        blk = np.repeat(frame[None], k, axis=0)
        blk[..., :2] += rng.normal(0.0, sigma, size=blk[..., :2].shape)
        return blk

    out = np.concatenate([block(poses[0], head), poses, block(poses[-1], tail)], axis=0)
    lab = np.concatenate([np.zeros(head, np.int64), labels, np.full(tail, NUM_CLASSES - 1, np.int64)])
    return out, lab


def mirror_left_right(poses: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lật ngang (phản chiếu qua tâm x của clip) và hoán đổi keypoint trái/phải."""
    poses, labels = _check(poses, labels)
    out = poses.copy()
    x = out[..., 0]
    if np.isfinite(x).any():
        c = np.nanmax(x) + np.nanmin(x)
        out[..., 0] = c - x
    for a, b in LR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out, labels.copy()


def jitter_keypoints(poses: np.ndarray, labels: np.ndarray, sigma_rel: float = 0.01,
                     rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Nhiễu Gauss độc lập mỗi keypoint, sigma = sigma_rel * độ dài thân."""
    poses, labels = _check(poses, labels)
    rng = rng or np.random.default_rng()
    out = poses.copy()
    out[..., :2] += rng.normal(0.0, sigma_rel * _torso_len(poses), size=out[..., :2].shape)
    return out, labels.copy()


def drop_frames(poses: np.ndarray, labels: np.ndarray, n_gaps: int = 3, max_gap: int = 4,
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mô phỏng mất phát hiện: đặt toàn bộ keypoint của vài đoạn ngắn thành NaN, conf=0."""
    poses, labels = _check(poses, labels)
    rng = rng or np.random.default_rng()
    out = poses.copy()
    n = len(out)
    for _ in range(max(0, n_gaps)):
        length = int(rng.integers(1, max(2, max_gap + 1)))
        start = int(rng.integers(0, max(1, n - length + 1)))
        out[start:start + length, :, :2] = np.nan
        out[start:start + length, :, 2] = 0.0
    return out, labels.copy()


@dataclass
class AugmentConfig:
    stretch_range: Tuple[float, float] = (0.8, 1.2)
    p_stretch: float = 0.8
    p_pad: float = 0.7
    max_pad_s: float = 1.5
    p_mirror: float = 0.5
    p_jitter: float = 0.8
    jitter_sigma_rel: float = 0.01
    p_drop: float = 0.5
    drop_gaps: Tuple[int, int] = (1, 4)
    drop_max_gap: int = 4
    fps: float = 30.0  # chỉ dùng đổi max_pad_s -> số frame

    def to_dict(self) -> dict:
        return {**self.__dict__, "stretch_range": list(self.stretch_range),
                "drop_gaps": list(self.drop_gaps)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AugmentConfig":
        if not d:
            return cls()
        d = dict(d)
        for k in ("stretch_range", "drop_gaps"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def augment_clip(poses: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                 cfg: Optional[AugmentConfig] = None, fps: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Áp một chuỗi biến đổi ngẫu nhiên (đồng bộ pose & nhãn)."""
    cfg = cfg or AugmentConfig()
    fps = fps or cfg.fps
    p, l = _check(poses, labels)
    if rng.random() < cfg.p_stretch:
        p, l = time_stretch(p, l, float(rng.uniform(*cfg.stretch_range)))
    if rng.random() < cfg.p_pad:
        mx = max(1, int(round(cfg.max_pad_s * fps)))
        p, l = pad_head_tail(p, l, int(rng.integers(0, mx + 1)), int(rng.integers(0, mx + 1)), rng)
    if rng.random() < cfg.p_mirror:
        p, l = mirror_left_right(p, l)
    if rng.random() < cfg.p_jitter:
        p, l = jitter_keypoints(p, l, cfg.jitter_sigma_rel, rng)
    if rng.random() < cfg.p_drop:
        p, l = drop_frames(p, l, int(rng.integers(cfg.drop_gaps[0], cfg.drop_gaps[1] + 1)),
                           cfg.drop_max_gap, rng)
    return p, l
