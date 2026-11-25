"""Đặc trưng theo ĐƠN VỊ CHẤM: ready = một đoạn sẵn sàng; serve = một lần phát (4 pha). Giá trị = trung vị có trọng số độ tin cậy."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from badminton3d.angles import choose_arm, ready_features, serve_features, wrist_reach
from badminton3d.cleaning import CleanPose
from badminton3d.segment import ReadySegmentation, ServeSegmentation


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    ok = ~np.isnan(values) & (weights > 0)
    if not ok.any():
        return float("nan")
    v, w = values[ok], weights[ok]
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def unit_stats(feat: dict, a: int, b: int) -> dict:
    """Thống kê một đặc trưng trên [a, b]: giá trị, độ tin cậy, độ phủ, tỉ lệ frame phải dùng giá trị thay thế."""
    val, rel, sub = feat["value"][a:b + 1], feat["rel"][a:b + 1], feat["substituted"][a:b + 1]
    ok = ~np.isnan(val) & (rel > 0)
    n = b - a + 1
    if not ok.any():
        return {"value": float("nan"), "rel": 0.0, "coverage": 0.0, "substituted": 0.0, "n_frames": n}
    cov = float(ok.mean())
    return {"value": weighted_median(val, rel), "rel": float(np.mean(rel[ok]) * cov), "coverage": cov,
            "substituted": float(np.mean(sub[ok])), "n_frames": n}


def ready_units(clean: CleanPose, seg: ReadySegmentation, cfg: dict) -> List[dict]:
    feats = ready_features(clean, cfg)
    units = []
    for k, (a, b) in enumerate(seg.episodes):
        units.append({"id": f"R{k + 1}", "frames": (a, b), "duration_s": (b - a + 1) / clean.fps,
                      "features": {n: unit_stats(f, a, b) for n, f in feats.items()}})
    return units


def serve_setup(clean: CleanPose, cfg: dict, racket_hand: str, frames: Optional[np.ndarray] = None) -> Tuple[Dict, np.ndarray, Dict]:
    arm = choose_arm(clean, racket_hand, cfg["quality"]["min_reliability_to_score"], frames)
    feats = serve_features(clean, cfg, arm)
    return arm, wrist_reach(clean, arm["side"])[0], feats


def serve_units(clean: CleanPose, seg: ServeSegmentation, feats: dict, cfg: dict) -> List[dict]:
    out = []
    for k, r in enumerate(seg.reps):
        out.append({"id": f"S{k + 1}", "frames": (r.prep_start, r.end), "duration_s": (r.end - r.prep_start + 1) / clean.fps,
                    "phase_frames": r.phases, "peak_frame": r.peak, "truncated": r.truncated, "no_prep": r.no_prep,
                    "phases": {ph: {n: unit_stats(f, a, b) for n, f in feats.items()} for ph, (a, b) in r.phases.items()}})
    return out
