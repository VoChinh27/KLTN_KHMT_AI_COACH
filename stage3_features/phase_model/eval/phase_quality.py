"""Biến "không chắc" thành "bắt buộc duyệt": cờ chất lượng cho một đề xuất phân pha."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from stage3_features.phase_model.support.phase_common import NUM_PHASES

FLAG_MISSING_PHASE = "missing_phase"
FLAG_DURATION = "duration_out_of_prior"
FLAG_LOW_VISIBLE = "low_visible_ratio"
FLAG_MANY_NAN = "many_nan_frames"
FLAG_MULTI_REP = "multi_rep_suspected"
FLAG_DOMAIN_SHIFT = "domain_shift"
FLAG_SHORT = "short_clip"

ALL_FLAGS = (FLAG_MISSING_PHASE, FLAG_DURATION, FLAG_LOW_VISIBLE, FLAG_MANY_NAN,
             FLAG_MULTI_REP, FLAG_DOMAIN_SHIFT, FLAG_SHORT)

# Ngưỡng thời lượng mặc định (giây) cho 4 pha. Nên thay bằng prior học từ nhãn
# (xem ``duration_priors_from_boundaries``) - train_phase_model lưu sẵn vào model.
DEFAULT_DURATION_PRIORS_S: Dict[int, Sequence[float]] = {
    1: (0.5, 15.0), 2: (0.3, 8.0), 3: (0.1, 3.0), 4: (0.2, 6.0),
}


@dataclass
class QualityConfig:
    min_phase_frames: int = 2
    min_visible_ratio: float = 0.70
    max_low_visible_frac: float = 0.30
    max_interp_frac: float = 0.25
    max_nan_frac: float = 0.15
    min_clip_s: float = 3.0
    pre_margin_s: float = 0.3
    post_margin_s: float = 0.3
    multi_rep_min_run_s: float = 0.2
    multi_rep_outside_mass: float = 0.15
    domain_shift_z: float = 3.0


def _col(X: np.ndarray, names: Sequence[str], name: str) -> Optional[np.ndarray]:
    try:
        return X[:, list(names).index(name)]
    except ValueError:
        return None


def _runs(mask: np.ndarray, min_len: int) -> int:
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int(np.sum((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= min_len))


# --------------------------------------------------------------------------- #
# Prior & thống kê miền dữ liệu (tính lúc huấn luyện)
# --------------------------------------------------------------------------- #
def duration_priors_from_boundaries(boundary_list: Sequence[Sequence[int]], fps_list: Sequence[float],
                                    lo_q: float = 5.0, hi_q: float = 95.0,
                                    lo_margin: float = 0.7, hi_margin: float = 1.4,
                                    min_samples: int = 5) -> Dict[int, List[float]]:
    priors: Dict[int, List[float]] = {}
    for k in range(1, NUM_PHASES + 1):
        d = [(b[k] - b[k - 1]) / f for b, f in zip(boundary_list, fps_list) if b[k] > b[k - 1]]
        if len(d) >= min_samples:
            lo, hi = np.percentile(d, [lo_q, hi_q])
            priors[k] = [float(max(0.05, lo * lo_margin)), float(hi * hi_margin)]
        else:
            priors[k] = [float(v) for v in DEFAULT_DURATION_PRIORS_S[k]]
    return priors


def shift_columns(names: Sequence[str]) -> List[str]:
    return [n for n in names if n.startswith("abs_") or n == "visible_ratio"]


def compute_feature_stats(X_list: Sequence[np.ndarray], names: Sequence[str]) -> Dict[str, Any]:
    """Trung bình/độ lệch chuẩn (qua clip) của median từng cột "miền dữ liệu"."""
    cols = shift_columns(names)
    idx = [list(names).index(c) for c in cols]
    meds = np.array([[np.nanmedian(X[:, i]) for i in idx] for X in X_list], dtype=float)
    return {"columns": cols, "mean": np.nanmean(meds, axis=0).tolist(),
            "std": np.nanstd(meds, axis=0).tolist()}


def domain_shift_score(X: np.ndarray, names: Sequence[str], stats: Dict[str, Any]) -> float:
    z = []
    for c, mu, sd in zip(stats["columns"], stats["mean"], stats["std"]):
        col = _col(X, names, c)
        if col is None or not np.isfinite(col).any():
            continue
        floor = max(sd, 0.15 * abs(mu), 1e-2)
        z.append(abs(np.nanmedian(col) - mu) / floor)
    return float(np.median(z)) if z else 0.0


# --------------------------------------------------------------------------- #
def compute_flags(boundaries: Sequence[int], n_frames: int, fps: float, *,
                  X: Optional[np.ndarray] = None, feature_names: Optional[Sequence[str]] = None,
                  posteriors: Optional[np.ndarray] = None,
                  duration_priors: Optional[Dict[Any, Sequence[float]]] = None,
                  feature_stats: Optional[Dict[str, Any]] = None,
                  cfg: Optional[QualityConfig] = None) -> List[str]:
    cfg = cfg or QualityConfig()
    b = [int(v) for v in boundaries]
    flags: List[str] = []
    priors = {int(k): v for k, v in (duration_priors or DEFAULT_DURATION_PRIORS_S).items()}
    lengths = [b[k] - b[k - 1] for k in range(1, NUM_PHASES + 1)]

    if any(L < cfg.min_phase_frames for L in lengths):
        flags.append(FLAG_MISSING_PHASE)

    for k, L in enumerate(lengths, start=1):
        if L >= cfg.min_phase_frames and k in priors:
            lo, hi = priors[k]
            if not (lo <= L / fps <= hi):
                flags.append(FLAG_DURATION)
                break

    active = slice(b[0], max(b[-1], b[0] + 1))
    if X is not None and feature_names is not None:
        vis = _col(X, feature_names, "visible_ratio")
        if vis is not None and len(vis[active]):
            v = vis[active]
            if np.nanmedian(v) < cfg.min_visible_ratio or np.mean(v < 0.5) > cfg.max_low_visible_frac:
                flags.append(FLAG_LOW_VISIBLE)
        interp = _col(X, feature_names, "interp_flag")
        nanf = _col(X, feature_names, "nan_frac")
        bad = (interp is not None and len(interp[active]) and np.mean(interp[active]) > cfg.max_interp_frac) \
            or (nanf is not None and len(nanf[active]) and np.mean(nanf[active]) > cfg.max_nan_frac)
        if bad:
            flags.append(FLAG_MANY_NAN)

    if posteriors is not None and len(posteriors):
        lab = posteriors.argmax(axis=1)
        min_run = max(3, int(round(cfg.multi_rep_min_run_s * fps)))
        multi = _runs(lab == 1, min_run) >= 2 or _runs(lab == 4, min_run) >= 2
        mass = posteriors[:, 1:5].sum(axis=1)
        outside = np.ones(len(mass), dtype=bool)
        outside[b[0]:b[-1]] = False
        if outside.any() and mass[outside].sum() / max(mass.sum(), 1e-9) > cfg.multi_rep_outside_mass:
            multi = True
        if multi:
            flags.append(FLAG_MULTI_REP)

    if X is not None and feature_names is not None and feature_stats:
        if domain_shift_score(X, feature_names, feature_stats) > cfg.domain_shift_z:
            flags.append(FLAG_DOMAIN_SHIFT)

    pre, post = b[0], n_frames - b[-1]
    if (n_frames / fps < cfg.min_clip_s or pre < cfg.pre_margin_s * fps
            or post < cfg.post_margin_s * fps):
        flags.append(FLAG_SHORT)

    return [f for f in ALL_FLAGS if f in flags]
