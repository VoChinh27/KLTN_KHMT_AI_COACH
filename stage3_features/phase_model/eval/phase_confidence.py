"""Độ tin cậy đã hiệu chỉnh cho biên pha.

* ``PlattCalibrator``: logistic 2 tham số (a, b) trên logit(điểm thô), fit trên dự đoán
  out-of-fold. Dùng Platt thay vì isotonic vì chỉ ~120 điểm biên.
  Có hiệu chỉnh nhãn của Platt và co (shrink) về phép đồng nhất (a=1, b=0).
* ``bootstrap_ci`` / ``paired_bootstrap_diff``: khoảng tin cậy (có thể theo cụm người).
* ``confidence_level``: cao / trung bình / thấp.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

LEVEL_HIGH = "cao"
LEVEL_MID = "trung bình"
LEVEL_LOW = "thấp"
LEVELS = (LEVEL_LOW, LEVEL_MID, LEVEL_HIGH)


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class PlattCalibrator:
    def __init__(self, l2: float = 1e-2, eps: float = 1e-4):
        self.l2 = l2
        self.eps = eps
        self.a_: float = 1.0
        self.b_: float = 0.0
        self.n_fit_: int = 0
        self.fitted_: bool = False

    def fit(self, scores: Sequence[float], correct: Sequence[float]) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(correct, dtype=float)
        m = np.isfinite(s) & np.isfinite(y)
        s, y = s[m], y[m]
        if s.size < 2 or len(np.unique(y > 0.5)) < 2:
            raise ValueError("Cần cả mẫu đúng và sai (>=2 điểm) để hiệu chỉnh Platt")
        n_pos = float(np.sum(y > 0.5))
        n_neg = float(len(y) - n_pos)
        t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        A = np.column_stack([_logit(s, self.eps), np.ones_like(s)])
        w0 = np.array([1.0, 0.0])
        w = w0.copy()
        for _ in range(100):
            p = _sigmoid(A @ w)
            g = A.T @ (p - t) + 2 * self.l2 * (w - w0)
            W = np.maximum(p * (1 - p), 1e-9)
            H = (A * W[:, None]).T @ A + 2 * self.l2 * np.eye(2)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.a_, self.b_ = float(w[0]), float(w[1])
        self.n_fit_ = int(s.size)
        self.fitted_ = True
        return self

    def predict(self, scores: Any) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        out = np.full(s.shape, np.nan)
        m = np.isfinite(s)
        out[m] = _sigmoid(self.a_ * _logit(s[m], self.eps) + self.b_)
        return out

    __call__ = predict

    # ---- lưu / nạp (JSON, không phụ thuộc pickle) ----
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "platt", "a": self.a_, "b": self.b_, "l2": self.l2, "eps": self.eps,
                "n_fit": self.n_fit_}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlattCalibrator":
        c = cls(l2=d.get("l2", 1e-2), eps=d.get("eps", 1e-4))
        c.a_, c.b_ = float(d["a"]), float(d["b"])
        c.n_fit_ = int(d.get("n_fit", 0))
        c.fitted_ = True
        return c

    def save(self, path: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "PlattCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Số đo hiệu chỉnh
# --------------------------------------------------------------------------- #
def brier_score(p: Sequence[float], y: Sequence[float]) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else float("nan")


def expected_calibration_error(p: Sequence[float], y: Sequence[float], n_bins: int = 5) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = ids == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _cluster_indices(groups: Optional[Sequence[Any]], n: int) -> list:
    if groups is None:
        return [np.array([i]) for i in range(n)]
    g = np.asarray(groups)
    return [np.flatnonzero(g == u) for u in np.unique(g)]


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 groups: Optional[Sequence[Any]] = None) -> Tuple[float, float, float]:
    """(ước lượng, cận dưới, cận trên). Nếu có ``groups`` -> bootstrap theo cụm (người)."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    if groups is not None:
        groups = np.asarray(groups)[keep]
    v = v[keep]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    clusters = _cluster_indices(groups, v.size)
    rng = np.random.default_rng(seed)
    est = float(stat(v))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        boots[i] = stat(v[np.concatenate([clusters[j] for j in pick])])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def paired_bootstrap_diff(baseline: Sequence[float], candidate: Sequence[float],
                          groups: Optional[Sequence[Any]] = None, n_boot: int = 5000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """So sánh sai số (nhỏ hơn = tốt hơn). diff = baseline - candidate (dương = candidate tốt hơn)."""
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    if b.shape != c.shape:
        raise ValueError("baseline và candidate phải cùng độ dài")
    keep = np.isfinite(b) & np.isfinite(c)
    d = (b - c)[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    if d.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "candidate_better": False}
    clusters = _cluster_indices(g, d.size)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        boots[i] = d[np.concatenate([clusters[j] for j in pick])].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": int(d.size), "mean_diff": float(d.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "p_value": float(np.mean(boots <= 0.0)),
            "candidate_better": bool(lo > 0.0)}


# --------------------------------------------------------------------------- #
# Mức tin cậy
# --------------------------------------------------------------------------- #
def confidence_level(p: float, high: float = 0.85, low: float = 0.60) -> str:
    if p is None or not np.isfinite(p):
        return LEVEL_LOW
    if p >= high:
        return LEVEL_HIGH
    if p >= low:
        return LEVEL_MID
    return LEVEL_LOW


def boundary_confidence_from_decode(decode_result: Any,
                                    calibrator: Optional[PlattCalibrator] = None) -> np.ndarray:
    """Lấy ``DecodeResult.boundary_confidence()`` rồi hiệu chỉnh (nếu có calibrator)."""
    raw = np.asarray(decode_result.boundary_confidence(), dtype=float)
    return calibrator.predict(raw) if calibrator is not None else raw
