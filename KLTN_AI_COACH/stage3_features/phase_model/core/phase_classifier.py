"""Tầng 1: phân loại pha theo từng frame -> hậu nghiệm (N, 6).

Bắt buộc ``class_weight='balanced'`` để hậu nghiệm xấp xỉ prior đều, khớp với
``prior_power=0`` của decoder (tầng 2).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np

from stage3_features.phase_model.support.phase_common import NUM_CLASSES

logger = logging.getLogger(__name__)

PROBA_EPS = 1e-6

DEFAULT_LGBM_PARAMS: Dict[str, Any] = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    verbose=-1, n_jobs=-1,
)
DEFAULT_LOGREG_PARAMS: Dict[str, Any] = dict(C=1.0, max_iter=3000)


def boundary_proximity_weights(labels: Sequence[int], fps: float, radius_s: float = 2.0,
                               peak: float = 4.0, base: float = 1.0) -> np.ndarray:
    """Trọng số tam giác: = ``peak`` tại biên, giảm tuyến tính về ``base`` ở ±radius_s."""
    lab = np.asarray(labels)
    n = len(lab)
    w = np.full(n, float(base))
    changes = np.flatnonzero(np.diff(lab) != 0) + 1
    if changes.size == 0:
        return w
    radius = max(1.0, radius_s * fps)
    idx = np.arange(n)
    pos = np.searchsorted(changes, idx)
    left = changes[np.clip(pos - 1, 0, len(changes) - 1)]
    right = changes[np.clip(pos, 0, len(changes) - 1)]
    dist = np.minimum(np.abs(idx - left), np.abs(idx - right)).astype(float)
    return base + (peak - base) * np.clip(1.0 - dist / radius, 0.0, 1.0)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


class FramePhaseClassifier:
    """LightGBM (mặc định) hoặc hồi quy logistic; luôn trả đủ ``n_classes`` cột."""

    def __init__(self, backend: str = "lightgbm", params: Optional[Dict[str, Any]] = None,
                 n_classes: int = NUM_CLASSES, random_state: int = 0):
        if backend not in ("lightgbm", "logreg"):
            raise ValueError("backend phải là 'lightgbm' hoặc 'logreg'")
        if backend == "lightgbm" and not _lightgbm_available():
            logger.warning("Không có lightgbm -> dùng logistic regression")
            backend = "logreg"
        self.backend = backend
        self.params = dict(params or {})
        self.n_classes = n_classes
        self.random_state = random_state
        self.model_: Any = None
        self.feature_names_: Optional[list] = None

    # ------------------------------------------------------------------ #
    def _make_model(self) -> Any:
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            p = {**DEFAULT_LGBM_PARAMS, **self.params}
            return LGBMClassifier(objective="multiclass", class_weight="balanced",
                                  random_state=self.random_state, **p)
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        p = {**DEFAULT_LOGREG_PARAMS, **self.params}
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=self.random_state, **p))

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X phải 2 chiều, nhận {X.shape}")
        return np.where(np.isinf(X), np.nan, X)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            feature_names: Optional[Sequence[str]] = None) -> "FramePhaseClassifier":
        X = self._clean(X)
        y = np.asarray(y, dtype=np.int64)
        if len(X) != len(y):
            raise ValueError("X và y lệch độ dài")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"Nhãn phải trong [0, {self.n_classes})")
        if len(np.unique(y)) < 2:
            raise ValueError("Cần ít nhất 2 lớp trong dữ liệu huấn luyện")
        model = self._make_model()
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if self.backend == "lightgbm":
                model.fit(X, y, sample_weight=sw)
            else:
                model.fit(X, y, logisticregression__sample_weight=sw)
        else:
            model.fit(X, y)
        self.model_ = model
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Chưa fit")
        X = self._clean(X)
        raw = self.model_.predict_proba(X)
        classes = np.asarray(self.model_.classes_, dtype=int)
        full = np.full((len(X), self.n_classes), PROBA_EPS, dtype=np.float64)
        full[:, classes] = raw
        full = np.clip(full, PROBA_EPS, None)
        return full / full.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    def save(self, path: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Any) -> "FramePhaseClassifier":
        obj = joblib.load(path)
        if not isinstance(obj, FramePhaseClassifier):
            raise TypeError(f"{path} không chứa FramePhaseClassifier")
        return obj
