"""Cổng vào duy nhất cho UI và pipeline: poses -> đề xuất phân pha.

Ghép: ``build_feature_matrix`` -> ``FramePhaseClassifier`` -> ``phase_decoder`` (tầng 2)
-> hiệu chỉnh độ tin cậy -> cờ chất lượng.

Giả định về ``stage3_features/phase_decoder.py`` (đã có):
  * ``PhaseDecoder(skip_penalty=..., emission_scale=..., prior_power=0.0)`` với
    ``.decode(posteriors, fps=...) -> DecodeResult`` (hoặc hàm module ``decode``);
  * ``DecodeResult`` có ``boundaries`` (5 chỉ số frame) hoặc đường đi theo frame
    (``path``/``labels``/``states``), và ``boundary_confidence()`` (5 giá trị).
Toàn bộ chỗ phụ thuộc nằm trong ``_run_decoder`` / ``_extract_result`` - chỉnh tại đó
nếu tên khác. Tham số không được decoder hỗ trợ sẽ tự bị lược bỏ.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from core.phase_schema import PHASE_ORDER as PHASE_NAMES
from stage3_features.phase_model.core.phase_classifier import FramePhaseClassifier
from stage3_features.phase_model.support.phase_common import (
    MARKED_AUTO, NUM_BOUNDARIES, boundaries_to_labels, labels_to_boundaries, make_payload,
    enforce_monotonic, read_json, utc_now_iso, write_json_atomic)
from stage3_features.phase_model.eval.phase_confidence import PlattCalibrator, confidence_level
from stage3_features.phase_model.core.phase_features import FeatureConfig, build_feature_matrix
from stage3_features.phase_model.eval.phase_quality import QualityConfig, compute_flags

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1
DEFAULT_MODELS_DIR = Path("models/phase_segmenter")
DEFAULT_DECODER_PARAMS: Dict[str, float] = {"skip_penalty": 2.0, "emission_scale": 1.0,
                                            "prior_power": 0.0}


# --------------------------------------------------------------------------- #
# Decoder adapter: shared contract for training and runtime
# --------------------------------------------------------------------------- #
def decode_posteriors(posteriors: np.ndarray, fps: float,
                      params: Optional[Dict[str, Any]] = None,
                      duration_priors: Optional[Dict[int, Sequence[float]]] = None
                      ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Giải mã posterior bằng HSMM; hậu nghiệm biên chỉ tính khi cần confidence."""
    from stage3_features.phase_model.core.phase_decoder import DurationPriors, PhaseDecoder, StatePrior
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    p = {**DEFAULT_DECODER_PARAMS, **(params or {})}
    priors = DurationPriors.default()
    if duration_priors:
        for phase_id, bounds in duration_priors.items():
            idx = int(phase_id) - 1
            if idx < 0 or idx >= len(PHASE_NAMES) or len(bounds) < 2:
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            if lo > 0 and hi >= lo:
                old = priors.priors[PHASE_NAMES[idx]]
                priors.priors[PHASE_NAMES[idx]] = StatePrior(
                    lo, hi, math.log(math.sqrt(lo * hi)), old.sigma_log
                )
    decoder = PhaseDecoder(
        priors=priors,
        skip_penalty=float(p.get("skip_penalty", 2.0)),
        emission_scale=float(p.get("emission_scale", 1.0)),
        prior_power=float(p.get("prior_power", 0.0)),
    )
    result = decoder.decode(np.asarray(posteriors, dtype=float), float(fps), compute_posteriors=False)
    if not result.success:
        raise RuntimeError("HSMM decode failed")
    path = result.state_path()
    b = enforce_monotonic(labels_to_boundaries(path), len(path))
    conf_map = result.boundary_confidence()
    conf = np.zeros(NUM_BOUNDARIES, dtype=float)
    for i, name in enumerate(PHASE_NAMES):
        conf[i] = float(conf_map.get(name, {}).get("start", 0.0))
    present = [name for name in PHASE_NAMES if not result.to_rep().phases[name].absent]
    if present:
        conf[-1] = float(conf_map.get(present[-1], {}).get("end", 0.0))
    return b, conf, path.astype(np.int64)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(models_dir: Any) -> Dict[str, Any]:
    p = Path(models_dir) / "registry.json"
    if not p.exists():
        return {"current": None, "versions": {}}
    return read_json(p)


def write_registry(models_dir: Any, registry: Dict[str, Any]) -> None:
    write_json_atomic(Path(models_dir) / "registry.json", registry)


# --------------------------------------------------------------------------- #
@dataclass
class PhaseProposal:
    rep: Dict[str, Any]                 # payload-like: boundaries_frames, phases, fps, n_frames
    boundary_conf: np.ndarray           # (5,) đã hiệu chỉnh nếu có calibrator
    flags: List[str]
    model_version: str
    posteriors: np.ndarray              # (N,6) hậu nghiệm tầng 1
    boundary_conf_raw: Optional[np.ndarray] = None
    confidence_levels: List[str] = field(default_factory=list)

    @property
    def boundaries(self) -> List[int]:
        return list(self.rep["boundaries_frames"])

    @property
    def windows(self) -> Dict[str, Optional[Tuple[int, int]]]:
        out = {}
        phases = self.rep.get("phases", {})
        for name in PHASE_NAMES:
            phase = phases.get(name, {})
            if phase.get("absent"):
                out[name] = None
            else:
                out[name] = (int(phase["start_frame"]), int(phase["end_frame"]))
        return out

    @property
    def version(self) -> str:
        return self.model_version

    @property
    def boundary_confidence(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for i, name in enumerate(PHASE_NAMES):
            out[name] = {"start": float(self.boundary_conf[i]),
                         "end": float(self.boundary_conf[i + 1]) if i < 3 else float(self.boundary_conf[-1])}
        return out

    def auto_meta(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "created": utc_now_iso(),
            "boundary_conf": [None if not np.isfinite(v) else round(float(v), 4)
                              for v in self.boundary_conf],
            "boundary_conf_raw": None if self.boundary_conf_raw is None else
            [None if not np.isfinite(v) else round(float(v), 4) for v in self.boundary_conf_raw],
            "confidence_level": list(self.confidence_levels),
            "flags": list(self.flags),
        }

    def to_payload(self, video: str, marked_by: str = MARKED_AUTO) -> Dict[str, Any]:
        payload = make_payload(video, self.rep["fps"], self.rep["n_frames"], self.boundaries,
                               marked_by, auto_meta=self.auto_meta())
        payload["auto_phases"] = payload["phases"]
        return payload


class PhaseSegmenter:
    def __init__(self, bundle: Dict[str, Any], model_version: str,
                 quality_cfg: Optional[QualityConfig] = None,
                 level_thresholds: Tuple[float, float] = (0.85, 0.60)):
        self.bundle = bundle
        self.model_version = model_version
        self.classifier: FramePhaseClassifier = bundle["classifier"]
        self.decoder_params: Dict[str, Any] = bundle.get("decoder_params", DEFAULT_DECODER_PARAMS)
        cal = bundle.get("calibrator")
        self.calibrator = PlattCalibrator.from_dict(cal) if cal else None
        self.feature_cfg = FeatureConfig.from_dict(bundle.get("feature_cfg"))
        self.feature_names: List[str] = list(bundle["feature_names"])
        self.duration_priors = bundle.get("duration_priors")
        self.feature_stats = bundle.get("feature_stats")
        self.tolerance_frames: int = int(bundle.get("tolerance_frames", 3))
        self.quality_cfg = quality_cfg or QualityConfig()
        self.level_thresholds = level_thresholds

    # ------------------------------------------------------------------ #
    @classmethod
    def from_registry(cls, path: Any = DEFAULT_MODELS_DIR, version: Optional[str] = None,
                      **kwargs: Any) -> "PhaseSegmenter":
        """``path``: thư mục chứa registry.json, hoặc trực tiếp file ``.joblib``."""
        p = Path(path)
        if p.is_file():
            return cls(joblib.load(p), p.stem, **kwargs)
        reg = load_registry(p)
        ver = version or reg.get("current")
        if not ver or ver not in reg.get("versions", {}):
            raise FileNotFoundError(f"Không có mô hình '{ver}' trong {p / 'registry.json'}")
        bundle = joblib.load(p / reg["versions"][ver]["model"])
        return cls(bundle, ver, **kwargs)

    # ------------------------------------------------------------------ #
    def propose(self, poses: np.ndarray, fps: float, is_coach: bool = False,
                front_side_hint: Optional[str] = None, video_id: str = "") -> PhaseProposal:
        X, names = build_feature_matrix(poses, fps, front_side_hint, is_coach=is_coach,
                                        cfg=self.feature_cfg)
        if names != self.feature_names:
            raise ValueError("Bộ đặc trưng khác lúc huấn luyện - cần huấn luyện lại mô hình")
        post = self.classifier.predict_proba(X)
        boundaries, raw_conf, _ = decode_posteriors(
            post, fps, self.decoder_params, duration_priors=self.duration_priors
        )
        conf = self.calibrator.predict(raw_conf) if self.calibrator else raw_conf
        n = len(post)
        flags = compute_flags(boundaries, n, fps, X=X, feature_names=names, posteriors=post,
                              duration_priors=self.duration_priors,
                              feature_stats=self.feature_stats, cfg=self.quality_cfg)
        hi, lo = self.level_thresholds
        levels = [confidence_level(c, hi, lo) for c in conf]
        rep = make_payload(video_id, fps, n, boundaries, MARKED_AUTO)
        return PhaseProposal(rep=rep, boundary_conf=np.asarray(conf, float), flags=flags,
                             model_version=self.model_version, posteriors=post,
                             boundary_conf_raw=np.asarray(raw_conf, float),
                             confidence_levels=levels)


