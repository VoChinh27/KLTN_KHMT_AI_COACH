"""CHUẨN TỪ HLV: từ 1–3 (hoặc nhiều hơn) video HLV tập đúng -> giá trị chuẩn + dung sai cho từng góc (và từng pha với serve)."""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from badminton3d.features import weighted_median


def _tol(values: List[float], floor: float, cfg: dict) -> float:
    s = cfg["scoring"]
    if len(values) >= 2:
        return max(floor, s["range_margin"] * (max(values) - min(values)) / 2.0)
    return floor * s["single_video_floor_factor"]


def _feature_ref(per_video: Dict[str, float], spec: dict, cfg: dict) -> Optional[dict]:
    vals = [v for v in per_video.values() if v == v]
    if not vals:
        return None
    return {"center": float(np.median(vals)), "tol": float(_tol(vals, spec["floor"], cfg)), "spread": [float(min(vals)), float(max(vals))],
            "n_videos": len(vals), "level": spec["level"], "unit": spec["unit"], "label": spec["label"], "per_video": {k: float(v) for k, v in per_video.items() if v == v}}


def _video_value(units_vals: List[tuple]) -> float:
    """units_vals = [(giá trị, trọng số)] của các đơn vị trong CÙNG một video -> trung vị có trọng số."""
    if not units_vals:
        return float("nan")
    v = np.array([x[0] for x in units_vals], float)
    w = np.array([x[1] for x in units_vals], float)
    return weighted_median(v, w)


def build_reference(exercise: str, videos: Dict[str, List[dict]], cfg: dict, meta: Optional[dict] = None) -> dict:
    """videos[stem] = danh sách đơn vị chấm (ready_units / serve_units) của video HLV đó."""
    min_rel = cfg["quality"]["min_reliability_to_score"]
    spec = cfg["features"][exercise]
    ref = {"exercise": exercise, "built_from": sorted(videos), "n_videos": len(videos), "built_on": date.today().isoformat(),
           "n_units": {k: len(v) for k, v in videos.items()}, "meta": meta or {}}
    if exercise == "ready":
        feats = {}
        for f, sp in spec.items():
            per = {}
            for stem, units in videos.items():
                per[stem] = _video_value([(u["features"][f]["value"], u["duration_s"] * u["features"][f]["rel"])
                                          for u in units if u["features"][f]["rel"] >= min_rel])
            r = _feature_ref(per, sp, cfg)
            if r:
                feats[f] = r
        ref["features"] = feats
    else:
        phases = {}
        for ph in spec["phase_weights"]:
            feats = {}
            for f, sp in spec.items():
                if f in ("phase_weights", "phase_labels"):
                    continue
                per = {}
                for stem, units in videos.items():
                    per[stem] = _video_value([(u["phases"][ph][f]["value"], u["phases"][ph][f]["rel"])
                                              for u in units if ph in u["phases"] and u["phases"][ph][f]["rel"] >= min_rel])
                r = _feature_ref(per, sp, cfg)
                if r:
                    feats[f] = r
            if feats:
                phases[ph] = feats
        ref["phases"] = phases
    return ref


def save_reference(ref: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, ensure_ascii=False, indent=2)


def load_reference(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chưa có chuẩn HLV: {path}. Chạy `python -m badminton3d build-reference` trước.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
