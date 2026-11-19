"""
stage4_scoring/scorer_factory.py
==================================
Mục đích : "Single source of truth" cho việc khởi tạo `Scorer`.

Trước đây logic dựng Scorer (đọc feature_weights, feature_specs, tolerance_map,
level_weights từ SportPlugin) nằm trong `Pipeline._get_scorer`, nên các module
khác (ví dụ stage5_eval/loso_cv_eval.py) phải tự viết lại một scorer riêng và
số liệu đánh giá không còn khớp pipeline chính.

Giờ MỌI nơi cần Scorer đều gọi `build_scorer(plugin, phase, fps_actual)`:
    - core.pipeline.Pipeline._get_scorer
    - stage5_eval.loso_cv_eval.LosoCvEvaluator._score_clip

Hành vi giữ NGUYÊN so với bản `Pipeline._get_scorer` cũ:
    - weights        : plugin.feature_weights(phase)
    - feature_specs  : {key: {level, scoring_type, unit, fps_reference}}
    - tolerance_map  : union(feature_specs, fixed_reference_ranges,
                       tolerance.overrides) -> plugin.tolerance_for(key)
    - level_weights  : plugin.level_weights() (nếu plugin có)
    - fps_actual     : cho scoring_type = frame_window
"""

from __future__ import annotations

import logging
from typing import Dict, Set

from core.sport_interface import SportPlugin
from stage4_scoring.scorer import Scorer

logger = logging.getLogger(__name__)

__all__ = ["build_scorer"]


def _collect_feature_specs(plugin: SportPlugin, phase: str) -> Dict[str, Dict]:
    """Chuẩn hóa feature_specs của plugin về dict thuần cho Scorer."""
    feature_specs: Dict[str, Dict] = {}
    if not hasattr(plugin, "feature_specs"):
        return feature_specs

    specs_obj = plugin.feature_specs(phase)
    for key, spec in specs_obj.items():
        if hasattr(spec, "level"):
            feature_specs[key] = {
                "level": spec.level,
                "scoring_type": spec.scoring_type,
                "unit": spec.unit,
                "fps_reference": getattr(spec, "fps_reference", None),
            }
        else:
            feature_specs[key] = spec
    return feature_specs


def _collect_tolerance_map(
    plugin: SportPlugin, phase: str, feature_specs: Dict[str, Dict]
) -> Dict[str, float]:
    """
    tolerance_map theo từng feature, trên UNION của:
        feature_specs  |  fixed_reference_ranges(phase)  |  tolerance.overrides
    Feature nào plugin.tolerance_for() không trả được thì bỏ qua (Scorer dùng
    tolerance mặc định của nó).
    """
    tolerance_map: Dict[str, float] = {}
    if not hasattr(plugin, "tolerance_for"):
        return tolerance_map

    all_keys: Set[str] = set(feature_specs.keys())
    try:
        all_keys |= {r.feature_key for r in plugin.fixed_reference_ranges(phase)}
    except Exception as exc:  # noqa: BLE001 — plugin không khai fixed ranges
        logger.debug("build_scorer: bỏ qua fixed_reference_ranges(%s): %s", phase, exc)

    config = getattr(plugin, "_config", None)
    if config is not None and hasattr(config, "tolerance"):
        overrides = (config.tolerance or {}).get("overrides", {}) or {}
        all_keys |= set(overrides.keys())

    for key in all_keys:
        try:
            tolerance_map[key] = plugin.tolerance_for(key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("build_scorer: tolerance_for(%s) lỗi: %s", key, exc)
    return tolerance_map


def build_scorer(plugin: SportPlugin, phase: str, fps_actual: int = 60) -> Scorer:
    """
    Khởi tạo Scorer cho 1 phase với đầy đủ cấu hình V2.1.

    Args:
        plugin     : SportPlugin của môn đang chấm.
        phase      : tên pha theo plugin (vao_cho / static / sprint_start /
                     drive_phase / ...). PHẢI trùng với `bundle.phase` và
                     `reference.phase` lúc gọi `scorer.score(bundle, reference)`.
        fps_actual : fps thực của video, cho scoring_type = frame_window.
    """
    weights = plugin.feature_weights(phase)
    feature_specs = _collect_feature_specs(plugin, phase)
    tolerance_map = _collect_tolerance_map(plugin, phase, feature_specs)
    level_weights = (
        plugin.level_weights() if hasattr(plugin, "level_weights") else {}
    )

    return Scorer(
        weights=weights,
        feature_specs=feature_specs,
        level_weights=level_weights,
        tolerance_map=tolerance_map,
        fps_actual=fps_actual,
    )