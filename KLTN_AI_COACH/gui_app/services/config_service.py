"""Đọc YAML môn: mô tả pha (mo_ta), chỉ số chấm theo pha, quality_gates. Không tự suy diễn."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from gui_app import constants as C

_FALLBACK_DESC = {
    "vao_cho": "Hai tay rộng bằng vai, chống đất. Gối sau chạm đất, gối trước gập ~100°.",
    "static": "Hông cao hơn vai 10–20cm. Gối trước ~90°, gối sau 120–140°.",
    "sprint_start": "Chân sau rời bàn đạp trước, thân lao về trước.",
    "drive_phase": "Độ dài bước tăng dần, thân dựng thẳng dần.",
}


class ConfigService:
    def __init__(self, yaml_path: Path):
        from core.sport_interface import SportConfig
        self.path = Path(yaml_path)
        self.cfg = SportConfig.from_yaml(str(self.path))

    def validate(self) -> tuple[List[str], List[str]]:
        """(errors, warnings). SportConfig.validate() theo docstring trả về DANH SÁCH CẢNH BÁO (không raise),
        nên được xếp vào warnings; lỗi cú pháp/thiếu trường bắt buộc đã raise ở from_yaml (bắt ở UI)."""
        errors: List[str] = []
        warns = list(self.cfg.validate())
        fn = getattr(self.cfg, "validation_warnings", None)
        if callable(fn):
            warns = list(fn())
        return errors, warns

    @property
    def min_fps(self) -> float:
        return float(self.cfg.quality_gates.get("min_fps_for_temporal", 30.0))

    @property
    def low_res_shoulder_px(self) -> float:
        return float(self.cfg.quality_gates.get("low_resolution_shoulder_width_px", 50.0))

    def phase_description(self, engine_phase: str) -> str:
        """Ưu tiên `mo_ta` trong technique_phases; fallback văn bản mặc định."""
        for _, spec in (self.cfg.technique_phases or {}).items():
            if spec.get("engine_phase") == engine_phase and spec.get("mo_ta"):
                return str(spec["mo_ta"])
        return _FALLBACK_DESC.get(engine_phase, "")

    def scored_features(self, engine_phase: str) -> Dict[str, float]:
        w = (self.cfg.feature_weights or {}).get(engine_phase, {}) or {}
        return {k: v for k, v in w.items() if v}

    def phase_weight(self, engine_phase: str) -> Optional[float]:
        return (self.cfg.phase_weights or {}).get(engine_phase)
