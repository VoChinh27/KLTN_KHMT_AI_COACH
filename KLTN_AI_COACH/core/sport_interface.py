"""
core/sport_interface.py
==========================
Mục đích : Định nghĩa interface chung cho MỌI môn thể thao. Đây là "điểm
           nối" duy nhất giữa phần lõi dùng chung và phần cấu hình riêng
           từng môn.

Nguyên tắc mở rộng:
    Thêm 1 môn mới = thêm 1 file config/sport_configs/<ten_mon>.yaml mới,
    KHÔNG cần viết thêm class Python nào — miễn là môn đó dùng đúng bộ
    đặc trưng (góc khớp) mà normalizer.py đã tính.

V2.1 - Bổ sung:
    - FeatureSpec, SubPhaseSpec dataclass để validate YAML.
    - API get_sub_phases / get_feature_specs / get_phase_weights.
    - Phân loại lỗi rõ ràng (TECHNIQUE / MEASUREMENT / INSUFFICIENT_DATA).
    - Validation schema cho YAML trước khi chạy pipeline.

V2.2 - Sửa bug config (SI1):
    - SportConfig.validate kiểm tra min < max trong fixed_reference_ranges.
      YAML min > max sẽ sinh warning rõ ràng (tránh ref.center sai, z_score
      đảo dấu, score vô nghĩa).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from stage1_data.capture_guide import CaptureThresholds

import logging; logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_METHOD = "median_iqr"

# Các scoring_type hợp lệ (khớp với spec v2.1 mục 3.5)
VALID_SCORING_TYPES = {"range", "boolean", "slope", "frame_window", "info"}
VALID_LEVELS = {"PRIMARY", "SECONDARY", "INFO"}


# =====================================================================
# DATACLASS SPEC - validate + truy xuất rõ ràng
# =====================================================================

@dataclass
class FeatureSpec:
    """Đặc tả 1 feature trong 1 sub-phase."""

    feature_key: str
    level: str                    # PRIMARY / SECONDARY / INFO
    scoring_type: str             # range / boolean / slope / frame_window / info
    unit: str = ""
    source: str = ""
    fps_reference: Optional[int] = None    # cho scoring_type=frame_window
    status: str = "active"                 # active / planned

    def is_info_only(self) -> bool:
        return self.level == "INFO" or self.scoring_type == "info"

    def is_valid(self) -> bool:
        return (
            self.level in VALID_LEVELS
            and self.scoring_type in VALID_SCORING_TYPES
        )


@dataclass
class SubPhaseSpec:
    """Đặc tả 1 sub-phase trong YAML."""

    phase_key: str                 # vao_cho / static / sprint_start / drive_phase
    ten_day_du: str = ""
    engine_phase: Optional[str] = None
    order: int = 0
    status: str = "active"         # active / planned
    features: Dict[str, FeatureSpec] = field(default_factory=dict)


@dataclass
class ReferenceRangeSpec:
    """1 dòng reference range (từ fixed_reference_ranges)."""

    feature_key: str
    min: float
    max: float
    unit: str = ""
    source_note: str = ""


# =====================================================================
# SportConfig - đọc YAML đầy đủ + validate
# =====================================================================

@dataclass
class SportConfig:
    """Dữ liệu thô đọc từ 1 file sport_configs/<ten_mon>.yaml."""

    sport: str
    phases: List[str]
    cyclic: Dict[str, bool]
    feature_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    capture: Dict[str, float] = field(default_factory=dict)
    reference_method: str = DEFAULT_REFERENCE_METHOD
    feedback_advice: Dict[str, Dict[str, str]] = field(default_factory=dict)
    technique_phases: Dict[str, Dict] = field(default_factory=dict)
    fixed_reference_ranges: Dict[str, List[Dict]] = field(default_factory=dict)
    feature_specs: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    phase_weights: Dict[str, float] = field(default_factory=dict)
    level_weights: Dict[str, float] = field(default_factory=dict)
    quality_gates: Dict[str, float] = field(default_factory=dict)
    tolerance: Dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "SportConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: Dict) -> "SportConfig":
        missing = [k for k in ("sport", "phases", "cyclic") if k not in raw]
        if missing:
            raise ValueError(
                f"Thiếu trường bắt buộc trong sport config: {missing}"
            )
        return cls(
            sport=raw["sport"],
            phases=list(raw["phases"]),
            cyclic=dict(raw["cyclic"]),
            feature_weights=raw.get("feature_weights", {}) or {},
            capture=raw.get("capture", {}) or {},
            reference_method=raw.get(
                "reference_method", DEFAULT_REFERENCE_METHOD
            ),
            feedback_advice=raw.get("feedback_advice", {}) or {},
            technique_phases=raw.get("technique_phases", {}) or {},
            fixed_reference_ranges=raw.get("fixed_reference_ranges", {}) or {},
            feature_specs=raw.get("feature_specs", {}) or {},
            phase_weights=raw.get("phase_weights", {}) or {},
            level_weights=raw.get("level_weights", {}) or {},
            quality_gates=raw.get("quality_gates", {}) or {},
            tolerance=raw.get("tolerance", {}) or {},
        )

    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """
        Kiểm tra tính nhất quán của YAML trước khi chạy pipeline.
        Trả về danh sách warning (rỗng = hợp lệ).
        KHÔNG raise exception — để pipeline quyết định có chạy tiếp hay không.

        V2.2 (SI1): thêm kiểm tra min < max trong fixed_reference_ranges.
        """
        warnings: List[str] = []

        # 1. Mỗi phase trong phases phải có cyclic + feature_specs (trừ planned)
        for phase in self.phases:
            if phase not in self.cyclic:
                warnings.append(
                    f"[config] Phase '{phase}' thiếu khai báo cyclic."
                )
            phase_specs = self.feature_specs.get(phase, {})
            phase_refs = {
                r["feature_key"]
                for r in self.fixed_reference_ranges.get(phase, [])
                if isinstance(r, dict) and "feature_key" in r
            }
            for feat_key, spec in phase_specs.items():
                status = spec.get("status", "active")
                if status == "planned":
                    continue
                # Feature có scoring phải có reference
                scoring_type = spec.get("scoring_type", "range")
                if scoring_type != "info" and feat_key not in phase_refs:
                    warnings.append(
                        f"[config] Feature '{feat_key}' (phase '{phase}') "
                        f"có scoring_type='{scoring_type}' nhưng thiếu "
                        f"reference range."
                    )
                # Level hợp lệ
                level = spec.get("level", "PRIMARY")
                if level not in VALID_LEVELS:
                    warnings.append(
                        f"[config] Feature '{feat_key}' (phase '{phase}') "
                        f"có level không hợp lệ: '{level}'."
                    )
                # Scoring type hợp lệ
                if scoring_type not in VALID_SCORING_TYPES:
                    warnings.append(
                        f"[config] Feature '{feat_key}' (phase '{phase}') "
                        f"có scoring_type không hợp lệ: '{scoring_type}'."
                    )

        # 2. INFO feature không được có weight > 0
        for phase, weights in self.feature_weights.items():
            for feat_key, w in weights.items():
                spec = self.feature_specs.get(phase, {}).get(feat_key, {})
                if (
                    spec.get("level") == "INFO"
                    or spec.get("scoring_type") == "info"
                ):
                    if w > 0:
                        warnings.append(
                            f"[config] Feature '{feat_key}' (phase '{phase}') "
                            f"là INFO nhưng có weight={w} > 0."
                        )

        # 3. phase_weights phải bao phủ hết phases đang active
        for phase in self.phases:
            spec_status = "active"
            for tp in self.technique_phases.values():
                if tp.get("engine_phase") == phase:
                    spec_status = tp.get("status", "active")
                    break
            if spec_status == "active" and phase not in self.phase_weights:
                warnings.append(
                    f"[config] Phase '{phase}' đang active nhưng thiếu "
                    f"phase_weights."
                )

        # 4. level_weights phải có đủ 3 level
        for lv in VALID_LEVELS:
            if lv not in self.level_weights:
                warnings.append(
                    f"[config] Thiếu level_weights cho '{lv}'."
                )

        # 5. SI1: fixed_reference_ranges — min phải < max
        for phase, ranges in self.fixed_reference_ranges.items():
            if not isinstance(ranges, list):
                warnings.append(
                    f"[config] fixed_reference_ranges['{phase}'] không phải "
                    f"list."
                )
                continue
            for i, item in enumerate(ranges):
                if not isinstance(item, dict):
                    warnings.append(
                        f"[config] fixed_reference_ranges['{phase}'][{i}] "
                        f"không phải dict."
                    )
                    continue
                feat = item.get("feature_key", f"<index {i}>")
                if "min" not in item or "max" not in item:
                    warnings.append(
                        f"[config] fixed_reference_ranges['{phase}'] "
                        f"feature '{feat}' thiếu min hoặc max."
                    )
                    continue
                try:
                    lo = float(item["min"])
                    hi = float(item["max"])
                except (TypeError, ValueError):
                    warnings.append(
                        f"[config] fixed_reference_ranges['{phase}'] "
                        f"feature '{feat}' min/max không phải số "
                        f"(min={item.get('min')!r}, max={item.get('max')!r})."
                    )
                    continue
                if lo > hi:
                    warnings.append(
                        f"[config] fixed_reference_ranges['{phase}'] "
                        f"feature '{feat}' có min ({lo}) > max ({hi}). "
                        f"Reference center và z-score sẽ sai dấu — hãy sửa YAML."
                    )
                elif lo == hi:
                    # V2.3: boolean features có min == max == 1.0 là ĐÚNG theo thiết kế
                    # (chỉ pass khi value == 1.0). Không cần cảnh báo.
                    scoring_type = "range"
                    for tp_key, tp_val in self.feature_specs.get(phase, {}).items():
                        if tp_key == feat:
                            scoring_type = tp_val.get("scoring_type", "range")
                            break
                    if scoring_type == "boolean":
                        continue  # skip warning
                    warnings.append(
                        f"[config] fixed_reference_ranges['{phase}'] "
                        f"feature '{feat}' có min == max ({lo}). "
                        f"Tolerance band bằng 0 — mọi giá trị lệch đều fail."
                    )
        return warnings


# =====================================================================
# SportPlugin ABC - giữ nguyên interface cũ
# =====================================================================

class SportPlugin(ABC):
    """Interface bắt buộc mọi môn thể thao phải cung cấp."""

    @property
    @abstractmethod
    def sport_name(self) -> str: ...

    @abstractmethod
    def phases(self) -> List[str]:
        """Danh sách tên pha của môn này."""

    @abstractmethod
    def is_cyclic(self, phase: str) -> bool:
        """True nếu pha này có chuyển động TUẦN HOÀN."""

    @abstractmethod
    def feature_weights(self, phase: str) -> Dict[str, float]:
        """Trọng số cho scorer.Scorer ở pha này."""

    @abstractmethod
    def capture_thresholds(self) -> CaptureThresholds:
        """Tiêu chí đặt camera riêng cho môn này."""

    @abstractmethod
    def reference_method(self) -> str:
        """'median_iqr' hoặc 'mean_std'."""

    def feedback_advice_overrides(self, phase: str) -> Dict[tuple, str]:
        """Câu gợi ý điều chỉnh riêng cho môn này."""
        return {}


# =====================================================================
# YamlSportPlugin - cài đặt đầy đủ theo v2.1
# =====================================================================

class YamlSportPlugin(SportPlugin):
    """
    Cài đặt SportPlugin DÙNG CHUNG cho mọi môn được khai báo thuần bằng YAML.
    V2.1: bổ sung API cho sub-phase, feature specs, phase weights,
          tolerance, quality gates.
    """

    def __init__(self, config: SportConfig):
        self._config = config
        for phase in config.phases:
            if phase not in config.cyclic:
                raise ValueError(
                    f"Sport config '{config.sport}': thiếu khai báo cyclic "
                    f"cho pha '{phase}'"
                )
        # Chạy validation — chỉ cảnh báo, không chặn
        self._validation_warnings = config.validate()
        for w in self._validation_warnings:
            logger.warning(w)

    @classmethod
    def from_yaml(cls, path: str) -> "YamlSportPlugin":
        return cls(SportConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # Thuộc tính cơ bản
    # ------------------------------------------------------------------
    @property
    def sport_name(self) -> str:
        return self._config.sport

    def phases(self) -> List[str]:
        return list(self._config.phases)

    def is_cyclic(self, phase: str) -> bool:
        self._check_phase(phase)
        return bool(self._config.cyclic[phase])

    def reference_method(self) -> str:
        return self._config.reference_method

    def capture_thresholds(self) -> CaptureThresholds:
        overrides = self._config.capture
        defaults = CaptureThresholds()
        return CaptureThresholds(**{**defaults.__dict__, **overrides})

    # ------------------------------------------------------------------
    # Feature weights + level weights
    # ------------------------------------------------------------------
    def feature_weights(self, phase: str) -> Dict[str, float]:
        self._check_phase(phase)
        return dict(self._config.feature_weights.get(phase, {}))

    def level_weights(self) -> Dict[str, float]:
        return dict(self._config.level_weights)

    def phase_weights(self) -> Dict[str, float]:
        return dict(self._config.phase_weights)

    # ------------------------------------------------------------------
    # Sub-phase / technique phase
    # ------------------------------------------------------------------
    def technique_phases(self) -> Dict[str, Dict]:
        return dict(self._config.technique_phases)

    def get_sub_phases(
        self, status_filter: Optional[str] = None
    ) -> List[SubPhaseSpec]:
        """
        Trả về danh sách SubPhaseSpec đã sort theo order.
        status_filter: 'active' | 'planned' | None (lấy hết).
        """
        result: List[SubPhaseSpec] = []
        for phase_key, tp in self._config.technique_phases.items():
            status = tp.get("status", "active")
            if status_filter is not None and status != status_filter:
                continue
            engine_phase = tp.get("engine_phase")
            sub_spec = SubPhaseSpec(
                phase_key=engine_phase or phase_key,
                ten_day_du=tp.get("ten_day_du", ""),
                engine_phase=engine_phase,
                order=int(tp.get("order", 0)),
                status=status,
                features=(
                    self.get_feature_specs(engine_phase)
                    if engine_phase
                    else {}
                ),
            )
            result.append(sub_spec)
        result.sort(key=lambda s: s.order)
        return result

    def get_active_engine_phases(self) -> List[str]:
        """Trả về danh sách engine_phase của các sub-phase status='active'."""
        active = [
            tp["engine_phase"]
            for tp in self._config.technique_phases.values()
            if tp.get("status", "active") == "active" and tp.get("engine_phase")
        ]
        return active

    # ------------------------------------------------------------------
    # Feature specs (canonical, theo spec v2.1)
    # ------------------------------------------------------------------
    def feature_specs(self, phase: str) -> Dict[str, FeatureSpec]:
        """
        Trả về Dict[feature_key, FeatureSpec] cho 1 phase.
        Bỏ qua các feature status='planned' (chưa implement).
        """
        self._check_phase(phase)
        raw = self._config.feature_specs.get(phase, {})
        result: Dict[str, FeatureSpec] = {}
        for feat_key, spec in raw.items():
            if spec.get("status") == "planned":
                continue
            feature_spec = FeatureSpec(
                feature_key=feat_key,
                level=spec.get("level", "PRIMARY"),
                scoring_type=spec.get("scoring_type", "range"),
                unit=spec.get("unit", ""),
                source=spec.get("source", ""),
                fps_reference=spec.get("fps_reference"),
                status=spec.get("status", "active"),
            )
            if feature_spec.is_valid():
                result[feat_key] = feature_spec
            else:
                logger.warning(
                    "[sport_interface] Bỏ qua feature '%s' (phase '%s'): "
                    "level='%s' scoring_type='%s' không hợp lệ",
                    feat_key, phase, feature_spec.level, feature_spec.scoring_type,
                )
        return result

    # Alias tương thích (một số caller dùng get_feature_specs)
    def get_feature_specs(self, phase: str) -> Dict[str, FeatureSpec]:
        return self.feature_specs(phase)

    # ------------------------------------------------------------------
    # Reference ranges
    # ------------------------------------------------------------------
    def fixed_reference_ranges(self, phase: str) -> List[ReferenceRangeSpec]:
        self._check_phase(phase)
        raw = self._config.fixed_reference_ranges.get(phase, [])
        result: List[ReferenceRangeSpec] = []
        for item in raw:
            if (
                "feature_key" not in item
                or "min" not in item
                or "max" not in item
            ):
                continue
            result.append(
                ReferenceRangeSpec(
                    feature_key=item["feature_key"],
                    min=float(item["min"]),
                    max=float(item["max"]),
                    unit=item.get("unit", ""),
                    source_note=item.get("source_note", ""),
                )
            )
        return result

    # ------------------------------------------------------------------
    # Feedback advice
    # ------------------------------------------------------------------
    def feedback_advice_overrides(
        self, phase: str
    ) -> Dict[Tuple[str, str], str]:
        """
        Trả Dict[(feature_key, direction), advice].

        YAML key dạng "feature_key:direction" → được partition thành tuple.
        Đã verify đúng format mà FeedbackGenerator kỳ vọng (PL3).
        """
        self._check_phase(phase)
        raw = self._config.feedback_advice.get(phase, {})
        result: Dict[Tuple[str, str], str] = {}
        for key, advice in raw.items():
            feature_key, _, direction = key.partition(":")
            if not direction:
                # Key không có ":direction" → bỏ qua hoặc coi direction=""
                continue
            result[(feature_key.strip(), direction.strip())] = advice
        return result

    # ------------------------------------------------------------------
    # Tolerance
    # ------------------------------------------------------------------
    def tolerance_for(self, feature_key: str) -> float:
        """
        Trả về tolerance cho 1 feature.
        Ưu tiên: overrides[feature_key] > default > 3.0.
        """
        tol = self._config.tolerance or {}
        overrides = tol.get("overrides", {}) or {}
        if feature_key in overrides:
            return float(overrides[feature_key])
        return float(tol.get("default", 3.0))

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------
    def quality_gates(self) -> Dict[str, float]:
        return dict(self._config.quality_gates)

    # ------------------------------------------------------------------
    # Validation warnings (public)
    # ------------------------------------------------------------------
    def validation_warnings(self) -> List[str]:
        return list(self._validation_warnings)

    # ------------------------------------------------------------------
    def _check_phase(self, phase: str) -> None:
        if phase not in self._config.phases:
            raise ValueError(
                f"Môn '{self.sport_name}' không có pha '{phase}' "
                f"(các pha hợp lệ: {self._config.phases})"
            )


# =====================================================================
# Loader
# =====================================================================

def load_sport_plugin(sport_config_dir: str, sport: str) -> YamlSportPlugin:
    """Nạp plugin theo tên môn, tìm file '<sport_config_dir>/<sport>.yaml'."""
    path = Path(sport_config_dir) / f"{sport}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy cấu hình cho môn '{sport}' tại {path}. "
            f"Để thêm môn mới, tạo file '{sport}.yaml' theo mẫu trong "
            f"config/sport_configs/running.yaml."
        )
    return YamlSportPlugin.from_yaml(str(path))