"""
stage4_scoring/scorer.py
==========================
Mục đích : Tính điểm từng chỉ số và điểm tổng cho MỘT học viên.

V2.1.3
    - Sprint start đọc canonical feature (ExplosiveStartSummary) rồi mới
      tới event_features. Alias at_trunk / at_knee_front được map sang
      tên reference, canonical thắng nếu trùng.
    - Boolean PRIMARY có trong reference mà vắng hoặc NaN -> 0 điểm,
      không bị biến mất khỏi mẫu.
    - PRIMARY bị NaN (không đo được) -> phase KHÔNG được chấm (is_scored=False),
      không phạt 0 điểm. SECONDARY NaN vẫn bỏ qua.
    - fps lấy từ bundle.fps của từng video.
    - Drive phase không fallback sang static_ranges.
    - Tolerance theo đơn vị khi YAML không đưa override.
    - Phase < 50% feature chấm được HOẶC thiếu PRIMARY -> is_scored = False, overall = NaN.
    - error_class phân biệt technique và measurement từ quality_flags.
    - Lệch phase không còn raise làm chết cả pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stage3_features.feature_extractor import (
    FeatureBundle,
    PHASE_DRIVE,
    PHASE_DYNAMIC,
    PHASE_INTO_POSITION,
    PHASE_SPRINT_START,
    PHASE_STATIC,
)
from stage4_scoring.reference_model import ReferenceModel, ReferenceRange

DEFAULT_SOFTNESS = 1.0
BOOLEAN_PASS_MIN = 0.75
MIN_SCORED_RATIO = 0.50

# Tên boolean của sprint_start. Dùng khi feature_specs không được truyền.
_KNOWN_BOOLEAN_FEATURES = frozenset({
    "front_leg_full_extension",
    "rear_leg_leaves_first",
    "arm_leg_opposition",
})

_SPRINT_ALIASES = {
    "at_trunk": "at_trunk_angle_vertical",
    "trunk": "at_trunk_angle_vertical",
    "at_trunk_angle_vertical": "at_trunk_angle_vertical",
    "at_knee_front": "at_knee_front_angle",
    "knee_front": "at_knee_front_angle",
    "at_knee_front_angle": "at_knee_front_angle",
    "at_leg_ground_angle": "leg_ground_angle",
    "leg_ground_angle": "leg_ground_angle",
    "at_front_leg_full_extension": "front_leg_full_extension",
    "front_leg_full_extension": "front_leg_full_extension",
    "at_rear_leg_leaves_first": "rear_leg_leaves_first",
    "rear_leg_leaves_first": "rear_leg_leaves_first",
    "at_arm_leg_opposition": "arm_leg_opposition",
    "arm_leg_opposition": "arm_leg_opposition",
    "at_first_step_timing": "first_step_timing",
    "first_step_timing": "first_step_timing",
}


@dataclass
class FeatureScore:
    feature_key: str
    student_value: float
    reference_center: float
    reference_spread: float
    z_score: float
    score_0_100: float
    weight: float
    unit: str = ""
    level: str = "PRIMARY"
    scoring_type: str = "range"
    is_info_only: bool = False
    deviation: float = 0.0
    is_in_range: bool = False
    is_in_tolerance: bool = False
    tolerance: float = 0.0
    error_hint: str = ""
    error_class: str = ""


@dataclass
class ScoreReport:
    phase: str
    overall_score: float
    feature_scores: List[FeatureScore] = field(default_factory=list)
    unscored_features: List[str] = field(default_factory=list)
    total_weight: float = 0.0
    earned_weight: float = 0.0
    n_primary_failed: int = 0
    n_secondary_failed: int = 0
    is_scored: bool = True
    phase_mismatch: bool = False
    notes: List[str] = field(default_factory=list)

    def get(self, feature_key: str) -> Optional[FeatureScore]:
        for fs in self.feature_scores:
            if fs.feature_key == feature_key:
                return fs
        return None


def z_score(value: float, ref: ReferenceRange) -> float:
    if np.isnan(value):
        return float("nan")
    spread = max(ref.spread, 1e-6)
    return float((value - ref.center) / spread)


def z_to_score(z: float, softness: float = DEFAULT_SOFTNESS) -> float:
    if np.isnan(z):
        return float("nan")
    return float(100.0 * math.exp(-0.5 * (z / softness) ** 2))


def _score_range_with_tolerance(
    value: float,
    ref_min: float,
    ref_max: float,
    tolerance: float,
) -> tuple:
    if np.isnan(value):
        return float("nan"), float("nan"), False, False

    if ref_min <= value <= ref_max:
        return 100.0, 0.0, True, True

    if value < ref_min:
        deviation = ref_min - value
    else:
        deviation = value - ref_max

    if tolerance <= 0:
        tolerance = 1e-6

    score = 100.0 * math.exp(-0.5 * (deviation / tolerance) ** 2)
    is_in_tol = deviation <= tolerance
    return float(score), float(deviation), False, is_in_tol


def _score_boolean(value: float, expected: float) -> tuple:
    """Pass chỉ khi boolean rõ: >= 0.75 nếu mong đợi đúng, <= 0.25 nếu mong đợi sai."""
    if np.isnan(value):
        return float("nan"), 0.0, False
    v = float(value)
    if expected >= 0.5:
        is_pass = v >= BOOLEAN_PASS_MIN
    else:
        is_pass = v <= (1.0 - BOOLEAN_PASS_MIN)
    return (100.0 if is_pass else 0.0), 0.0, is_pass


def _score_slope(
    value: float,
    ref_min: float,
    ref_max: float,
    tolerance: float = 0.0,
) -> tuple:
    return _score_range_with_tolerance(value, ref_min, ref_max, tolerance)


def _score_frame_window(
    value: float,
    ref_min: float,
    ref_max: float,
    fps_reference: int,
    fps_actual: float,
    tolerance: float,
) -> tuple:
    if np.isnan(value):
        return float("nan"), 0.0, False, False
    if fps_actual <= 0 or fps_reference <= 0:
        return float("nan"), float("nan"), False, False
    scale = float(fps_actual) / float(fps_reference)
    return _score_range_with_tolerance(
        value, ref_min * scale, ref_max * scale, tolerance,
    )


def _as_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


class Scorer:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        feature_specs: Optional[Dict[str, Dict]] = None,
        level_weights: Optional[Dict[str, float]] = None,
        softness: float = DEFAULT_SOFTNESS,
        default_weight: float = 1.0,
        tolerance_map: Optional[Dict[str, float]] = None,
        default_tolerance: float = 8.0,
        default_slope_tolerance: float = 0.05,
        default_ratio_tolerance: float = 0.15,
        fps_actual: int = 60,
    ):
        self.weights = weights or {}
        self.feature_specs = feature_specs or {}
        self.level_weights = level_weights or {
            "PRIMARY": 1.0, "SECONDARY": 0.5, "INFO": 0.0
        }
        self.softness = softness
        self.default_weight = default_weight
        self.tolerance_map = tolerance_map or {}
        self.default_tolerance = default_tolerance
        self.default_slope_tolerance = default_slope_tolerance
        self.default_ratio_tolerance = default_ratio_tolerance
        self.fps_actual = fps_actual

    def score(self, bundle: FeatureBundle, reference: ReferenceModel) -> ScoreReport:
        phase = bundle.phase
        mismatch = phase != reference.phase
        notes: List[str] = []
        if mismatch:
            notes.append(
                f"phase lệch: bundle='{phase}' reference='{reference.phase}'. "
                "Vẫn chấm theo bundle, không dừng pipeline."
            )

        if phase in (PHASE_INTO_POSITION, PHASE_STATIC):
            report = self._score_static(bundle, reference)
        elif phase == PHASE_DYNAMIC:
            report = self._score_dynamic(bundle, reference)
        elif phase == PHASE_SPRINT_START:
            report = self._score_sprint_start(bundle, reference)
        elif phase == PHASE_DRIVE:
            report = self._score_drive_phase(bundle, reference)
        else:
            raise ValueError(f"phase không hỗ trợ: {phase}")

        report.phase_mismatch = mismatch
        report.notes.extend(notes)
        return report

    def _fps_of(self, bundle: FeatureBundle) -> float:
        fps = _as_float(getattr(bundle, "fps", float("nan")))
        if np.isfinite(fps) and fps > 0:
            return fps
        return float(self.fps_actual)

    def _score_static(self, bundle: FeatureBundle, reference: ReferenceModel) -> ScoreReport:
        if bundle.static_summary is None:
            return self._empty_report(
                bundle.phase,
                "không có đoạn giữ tư thế đủ dài để chấm",
            )
        student_values = (
            bundle.static_summary.feature_mean
            if bundle.static_summary.feature_mean
            else bundle.static_summary.angle_mean
        )
        ranges = (
            reference.phase_ranges
            if bundle.phase == PHASE_INTO_POSITION
            else reference.static_ranges
        )
        return self._build_report(
            phase=bundle.phase,
            student_values=dict(student_values or {}),
            reference_ranges=ranges or {},
            default_scoring_type="range",
            fps_actual=self._fps_of(bundle),
            quality_flags=dict(bundle.quality_flags or {}),
        )

    def _score_dynamic(self, bundle: FeatureBundle, reference: ReferenceModel) -> ScoreReport:
        if bundle.dynamic_summary is None:
            return self._empty_report(PHASE_DYNAMIC, "dynamic_summary rỗng")
        d = bundle.dynamic_summary
        student_values: Dict[str, float] = {"cadence_spm": d.cadence_spm_mean}
        for angle_key, rom in d.rom_by_angle.items():
            student_values[f"rom_{angle_key}"] = rom
        for sym_name, sym_val in d.symmetry_index.items():
            student_values[f"symmetry_{sym_name}"] = sym_val

        report = self._build_report(
            phase=PHASE_DYNAMIC,
            student_values=student_values,
            reference_ranges=reference.dynamic_ranges or {},
            default_scoring_type="range",
            fps_actual=self._fps_of(bundle),
            quality_flags=dict(bundle.quality_flags or {}),
        )
        if bundle.reference_similarity:
            for angle_key, sim_score in bundle.reference_similarity.items():
                weight = self.weights.get(f"dtw_{angle_key}", self.default_weight)
                report.feature_scores.append(
                    FeatureScore(
                        feature_key=f"dtw_{angle_key}",
                        student_value=sim_score,
                        reference_center=100.0,
                        reference_spread=float("nan"),
                        z_score=float("nan"),
                        score_0_100=sim_score,
                        weight=weight,
                        unit="điểm",
                        scoring_type="dtw",
                        error_class="technique" if sim_score < 60.0 else "",
                        error_hint="technique" if sim_score < 60.0 else "",
                    )
                )
            report.overall_score = self._weighted_average(report.feature_scores)
            self._refresh_aggregates(report, reference.dynamic_ranges or {})
        return report

    def _score_sprint_start(self, bundle: FeatureBundle, reference: ReferenceModel) -> ScoreReport:
        student_values = self._sprint_student_values(bundle)
        if not student_values and bundle.sprint_start_summary is None:
            return self._empty_report(
                PHASE_SPRINT_START,
                "không có sprint_start_summary lẫn canonical sub-phase",
            )
        return self._build_report(
            phase=PHASE_SPRINT_START,
            student_values=student_values,
            reference_ranges=reference.sprint_start_ranges or {},
            default_scoring_type="range",
            fps_actual=self._fps_of(bundle),
            quality_flags=dict(bundle.quality_flags or {}),
        )

    def _score_drive_phase(self, bundle: FeatureBundle, reference: ReferenceModel) -> ScoreReport:
        if bundle.static_summary is None:
            return self._empty_report(
                PHASE_DRIVE,
                "không có trend values (sub-phase quá ngắn hoặc không đo được vai)",
            )
        ranges = getattr(reference, "drive_phase_ranges", None) or {}
        if not ranges:
            report = self._empty_report(
                PHASE_DRIVE,
                "reference thiếu drive_phase_ranges; không dùng static_ranges để tránh chấm nhầm",
            )
            return report
        return self._build_report(
            phase=PHASE_DRIVE,
            student_values=dict(bundle.static_summary.feature_mean or {}),
            reference_ranges=ranges,
            default_scoring_type="slope",
            fps_actual=self._fps_of(bundle),
            quality_flags=dict(bundle.quality_flags or {}),
        )

    def _sprint_student_values(self, bundle: FeatureBundle) -> Dict[str, float]:
        """Gộp snapshot legacy và canonical. Canonical ghi đè alias."""
        merged: Dict[str, float] = {}
        summary = bundle.sprint_start_summary
        if summary is not None:
            raw = dict(summary.event_features or {})
            if not raw and summary.angles_at_launch:
                raw = {f"at_{k}": v for k, v in summary.angles_at_launch.items()}
            for key, value in raw.items():
                number = _as_float(value)
                merged[key] = number
                alias = _SPRINT_ALIASES.get(key)
                if alias is not None and alias not in merged:
                    merged[alias] = number

        sub = None
        if bundle.sub_phases:
            sub = bundle.sub_phases.get(PHASE_SPRINT_START)
        feature_values = getattr(sub, "feature_values", None) if sub is not None else None
        if feature_values:
            for key, value in feature_values.items():
                merged[key] = _as_float(value)
        return merged

    def _spec_of(self, key: str, default_scoring_type: str) -> tuple:
        spec = self.feature_specs.get(key, {})
        scoring_type = spec.get("scoring_type", default_scoring_type)
        if key in _KNOWN_BOOLEAN_FEATURES and "scoring_type" not in spec:
            scoring_type = "boolean"
        level = spec.get("level", "PRIMARY")
        return spec, scoring_type, level

    def _tolerance_for(
        self,
        key: str,
        scoring_type: str,
        unit: str,
    ) -> float:
        if key in self.tolerance_map:
            return float(self.tolerance_map[key])
        if scoring_type == "slope":
            return float(self.default_slope_tolerance)
        if scoring_type == "frame_window":
            return 1.0
        return self._default_tolerance_for_unit(unit)

    def _default_tolerance_for_unit(self, unit: str) -> float:
        text = (unit or "").strip().lower()
        if text in {"°", "deg", "degree", "degrees"}:
            return float(self.default_tolerance)
        if "ratio" in text:
            return float(self.default_ratio_tolerance)
        if "slope" in text:
            return float(self.default_slope_tolerance)
        if text in {"frame", "frames"}:
            return 1.0
        return float(self.default_tolerance)

    def _error_class(
        self,
        feature_key: str,
        score: float,
        flags: Dict[str, bool],
        scoring_type: str,
    ) -> str:
        if np.isnan(score) or score >= 60.0:
            return ""
        if not flags:
            return "technique"
        key = feature_key.lower()
        measurement = False
        if flags.get("low_resolution") or flags.get("insufficient_data"):
            measurement = True
        if "wrist" in key and flags.get("wrist_low_confidence"):
            measurement = True
        if any(tok in key for tok in ("knee", "ankle", "rear_leg", "foot")) and (
            flags.get("knee_back_low_confidence") or flags.get("rear_ankle_low_confidence")
        ):
            measurement = True
        if flags.get("low_fps") and (
            scoring_type == "frame_window" or "timing" in key or "cadence" in key
        ):
            measurement = True
        if flags.get("drive_phase_unavailable") and scoring_type == "slope":
            measurement = True
        return "measurement" if measurement else "technique"

    def _build_report(
        self,
        phase: str,
        student_values: Dict[str, float],
        reference_ranges: Dict[str, ReferenceRange],
        default_scoring_type: str = "range",
        fps_actual: Optional[float] = None,
        quality_flags: Optional[Dict[str, bool]] = None,
    ) -> ScoreReport:
        feature_scores: List[FeatureScore] = []
        unscored: List[str] = []
        flags = quality_flags or {}
        fps = float(self.fps_actual if fps_actual is None else fps_actual)
        seen = set()

        items = list(student_values.items())
        for key in reference_ranges:
            if key not in student_values:
                items.append((key, float("nan")))

        for key, raw_value in items:
            if key in seen:
                continue
            seen.add(key)
            ref = reference_ranges.get(key)
            if ref is None:
                unscored.append(key)
                continue

            spec, scoring_type, level = self._spec_of(key, default_scoring_type)
            is_info = scoring_type == "info" or level == "INFO"
            value = _as_float(raw_value)
            missing_primary = (not is_info) and level == "PRIMARY" and not np.isfinite(value)

            if is_info:
                score = float("nan")
                deviation = 0.0
                in_range = False
                in_tol = False
                tolerance = 0.0
                z = float("nan")
            elif missing_primary:
                # Có trong reference mà không đo được: tính 0, không xóa khỏi mẫu.
                score = float("nan")
                deviation = float("nan")
                in_range = False
                in_tol = False
                tolerance = 0.0
                z = float("nan")
            elif scoring_type == "boolean":
                expected = 1.0 if ref.center >= 0.5 else 0.0
                score, deviation, in_range = _score_boolean(value, expected)
                in_tol = in_range
                tolerance = BOOLEAN_PASS_MIN
                z = 0.0 if in_range else 1.0
            elif scoring_type == "slope":
                tolerance = self._tolerance_for(key, "slope", ref.unit)
                score, deviation, in_range, in_tol = _score_slope(
                    value, ref.min_value, ref.max_value, tolerance
                )
                z = float("nan")
            elif scoring_type == "frame_window":
                tolerance = self._tolerance_for(key, "frame_window", ref.unit)
                fps_ref = int(spec.get("fps_reference", 60))
                score, deviation, in_range, in_tol = _score_frame_window(
                    value, ref.min_value, ref.max_value, fps_ref, fps, tolerance
                )
                z = float("nan")
            else:
                tolerance = self._tolerance_for(key, "range", ref.unit)
                score, deviation, in_range, in_tol = _score_range_with_tolerance(
                    value, ref.min_value, ref.max_value, tolerance
                )
                z = z_score(value, ref) if not in_range else 0.0

            base_weight = self.weights.get(key)
            if base_weight is None:
                base_weight = self.level_weights.get(level, self.default_weight)
            weight = 0.0 if is_info else base_weight

            error_class = self._error_class(key, score, flags, scoring_type)
            if missing_primary and error_class == "technique":
                error_class = "data_quality"

            feature_scores.append(
                FeatureScore(
                    feature_key=key,
                    student_value=value,
                    reference_center=ref.center,
                    reference_spread=ref.spread,
                    z_score=z,
                    score_0_100=score,
                    weight=weight,
                    unit=ref.unit,
                    level=level,
                    scoring_type=scoring_type,
                    is_info_only=is_info,
                    deviation=deviation if np.isfinite(deviation) else 0.0,
                    is_in_range=in_range,
                    is_in_tolerance=in_tol,
                    tolerance=tolerance,
                    error_hint=error_class,
                    error_class=error_class,
                )
            )

        report = ScoreReport(
            phase=phase,
            overall_score=float("nan"),
            feature_scores=feature_scores,
            unscored_features=unscored,
        )
        self._refresh_aggregates(report, reference_ranges)
        return report

    def _refresh_aggregates(
        self,
        report: ScoreReport,
        reference_ranges: Dict[str, ReferenceRange],
    ) -> None:
        valid = [
            fs for fs in report.feature_scores
            if not fs.is_info_only and np.isfinite(fs.score_0_100)
        ]
        expected = []
        for key in reference_ranges:
            _spec, scoring_type, level = self._spec_of(key, "range")
            if scoring_type == "info" or level == "INFO":
                continue
            expected.append(key)
        ratio = (len(valid) / len(expected)) if expected else 0.0

        # Đếm PRIMARY không đo được (student_value NaN): phải bỏ phase, không chấm trên subset
        primary_missing_keys = [
            fs.feature_key for fs in report.feature_scores
            if fs.level == "PRIMARY"
            and not fs.is_info_only
            and not np.isfinite(fs.student_value)
        ]
        has_primary_missing = len(primary_missing_keys) > 0

        report.is_scored = (
            bool(expected)
            and ratio >= MIN_SCORED_RATIO
            and bool(valid)
            and not has_primary_missing
        )
        report.overall_score = (
            self._weighted_average(report.feature_scores)
            if report.is_scored else float("nan")
        )
        report.total_weight = sum(fs.weight for fs in valid)
        report.earned_weight = sum(fs.weight * fs.score_0_100 / 100.0 for fs in valid)
        report.n_primary_failed = sum(
            1 for fs in report.feature_scores
            if fs.level == "PRIMARY" and np.isfinite(fs.score_0_100) and fs.score_0_100 < 60.0
        )
        report.n_secondary_failed = sum(
            1 for fs in report.feature_scores
            if fs.level == "SECONDARY" and np.isfinite(fs.score_0_100) and fs.score_0_100 < 60.0
        )
        if not report.is_scored:
            if has_primary_missing:
                report.notes.append(
                    "Không đo được chỉ số PRIMARY: "
                    + ", ".join(primary_missing_keys)
                    + "; phase không được chấm (is_scored=False)."
                )
            else:
                report.notes.append(
                    f"phase không đủ feature hợp lệ ({len(valid)}/{len(expected)}); "
                    "overall_score = NaN, pipeline nên bỏ qua phase này"
                )

    def _empty_report(self, phase: str, note: str) -> ScoreReport:
        return ScoreReport(
            phase=phase,
            overall_score=float("nan"),
            is_scored=False,
            notes=[note],
        )

    @staticmethod
    def _weighted_average(feature_scores: List[FeatureScore]) -> float:
        valid = [
            fs for fs in feature_scores
            if not np.isnan(fs.score_0_100) and not fs.is_info_only
        ]
        if not valid:
            return float("nan")
        total_weight = sum(fs.weight for fs in valid)
        if total_weight <= 0:
            return float(np.mean([fs.score_0_100 for fs in valid]))
        return float(
            sum(fs.score_0_100 * fs.weight for fs in valid) / total_weight
        )