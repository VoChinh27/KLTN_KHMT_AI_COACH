"""
stage3_features/feature_extractor.py
=======================================
Mục đích : Tổng hợp bộ đặc trưng CUỐI CÙNG theo từng sub-phase động tác,
           ghép đặc trưng đã chuẩn hóa (normalizer.py) với thông tin chu kỳ
           (cycle_segmenter.py) và loại bỏ đoạn bất thường (anomaly_detector.py).

V2.1 - Bổ sung:
    - Canonical feature naming (spec_key ↔ internal_key) theo bảng 6.1.
    - 4 sub-phase summary: IntoPosition, SetPosition, ExplosiveStart, DrivePhase.
    - Method extract_sprint_start_full() tự động segment 4 sub-phase.
    - Metadata (unit, level, source) cho mỗi feature.
    - Giữ backward compat: static_summary, sprint_start_summary cũ vẫn còn.

V2.1.1 - Hoàn thiện drive_phase:
    - _build_drive_phase_summary() gọi temporal_utils.compute_drive_phase_trends()
      thay vì trả NaN (Đợt 4).
    - Xử lý 2 ankle_y riêng biệt (không lấy trung bình, tránh triệt tiêu dao động).
    - Tích hợp quality_flags từ bundle vào output.

V2.1.2 - Hỗ trợ file phases đánh dấu thủ công:
    - extract_sprint_start_from_windows() nhận windows do caller cung cấp
      (thay vì tự động segment). Dùng bởi pipeline._try_load_manual_phases().
    - Cấu trúc bundle giống hệt extract_sprint_start_full(), chỉ khác nguồn
      boundaries.

Quy tắc quan trọng: CHỈ tính đặc trưng trên các đoạn ĐÃ ĐƯỢC anomaly_detector
xác nhận hợp lệ. Không suy diễn/nội suy số liệu cho đoạn bị loại.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.pose_schema import PoseFrame
from stage3_features.anomaly_detector import (
    AnomalyDetector, AnomalyThresholds, FrameState, compute_center_of_mass,
    compute_kinematics, merge_valid_segments, sprint_start_thresholds,
    summarize_labels,
)
from stage3_features.cycle_segmenter import (
    GaitCycle, GaitCycleSegmenter, dtw_similarity_score,
)
from stage3_features.discrete_event_segmenter import (
    DiscreteEventSegmenter, flatten_event_features, _detect_hand_off,
)
from stage3_features.normalizer import NormalizedFrame, Normalizer, to_dataframe

PHASE_INTO_POSITION = "vao_cho"
PHASE_STATIC = "static"
PHASE_DYNAMIC = "dynamic"
PHASE_SPRINT_START = "sprint_start"
PHASE_DRIVE = "drive_phase"
_VALID_PHASES = (
    PHASE_INTO_POSITION, PHASE_STATIC, PHASE_DYNAMIC,
    PHASE_SPRINT_START, PHASE_DRIVE,
)

# Aggregate: bo frame keypoint yeu, khong lay mean tren du lieu ban.
AGGREGATE_MIN_CONF = 0.45
AGGREGATE_MIN_SAMPLES = 5
AGGREGATE_MIN_VALID_RATIO = 0.30
# Goc goi "duoi" tai launch. 170° la moc giai phau, khong dat duoc tren 30fps.
FRONT_LEG_FULL_EXTENSION_DEG = 140.0

_JOINT_TOKENS = (
    "wrist", "ankle", "knee", "hip", "elbow", "shoulder",
)


# =====================================================================
# CANONICAL FEATURE CATALOG (spec v2.1 - bang 6.1)
# Ánh xạ spec_key (tên trong tài liệu) <-> internal_key (tên code cũ)
# =====================================================================

SUBPHASE_FEATURE_CATALOG: Dict[str, Dict[str, Dict[str, str]]] = {
    PHASE_INTO_POSITION: {
        "knee_front_angle": {
            "internal": "knee_front", "unit": "°", "level": "PRIMARY",
        },
        "knee_back_angle": {
            "internal": "knee_back", "unit": "°", "level": "PRIMARY",
        },
        "wrist_width_ratio": {
            "internal": "wrist_width", "unit": "ratio", "level": "PRIMARY",
        },
        "trunk_thigh_angle": {
            "internal": "trunk_thigh_angle", "unit": "°", "level": "PRIMARY",
        },
        "elbow_left": {"internal": "elbow_left", "unit": "°", "level": "SECONDARY"},
        "elbow_right": {"internal": "elbow_right", "unit": "°", "level": "SECONDARY"},
        "head_trunk_alignment": {
            "internal": "head_trunk_alignment", "unit": "°", "level": "SECONDARY",
        },
    },
    PHASE_STATIC: {
        "knee_front_angle": {"internal": "knee_front", "unit": "°", "level": "PRIMARY"},
        "knee_back_angle": {"internal": "knee_back", "unit": "°", "level": "PRIMARY"},
        "hip_above_shoulder_norm": {
            "internal": "hip_above_shoulder", "unit": "ratio", "level": "PRIMARY",
        },
        "shoulder_forward_norm": {
            "internal": "shoulder_forward", "unit": "ratio", "level": "SECONDARY",
        },
        "elbow_left": {"internal": "elbow_left", "unit": "°", "level": "SECONDARY"},
        "elbow_right": {"internal": "elbow_right", "unit": "°", "level": "SECONDARY"},
        "trunk_thigh_angle": {
            "internal": "trunk_thigh_angle", "unit": "°", "level": "PRIMARY",
        },
    },
    PHASE_SPRINT_START: {
        "at_trunk_angle_vertical": {
            "internal": "at_trunk", "unit": "°", "level": "PRIMARY",
        },
        "front_leg_full_extension": {
            "internal": "front_leg_full_extension", "unit": "bool", "level": "PRIMARY",
        },
        "rear_leg_leaves_first": {
            "internal": "rear_leg_leaves_first", "unit": "bool", "level": "PRIMARY",
        },
        "arm_leg_opposition": {
            "internal": "arm_leg_opposition", "unit": "bool", "level": "PRIMARY",
        },
        "leg_ground_angle": {
            "internal": "leg_ground_angle", "unit": "°", "level": "PRIMARY",
        },
        "at_knee_front_angle": {
            "internal": "at_knee_front", "unit": "°", "level": "SECONDARY",
        },
        "first_step_timing": {
            "internal": "first_step_timing", "unit": "frame", "level": "SECONDARY",
        },
    },
    PHASE_DRIVE: {
        "step_length_trend": {
            "internal": "step_length_trend", "unit": "slope", "level": "SECONDARY",
        },
        "cadence_trend": {
            "internal": "cadence_trend", "unit": "slope", "level": "SECONDARY",
        },
        "contact_position_convergence": {
            "internal": "contact_position_convergence", "unit": "slope", "level": "SECONDARY",
        },
        "trunk_angle_vertical_trend": {
            "internal": "trunk_angle_vertical_trend", "unit": "deg/step", "level": "SECONDARY",
        },
    },
}


def canonical_to_internal(phase: str, feature_key: str) -> Optional[str]:
    """Map spec_key -> internal_key cho 1 sub-phase. None neu khong co."""
    catalog = SUBPHASE_FEATURE_CATALOG.get(phase, {})
    spec = catalog.get(feature_key)
    return spec["internal"] if spec else None


def internal_to_canonical(phase: str, internal_key: str) -> Optional[str]:
    """Map internal_key -> spec_key."""
    catalog = SUBPHASE_FEATURE_CATALOG.get(phase, {})
    for spec_key, spec in catalog.items():
        if spec["internal"] == internal_key:
            return spec_key
    return None


# =====================================================================
# SUMMARY DATACLASS CŨ (giữ backward compat)
# =====================================================================

@dataclass
class StaticPhaseSummary:
    segment_start_frame: int
    segment_end_frame: int
    duration_s: float
    angle_mean: Dict[str, float]
    angle_std: Dict[str, float]
    ratio_mean: Dict[str, float] = field(default_factory=dict)
    ratio_std: Dict[str, float] = field(default_factory=dict)
    feature_mean: Dict[str, float] = field(default_factory=dict)
    feature_std: Dict[str, float] = field(default_factory=dict)


@dataclass
class DynamicPhaseSummary:
    num_valid_cycles: int
    cadence_spm_mean: float
    cadence_spm_std: float
    rom_by_angle: Dict[str, float]
    symmetry_index: Dict[str, float]
    mean_curve_by_angle: Dict[str, np.ndarray]


@dataclass
class SprintStartPhaseSummary:
    segment_start_frame: int
    segment_end_frame: int
    duration_s: float
    launch_frame_idx: int
    launch_timestamp_s: float
    peak_acceleration: float
    angles_at_launch: Dict[str, float]
    trunk_angle_series: List[float]
    event_features: Dict[str, float] = field(default_factory=dict)
    launch_detection_method: str = "legacy_peak_acceleration"
    launch_confidence: str = "low"


# =====================================================================
# SUB-PHASE SUMMARY MỚI (spec v2.1)
# =====================================================================

@dataclass
class SubPhaseWindow:
    """Cua so frame cho 1 sub-phase."""
    phase: str
    start_frame: int
    end_frame: int
    duration_s: float


@dataclass
class IntoPositionSummary:
    """Sub-phase 1 - Vao cho (TINH)."""
    window: SubPhaseWindow
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_metadata: Dict[str, Dict] = field(default_factory=dict)
    feature_std: Dict[str, float] = field(default_factory=dict)


@dataclass
class SetPositionSummary:
    """Sub-phase 2 - San sang (TINH)."""
    window: SubPhaseWindow
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_metadata: Dict[str, Dict] = field(default_factory=dict)
    feature_std: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExplosiveStartSummary:
    """Sub-phase 3 - Boc phat (DONG)."""
    window: SubPhaseWindow
    launch_frame_idx: int
    launch_timestamp_s: float
    peak_acceleration: float
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_metadata: Dict[str, Dict] = field(default_factory=dict)
    events: Dict[str, Optional[int]] = field(default_factory=dict)
    launch_detection_method: str = "legacy_peak_acceleration"
    launch_confidence: str = "low"


@dataclass
class DrivePhaseSummary:
    """Sub-phase 4 - Chay lao 25-30m dau (DONG)."""
    window: SubPhaseWindow
    num_steps: int
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_metadata: Dict[str, Dict] = field(default_factory=dict)
    step_length_trend: float = float("nan")
    cadence_trend: float = float("nan")
    contact_position_convergence: float = float("nan")
    trunk_angle_vertical_trend: float = float("nan")


# =====================================================================
# FEATURE BUNDLE
# =====================================================================

@dataclass
class FeatureBundle:
    phase: str
    fps: float
    total_frames: int
    valid_frame_ratio: float
    state_summary: Dict[str, float]
    normalized_df: pd.DataFrame
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    # --- Backward compat ---
    static_summary: Optional[StaticPhaseSummary] = None
    dynamic_summary: Optional[DynamicPhaseSummary] = None
    sprint_start_summary: Optional[SprintStartPhaseSummary] = None
    cycles: List[GaitCycle] = field(default_factory=list)
    reference_similarity: Optional[Dict[str, float]] = None

    # --- V2.1: 4 sub-phase (dung cho pipeline full sprint_start) ---
    sub_phases: Dict[str, object] = field(default_factory=dict)
    sub_phase_order: List[str] = field(default_factory=list)


# =====================================================================
# HELPERS
# =====================================================================

def _longest_valid_run(
    labels, valid_states: Sequence[FrameState]
) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    run_start: Optional[int] = None
    for i, lb in enumerate(labels + [None]):
        in_valid = lb is not None and lb.state in valid_states
        if in_valid and run_start is None:
            run_start = i
        elif not in_valid and run_start is not None:
            cand = (labels[run_start].frame_idx, labels[i - 1].frame_idx)
            if best is None or (cand[1] - cand[0]) > (best[1] - best[0]):
                best = cand
            run_start = None
    return best


_SYMMETRY_PAIRS = [
    ("knee_left", "knee_right"),
    ("hip_left", "hip_right"),
    ("elbow_left", "elbow_right"),
    ("ankle_left", "ankle_right"),
]


def _joints_for_feature(name: str) -> Tuple[str, ...]:
    """Suy ra keypoint bat buoc tu ten feature. Khong doan duoc -> ()."""
    lower = name.lower()
    side = None
    if "left" in lower and "right" not in lower:
        side = "left"
    elif "right" in lower and "left" not in lower:
        side = "right"
    joints: List[str] = []
    for token in _JOINT_TOKENS:
        if token not in lower:
            continue
        if side is None:
            joints.extend((f"left_{token}", f"right_{token}"))
        else:
            joints.append(f"{side}_{token}")
    # trunk / head khong co 1 joint duy nhat o day — bo loc o tang pose tong.
    if "trunk" in lower or "head" in lower:
        joints.extend(("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    # front/back leg features can ca hai ben vi chua biet side luc aggregate.
    if ("front" in lower or "back" in lower) and not joints:
        joints.extend((
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ))
    return tuple(dict.fromkeys(joints))


def _frame_joints_confident(pf: PoseFrame, joints: Sequence[str], min_conf: float) -> bool:
    if not joints:
        return True
    for joint in joints:
        try:
            conf = float(pf.get_conf(joint))
        except Exception:
            return False
        if np.isnan(conf) or conf < min_conf:
            return False
    return True


def _index_poses(sequence: Optional[Sequence[PoseFrame]]) -> Dict[int, PoseFrame]:
    if not sequence:
        return {}
    return {pf.frame_idx: pf for pf in sequence}


def _arm_leg_opposition(pf: PoseFrame, min_conf: float = AGGREGATE_MIN_CONF) -> float:
    """
    1.0 neu tay va chan trai/phai dao chieu tren truc ngang (opposition),
    0.0 neu cung chieu, NaN neu keypoint yeu hoac lech qua nho.
    Khong gia dinh camera quay tu trai sang phai.
    """
    joints = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    if not _frame_joints_confident(pf, joints, min_conf):
        return float("nan")
    try:
        arm = float(pf.get("left_wrist")[0]) - float(pf.get("right_wrist")[0])
        leg = float(pf.get("left_ankle")[0]) - float(pf.get("right_ankle")[0])
    except Exception:
        return float("nan")
    if abs(arm) < 1.0 or abs(leg) < 1.0:
        return float("nan")
    return 1.0 if arm * leg < 0.0 else 0.0


def _rear_leg_leaves_first(
    rear_frame: Optional[int], front_frame: Optional[int],
) -> float:
    """1.0 neu chan sau roi truoc chan truoc. Thieu 1 moc -> NaN (khong fail cung)."""
    if rear_frame is None or front_frame is None:
        return float("nan")
    return 1.0 if rear_frame < front_frame else 0.0


def _aggregate_feature_window(
    normalized: List[NormalizedFrame],
    phase: str,
    use_mean: bool = True,
    poses: Optional[Sequence[PoseFrame]] = None,
    min_conf: float = AGGREGATE_MIN_CONF,
    min_samples: int = AGGREGATE_MIN_SAMPLES,
    min_valid_ratio: float = AGGREGATE_MIN_VALID_RATIO,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict]]:
    """
    Thong ke feature trong cua so.

    use_mean=False -> median (robust voi frame bien).
    Frame co keypoint lien quan < min_conf bi loai.
    So frame con lai < min_samples hoac < min_valid_ratio * window -> NaN.
    """
    del phase  # catalog remap lam o buoc sau
    if not normalized:
        return {}, {}, {}

    window = normalized
    pose_by_frame = _index_poses(poses)
    n = len(window)
    need = max(min_samples, int(np.ceil(min_valid_ratio * n)))

    angle_names = list(window[0].angles.keys())
    ratio_names = list(window[0].ratios.keys())

    feature_values: Dict[str, float] = {}
    feature_std: Dict[str, float] = {}
    feature_metadata: Dict[str, Dict] = {}

    def _reduce(name: str, kind: str, getter) -> None:
        joints = _joints_for_feature(name)
        kept: List[float] = []
        for nf in window:
            raw = getter(nf)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            if pose_by_frame:
                pf = pose_by_frame.get(nf.frame_idx)
                if pf is None or not _frame_joints_confident(pf, joints, min_conf):
                    continue
            kept.append(float(raw))
        meta = {"unit": "°" if kind == "angle" else "ratio", "type": kind}
        if len(kept) < need:
            feature_values[name] = float("nan")
            feature_std[name] = float("nan")
            meta["low_confidence_rejected"] = True
            meta["valid_frames"] = len(kept)
            feature_metadata[name] = meta
            return
        arr = np.asarray(kept, dtype=np.float64)
        feature_values[name] = float(np.mean(arr) if use_mean else np.median(arr))
        feature_std[name] = float(np.std(arr))
        meta["valid_frames"] = len(kept)
        feature_metadata[name] = meta

    for name in angle_names:
        _reduce(name, "angle", lambda nf, name=name: nf.angles.get(name, float("nan")))
    for name in ratio_names:
        _reduce(name, "ratio", lambda nf, name=name: nf.ratios.get(name, float("nan")))

    return feature_values, feature_std, feature_metadata


def _remap_to_canonical(
    raw_features: Dict[str, float],
    phase: str,
    fill_missing: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Doi internal_key -> spec_key theo catalog. Key thieu -> NaN neu fill_missing."""
    catalog = SUBPHASE_FEATURE_CATALOG.get(phase, {})
    result: Dict[str, float] = {}
    metadata: Dict[str, Dict] = {}
    for spec_key, spec in catalog.items():
        internal_key = spec["internal"]
        meta = {
            "unit": spec["unit"],
            "level": spec["level"],
            "internal_key": internal_key,
        }
        if internal_key in raw_features:
            result[spec_key] = raw_features[internal_key]
        elif fill_missing:
            result[spec_key] = float("nan")
            meta["missing"] = True
        else:
            continue
        metadata[spec_key] = meta
    return result, metadata


def _catalog_nan(phase: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict]]:
    catalog = SUBPHASE_FEATURE_CATALOG.get(phase, {})
    values = {key: float("nan") for key in catalog}
    std = {key: float("nan") for key in catalog}
    meta = {
        key: {
            "unit": spec["unit"],
            "level": spec["level"],
            "internal_key": spec["internal"],
            "window_too_short": True,
        }
        for key, spec in catalog.items()
    }
    return values, std, meta


# =====================================================================
# FEATURE EXTRACTOR
# =====================================================================

class FeatureExtractor:
    def __init__(
        self,
        anomaly_thresholds: Optional[AnomalyThresholds] = None,
        sprint_start_anomaly_thresholds: Optional[AnomalyThresholds] = None,
        cycle_feature_keys: Sequence[str] = (
            "angle_knee_left", "angle_knee_right",
            "angle_hip_left", "angle_hip_right",
            "angle_elbow_left", "angle_elbow_right",
            "angle_trunk",
        ),
        num_resample_points: int = 101,
    ):
        self.normalizer = Normalizer()
        self.anomaly_detector = AnomalyDetector(anomaly_thresholds)
        self.sprint_start_anomaly_detector = AnomalyDetector(
            sprint_start_anomaly_thresholds or sprint_start_thresholds()
        )
        self.cycle_segmenter = GaitCycleSegmenter(
            feature_keys=cycle_feature_keys, num_resample_points=num_resample_points
        )
        self.discrete_event_segmenter = DiscreteEventSegmenter()

    # ------------------------------------------------------------------
    def extract(
        self,
        sequence: List[PoseFrame],
        fps: float,
        phase: str,
        reference_cycle: Optional[GaitCycle] = None,
        front_side_hint: Optional[str] = None,
    ) -> FeatureBundle:
        if phase not in _VALID_PHASES:
            raise ValueError(f"phase phải là một trong {_VALID_PHASES}, nhận '{phase}'")
        if not sequence:
            raise ValueError("sequence rỗng — không có dữ liệu để trích xuất đặc trưng")

        normalized = self.normalizer.normalize_sequence(sequence, front_side_hint=front_side_hint)
        detector = (
            self.sprint_start_anomaly_detector
            if phase in (PHASE_SPRINT_START, PHASE_DRIVE)
            else self.anomaly_detector
        )
        labels = detector.detect(sequence, normalized, fps)
        state_summary = summarize_labels(labels)
        valid_frame_ratio = (
            state_summary.get(FrameState.STATIC.value, 0.0)
            + state_summary.get(FrameState.MOVING.value, 0.0)
        )

        df = to_dataframe(normalized)
        df["state"] = [lb.state.value for lb in labels]
        quality_flags = self._build_quality_flags(sequence, normalized, fps, state_summary)

        bundle = FeatureBundle(
            phase=phase,
            fps=fps,
            total_frames=len(sequence),
            valid_frame_ratio=valid_frame_ratio,
            state_summary=state_summary,
            normalized_df=df,
            quality_flags=quality_flags,
        )

        if phase in (PHASE_INTO_POSITION, PHASE_STATIC):
            bundle.static_summary = self._build_static_summary(labels, normalized)
        elif phase == PHASE_SPRINT_START:
            bundle.sprint_start_summary = self._build_sprint_start_summary(
                sequence, labels, normalized, fps
            )
        elif phase == PHASE_DYNAMIC:
            ankle_y = np.array(
                [pf.get("left_ankle")[1] for pf in sequence], dtype=np.float64
            )
            valid_segments = merge_valid_segments(
                labels, keep_states=(FrameState.MOVING,),
                min_segment_len_frames=self.anomaly_detector.th.min_segment_len_frames,
            )
            cycles = self._segment_cycles_within_valid_ranges(
                ankle_y, normalized, fps, valid_segments
            )
            bundle.cycles = cycles
            bundle.dynamic_summary = self._build_dynamic_summary(cycles)
            if reference_cycle is not None and cycles:
                bundle.reference_similarity = self._compare_to_reference(
                    cycles, reference_cycle
                )

        return bundle

    # ------------------------------------------------------------------
    def extract_sprint_start_full(
        self,
        sequence: List[PoseFrame],
        fps: float,
        front_side_hint: Optional[str] = None,
    ) -> FeatureBundle:
        """
        Trich xuat day du 4 sub-phase cho 1 video sprint_start.
        Auto segmentation (heuristic).
        """
        if not sequence:
            raise ValueError("sequence rỗng")

        normalized = self.normalizer.normalize_sequence(
            sequence, front_side_hint=front_side_hint
        )
        labels = self.sprint_start_anomaly_detector.detect(sequence, normalized, fps)
        state_summary = summarize_labels(labels)

        df = to_dataframe(normalized)
        df["state"] = [lb.state.value for lb in labels]
        quality_flags = self._build_quality_flags(sequence, normalized, fps, state_summary)

        bundle = FeatureBundle(
            phase="sprint_start_full",
            fps=fps,
            total_frames=len(sequence),
            valid_frame_ratio=(
                state_summary.get(FrameState.STATIC.value, 0.0)
                + state_summary.get(FrameState.MOVING.value, 0.0)
            ),
            state_summary=state_summary,
            normalized_df=df,
            quality_flags=quality_flags,
        )

        boundaries = self._detect_sub_phase_boundaries(sequence, normalized, fps, labels)
        bundle.sub_phase_order = [
            PHASE_INTO_POSITION, PHASE_STATIC, PHASE_SPRINT_START, PHASE_DRIVE,
        ]

        bundle.sub_phases[PHASE_INTO_POSITION] = self._build_into_position_summary(
            boundaries[PHASE_INTO_POSITION], normalized, fps, sequence
        )
        bundle.sub_phases[PHASE_STATIC] = self._build_set_position_summary(
            boundaries[PHASE_STATIC], normalized, fps, sequence
        )
        bundle.sub_phases[PHASE_SPRINT_START] = self._build_explosive_start_summary(
            boundaries[PHASE_SPRINT_START], sequence, normalized, fps, labels
        )
        bundle.sub_phases[PHASE_DRIVE] = self._build_drive_phase_summary(
            boundaries[PHASE_DRIVE], sequence, normalized, fps, bundle.quality_flags
        )

        bundle.static_summary = self._build_static_summary(labels, normalized)

        return bundle

    # ------------------------------------------------------------------
    def extract_sprint_start_from_windows(
        self,
        sequence: List[PoseFrame],
        fps: float,
        windows: Dict[str, Tuple[int, int]],
        front_side_hint: Optional[str] = None,
    ) -> FeatureBundle:
        """
        Variant của extract_sprint_start_full() nhưng dùng WINDOWS do caller
        cung cấp (thường là từ file phases đánh dấu thủ công) thay vì tự
        động segment.

        Args:
            sequence: List PoseFrame đã smoothing.
            fps: FPS video.
            windows: Dict[phase_name -> (start_frame, end_frame)].
                     Phải có đủ 4 phase: vao_cho, static, sprint_start, drive_phase.

        Returns:
            FeatureBundle giống hệt extract_sprint_start_full().
        """
        if not sequence:
            raise ValueError("sequence rỗng")

        normalized = self.normalizer.normalize_sequence(sequence, front_side_hint=front_side_hint)
        labels = self.sprint_start_anomaly_detector.detect(sequence, normalized, fps)
        state_summary = summarize_labels(labels)

        df = to_dataframe(normalized)
        df["state"] = [lb.state.value for lb in labels]
        quality_flags = self._build_quality_flags(sequence, normalized, fps, state_summary)

        bundle = FeatureBundle(
            phase="sprint_start_full",
            fps=fps,
            total_frames=len(sequence),
            valid_frame_ratio=(
                state_summary.get(FrameState.STATIC.value, 0.0)
                + state_summary.get(FrameState.MOVING.value, 0.0)
            ),
            state_summary=state_summary,
            normalized_df=df,
            quality_flags=quality_flags,
        )

        # Chuyển (start, end) tuple -> SubPhaseWindow
        boundaries: Dict[str, SubPhaseWindow] = {}
        for phase in (PHASE_INTO_POSITION, PHASE_STATIC,
                      PHASE_SPRINT_START, PHASE_DRIVE):
            if phase not in windows:
                raise ValueError(
                    f"windows thiếu phase '{phase}'. Cần đủ 4 phase: "
                    f"{PHASE_INTO_POSITION}, {PHASE_STATIC}, "
                    f"{PHASE_SPRINT_START}, {PHASE_DRIVE}"
                )
            start, end = windows[phase]
            boundaries[phase] = SubPhaseWindow(
                phase=phase,
                start_frame=int(start),
                end_frame=int(end),
                duration_s=max(0.0, (int(end) - int(start)) / fps),
            )

        bundle.sub_phase_order = [
            PHASE_INTO_POSITION, PHASE_STATIC, PHASE_SPRINT_START, PHASE_DRIVE,
        ]

        bundle.sub_phases[PHASE_INTO_POSITION] = self._build_into_position_summary(
            boundaries[PHASE_INTO_POSITION], normalized, fps, sequence
        )
        bundle.sub_phases[PHASE_STATIC] = self._build_set_position_summary(
            boundaries[PHASE_STATIC], normalized, fps, sequence
        )
        bundle.sub_phases[PHASE_SPRINT_START] = self._build_explosive_start_summary(
            boundaries[PHASE_SPRINT_START], sequence, normalized, fps, labels
        )
        bundle.sub_phases[PHASE_DRIVE] = self._build_drive_phase_summary(
            boundaries[PHASE_DRIVE], sequence, normalized, fps, bundle.quality_flags
        )

        bundle.static_summary = self._build_static_summary(labels, normalized)

        return bundle

    # ------------------------------------------------------------------
    def _detect_sub_phase_boundaries(
        self,
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
        labels,
    ) -> Dict[str, SubPhaseWindow]:
        """Phat hien moc chia 4 sub-phase (heuristic fallback)."""
        n = len(sequence)
        if n < 4:
            empty = SubPhaseWindow(
                phase="", start_frame=0, end_frame=n - 1, duration_s=n / fps
            )
            return {
                PHASE_INTO_POSITION: empty, PHASE_STATIC: empty,
                PHASE_SPRINT_START: empty, PHASE_DRIVE: empty,
            }

        # --- Heuristic 1: hand_off (cung logic confidence + debounce voi segmenter) ---
        hand_off = _detect_hand_off(
            sequence, normalized, self.discrete_event_segmenter.th,
        )
        if hand_off is None:
            hand_off = sequence[n // 2].frame_idx

        # --- Heuristic 2: hip_start ---
        hip_y_series = []
        for nf in normalized:
            h = nf.ratios.get("hip_above_shoulder", float("nan"))
            hip_y_series.append(h)
        hip_y_arr = np.array(hip_y_series, dtype=np.float64)

        hip_start = 0
        for i, v in enumerate(hip_y_arr):
            if not np.isnan(v) and v > 0.10:
                hip_start = normalized[i].frame_idx
                break

        # --- launch position ---
        events = self.discrete_event_segmenter.detect_events(
            sequence, normalized, fps, segment=(0, n - 1)
        )
        if events:
            launch_frame = events[0].launch_frame_idx
        else:
            launch_frame = hand_off + 4

        launch_frame = min(launch_frame, n - 1)
        drive_start = min(launch_frame + 8, n - 1)

        windows = {
            PHASE_INTO_POSITION: SubPhaseWindow(
                phase=PHASE_INTO_POSITION,
                start_frame=sequence[0].frame_idx,
                end_frame=max(hip_start - 1, sequence[0].frame_idx),
                duration_s=0.0,
            ),
            PHASE_STATIC: SubPhaseWindow(
                phase=PHASE_STATIC,
                start_frame=hip_start,
                end_frame=max(hand_off - 1, hip_start),
                duration_s=0.0,
            ),
            PHASE_SPRINT_START: SubPhaseWindow(
                phase=PHASE_SPRINT_START,
                start_frame=hand_off,
                end_frame=max(drive_start - 1, hand_off),
                duration_s=0.0,
            ),
            PHASE_DRIVE: SubPhaseWindow(
                phase=PHASE_DRIVE,
                start_frame=drive_start,
                end_frame=sequence[-1].frame_idx,
                duration_s=0.0,
            ),
        }
        for w in windows.values():
            w.duration_s = max(0.0, (w.end_frame - w.start_frame) / fps)
        return windows

    # ------------------------------------------------------------------
    @staticmethod
    def _slice_normalized(
        normalized: List[NormalizedFrame], window: SubPhaseWindow
    ) -> List[NormalizedFrame]:
        return [
            nf for nf in normalized
            if window.start_frame <= nf.frame_idx <= window.end_frame
        ]

    @staticmethod
    def _slice_sequence(
        sequence: Optional[Sequence[PoseFrame]], window: SubPhaseWindow
    ) -> List[PoseFrame]:
        if not sequence:
            return []
        return [
            pf for pf in sequence
            if window.start_frame <= pf.frame_idx <= window.end_frame
        ]

    def _build_into_position_summary(
        self, window: SubPhaseWindow, normalized: List[NormalizedFrame], fps: float,
        sequence: Optional[List[PoseFrame]] = None,
    ) -> IntoPositionSummary:
        del fps
        sliced = self._slice_normalized(normalized, window)
        if len(sliced) < AGGREGATE_MIN_SAMPLES:
            canonical_values, canonical_std, canonical_meta = _catalog_nan(PHASE_INTO_POSITION)
            return IntoPositionSummary(
                window=window,
                feature_values=canonical_values,
                feature_metadata=canonical_meta,
                feature_std=canonical_std,
            )
        poses = self._slice_sequence(sequence, window) if sequence else None
        raw_values, raw_std, _ = _aggregate_feature_window(
            sliced, PHASE_INTO_POSITION, use_mean=False, poses=poses,
        )
        canonical_values, canonical_meta = _remap_to_canonical(
            raw_values, PHASE_INTO_POSITION
        )
        canonical_std = {
            spec_key: raw_std.get(spec["internal"], float("nan"))
            for spec_key, spec in SUBPHASE_FEATURE_CATALOG[PHASE_INTO_POSITION].items()
        }
        return IntoPositionSummary(
            window=window,
            feature_values=canonical_values,
            feature_metadata=canonical_meta,
            feature_std=canonical_std,
        )

    def _build_set_position_summary(
        self, window: SubPhaseWindow, normalized: List[NormalizedFrame], fps: float,
        sequence: Optional[List[PoseFrame]] = None,
    ) -> SetPositionSummary:
        del fps
        sliced = self._slice_normalized(normalized, window)
        if len(sliced) < AGGREGATE_MIN_SAMPLES:
            canonical_values, canonical_std, canonical_meta = _catalog_nan(PHASE_STATIC)
            return SetPositionSummary(
                window=window,
                feature_values=canonical_values,
                feature_metadata=canonical_meta,
                feature_std=canonical_std,
            )
        poses = self._slice_sequence(sequence, window) if sequence else None
        raw_values, raw_std, _ = _aggregate_feature_window(
            sliced, PHASE_STATIC, use_mean=False, poses=poses,
        )
        canonical_values, canonical_meta = _remap_to_canonical(
            raw_values, PHASE_STATIC
        )
        canonical_std = {
            spec_key: raw_std.get(spec["internal"], float("nan"))
            for spec_key, spec in SUBPHASE_FEATURE_CATALOG[PHASE_STATIC].items()
        }
        return SetPositionSummary(
            window=window,
            feature_values=canonical_values,
            feature_metadata=canonical_meta,
            feature_std=canonical_std,
        )

    def _build_explosive_start_summary(
        self,
        window: SubPhaseWindow,
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
        labels,
    ) -> ExplosiveStartSummary:
        del labels
        sliced = self._slice_normalized(normalized, window)
        poses = self._slice_sequence(sequence, window)
        raw_values, _, _ = _aggregate_feature_window(
            sliced, PHASE_SPRINT_START, use_mean=False, poses=poses,
        )
        canonical_values, canonical_meta = _remap_to_canonical(
            raw_values, PHASE_SPRINT_START
        )

        events = self.discrete_event_segmenter.detect_events(
            sequence, normalized, fps,
            segment=(window.start_frame, window.end_frame),
        )

        launch_frame: Optional[int] = None
        launch_timestamp = 0.0
        method = "fallback_no_event"
        confidence = "low"
        peak_accel = float("nan")
        event_map: Dict[str, Optional[int]] = {
            "hand_off": None,
            "onset_of_motion": None,
            "rear_foot_off": None,
            "front_foot_off": None,
            "launch": None,
        }

        sub_seq = poses
        sub_norm = sliced
        accel = np.array([], dtype=np.float64)
        if len(sub_seq) >= 3 and len(sub_norm) == len(sub_seq):
            com = compute_center_of_mass(sub_seq)
            body_scale = np.array([nf.body_scale for nf in sub_norm], dtype=np.float64)
            _vel, accel = compute_kinematics(com, body_scale, fps)

        if events:
            event = events[0]
            launch_frame = event.launch_frame_idx
            method = event.launch_detection_method or "discrete_event_segmenter"
            confidence = event.launch_confidence or "medium"
            if not np.isnan(event.peak_acceleration):
                peak_accel = float(event.peak_acceleration)
            event_features = flatten_event_features(event)
            self._merge_launch_snapshot(canonical_values, canonical_meta, event_features)
            event_map.update({
                "hand_off": event.hand_off_frame,
                "onset_of_motion": event.onset_of_motion_frame,
                "rear_foot_off": event.rear_foot_off_frame,
                "front_foot_off": event.front_foot_off_frame,
                "launch": launch_frame,
            })
            canonical_values["rear_leg_leaves_first"] = _rear_leg_leaves_first(
                event.rear_foot_off_frame, event.front_foot_off_frame,
            )
            if event.front_foot_off_frame is not None and launch_frame is not None:
                canonical_values["first_step_timing"] = float(
                    event.front_foot_off_frame - launch_frame
                )
            launch_pos = next(
                (i for i, pf in enumerate(sequence) if pf.frame_idx == launch_frame),
                None,
            )
            if launch_pos is not None:
                launch_timestamp = sequence[launch_pos].timestamp_s
                canonical_values["arm_leg_opposition"] = _arm_leg_opposition(sequence[launch_pos])
            else:
                canonical_values["arm_leg_opposition"] = float("nan")
        else:
            # Khong co event: lay frame gia toc lon nhat trong window lam launch.
            finite = np.isfinite(accel)
            if finite.any():
                local_pos = int(np.argmax(np.where(finite, accel, -np.inf)))
                launch_frame = sub_seq[local_pos].frame_idx
                launch_timestamp = sub_seq[local_pos].timestamp_s
                peak_accel = float(accel[local_pos])
                method = "fallback_peak_acceleration"
                confidence = "low"
                snap = _aggregate_feature_window(
                    sub_norm[local_pos:local_pos + 1],
                    PHASE_SPRINT_START,
                    use_mean=True,
                    poses=sub_seq[local_pos:local_pos + 1],
                    min_samples=1,
                    min_valid_ratio=0.0,
                )[0]
                flat = {f"at_{key}": val for key, val in snap.items()}
                self._merge_launch_snapshot(canonical_values, canonical_meta, flat)
                canonical_values["arm_leg_opposition"] = _arm_leg_opposition(sub_seq[local_pos])
                event_map["launch"] = launch_frame
            else:
                # Khong co tin hieu: de feature NaN, khong gia lap launch = dau window.
                method = "fallback_no_event"
                confidence = "none"
                launch_frame = window.start_frame
                launch_timestamp = sub_seq[0].timestamp_s if sub_seq else 0.0

        if np.isnan(peak_accel) and len(accel):
            finite = accel[np.isfinite(accel)]
            if len(finite):
                peak_accel = float(np.max(finite))

        canonical_meta.setdefault(
            "rear_leg_leaves_first",
            {"unit": "bool", "level": "PRIMARY", "internal_key": "rear_leg_leaves_first"},
        )
        canonical_meta.setdefault(
            "arm_leg_opposition",
            {"unit": "bool", "level": "PRIMARY", "internal_key": "arm_leg_opposition"},
        )
        canonical_values.setdefault("rear_leg_leaves_first", float("nan"))
        canonical_values.setdefault("arm_leg_opposition", float("nan"))

        return ExplosiveStartSummary(
            window=window,
            launch_frame_idx=launch_frame if launch_frame is not None else window.start_frame,
            launch_timestamp_s=launch_timestamp,
            peak_acceleration=peak_accel,
            feature_values=canonical_values,
            feature_metadata=canonical_meta,
            events=event_map,
            launch_detection_method=method,
            launch_confidence=confidence,
        )

    @staticmethod
    def _merge_launch_snapshot(
        canonical_values: Dict[str, float],
        canonical_meta: Dict[str, Dict],
        event_features: Dict[str, float],
    ) -> None:
        """Ghi at_* tu snapshot launch. Khong ghi de bang NaN."""
        trunk = event_features.get("at_trunk_angle_vertical", event_features.get("at_trunk"))
        if trunk is not None and not (isinstance(trunk, float) and np.isnan(trunk)):
            canonical_values["at_trunk_angle_vertical"] = float(trunk)
            canonical_meta["at_trunk_angle_vertical"] = {
                "unit": "°", "level": "PRIMARY", "internal_key": "at_trunk",
            }

        knee = event_features.get("at_knee_front", event_features.get("at_knee_front_angle"))
        if knee is not None and not (isinstance(knee, float) and np.isnan(knee)):
            canonical_values["at_knee_front_angle"] = float(knee)
            canonical_meta["at_knee_front_angle"] = {
                "unit": "°", "level": "SECONDARY", "internal_key": "at_knee_front",
            }
            canonical_values["front_leg_full_extension"] = (
                1.0 if float(knee) >= FRONT_LEG_FULL_EXTENSION_DEG else 0.0
            )
            canonical_meta["front_leg_full_extension"] = {
                "unit": "bool",
                "level": "PRIMARY",
                "internal_key": "front_leg_full_extension",
                "threshold_deg": FRONT_LEG_FULL_EXTENSION_DEG,
            }

        leg_ground = event_features.get(
            "at_leg_ground_angle", event_features.get("leg_ground_angle"),
        )
        if leg_ground is not None and not (isinstance(leg_ground, float) and np.isnan(leg_ground)):
            canonical_values["leg_ground_angle"] = float(leg_ground)
            canonical_meta["leg_ground_angle"] = {
                "unit": "°", "level": "PRIMARY", "internal_key": "leg_ground_angle",
            }

    # ------------------------------------------------------------------
    def _build_drive_phase_summary(
        self,
        window: SubPhaseWindow,
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
        quality_flags: Optional[Dict[str, bool]] = None,
    ) -> DrivePhaseSummary:
        """
        Sub-phase 4 - Chay lao (25-30m dau).
        Tinh 4 trend features theo spec v2.1 muc 4.1 + 7.7.
        """
        sliced_norm = self._slice_normalized(normalized, window)
        n = len(sliced_norm)

        empty_result = DrivePhaseSummary(
            window=window,
            num_steps=0,
            feature_values={},
            feature_metadata={},
            step_length_trend=float("nan"),
            cadence_trend=float("nan"),
            contact_position_convergence=float("nan"),
            trunk_angle_vertical_trend=float("nan"),
        )
        if n < 5:
            return empty_result

        # --- Map frame_idx -> pos trong sliced_norm ---
        frame_idx_to_pos = {nf.frame_idx: i for i, nf in enumerate(sliced_norm)}

        # --- Trich tin hieu 1D qua cac frame drive_phase ---
        ankle_y_left  = np.full(n, np.nan, dtype=np.float64)
        ankle_y_right = np.full(n, np.nan, dtype=np.float64)
        body_x        = np.full(n, np.nan, dtype=np.float64)
        trunk_angle   = np.full(n, np.nan, dtype=np.float64)

        for pf in sequence:
            pos = frame_idx_to_pos.get(pf.frame_idx)
            if pos is None:
                continue
            try:
                ankle_y_left[pos]  = float(pf.get("left_ankle")[1])
                ankle_y_right[pos] = float(pf.get("right_ankle")[1])
                ms_x = (float(pf.get("left_shoulder")[0])
                        + float(pf.get("right_shoulder")[0])) / 2.0
                mh_x = (float(pf.get("left_hip")[0])
                        + float(pf.get("right_hip")[0])) / 2.0
                body_x[pos] = (ms_x + mh_x) / 2.0
            except Exception:
                continue

        for nf in sliced_norm:
            pos = frame_idx_to_pos.get(nf.frame_idx)
            if pos is None:
                continue
            trunk_angle[pos] = nf.angles.get("trunk_angle_vertical", float("nan"))

        # --- Shoulder width (px) de normalize step length / contact offset ---
        sw_list: List[float] = []
        for nf in sliced_norm:
            sw = nf.ratios.get("shoulder_width", float("nan")) * nf.body_scale
            if not np.isnan(sw) and sw > 0:
                sw_list.append(sw)
        shoulder_width_px = float(np.median(sw_list)) if sw_list else float("nan")
        if not np.isfinite(shoulder_width_px) or shoulder_width_px <= 0:
            if quality_flags is not None:
                quality_flags["drive_phase_unavailable"] = True
            return empty_result

        # --- Goi temporal_utils tinh 4 trend features ---
        from stage3_features.temporal_utils import compute_drive_phase_trends
        trend = compute_drive_phase_trends(
            ankle_y_left=ankle_y_left,
            ankle_y_right=ankle_y_right,
            body_x_series=body_x,
            trunk_angle_series=trunk_angle,
            shoulder_width_px=shoulder_width_px,
            fps=fps,
            max_steps=8,
        )

        return DrivePhaseSummary(
            window=window,
            num_steps=trend.n_steps,
            feature_values={
                "step_length_trend":            trend.step_length_trend,
                "cadence_trend":                trend.cadence_trend,
                "contact_position_convergence": trend.contact_convergence,
                "trunk_angle_vertical_trend":   trend.trunk_trend,
            },
            feature_metadata={
                "step_length_trend": {
                    "unit": "slope", "level": "SECONDARY",
                    "significant": trend.step_length_significant,
                },
                "cadence_trend": {
                    "unit": "slope", "level": "SECONDARY",
                    "significant": trend.cadence_significant,
                    "note": "khong cham neu FPS < 60 (spec 7.7)",
                },
                "contact_position_convergence": {
                    "unit": "slope", "level": "SECONDARY",
                    "significant": trend.contact_significant,
                },
                "trunk_angle_vertical_trend": {
                    "unit": "deg/step", "level": "SECONDARY",
                    "significant": trend.trunk_significant,
                },
            },
            step_length_trend=trend.step_length_trend,
            cadence_trend=trend.cadence_trend,
            contact_position_convergence=trend.contact_convergence,
            trunk_angle_vertical_trend=trend.trunk_trend,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_quality_flags(
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
        state_summary: Dict[str, float],
    ) -> Dict[str, bool]:
        flags: Dict[str, bool] = {
            "low_fps": fps < 60.0,
            "insufficient_data": state_summary.get(FrameState.INVALID.value, 0.0) > 0.10,
            "low_resolution": False,
            "wrist_low_confidence": False,
            "knee_back_low_confidence": False,
            "rear_ankle_low_confidence": False,
        }
        shoulder_widths = [
            nf.ratios.get("shoulder_width", float("nan")) * nf.body_scale
            for nf in normalized
        ]
        shoulder_arr = np.array(shoulder_widths, dtype=np.float64)
        shoulder_arr = shoulder_arr[~np.isnan(shoulder_arr)]
        if len(shoulder_arr):
            flags["low_resolution"] = bool(float(np.nanmedian(shoulder_arr)) < 50.0)

        def low_conf(joint_names, threshold, mode="mean"):
            misses = []
            for pf in sequence:
                vals = [pf.get_conf(j) for j in joint_names if j in pf.joint_names]
                if vals:
                    agg = float(np.mean(vals)) if mode == "mean" else float(np.min(vals))
                    misses.append(agg < threshold)
            return bool(np.mean(misses) > 0.20) if misses else True

        flags["wrist_low_confidence"] = low_conf(("left_wrist", "right_wrist"), 0.5, mode="mean")
        flags["knee_back_low_confidence"] = low_conf(("left_knee", "right_knee"), 0.5, mode="mean")
        flags["rear_ankle_low_confidence"] = low_conf(("left_ankle", "right_ankle"), 0.4, mode="min")
        return flags

    # ------------------------------------------------------------------
    @staticmethod
    def _build_static_summary(labels, normalized: List[NormalizedFrame]):
        static_segments = merge_valid_segments(
            labels, keep_states=(FrameState.STATIC,), min_segment_len_frames=1
        )
        if not static_segments:
            return None
        start, end, _ = max(static_segments, key=lambda seg: seg[1] - seg[0])
        window = [nf for nf in normalized if start <= nf.frame_idx <= end]
        if not window:
            return None

        angle_names = window[0].angles.keys()
        angle_mean, angle_std = {}, {}
        for name in angle_names:
            values = np.array([nf.angles[name] for nf in window], dtype=np.float64)
            values = values[~np.isnan(values)]
            angle_mean[name] = float(np.mean(values)) if len(values) else float("nan")
            angle_std[name] = float(np.std(values)) if len(values) else float("nan")

        ratio_names = window[0].ratios.keys()
        ratio_mean, ratio_std = {}, {}
        for name in ratio_names:
            values = np.array([nf.ratios[name] for nf in window], dtype=np.float64)
            values = values[~np.isnan(values)]
            ratio_mean[name] = float(np.mean(values)) if len(values) else float("nan")
            ratio_std[name] = float(np.std(values)) if len(values) else float("nan")

        feature_mean = {**angle_mean, **ratio_mean}
        feature_std = {**angle_std, **ratio_std}
        duration_s = window[-1].timestamp_s - window[0].timestamp_s
        return StaticPhaseSummary(
            segment_start_frame=start, segment_end_frame=end,
            duration_s=duration_s, angle_mean=angle_mean, angle_std=angle_std,
            ratio_mean=ratio_mean, ratio_std=ratio_std,
            feature_mean=feature_mean, feature_std=feature_std,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_sprint_start_summary(
        sequence: List[PoseFrame],
        labels,
        normalized: List[NormalizedFrame],
        fps: float,
    ) -> Optional[SprintStartPhaseSummary]:
        run = _longest_valid_run(labels, (FrameState.STATIC, FrameState.MOVING))
        if run is None:
            return None
        start, end = run

        frame_idx_to_pos = {pf.frame_idx: i for i, pf in enumerate(sequence)}
        pos_start, pos_end = frame_idx_to_pos[start], frame_idx_to_pos[end]
        sub_sequence = sequence[pos_start : pos_end + 1]
        sub_normalized = normalized[pos_start : pos_end + 1]
        if len(sub_sequence) < 3:
            return None

        events = DiscreteEventSegmenter().detect_events(
            sequence, normalized, fps, segment=(start, end)
        )
        event = events[0] if events else None

        com = compute_center_of_mass(sub_sequence)
        body_scale = np.array([nf.body_scale for nf in sub_normalized], dtype=np.float64)
        _velocity, acceleration = compute_kinematics(com, body_scale, fps)

        if event is not None:
            launch_frame_idx = event.launch_frame_idx
            launch_pos = next(
                (i for i, pf in enumerate(sub_sequence) if pf.frame_idx == launch_frame_idx),
                0,
            )
            event_features = flatten_event_features(event)
            if "at_trunk" in event_features:
                event_features["at_trunk_angle_vertical"] = event_features["at_trunk"]
            if "at_knee_front" in event_features:
                event_features["at_knee_front_angle"] = event_features["at_knee_front"]
                event_features["front_leg_full_extension"] = (
                    1.0 if event_features["at_knee_front"] >= 170.0 else 0.0
                )
            method = "discrete_event_segmenter"
            confidence = "medium"
        else:
            accel_for_argmax = np.where(np.isnan(acceleration), -np.inf, acceleration)
            launch_pos = int(np.argmax(accel_for_argmax))
            event_features = {}
            method = "fallback_peak_acceleration"
            confidence = "low"

        launch_pose = sub_sequence[launch_pos]
        launch_nf = sub_normalized[launch_pos]
        peak_accel = acceleration[launch_pos]

        return SprintStartPhaseSummary(
            segment_start_frame=start,
            segment_end_frame=end,
            duration_s=sub_sequence[-1].timestamp_s - sub_sequence[0].timestamp_s,
            launch_frame_idx=launch_pose.frame_idx,
            launch_timestamp_s=launch_pose.timestamp_s,
            peak_acceleration=float(peak_accel) if not np.isnan(peak_accel) else float("nan"),
            angles_at_launch=dict(launch_nf.angles),
            trunk_angle_series=[
                nf.angles.get(
                    "trunk_angle_vertical",
                    nf.angles.get("trunk", float("nan")),
                )
                for nf in sub_normalized
            ],
            event_features=event_features,
            launch_detection_method=method,
            launch_confidence=confidence,
        )

    # ------------------------------------------------------------------
    def _segment_cycles_within_valid_ranges(
        self,
        ankle_y_signal: np.ndarray,
        normalized: List[NormalizedFrame],
        fps: float,
        valid_segments: List[Tuple[int, int, FrameState]],
    ) -> List[GaitCycle]:
        all_cycles: List[GaitCycle] = []
        frame_idx_to_pos = {nf.frame_idx: i for i, nf in enumerate(normalized)}
        for start, end, _ in valid_segments:
            pos_start, pos_end = frame_idx_to_pos[start], frame_idx_to_pos[end]
            sub_normalized = normalized[pos_start : pos_end + 1]
            sub_ankle_y = ankle_y_signal[pos_start : pos_end + 1]
            cycles = self.cycle_segmenter.segment_from_ankle_y(
                sub_ankle_y, sub_normalized, fps
            )
            for c in cycles:
                c.cycle_index = len(all_cycles)
                all_cycles.append(c)
        return all_cycles

    # ------------------------------------------------------------------
    @staticmethod
    def _build_dynamic_summary(cycles: List[GaitCycle]) -> Optional[DynamicPhaseSummary]:
        if not cycles:
            return None
        cadences = np.array([c.cadence_spm for c in cycles], dtype=np.float64)
        feature_keys = list(cycles[0].resampled_features.keys())

        rom_by_angle: Dict[str, float] = {}
        mean_curve_by_angle: Dict[str, np.ndarray] = {}
        for key in feature_keys:
            stacked = np.stack([c.resampled_features[key] for c in cycles], axis=0)
            mean_curve = np.nanmean(stacked, axis=0)
            mean_curve_by_angle[key] = mean_curve
            rom_by_angle[key] = float(np.nanmax(mean_curve) - np.nanmin(mean_curve))

        symmetry_index: Dict[str, float] = {}
        for left_name, right_name in _SYMMETRY_PAIRS:
            left_key, right_key = f"angle_{left_name}", f"angle_{right_name}"
            if left_key not in rom_by_angle or right_key not in rom_by_angle:
                continue
            l, r = rom_by_angle[left_key], rom_by_angle[right_key]
            denom = (l + r) / 2.0
            symmetry_index[f"{left_name.split('_')[0]}"] = (
                float(abs(l - r) / denom * 100.0) if denom > 1e-6 else float("nan")
            )

        return DynamicPhaseSummary(
            num_valid_cycles=len(cycles),
            cadence_spm_mean=float(np.mean(cadences)),
            cadence_spm_std=float(np.std(cadences)),
            rom_by_angle=rom_by_angle,
            symmetry_index=symmetry_index,
            mean_curve_by_angle=mean_curve_by_angle,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compare_to_reference(
        cycles: List[GaitCycle], reference_cycle: GaitCycle
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for key, ref_curve in reference_cycle.resampled_features.items():
            student_curves = [
                c.resampled_features.get(key)
                for c in cycles if key in c.resampled_features
            ]
            if not student_curves:
                continue
            student_mean = np.nanmean(np.stack(student_curves, axis=0), axis=0)
            scale = max(1e-6, float(np.nanmax(ref_curve) - np.nanmin(ref_curve)))
            scores[key] = dtw_similarity_score(student_mean, ref_curve, scale=scale)
        return scores


# =====================================================================
# SERIALIZATION
# =====================================================================

def bundle_to_summary_dict(bundle: FeatureBundle) -> Dict:
    """Chuyển FeatureBundle thành dict phẳng, dễ serialize (JSON)."""
    out: Dict = {
        "phase": bundle.phase,
        "fps": bundle.fps,
        "total_frames": bundle.total_frames,
        "valid_frame_ratio": bundle.valid_frame_ratio,
        "state_summary": bundle.state_summary,
        "quality_flags": bundle.quality_flags,
    }
    if bundle.static_summary is not None:
        out["static"] = {
            "duration_s": bundle.static_summary.duration_s,
            "angle_mean": bundle.static_summary.angle_mean,
            "angle_std": bundle.static_summary.angle_std,
            "ratio_mean": bundle.static_summary.ratio_mean,
            "ratio_std": bundle.static_summary.ratio_std,
            "feature_mean": bundle.static_summary.feature_mean,
            "feature_std": bundle.static_summary.feature_std,
        }
    if bundle.dynamic_summary is not None:
        d = bundle.dynamic_summary
        out["dynamic"] = {
            "num_valid_cycles": d.num_valid_cycles,
            "cadence_spm_mean": d.cadence_spm_mean,
            "cadence_spm_std": d.cadence_spm_std,
            "rom_by_angle": d.rom_by_angle,
            "symmetry_index": d.symmetry_index,
        }
    if bundle.reference_similarity is not None:
        out["reference_similarity"] = bundle.reference_similarity
    if bundle.sprint_start_summary is not None:
        s = bundle.sprint_start_summary
        out["sprint_start"] = {
            "duration_s": s.duration_s,
            "launch_frame_idx": s.launch_frame_idx,
            "launch_timestamp_s": s.launch_timestamp_s,
            "peak_acceleration": s.peak_acceleration,
            "angles_at_launch": s.angles_at_launch,
            "event_features": s.event_features,
            "launch_detection_method": s.launch_detection_method,
            "launch_confidence": s.launch_confidence,
        }
    if bundle.sub_phases:
        out["sub_phases"] = {}
        for phase, summary in bundle.sub_phases.items():
            sp: Dict = {"phase": phase}
            if hasattr(summary, "window"):
                sp["window"] = {
                    "start": summary.window.start_frame,
                    "end": summary.window.end_frame,
                    "duration_s": summary.window.duration_s,
                }
            if hasattr(summary, "feature_values"):
                sp["feature_values"] = summary.feature_values
            if hasattr(summary, "feature_metadata"):
                sp["feature_metadata"] = summary.feature_metadata
            out["sub_phases"][phase] = sp
    return out