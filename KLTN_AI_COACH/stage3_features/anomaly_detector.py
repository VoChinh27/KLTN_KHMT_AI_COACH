"""
stage3_features/anomaly_detector.py
=====================================
Mục đích : Phát hiện đoạn bất thường để LOẠI BỎ trước khi tính đặc trưng.

V2.1 - Bổ sung:
    - QualityFlags dataclass: tap trung tat ca co chat luong du lieu.
    - compute_quality_flags(): tinh flags theo spec v2.1 muc 7.3.
    - Cac nguong lay tu quality_gates cua sport config.

Nguyên tắc: CHỈ LOẠI BỎ, không "sửa" hay suy diễn số liệu cho đoạn bất thường.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.pose_schema import PoseFrame, JOINT_GROUPS
from stage3_features.normalizer import NormalizedFrame


class FrameState(str, Enum):
    STATIC = "static"
    MOVING = "moving"
    TRANSITION = "transition"
    INVALID = "invalid"


@dataclass
class FrameLabel:
    frame_idx: int
    state: FrameState
    reason: Optional[str] = None


@dataclass
class AnomalyThresholds:
    static_velocity_max: float = 0.6
    moving_velocity_min: float = 0.6
    fall_velocity_min: float = 3.5
    fall_acceleration_min: float = 6.0
    fall_trunk_angle_deg: float = 55.0
    min_visible_ratio: float = 0.6
    z_score_window: int = 15
    z_score_threshold: float = 3.0
    min_segment_len_frames: int = 5


# =====================================================================
# V2.1 - QUALITY FLAGS
# =====================================================================

@dataclass
class QualityFlags:
    """
    Co chat luong du lieu - theo spec v2.1 muc 7.3 + 7.8.
    Bat ky flag nao True -> pipeline co the skip 1 so feature hoac giam confidence.
    """

    # --- Du lieu co ban ---
    low_fps: bool = False                     # FPS < 60
    low_resolution: bool = False              # shoulder_width < 50px
    insufficient_data: bool = False           # NaN > 10% sub-phase
    perspective_warning: bool = False         # camera lech > 15 do
    out_of_frame: bool = False                # VDV ra khoi khung
    multi_person: bool = False                # nhieu nguoi trong khung

    # --- Confidence theo nhom joint ---
    wrist_low_confidence: bool = False        # wrist_conf < 0.5 o > 20% frame
    knee_back_low_confidence: bool = False    # knee_conf < 0.5 o > 20% frame
    rear_ankle_low_confidence: bool = False   # ankle_conf < 0.4 o > 20% frame

    # --- Su kien ---
    hand_off_not_found: bool = False
    launch_not_found: bool = False
    leg_order_violation: bool = False
    pose_unstable: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "low_fps": self.low_fps,
            "low_resolution": self.low_resolution,
            "insufficient_data": self.insufficient_data,
            "perspective_warning": self.perspective_warning,
            "out_of_frame": self.out_of_frame,
            "multi_person": self.multi_person,
            "wrist_low_confidence": self.wrist_low_confidence,
            "knee_back_low_confidence": self.knee_back_low_confidence,
            "rear_ankle_low_confidence": self.rear_ankle_low_confidence,
            "hand_off_not_found": self.hand_off_not_found,
            "launch_not_found": self.launch_not_found,
            "leg_order_violation": self.leg_order_violation,
            "pose_unstable": self.pose_unstable,
        }

    def active_flags(self) -> List[str]:
        return [k for k, v in self.to_dict().items() if v]


def sprint_start_thresholds(base: Optional[AnomalyThresholds] = None) -> AnomalyThresholds:
    """Bộ ngưỡng riêng cho pha bật xuất phát - xem docstring chi tiết trong spec v2.1."""
    base = base or AnomalyThresholds()
    return AnomalyThresholds(
        static_velocity_max=base.static_velocity_max,
        moving_velocity_min=base.moving_velocity_min,
        fall_velocity_min=base.fall_velocity_min * 4.0,
        fall_acceleration_min=base.fall_acceleration_min * 5.0,
        fall_trunk_angle_deg=float("inf"),
        min_visible_ratio=base.min_visible_ratio,
        z_score_window=base.z_score_window,
        z_score_threshold=base.z_score_threshold * 2.0,
        min_segment_len_frames=max(1, base.min_segment_len_frames - 2),
    )


# =====================================================================
# Kinematics helpers
# =====================================================================

def compute_center_of_mass(sequence: List[PoseFrame]) -> np.ndarray:
    com = np.zeros((len(sequence), 2), dtype=np.float64)
    for i, pf in enumerate(sequence):
        mid_shoulder = (pf.get("left_shoulder") + pf.get("right_shoulder")) / 2.0
        mid_hip = (pf.get("left_hip") + pf.get("right_hip")) / 2.0
        com[i] = (mid_shoulder + mid_hip) / 2.0
    return com


def compute_kinematics(
    com_xy: np.ndarray, body_scale: np.ndarray, fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fps
    velocity_px = np.gradient(com_xy, axis=0) / dt
    velocity_mag_px = np.linalg.norm(velocity_px, axis=1)
    safe_scale = np.where(body_scale > 1e-6, body_scale, np.nan)
    velocity_norm = velocity_mag_px / safe_scale
    acceleration_norm = np.gradient(velocity_norm) / dt
    return velocity_norm, np.abs(acceleration_norm)


def rolling_z_score(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    z = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = signal[lo:hi]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) < 3:
            z[i] = 0.0
            continue
        mu, sigma = np.mean(window_vals), np.std(window_vals)
        z[i] = 0.0 if sigma < 1e-9 else (signal[i] - mu) / sigma
    return z


# =====================================================================
# V2.1 - Tinh QualityFlags
# =====================================================================

def compute_quality_flags(
    sequence: List[PoseFrame],
    normalized: List[NormalizedFrame],
    fps: float,
    quality_gates: Optional[Dict[str, float]] = None,
) -> QualityFlags:
    """
    Tinh tat ca quality flags theo spec v2.1.
    quality_gates: lay tu sport_config.quality_gates (neu co).
    """
    gates = quality_gates or {}
    wrist_conf_min = gates.get("wrist_conf_min", 0.5)
    knee_conf_min = gates.get("knee_back_conf_min", 0.5)
    ankle_conf_min = gates.get("rear_ankle_conf_min", 0.4)
    low_conf_ratio = gates.get("low_conf_ratio_threshold", 0.20)
    min_fps = gates.get("min_fps_for_temporal", 60.0)
    min_shoulder_px = gates.get("low_resolution_shoulder_width_px", 50.0)
    max_nan_ratio = gates.get("max_nan_ratio_per_subphase", 0.10)

    flags = QualityFlags()

    # --- FPS ---
    flags.low_fps = fps < min_fps

    # --- Resolution ---
    shoulder_px_list = []
    for nf in normalized:
        sw = nf.ratios.get("shoulder_width", float("nan")) * nf.body_scale
        if not np.isnan(sw):
            shoulder_px_list.append(sw)
    if shoulder_px_list:
        flags.low_resolution = bool(float(np.median(shoulder_px_list)) < min_shoulder_px)

    # --- Insufficient data (NaN ratio) ---
    if normalized:
        n_nan_total = 0
        n_total = 0
        for nf in normalized:
            for v in list(nf.angles.values()) + list(nf.ratios.values()):
                n_total += 1
                if isinstance(v, float) and np.isnan(v):
                    n_nan_total += 1
        if n_total > 0:
            flags.insufficient_data = (n_nan_total / n_total) > max_nan_ratio

    # --- Low confidence theo nhom ---
    def _low_conf_ratio(group: str, threshold: float) -> bool:
        miss_count = 0
        valid_count = 0
        for pf in sequence:
            conf = pf.get_group_conf(group)
            if np.isnan(conf):
                continue
            valid_count += 1
            if conf < threshold:
                miss_count += 1
        if valid_count == 0:
            return True
        return (miss_count / valid_count) > low_conf_ratio

    flags.wrist_low_confidence = _low_conf_ratio("wrist", wrist_conf_min)
    flags.knee_back_low_confidence = _low_conf_ratio("knee", knee_conf_min)
    flags.rear_ankle_low_confidence = _low_conf_ratio("ankle", ankle_conf_min)

    # --- Pose unstable: goc nhay > 30 do giua 2 frame lien tiep ---
    if len(normalized) >= 2:
        unstable_count = 0
        for i in range(1, len(normalized)):
            prev = normalized[i - 1].angles.get("trunk", float("nan"))
            cur = normalized[i].angles.get("trunk", float("nan"))
            if not np.isnan(prev) and not np.isnan(cur):
                if abs(cur - prev) > 30.0:
                    unstable_count += 1
        if len(normalized) > 1 and unstable_count / (len(normalized) - 1) > 0.10:
            flags.pose_unstable = True

    return flags


# =====================================================================
# AnomalyDetector
# =====================================================================

class AnomalyDetector:
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        self.th = thresholds or AnomalyThresholds()

    def detect(
        self,
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
    ) -> List[FrameLabel]:
        if len(sequence) != len(normalized):
            raise ValueError("sequence và normalized phải cùng độ dài")
        if not sequence:
            return []

        com = compute_center_of_mass(sequence)
        body_scale = np.array([nf.body_scale for nf in normalized], dtype=np.float64)
        velocity, acceleration = compute_kinematics(com, body_scale, fps)
        vel_z = rolling_z_score(velocity, self.th.z_score_window)

        labels: List[FrameLabel] = []
        for i, (pf, nf) in enumerate(zip(sequence, normalized)):
            visible_ratio = pf.visible_ratio()
            trunk_angle = nf.angles.get("trunk", float("nan"))

            if not nf.is_valid or visible_ratio < self.th.min_visible_ratio:
                labels.append(FrameLabel(
                    frame_idx=pf.frame_idx, state=FrameState.INVALID,
                    reason=nf.invalid_reason or f"visible_ratio={visible_ratio:.2f}",
                ))
                continue

            is_fall_like = (
                (not np.isnan(trunk_angle) and trunk_angle >= self.th.fall_trunk_angle_deg)
                or velocity[i] >= self.th.fall_velocity_min
                or acceleration[i] >= self.th.fall_acceleration_min
                or abs(vel_z[i]) >= self.th.z_score_threshold
            )
            if is_fall_like:
                reasons = []
                if not np.isnan(trunk_angle) and trunk_angle >= self.th.fall_trunk_angle_deg:
                    reasons.append(f"trunk_angle={trunk_angle:.0f}°")
                if velocity[i] >= self.th.fall_velocity_min:
                    reasons.append(f"velocity={velocity[i]:.2f}")
                if acceleration[i] >= self.th.fall_acceleration_min:
                    reasons.append(f"accel={acceleration[i]:.2f}")
                if abs(vel_z[i]) >= self.th.z_score_threshold:
                    reasons.append(f"z={vel_z[i]:.1f}")
                labels.append(FrameLabel(
                    frame_idx=pf.frame_idx, state=FrameState.INVALID,
                    reason="Nghi ngã/va chạm: " + "; ".join(reasons),
                ))
                continue

            if velocity[i] <= self.th.static_velocity_max:
                labels.append(FrameLabel(frame_idx=pf.frame_idx, state=FrameState.STATIC))
            elif velocity[i] >= self.th.moving_velocity_min:
                labels.append(FrameLabel(frame_idx=pf.frame_idx, state=FrameState.MOVING))
            else:
                labels.append(FrameLabel(frame_idx=pf.frame_idx, state=FrameState.TRANSITION))

        return labels


def merge_valid_segments(
    labels: List[FrameLabel],
    keep_states: Sequence[FrameState] = (FrameState.STATIC, FrameState.MOVING),
    min_segment_len_frames: int = 5,
) -> List[Tuple[int, int, FrameState]]:
    segments: List[Tuple[int, int, FrameState]] = []
    if not labels:
        return segments
    start_idx = 0
    for i in range(1, len(labels) + 1):
        same_state = i < len(labels) and labels[i].state == labels[start_idx].state
        if same_state:
            continue
        state = labels[start_idx].state
        end_idx = i - 1
        if state in keep_states and (end_idx - start_idx + 1) >= min_segment_len_frames:
            segments.append((labels[start_idx].frame_idx, labels[end_idx].frame_idx, state))
        start_idx = i
    return segments


def summarize_labels(labels: List[FrameLabel]) -> dict:
    total = len(labels)
    if total == 0:
        return {}
    counts = {state.value: 0 for state in FrameState}
    for lb in labels:
        counts[lb.state.value] += 1
    return {k: v / total for k, v in counts.items()}