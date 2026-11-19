"""
stage3_features/discrete_event_segmenter.py
=============================================
Mục đích : Phát hiện sự kiện rời rạc cho sprint_start theo Spec v2.1.

V2.1 - Bổ sung:
    - 5 sự kiện: hand_off, onset_of_motion, rear_foot_off, front_foot_off, launch.
    - Launch detection 3-tier fallback chain (thuần pose, KHÔNG dùng audio).
    - Fallback rear_foot_off bằng hip_thigh_angle khi rear_ankle_conf thấp.
    - Output kèm launch_detection_method + launch_confidence + quality flags.

V2.2 - Sửa lỗi (theo "Danh sách lỗi cần sửa"):
    #1  Tính 2 PRIMARY boolean còn thiếu, đưa vào DiscreteEvent.derived_features:
          rear_leg_leaves_first  (so rear_foot_off vs front_foot_off)
          arm_leg_opposition     (thiết kế mới, xem _compute_arm_leg_opposition)
        Đồng thời tính thêm leg_ground_angle và front_knee_peak_extension.
    #2  detect_events KHÔNG raise khi segment ngoài dải video: clamp + log
        warning; rỗng sau clamp -> trả [].
    #8  Không còn phụ thuộc mù vào key hip_back/thigh_back/hip_thigh_angle:
        nếu Normalizer không sinh -> tự tính từ keypoint (xem _build_signals).
        Chỉ đọc key GÓC (angles), không nhầm với tỉ lệ cùng tên (ratios).
    #9  tier1_rear_knee_min: 150° -> 130°.
    #10 hand_off có debounce (>=3 frame liên tiếp, có biên an toàn).
    #12 onset: nếu chưa đủ frame baseline trước hand_off -> trả None.
    #13 Thống nhất key thân người: ưu tiên trunk_angle_vertical, fallback trunk.
    #14 So sánh COM theo |Δx| (không phụ thuộc chiều chạy/hướng camera).
    Bổ sung: tham số hand_off_hint (caller đã biết biên pha), quality_flags
    chấp nhận cả đối tượng QualityFlags lẫn dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.geometry import angle_at_vertex, angle_to_vertical
from core.pose_schema import PoseFrame
from stage3_features.anomaly_detector import (
    compute_center_of_mass,
    compute_kinematics,
    QualityFlags,
)
from stage3_features.normalizer import NormalizedFrame

logger = logging.getLogger(__name__)

# Thứ tự ưu tiên key góc thân (Bug #13): spec v2.1 dùng trunk_angle_vertical.
TRUNK_KEYS: Tuple[str, ...] = ("trunk_angle_vertical", "trunk")

FlagsLike = Union[QualityFlags, Dict[str, bool], None]


# =====================================================================
# Thresholds
# =====================================================================

@dataclass
class DiscreteEventThresholds:
    # --- Cửa sổ lấy feature quanh sự kiện ---
    pre_event_window_frames: int = 5
    post_event_window_frames: int = 8

    # --- Phát hiện sự kiện ---
    min_peak_prominence_ratio: float = 0.3
    min_peak_acceleration: float = 2.0
    max_events_per_segment: int = 1  # giữ để tương thích API; hiện chỉ 1 sự kiện

    # --- hand_off ---
    hand_off_conf_min: float = 0.5
    hand_off_debounce_frames: int = 3      # Bug #10: số frame liên tiếp tối thiểu
    hand_off_margin_ratio: float = 0.05    # wrist_y < hip_y - 5% body_scale

    # --- onset_of_motion ---
    onset_std_factor: float = 3.0
    onset_abs_threshold_px: float = 5.0
    onset_window_after_hand_off: int = 12
    onset_min_baseline_frames: int = 5     # Bug #12

    # --- rear_foot_off / front_foot_off ---
    foot_off_delta_ratio: float = 0.05     # Δ = 5% shoulder_width
    rear_ankle_conf_fallback: float = 0.4
    hip_thigh_std_factor: float = 3.0

    # --- launch 3-tier ---
    tier1_torso_min: float = 30.0
    tier1_torso_max: float = 55.0
    tier1_rear_knee_min: float = 130.0     # Bug #9: 150 -> 130
    tier1_hip_omega_factor: float = 0.7
    tier1_com_delta_ratio: float = 0.02    # Bug #14: |Δx_COM| > 2% shoulder_width
    tier1_min_score: int = 4

    tier2_combined_min: float = 0.6

    tier3_com_delta_ratio: float = 0.02
    tier3_torso_delta: float = 5.0
    tier3_hip_omega_factor: float = 1.5

    # --- Cửa sổ tìm launch ---
    launch_window_after_onset: int = 8
    launch_max_offset_from_hand_off: int = 12

    # --- arm_leg_opposition (Bug #1) ---
    arm_conf_min: float = 0.4
    arm_opposition_min_diff_ratio: float = 0.10  # chênh lệch tối thiểu, theo body_scale


# =====================================================================
# Output dataclass
# =====================================================================

@dataclass
class DiscreteEvent:
    """1 sự kiện bộc phát với đầy đủ metadata."""
    event_index: int
    segment_start_frame: int
    segment_end_frame: int
    launch_frame_idx: int
    launch_timestamp_s: float
    peak_acceleration: float

    # 5 mốc sự kiện
    hand_off_frame: Optional[int] = None
    onset_of_motion_frame: Optional[int] = None
    rear_foot_off_frame: Optional[int] = None
    front_foot_off_frame: Optional[int] = None

    # Feature snapshots
    pre_event_features: Dict[str, float] = field(default_factory=dict)
    at_event_features: Dict[str, float] = field(default_factory=dict)
    post_event_features: Dict[str, float] = field(default_factory=dict)

    # Đặc trưng suy ra từ SỰ KIỆN (Bug #1). Khóa:
    #   rear_leg_leaves_first, arm_leg_opposition  (1.0 / 0.0 / NaN = không xác định)
    #   leg_ground_angle (độ), front_knee_peak_extension (độ)
    derived_features: Dict[str, float] = field(default_factory=dict)

    # Bối cảnh
    trunk_angle_series: List[float] = field(default_factory=list)
    timestamps_s: List[float] = field(default_factory=list)

    # Metadata
    launch_detection_method: str = "unknown"
    launch_confidence: str = "low"
    quality_flags: FlagsLike = None

    def to_dict(self) -> Dict:
        return {
            "event_index": self.event_index,
            "segment_start_frame": self.segment_start_frame,
            "segment_end_frame": self.segment_end_frame,
            "hand_off_frame": self.hand_off_frame,
            "onset_of_motion_frame": self.onset_of_motion_frame,
            "rear_foot_off_frame": self.rear_foot_off_frame,
            "front_foot_off_frame": self.front_foot_off_frame,
            "launch_frame_idx": self.launch_frame_idx,
            "launch_timestamp_s": self.launch_timestamp_s,
            "peak_acceleration": self.peak_acceleration,
            "launch_detection_method": self.launch_detection_method,
            "launch_confidence": self.launch_confidence,
            "pre_event_features": self.pre_event_features,
            "at_event_features": self.at_event_features,
            "post_event_features": self.post_event_features,
            "derived_features": self.derived_features,
            "quality_flags": _flags_to_dict(self.quality_flags),
        }


def flatten_event_features(event: DiscreteEvent) -> Dict[str, float]:
    """Gộp 3 snapshot thành dict phẳng với prefix pre_/at_/post_.
    (derived_features KHÔNG nằm trong đây — đọc trực tiếp event.derived_features.)"""
    out: Dict[str, float] = {}
    for prefix, features in (
        ("pre", event.pre_event_features),
        ("at", event.at_event_features),
        ("post", event.post_event_features),
    ):
        for key, val in features.items():
            out[f"{prefix}_{key}"] = val
    return out


def mean_features(frames: Sequence[NormalizedFrame]) -> Dict[str, float]:
    """Trung bình các góc/tỉ lệ (bỏ NaN) qua nhiều NormalizedFrame."""
    if not frames:
        return {}
    keys = set()
    for nf in frames:
        keys.update(nf.angles.keys())
        keys.update(nf.ratios.keys())
    out: Dict[str, float] = {}
    for key in keys:
        values = []
        for nf in frames:
            if key in nf.angles:
                values.append(nf.angles[key])
            elif key in nf.ratios:
                values.append(nf.ratios[key])
        arr = np.array(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        out[key] = float(np.mean(arr)) if len(arr) else float("nan")
    return out


_mean_features = mean_features  # alias tương thích mã cũ


# =====================================================================
# Helpers chung
# =====================================================================

def _flag_set(flags: FlagsLike, name: str, value: bool = True) -> None:
    if flags is None:
        return
    if isinstance(flags, dict):
        flags[name] = value
        return
    try:
        setattr(flags, name, value)
    except Exception:  # noqa: BLE001
        pass


def _flag_get(flags: FlagsLike, name: str, default: bool = False) -> bool:
    if flags is None:
        return default
    if isinstance(flags, dict):
        return bool(flags.get(name, default))
    return bool(getattr(flags, name, default))


def _flags_to_dict(flags: FlagsLike) -> Dict[str, Any]:
    if flags is None:
        return {}
    if isinstance(flags, dict):
        return dict(flags)
    to_dict = getattr(flags, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def _ang(nf: NormalizedFrame, keys: Sequence[str]) -> float:
    """Lấy GÓC (chỉ trong nf.angles) theo thứ tự ưu tiên key. NaN nếu không có.
    Cố ý không đọc nf.ratios để không nhầm 'thigh_back' (tỉ lệ chiều dài)
    với góc (Bug #8)."""
    for k in keys:
        v = nf.angles.get(k)
        if v is not None and not np.isnan(v):
            return float(v)
    return float("nan")


def _shoulder_px(nf: NormalizedFrame) -> float:
    sw = nf.ratios.get("shoulder_width", float("nan")) * nf.body_scale
    return float(sw) if (not np.isnan(sw) and sw > 0) else float("nan")


def _pos_of(sequence: List[PoseFrame], frame_idx: Optional[int]) -> Optional[int]:
    if frame_idx is None:
        return None
    for i, pf in enumerate(sequence):
        if pf.frame_idx == frame_idx:
            return i
    return None


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    a_min, a_max = float(finite.min()), float(finite.max())
    if a_max - a_min < 1e-9:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def _resolve_segment(
    sequence: List[PoseFrame], segment: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """Đổi (start_frame, end_frame) theo frame_idx gốc thành (pos_start, pos_end)
    theo vị trí trong `sequence`. KHÔNG raise (Bug #2): clamp vào dải frame
    thực có, log warning; trả None nếu rỗng sau clamp."""
    n = len(sequence)
    if segment is None:
        return 0, n - 1

    start, end = int(segment[0]), int(segment[1])
    frame_ids = np.array([pf.frame_idx for pf in sequence], dtype=np.int64)
    lo, hi = int(frame_ids.min()), int(frame_ids.max())

    c_start, c_end = max(start, lo), min(end, hi)
    if (c_start, c_end) != (start, end):
        logger.warning(
            "segment=(%d,%d) lệch dải frame video [%d,%d] -> clamp thành (%d,%d). "
            "Kiểm tra lại file phases.", start, end, lo, hi, c_start, c_end,
        )
    if c_start > c_end:
        logger.warning("segment=(%d,%d) nằm hoàn toàn ngoài video [%d,%d] -> bỏ qua.",
                       start, end, lo, hi)
        return None

    positions = np.where((frame_ids >= c_start) & (frame_ids <= c_end))[0]
    if positions.size == 0:
        logger.warning("segment=(%d,%d) không chứa frame nào của video.", c_start, c_end)
        return None
    return int(positions[0]), int(positions[-1])


# =====================================================================
# Tín hiệu 1D dùng chung (tính 1 lần cho mỗi lần detect_events)
# =====================================================================

@dataclass
class _Signals:
    com_x: np.ndarray
    hip_back: np.ndarray      # góc hông chân sau
    thigh_back: np.ndarray    # góc đùi chân sau so với phương thẳng đứng
    hip_thigh: np.ndarray     # dùng cho fallback rear_foot_off
    trunk: np.ndarray         # góc thân so với phương thẳng đứng
    knee_back: np.ndarray     # góc gối chân sau


def _build_signals(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame], front_side: str
) -> _Signals:
    """Ưu tiên key do Normalizer sinh; nếu thiếu (NaN) thì tự tính từ keypoint
    của chân sau để launch/rear_foot_off không bị 'mù' (Bug #8)."""
    back = "left" if front_side == "right" else "right"
    n = len(sequence)

    hip_pose = np.full(n, np.nan, dtype=np.float64)
    thigh_pose = np.full(n, np.nan, dtype=np.float64)
    for i, pf in enumerate(sequence):
        try:
            sh = np.asarray(pf.get(f"{back}_shoulder"), dtype=np.float64)
            hp = np.asarray(pf.get(f"{back}_hip"), dtype=np.float64)
            kn = np.asarray(pf.get(f"{back}_knee"), dtype=np.float64)
            hip_pose[i] = angle_at_vertex(sh, hp, kn)
            thigh_pose[i] = angle_to_vertical(kn - hp)
        except Exception:  # noqa: BLE001
            continue

    def series(keys: Sequence[str], fallback: Optional[np.ndarray] = None) -> np.ndarray:
        arr = np.array([_ang(nf, keys) for nf in normalized], dtype=np.float64)
        if fallback is not None:
            mask = np.isnan(arr)
            arr[mask] = fallback[mask]
        return arr

    hip_back = series(("hip_back",), hip_pose)
    thigh_back = series(("thigh_back",), thigh_pose)
    hip_thigh = series(("hip_thigh_angle",), hip_back)
    trunk = series(TRUNK_KEYS)
    knee_back = series(("knee_back",))
    com_x = compute_center_of_mass(sequence)[:, 0]
    return _Signals(com_x, hip_back, thigh_back, hip_thigh, trunk, knee_back)


# =====================================================================
# Các sự kiện
# =====================================================================

def _detect_hand_off(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame],
    th: DiscreteEventThresholds,
) -> Optional[int]:
    """hand_off = frame ĐẦU của chuỗi >= hand_off_debounce_frames frame liên tiếp
    thỏa wrist_y < hip_y - margin (Bug #10). Frame có confidence thấp ngắt chuỗi."""
    run_start: Optional[int] = None
    run_len = 0
    for pf, nf in zip(sequence, normalized):
        ok = False
        try:
            wrist_conf = pf.get_group_conf("wrist")
            hip_conf = pf.get_group_conf("hip")
            if not (np.isnan(wrist_conf) or wrist_conf < th.hand_off_conf_min
                    or np.isnan(hip_conf) or hip_conf < th.hand_off_conf_min):
                wrist_y = float(np.mean([pf.get("left_wrist")[1], pf.get("right_wrist")[1]]))
                hip_y = float(np.mean([pf.get("left_hip")[1], pf.get("right_hip")[1]]))
                margin = th.hand_off_margin_ratio * nf.body_scale
                if np.isnan(margin):
                    margin = 0.0
                ok = wrist_y < hip_y - margin
        except Exception:  # noqa: BLE001
            ok = False

        if ok:
            if run_len == 0:
                run_start = pf.frame_idx
            run_len += 1
            if run_len >= th.hand_off_debounce_frames:
                return run_start
        else:
            run_len, run_start = 0, None
    return None


def find_hand_off_frame(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame],
    thresholds: Optional[DiscreteEventThresholds] = None,
) -> Optional[int]:
    """API công khai: dùng CHUNG một định nghĩa hand_off cho cả segmenter và
    feature_extractor (Bug #4)."""
    return _detect_hand_off(sequence, normalized, thresholds or DiscreteEventThresholds())


def _detect_onset_of_motion(
    sequence: List[PoseFrame], signals: _Signals, hand_off_pos: int,
    th: DiscreteEventThresholds,
) -> Optional[int]:
    """onset = frame đầu tiên sau hand_off khi COM_x dịch chuyển mạnh so với
    baseline trước hand_off. Không đủ baseline -> None (Bug #12)."""
    if hand_off_pos < th.onset_min_baseline_frames:
        return None

    com_x = signals.com_x
    baseline_std = float(np.nanstd(com_x[:hand_off_pos]))
    if np.isnan(baseline_std) or baseline_std < 1.0:
        baseline_std = 1.0

    threshold = max(th.onset_std_factor * baseline_std, th.onset_abs_threshold_px)
    base_x = com_x[hand_off_pos]
    max_pos = min(len(sequence) - 1, hand_off_pos + th.onset_window_after_hand_off)
    for i in range(hand_off_pos + 1, max_pos + 1):
        if abs(com_x[i] - base_x) > threshold:
            return sequence[i].frame_idx
    return None


def _foot_off_by_ankle(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame],
    hand_off_pos: int, side: str, th: DiscreteEventThresholds,
) -> Optional[int]:
    """Chân rời đất = ankle_y giảm (đi lên ảnh) vượt delta so với mức nền
    (trung vị vài frame ngay trước/tại hand_off)."""
    try:
        lo = max(0, hand_off_pos - 3)
        ref_ys = [float(sequence[i].get(f"{side}_ankle")[1]) for i in range(lo, hand_off_pos + 1)]
        block_y = float(np.nanmedian(ref_ys))
        shoulder_px = _shoulder_px(normalized[hand_off_pos])
        if np.isnan(block_y) or np.isnan(shoulder_px):
            return None
        delta = th.foot_off_delta_ratio * shoulder_px
        for pos in range(hand_off_pos, len(sequence)):
            if float(sequence[pos].get(f"{side}_ankle")[1]) < block_y - delta:
                return sequence[pos].frame_idx
    except Exception:  # noqa: BLE001
        return None
    return None


def _detect_rear_foot_off(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame], signals: _Signals,
    hand_off_pos: int, front_side: str, th: DiscreteEventThresholds, flags: FlagsLike,
) -> Tuple[Optional[int], str]:
    """Trả về (frame_idx, method) với method ∈ {ankle, hip_thigh_fallback, not_found}."""
    back_side = "left" if front_side == "right" else "right"

    if not _flag_get(flags, "rear_ankle_low_confidence"):
        found = _foot_off_by_ankle(sequence, normalized, hand_off_pos, back_side, th)
        if found is not None:
            return found, "ankle"

    # --- Fallback: hip_thigh_angle lệch mạnh so với baseline trước hand_off ---
    arr = signals.hip_thigh
    baseline = arr[:hand_off_pos] if hand_off_pos >= 3 else arr[:max(1, hand_off_pos)]
    baseline = baseline[~np.isnan(baseline)]
    if len(baseline) < 3:
        return None, "not_found"

    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    if baseline_std < 1e-6:
        baseline_std = 1.0
    threshold = th.hip_thigh_std_factor * baseline_std
    for i in range(hand_off_pos, len(sequence)):
        v = arr[i]
        if not np.isnan(v) and abs(v - baseline_mean) > threshold:
            return sequence[i].frame_idx, "hip_thigh_fallback"
    return None, "not_found"


def _detect_front_foot_off(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame],
    hand_off_pos: int, front_side: str, th: DiscreteEventThresholds,
) -> Optional[int]:
    return _foot_off_by_ankle(sequence, normalized, hand_off_pos, front_side, th)


# =====================================================================
# LAUNCH DETECTION - 3 TIER
# =====================================================================

def _tier1_multi_signal(
    signals: _Signals, normalized: List[NormalizedFrame],
    launch_window: List[int], hand_off_pos: int, th: DiscreteEventThresholds,
) -> Tuple[Optional[int], float]:
    """Tier 1: chấm điểm 4 tín hiệu (tối đa 6), chấp nhận nếu >= tier1_min_score."""
    window_pos = [i for i in launch_window if 0 <= i < len(normalized)]
    if not window_pos:
        return None, 0.0

    hip_arr = signals.hip_back[window_pos]
    if len(hip_arr) >= 2 and not np.all(np.isnan(hip_arr)):
        hip_omega = np.abs(np.gradient(hip_arr))
    else:
        hip_omega = np.full(len(window_pos), np.nan)
    finite_omega = hip_omega[~np.isnan(hip_omega)]
    max_hip_omega = float(finite_omega.max()) if finite_omega.size else 0.0

    # Bug #14: dùng |Δx| nên không phụ thuộc chiều chạy / hướng camera
    shoulder_px = _shoulder_px(normalized[hand_off_pos])
    com_thr = (th.tier1_com_delta_ratio * shoulder_px
               if not np.isnan(shoulder_px) else th.onset_abs_threshold_px)
    com_ref = signals.com_x[hand_off_pos]

    best_pos: Optional[int] = None
    best_score = -1
    best_abs_offset = float("inf")
    for k, pos in enumerate(window_pos):
        score = 0
        t = signals.trunk[pos]
        if not np.isnan(t) and th.tier1_torso_min <= t <= th.tier1_torso_max:
            score += 2
        rk = signals.knee_back[pos]
        if not np.isnan(rk) and rk >= th.tier1_rear_knee_min:
            score += 2
        if max_hip_omega > 1e-6 and hip_omega[k] > th.tier1_hip_omega_factor * max_hip_omega:
            score += 1
        if abs(signals.com_x[pos] - com_ref) > com_thr:
            score += 1

        abs_offset = abs(pos - hand_off_pos)
        if score > best_score or (score == best_score and abs_offset < best_abs_offset):
            best_score, best_pos, best_abs_offset = score, pos, abs_offset

    if best_score >= th.tier1_min_score:
        return best_pos, float(best_score)
    return None, float(best_score)


def _tier2_hip_thigh_omega(
    signals: _Signals, launch_window: List[int], th: DiscreteEventThresholds,
) -> Tuple[Optional[int], float]:
    """Tier 2: hip_omega * thigh_omega (chuẩn hóa 0-1), chấp nhận nếu > tier2_combined_min."""
    window_pos = [i for i in launch_window if 0 <= i < len(signals.hip_back)]
    if len(window_pos) < 2:
        return None, 0.0

    hip_arr = signals.hip_back[window_pos]
    thigh_arr = signals.thigh_back[window_pos]
    if np.all(np.isnan(hip_arr)) or np.all(np.isnan(thigh_arr)):
        return None, 0.0

    combined = _normalize_to_01(np.abs(np.gradient(hip_arr))) * \
        _normalize_to_01(np.abs(np.gradient(thigh_arr)))
    if np.all(np.isnan(combined)):
        return None, 0.0

    best_idx = int(np.nanargmax(combined))
    best_val = float(combined[best_idx])
    if best_val > th.tier2_combined_min:
        return window_pos[best_idx], best_val
    return None, best_val


def _tier3_first_movement(
    signals: _Signals, normalized: List[NormalizedFrame], sequence_len: int,
    hand_off_pos: int, launch_window: List[int], th: DiscreteEventThresholds,
) -> Optional[int]:
    """Tier 3: frame đầu tiên sau hand_off thỏa 1 trong 3 điều kiện."""
    if hand_off_pos >= len(signals.com_x):
        return None
    base_x = signals.com_x[hand_off_pos]

    shoulder_px = _shoulder_px(normalized[hand_off_pos])
    com_delta = (th.tier3_com_delta_ratio * shoulder_px
                 if not np.isnan(shoulder_px) else th.onset_abs_threshold_px)
    base_torso = signals.trunk[hand_off_pos]

    hip_hist = signals.hip_back[max(0, hand_off_pos - 5):hand_off_pos]
    baseline_hip_omega = 0.0
    if len(hip_hist) >= 2 and not np.all(np.isnan(hip_hist)):
        grad = np.abs(np.gradient(hip_hist))
        if np.any(~np.isnan(grad)):
            baseline_hip_omega = float(np.nanmean(grad))

    for pos in launch_window:
        if pos <= hand_off_pos or pos >= sequence_len:
            continue
        if abs(signals.com_x[pos] - base_x) > com_delta:
            return pos
        torso = signals.trunk[pos]
        if not np.isnan(torso) and not np.isnan(base_torso) \
                and abs(torso - base_torso) > th.tier3_torso_delta:
            return pos
        if pos > 0:
            curr, prev = signals.hip_back[pos], signals.hip_back[pos - 1]
            if not np.isnan(curr) and not np.isnan(prev):
                if abs(curr - prev) > th.tier3_hip_omega_factor * max(baseline_hip_omega, 0.1):
                    return pos
    return None


def _detect_launch(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame], signals: _Signals,
    hand_off_frame: Optional[int], onset_frame: Optional[int], th: DiscreteEventThresholds,
) -> Tuple[Optional[int], str, str]:
    """Launch detection 3-tier. Trả về (frame_idx, method, confidence)."""
    hand_off_pos = _pos_of(sequence, hand_off_frame)
    if hand_off_pos is None:
        return None, "hand_off_not_found", "none"

    onset_pos = _pos_of(sequence, onset_frame)
    if onset_pos is None:
        onset_pos = hand_off_pos

    window_start = max(onset_pos, hand_off_pos)
    window_end = min(
        len(sequence) - 1,
        onset_pos + th.launch_window_after_onset,
        hand_off_pos + th.launch_max_offset_from_hand_off,
    )
    launch_window = list(range(window_start, window_end + 1))
    if len(launch_window) < 2:
        return None, "launch_not_found", "none"

    t1_pos, _ = _tier1_multi_signal(signals, normalized, launch_window, hand_off_pos, th)
    if t1_pos is not None:
        return sequence[t1_pos].frame_idx, "tier1_multi_signal", "high"

    t2_pos, _ = _tier2_hip_thigh_omega(signals, launch_window, th)
    if t2_pos is not None:
        return sequence[t2_pos].frame_idx, "tier2_hip_thigh_omega", "medium"

    t3_pos = _tier3_first_movement(
        signals, normalized, len(sequence), hand_off_pos, launch_window, th
    )
    if t3_pos is not None:
        return sequence[t3_pos].frame_idx, "tier3_first_movement", "low"

    return None, "launch_not_found", "none"


# =====================================================================
# ĐẶC TRƯNG SUY RA TỪ SỰ KIỆN (Bug #1)
# =====================================================================

def _compute_arm_leg_opposition(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame], signals: _Signals,
    front_side: str, launch_pos: int, ref_pos: int, th: DiscreteEventThresholds,
) -> float:
    """Đối kháng tay-chân lúc bật: chân SAU là chân đưa lên trước đầu tiên, nên
    tay ĐỐI DIỆN với nó (= tay cùng phía CHÂN TRƯỚC) phải vung ra phía trước,
    còn tay phía chân sau vung ra sau.

    Cách đo: trên cửa sổ [launch, launch+post], với mỗi bên lấy độ lệch ngang
    (wrist_x - shoulder_x) theo CHIỀU CHẠY. Chiều chạy suy ra từ dịch chuyển COM
    (không giả định trái->phải). Tay "trước" hơn tay kia vượt ngưỡng:
        tay trước thuộc phía chân trước -> 1.0 ; ngược lại -> 0.0 ; không đủ
        chắc chắn / thiếu dữ liệu -> NaN.
    LƯU Ý: đây là THIẾT KẾ MỚI (spec chưa có công thức) - cần đối chiếu với
    video HLV đã gán nhãn trước khi dùng để chấm chính thức.
    """
    back_side = "left" if front_side == "right" else "right"
    end = min(len(sequence) - 1, launch_pos + th.post_event_window_frames)
    if end <= launch_pos:
        return float("nan")

    dx = signals.com_x[end] - signals.com_x[ref_pos]
    if np.isnan(dx) or abs(dx) < 1e-6:
        return float("nan")
    direction = 1.0 if dx > 0 else -1.0

    rel: Dict[str, List[float]] = {front_side: [], back_side: []}
    for pos in range(launch_pos, end + 1):
        pf = sequence[pos]
        for side in (front_side, back_side):
            try:
                if (pf.get_conf(f"{side}_wrist") < th.arm_conf_min
                        or pf.get_conf(f"{side}_shoulder") < th.arm_conf_min):
                    continue
                rel[side].append(
                    (float(pf.get(f"{side}_wrist")[0]) - float(pf.get(f"{side}_shoulder")[0]))
                    * direction
                )
            except Exception:  # noqa: BLE001
                continue

    if not rel[front_side] or not rel[back_side]:
        return float("nan")

    diff = float(np.mean(rel[front_side]) - np.mean(rel[back_side]))
    thr = th.arm_opposition_min_diff_ratio * normalized[launch_pos].body_scale
    if np.isnan(thr) or np.isnan(diff):
        return float("nan")
    if diff > thr:
        return 1.0
    if diff < -thr:
        return 0.0
    return float("nan")


def _compute_derived_features(
    sequence: List[PoseFrame], normalized: List[NormalizedFrame], signals: _Signals,
    front_side: str, hand_off_frame: Optional[int],
    rear_off: Optional[int], front_off: Optional[int],
    launch_pos: int, th: DiscreteEventThresholds,
) -> Dict[str, float]:
    nan = float("nan")
    derived: Dict[str, float] = {}

    # rear_leg_leaves_first: thiếu 1 trong 2 mốc -> NaN (KHÔNG coi là fail)
    if rear_off is None or front_off is None:
        derived["rear_leg_leaves_first"] = nan
    else:
        derived["rear_leg_leaves_first"] = 1.0 if rear_off < front_off else 0.0

    # arm_leg_opposition
    hand_off_pos = _pos_of(sequence, hand_off_frame)
    ref_pos = hand_off_pos if hand_off_pos is not None else max(0, launch_pos - 5)
    derived["arm_leg_opposition"] = _compute_arm_leg_opposition(
        sequence, normalized, signals, front_side, launch_pos, ref_pos, th
    )

    # leg_ground_angle: góc giữa đường hông->cổ chân của chân TRƯỚC (chân đạp)
    # và mặt đất (phương ngang) tại frame launch. [0°, 90°]. ĐỊNH NGHĨA TẠM -
    # đối chiếu lại với spec.
    try:
        pf = sequence[launch_pos]
        vec = np.asarray(pf.get(f"{front_side}_ankle"), dtype=np.float64) - \
            np.asarray(pf.get(f"{front_side}_hip"), dtype=np.float64)
        derived["leg_ground_angle"] = float(
            np.degrees(np.arctan2(abs(vec[1]), abs(vec[0])))
        )
    except Exception:  # noqa: BLE001
        derived["leg_ground_angle"] = nan

    # front_knee_peak_extension: góc gối trước LỚN NHẤT từ launch đến hết cửa sổ sau
    # (chân trước duỗi khi đạp, đỉnh duỗi thường nằm SAU frame launch).
    post_hi = min(len(normalized), launch_pos + th.post_event_window_frames + 1)
    vals = [_ang(nf, ("knee_front",)) for nf in normalized[launch_pos:post_hi]]
    vals = [v for v in vals if not np.isnan(v)]
    derived["front_knee_peak_extension"] = float(max(vals)) if vals else nan
    return derived


# =====================================================================
# SEGMENTER
# =====================================================================

class DiscreteEventSegmenter:
    """Phát hiện sự kiện rời rạc cho sprint_start theo spec v2.1 (bản sửa V2.2)."""

    def __init__(self, thresholds: Optional[DiscreteEventThresholds] = None):
        self.th = thresholds or DiscreteEventThresholds()

    def detect_events(
        self,
        sequence: List[PoseFrame],
        normalized: List[NormalizedFrame],
        fps: float,
        segment: Optional[Tuple[int, int]] = None,
        front_side: str = "right",
        quality_flags: FlagsLike = None,
        hand_off_hint: Optional[int] = None,
    ) -> List[DiscreteEvent]:
        """
        segment: (start_frame, end_frame) theo frame_idx GỐC. Lệch dải video thì
            được clamp (log warning), KHÔNG raise.
        front_side: 'left' | 'right' - chân nào là chân trước.
        quality_flags: QualityFlags hoặc dict; được cập nhật tại chỗ.
        hand_off_hint: frame hand_off do caller đã biết (vd. đầu cửa sổ
            sprint_start); nếu nằm trong đoạn phân tích thì được ưu tiên dùng
            thay cho tự phát hiện.
        """
        if len(sequence) != len(normalized):
            raise ValueError("sequence và normalized phải cùng độ dài")
        if front_side not in ("left", "right"):
            raise ValueError(f"front_side phải là 'left' hoặc 'right', nhận '{front_side}'")
        if len(sequence) < 3:
            return []

        bounds = _resolve_segment(sequence, segment)
        if bounds is None:
            _flag_set(quality_flags, "launch_not_found")
            return []
        pos_start, pos_end = bounds
        sub_sequence = sequence[pos_start: pos_end + 1]
        sub_normalized = normalized[pos_start: pos_end + 1]
        if len(sub_sequence) < 3:
            return []

        th = self.th
        com = compute_center_of_mass(sub_sequence)
        body_scale = np.array([nf.body_scale for nf in sub_normalized], dtype=np.float64)
        _velocity, acceleration = compute_kinematics(com, body_scale, fps)
        signals = _build_signals(sub_sequence, sub_normalized, front_side)

        # --- 5 sự kiện ---
        hand_off: Optional[int] = None
        if (hand_off_hint is not None
                and sub_sequence[0].frame_idx <= hand_off_hint <= sub_sequence[-1].frame_idx):
            hand_off = int(hand_off_hint)
        else:
            hand_off = _detect_hand_off(sub_sequence, sub_normalized, th)
        hand_off_pos = _pos_of(sub_sequence, hand_off)
        if hand_off_pos is None:
            hand_off = None

        onset = None
        rear_foot_off, front_foot_off = None, None
        if hand_off_pos is not None:
            onset = _detect_onset_of_motion(sub_sequence, signals, hand_off_pos, th)
            rear_foot_off, _rear_method = _detect_rear_foot_off(
                sub_sequence, sub_normalized, signals, hand_off_pos,
                front_side, th, quality_flags,
            )
            front_foot_off = _detect_front_foot_off(
                sub_sequence, sub_normalized, hand_off_pos, front_side, th
            )

        launch_frame, launch_method, launch_conf = _detect_launch(
            sub_sequence, sub_normalized, signals, hand_off, onset, th
        )

        # --- Quality flags ---
        if hand_off is None:
            _flag_set(quality_flags, "hand_off_not_found")
        if launch_frame is None:
            _flag_set(quality_flags, "launch_not_found")
        if (rear_foot_off is not None and front_foot_off is not None
                and rear_foot_off >= front_foot_off):
            _flag_set(quality_flags, "leg_order_violation")

        # --- Peak acceleration + fallback ---
        peak_accel = float("nan")
        if launch_frame is not None:
            launch_pos = _pos_of(sub_sequence, launch_frame)
            launch_pos = 0 if launch_pos is None else launch_pos
            if 0 <= launch_pos < len(acceleration):
                peak_accel = float(acceleration[launch_pos])
        else:
            accel_clean = np.where(np.isnan(acceleration), -np.inf, acceleration)
            peak_val = float(np.max(accel_clean))
            # Không có đỉnh gia tốc đủ lớn -> không có sự kiện thật. KHÔNG tạo event giả.
            if not np.isfinite(peak_val) or peak_val < th.min_peak_acceleration:
                _flag_set(quality_flags, "launch_not_found")
                return []
            launch_pos = int(np.argmax(accel_clean))
            launch_frame = sub_sequence[launch_pos].frame_idx
            launch_method = "fallback_peak_acceleration"
            launch_conf = "low"
            peak_accel = peak_val
            _flag_set(quality_flags, "launch_fallback_used")

        pre_lo = max(0, launch_pos - th.pre_event_window_frames)
        post_hi = min(len(sub_sequence), launch_pos + th.post_event_window_frames + 1)

        derived = _compute_derived_features(
            sub_sequence, sub_normalized, signals, front_side, hand_off,
            rear_foot_off, front_foot_off, launch_pos, th,
        )

        event = DiscreteEvent(
            event_index=0,
            segment_start_frame=sub_sequence[0].frame_idx,
            segment_end_frame=sub_sequence[-1].frame_idx,
            launch_frame_idx=launch_frame,
            launch_timestamp_s=sub_sequence[launch_pos].timestamp_s,
            peak_acceleration=peak_accel,
            hand_off_frame=hand_off,
            onset_of_motion_frame=onset,
            rear_foot_off_frame=rear_foot_off,
            front_foot_off_frame=front_foot_off,
            pre_event_features=mean_features(sub_normalized[pre_lo:launch_pos]),
            at_event_features=mean_features(sub_normalized[launch_pos: launch_pos + 1]),
            post_event_features=mean_features(sub_normalized[launch_pos + 1: post_hi]),
            derived_features=derived,
            trunk_angle_series=[_ang(nf, TRUNK_KEYS) for nf in sub_normalized],
            timestamps_s=[pf.timestamp_s for pf in sub_sequence],
            launch_detection_method=launch_method,
            launch_confidence=launch_conf,
            quality_flags=quality_flags,
        )
        return [event]