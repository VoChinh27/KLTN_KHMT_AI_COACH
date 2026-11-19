"""
stage2_pose/signal_smoother.py
===============================
Mục đích : Giảm nhiễu (jitter) giữa các frame và xử lý keypoint bị mất
           (confidence thấp) trước khi đưa vào chuẩn hóa (Giai đoạn 3).
Input    : List[PoseFrame] đã được tracker.py lọc còn 1 người/frame.
Output   : List[PoseFrame] đã làm sạch (xy đã nội suy + làm mượt)
           -> đưa vào normalizer.py (Giai đoạn 3).

Gồm 2 bước độc lập, có thể bật/tắt riêng:
    1. interpolate_missing  : nội suy tuyến tính theo thời gian cho các
       keypoint có confidence < ngưỡng. KHÔNG extrapolate đầu/cuối (SS1).
    2. smooth_savgol        : lọc Savitzky-Golay theo từng khớp, từng trục
       (x, y). Xử lý NaN bằng cách smooth từng đoạn liên tục (SS2).

Ngoài ra cung cấp KalmanPoint2D như một lựa chọn thay thế khi cần lọc theo
kiểu online (streaming), thay vì lọc theo cả cửa sổ như Savitzky-Golay.

V2.2 - Sửa bug smoother (SS1–SS3):
    - SS1: interpolate_missing không còn extrapolate hằng số ở đầu/cuối
      chuỗi. Frame trước điểm valid đầu tiên / sau điểm valid cuối cùng
      được giữ NaN.
    - SS2: smooth_savgol smooth từng đoạn không-NaN riêng biệt thay vì
      bỏ qua toàn bộ keypoint khi có bất kỳ NaN nào.
    - SS3: SignalSmoother.process log cảnh báo khi frame_idx có lỗ hổng
      (tracker drop frame) — hỗ trợ debug crash HLV08 / PL1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from core.pose_schema import CONF_THRESHOLD_DEFAULT, PoseFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Nội suy điểm mất / confidence thấp (SS1: không extrapolate)
# ---------------------------------------------------------------------------
def interpolate_missing(
    xy_seq: np.ndarray,
    conf_seq: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD_DEFAULT,
    max_gap: int = 10,
) -> np.ndarray:
    """
    Nội suy tuyến tính theo trục thời gian cho các keypoint có
    conf < conf_threshold.

    Tham số:
        xy_seq   : (num_frames, num_joints, 2)
        conf_seq : (num_frames, num_joints)
        max_gap  : số frame liên tiếp mất tối đa được phép nội suy; nếu một
                   khớp mất liên tục quá max_gap frame, đoạn đó được GIỮ
                   NGUYÊN là NaN để anomaly_detector.py (Giai đoạn 3) có thể
                   phát hiện và loại bỏ, thay vì bịa dữ liệu không đáng tin.

    V2.2 (SS1):
        - CHỈ nội suy *giữa* các điểm valid (không extrapolate đầu/cuối).
        - Frame 0 .. idx_valid[0]-1 và idx_valid[-1]+1 .. N-1 giữ NaN.
        - Tránh bịa pose khi VĐV chưa vào khung (đầu clip) hoặc đã rời
          khung (cuối clip).

    Trả về mảng xy đã nội suy, cùng shape với xy_seq.
    """
    num_frames, num_joints, _ = xy_seq.shape
    out = xy_seq.copy().astype(np.float64)
    valid = conf_seq >= conf_threshold

    for j in range(num_joints):
        valid_j = valid[:, j]
        if valid_j.all():
            continue
        if not valid_j.any():
            out[:, j, :] = np.nan
            continue

        idx_valid = np.where(valid_j)[0]
        first_valid = int(idx_valid[0])
        last_valid = int(idx_valid[-1])

        # --- Chỉ nội suy trong đoạn [first_valid, last_valid] ---
        idx_interp = np.arange(first_valid, last_valid + 1)
        for axis in range(2):
            series = xy_seq[:, j, axis].astype(np.float64)
            # np.interp chỉ nhận xp tăng dần; idx_valid đã sorted
            interpolated = np.interp(
                idx_interp,
                idx_valid.astype(np.float64),
                series[idx_valid],
            )
            out[first_valid : last_valid + 1, j, axis] = interpolated

        # Đầu chuỗi (trước first_valid) → NaN (không extrapolate)
        if first_valid > 0:
            out[:first_valid, j, :] = np.nan

        # Cuối chuỗi (sau last_valid) → NaN
        if last_valid < num_frames - 1:
            out[last_valid + 1 :, j, :] = np.nan

        # Đánh dấu lại NaN cho các đoạn gap quá dài *bên trong* vùng valid
        gap_start: Optional[int] = None
        for i in range(first_valid, last_valid + 1):
            if not valid_j[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_len = i - gap_start
                    if gap_len > max_gap:
                        out[gap_start:i, j, :] = np.nan
                    gap_start = None
        # Gap kéo dài tới last_valid (nếu last_valid vẫn invalid — không xảy ra
        # vì last_valid là valid, nhưng giữ logic an toàn)
        if gap_start is not None:
            gap_len = (last_valid + 1) - gap_start
            if gap_len > max_gap:
                out[gap_start : last_valid + 1, j, :] = np.nan

    return out


# ---------------------------------------------------------------------------
# 2. Làm mượt Savitzky-Golay (SS2: smooth từng đoạn không-NaN)
# ---------------------------------------------------------------------------
def _smooth_segment(
    series: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """
    Smooth 1 đoạn 1-D không chứa NaN. Trả về bản copy đã lọc.
    Nếu đoạn quá ngắn → trả nguyên bản.
    """
    n = len(series)
    wl = window_length
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl % 2 == 0:
        wl -= 1
    if wl <= polyorder or wl < 3:
        return series.copy()
    return savgol_filter(series, window_length=wl, polyorder=polyorder)


def _contiguous_finite_segments(
    series: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Trả danh sách (start, end) inclusive của các đoạn liên tục không-NaN.
    """
    finite = np.isfinite(series)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, ok in enumerate(finite):
        if ok:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(series) - 1))
    return segments


def smooth_savgol(
    xy_seq: np.ndarray,
    window_length: int = 9,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Áp dụng bộ lọc Savitzky-Golay theo từng khớp, từng trục (x, y).

    window_length phải là số lẻ và > polyorder; nếu chuỗi ngắn hơn
    window_length, hàm tự giảm window_length về giá trị lẻ lớn nhất phù hợp
    (tránh crash khi video/đoạn phân đoạn quá ngắn).

    V2.2 (SS2):
        - Không còn bỏ qua toàn bộ keypoint khi có bất kỳ NaN nào.
        - Chia series thành các đoạn liên tục không-NaN, smooth từng đoạn
          (nếu đủ dài). Đoạn NaN giữ nguyên NaN.
        → Keypoint miss một đoạn giữa vẫn được smooth ở hai phía.
    """
    num_frames = xy_seq.shape[0]
    if num_frames == 0:
        return xy_seq.copy()

    out = xy_seq.copy().astype(np.float64)
    num_joints = xy_seq.shape[1]

    for j in range(num_joints):
        for axis in range(2):
            series = xy_seq[:, j, axis].astype(np.float64)
            if not np.isnan(series).any():
                # Không có NaN → smooth cả chuỗi một lần (nhanh hơn)
                out[:, j, axis] = _smooth_segment(
                    series, window_length, polyorder
                )
                continue

            # Có NaN → smooth từng đoạn liên tục
            for seg_start, seg_end in _contiguous_finite_segments(series):
                seg = series[seg_start : seg_end + 1]
                out[seg_start : seg_end + 1, j, axis] = _smooth_segment(
                    seg, window_length, polyorder
                )
            # Vị trí NaN gốc giữ nguyên (đã là NaN trong out copy từ xy_seq
            # nếu input có NaN; đảm bảo rõ ràng)
            nan_mask = np.isnan(series)
            out[nan_mask, j, axis] = np.nan

    return out


# ---------------------------------------------------------------------------
# 3. Kalman filter 2D đơn giản (constant-velocity) - lựa chọn thay thế online
# ---------------------------------------------------------------------------
@dataclass
class KalmanPoint2D:
    """Kalman filter constant-velocity cho 1 điểm (x, y), dùng khi cần lọc
    theo kiểu streaming (xử lý từng frame khi video đang quay), khác với
    Savitzky-Golay vốn cần cả cửa sổ dữ liệu.
    """

    process_noise: float = 1e-2
    measurement_noise: float = 1e-1
    _x: Optional[np.ndarray] = None  # state: [x, y, vx, vy]
    _P: Optional[np.ndarray] = None  # covariance

    def _init_state(self, measurement: np.ndarray) -> None:
        self._x = np.array(
            [measurement[0], measurement[1], 0.0, 0.0], dtype=np.float64
        )
        self._P = np.eye(4) * 10.0

    def update(self, measurement: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Nhận 1 phép đo (x, y) mới, trả về vị trí đã lọc (x, y)."""
        if self._x is None:
            self._init_state(measurement)
            return measurement

        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        Q = np.eye(4) * self.process_noise
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * self.measurement_noise

        # predict
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # update
        y = measurement - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ y
        self._P = (np.eye(4) - K @ H) @ P_pred

        return self._x[:2].astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Class tổng hợp: pipeline hoàn chỉnh cho 1 chuỗi PoseFrame
# ---------------------------------------------------------------------------
def _report_frame_idx_gaps(sequence: List[PoseFrame]) -> int:
    """
    SS3: Phát hiện lỗ hổng frame_idx (tracker drop frame).
    Trả về số frame bị thiếu trong khoảng [min, max].
    Log cảnh báo nếu có gap.
    """
    if len(sequence) < 2:
        return 0
    indices = [pf.frame_idx for pf in sequence]
    lo, hi = min(indices), max(indices)
    expected = hi - lo + 1
    actual = len(set(indices))
    n_missing = expected - actual
    if n_missing > 0:
        # Liệt kê một vài gap để debug (không dump hết nếu nhiều)
        present = set(indices)
        gaps: List[int] = []
        for i in range(lo, hi + 1):
            if i not in present:
                gaps.append(i)
                if len(gaps) >= 8:
                    break
        extra = f" … (+{n_missing - len(gaps)} nữa)" if n_missing > len(gaps) else ""
        logger.warning(
            "[smoother] Phát hiện %d frame_idx bị drop trong [%d, %d] "
            "(tracker có thể đã loại frame không detect được người). "
            "Ví dụ gap: %s%s. "
            "Manual phases trỏ vào frame bị drop sẽ bị từ chối bởi "
            "pipeline._validate_windows_in_range.",
            n_missing, lo, hi, gaps, extra,
        )
    # Kiểm tra trùng frame_idx
    if actual < len(indices):
        logger.warning(
            "[smoother] Có frame_idx trùng lặp (%d entry, %d unique).",
            len(indices), actual,
        )
    return n_missing


class SignalSmoother:
    """Kết hợp nội suy + làm mượt thành 1 bước xử lý cho toàn bộ video.

    process(sequence) -> sequence mới (không sửa in-place) với xy đã sạch.
    conf được giữ nguyên (dùng làm tham khảo ở anomaly_detector.py).
    frame_idx được GIỮ NGUYÊN (không renumber) — quan trọng để khớp
    manual phases file.

    V2.2:
        - Không extrapolate đầu/cuối (SS1).
        - Smooth từng đoạn khi có NaN (SS2).
        - Log gap frame_idx (SS3).
    """

    def __init__(
        self,
        conf_threshold: float = CONF_THRESHOLD_DEFAULT,
        max_gap: int = 10,
        window_length: int = 9,
        polyorder: int = 2,
        use_savgol: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.max_gap = max_gap
        self.window_length = window_length
        self.polyorder = polyorder
        self.use_savgol = use_savgol

    def process(self, sequence: List[PoseFrame]) -> List[PoseFrame]:
        if not sequence:
            return []

        # SS3: báo cáo lỗ hổng frame_idx (không sửa, chỉ log)
        _report_frame_idx_gaps(sequence)

        joint_names = sequence[0].joint_names
        xy_seq = np.stack([pf.xy for pf in sequence], axis=0)
        conf_seq = np.stack([pf.conf for pf in sequence], axis=0)

        xy_clean = interpolate_missing(
            xy_seq,
            conf_seq,
            conf_threshold=self.conf_threshold,
            max_gap=self.max_gap,
        )
        fps = 30.0
        if len(sequence) >= 2:
            dt = sequence[1].timestamp_s - sequence[0].timestamp_s
            if dt > 1e-6:
                fps = 1.0 / dt
                
        if self.use_savgol:
            # Adaptive window: ~0.15s cho mọi FPS
            adaptive_wl = max(5, int(round(0.15 * fps)))
            if adaptive_wl % 2 == 0:
                adaptive_wl += 1
            xy_clean = smooth_savgol(xy_clean, window_length=adaptive_wl, polyorder=self.polyorder)

        out: List[PoseFrame] = []
        for i, pf in enumerate(sequence):
            out.append(
                PoseFrame(
                    frame_idx=pf.frame_idx,       # giữ nguyên — không renumber
                    timestamp_s=pf.timestamp_s,
                    xy=xy_clean[i].astype(np.float32)
                    if xy_clean.dtype != np.float32
                    else xy_clean[i],
                    conf=pf.conf,
                    joint_names=joint_names,
                    bbox=pf.bbox,
                    track_id=pf.track_id,
                )
            )
        return out