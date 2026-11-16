"""
stage3_features/temporal_utils.py
===================================
Mục đích : Cung cấp hạ tầng dùng chung cho MỌI temporal feature (trend, slope,
           step detection, cadence) — theo Spec v2.1 mục 7.6 + 7.7.

Vấn đề cần giải quyết:
    - Video < 60fps (đặc biệt 30fps) -> tín hiệu pose bị "thô", slope rất nhiễu.
    - Cần làm mịn tín hiệu (Savitzky-Golay / Moving Average) TRƯỚC khi tính slope.
    - Slope phải so với baseline std để tránh kết luận "tăng/giảm" trên nhiễu.
    - Phát hiện bước chạy (step) và tiếp đất (contact) từ ankle_y signal.

Nguyên tắc thiết kế:
    - Pure functions, không state, không phụ thuộc module khác.
    - Mọi slope function đều trả về kèm cờ "significant" (dựa vào baseline std).
    - Filter tự chọn theo FPS (spec 7.7). Có override thủ công nếu cần.

KHÔNG dùng module này để chấm điểm — chỉ cung cấp tín hiệu thô. Scoring ở
stage4_scoring/scorer.py sẽ quyết định pass/fail dựa trên tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =====================================================================
# 1. NaN handling
# =====================================================================

def fill_nan_gaps(
    signal: np.ndarray,
    max_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nội suy tuyến tính các đoạn NaN ngắn (<= max_gap frame).

    Args:
        signal: mảng 1D, có thể chứa NaN.
        max_gap: gap tối đa được nội suy. Gap dài hơn -> giữ nguyên NaN.

    Returns:
        (filled_signal, was_filled): mảng đã nội suy + mask bool báo frame
        nào đã được nội suy.
    """
    arr = np.asarray(signal, dtype=np.float64).copy()
    n = len(arr)
    was_filled = np.zeros(n, dtype=bool)

    if n == 0 or not np.any(np.isnan(arr)):
        return arr, was_filled

    # Tìm các đoạn NaN liên tiếp
    is_nan = np.isnan(arr)
    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        # Bắt đầu 1 gap
        gap_start = i
        while i < n and is_nan[i]:
            i += 1
        gap_end = i - 1
        gap_len = gap_end - gap_start + 1

        # Nếu gap <= max_gap -> nội suy
        if gap_len <= max_gap and gap_start > 0 and gap_end < n - 1:
            left_val = arr[gap_start - 1]
            right_val = arr[gap_end + 1]
            if not np.isnan(left_val) and not np.isnan(right_val):
                for k in range(gap_start, gap_end + 1):
                    t = (k - gap_start + 1) / (gap_len + 1)
                    arr[k] = left_val * (1 - t) + right_val * t
                    was_filled[k] = True

    return arr, was_filled


# =====================================================================
# 2. Smoothing filters
# =====================================================================

def smooth_signal(
    signal: np.ndarray,
    fps: float,
    method: str = "savgol",
    polyorder: int = 2,
) -> np.ndarray:
    """
    Làm mịn tín hiệu theo spec v2.1 mục 7.7.

    Args:
        signal: mảng 1D (đã fill NaN nếu cần).
        fps: FPS video. Quyết định window size.
        method: 'savgol' (khuyến nghị) hoặc 'moving_average'.
        polyorder: bậc đa thức cho Savitzky-Golay.

    Returns:
        Mảng đã làm mịn, cùng shape. NaN giữ nguyên vị trí (không được nội suy).

    Window size (spec 7.7):
        FPS <  60 -> window = 5
        FPS < 120 -> window = 7
        FPS >= 120 -> window = 9
    """
    arr = np.asarray(signal, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return arr.copy()

    # Chọn window theo FPS (spec 7.7)
    if fps < 60:
        window = 5
    elif fps < 120:
        window = 7
    else:
        window = 9

    # Đảm bảo window lẻ và <= n
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 3:
        return arr.copy()

    # Xử lý NaN: chỉ filter các đoạn không NaN
    if not np.any(np.isnan(arr)):
        if method == "savgol" and _HAS_SCIPY:
            try:
                return savgol_filter(arr, window_length=window, polyorder=polyorder)
            except Exception:
                # Fallback: moving average nếu savgol fail
                pass
        # Moving average
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    # Có NaN: filter từng đoạn liên tiếp không NaN
    result = arr.copy()
    is_nan = np.isnan(arr)
    i = 0
    while i < n:
        if is_nan[i]:
            i += 1
            continue
        seg_start = i
        while i < n and not is_nan[i]:
            i += 1
        seg_end = i - 1
        seg = arr[seg_start : seg_end + 1]
        if len(seg) >= 3:
            w = min(window, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            if w >= 3:
                if method == "savgol" and _HAS_SCIPY:
                    try:
                        result[seg_start : seg_end + 1] = savgol_filter(
                            seg, window_length=w, polyorder=min(polyorder, w - 1)
                        )
                        continue
                    except Exception:
                        pass
                kernel = np.ones(w) / w
                result[seg_start : seg_end + 1] = np.convolve(
                    seg, kernel, mode="same"
                )
    return result


# =====================================================================
# 3. Slope + significance
# =====================================================================

@dataclass
class SlopeResult:
    """Kết quả tính slope kèm thông tin ý nghĩa thống kê."""
    slope: float                # độ dốc (đơn vị: /frame)
    intercept: float
    r_squared: float            # độ khớp tuyến tính [0, 1]
    baseline_std: float         # std của tín hiệu gốc
    significant: bool           # slope có vượt ngưỡng nhiễu không
    n_samples: int


def compute_slope(
    signal: np.ndarray,
    noise_factor: float = 0.5,
) -> SlopeResult:
    """
    Tính slope bằng least-squares + đánh giá ý nghĩa thống kê.

    Args:
        signal: mảng 1D (đã làm mịn).
        noise_factor: hệ số ngưỡng nhiễu. Slope phải > noise_factor * std mới
                      được coi là "significant". Mặc định 0.5 -> slope phải
                      lớn hơn nửa std để có ý nghĩa.

    Returns:
        SlopeResult.
    """
    arr = np.asarray(signal, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n < 3:
        return SlopeResult(
            slope=float("nan"), intercept=float("nan"), r_squared=float("nan"),
            baseline_std=float("nan"), significant=False, n_samples=n,
        )

    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, arr, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((arr - y_pred) ** 2))
    ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    baseline_std = float(np.std(arr))

    # Slope significant nếu |slope| * n > noise_factor * baseline_std
    # (tổng thay đổi qua n frame lớn hơn ngưỡng nhiễu)
    threshold = noise_factor * baseline_std
    total_change = abs(slope) * n
    significant = total_change > threshold

    return SlopeResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        baseline_std=baseline_std,
        significant=significant,
        n_samples=n,
    )


def compute_trend_percent(
    values: Sequence[float],
    noise_factor: float = 0.5,
) -> Tuple[float, bool]:
    """
    Tính % thay đổi tổng của chuỗi (dùng cho cadence_trend, step_length_trend).
    Vd: [1.0, 1.1, 1.2] -> trend = (1.2-1.0)/1.0 = +20%.

    Returns:
        (trend_percent, is_significant): trend tính theo (last - first) / first,
        significant nếu trend vượt ngưỡng nhiễu.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), False

    first, last = arr[0], arr[-1]
    if abs(first) < 1e-9:
        return float("nan"), False

    trend = (last - first) / abs(first)
    baseline_std = float(np.std(arr))
    mean_val = float(np.mean(arr)) if len(arr) else 0.0
    threshold = noise_factor * baseline_std / max(abs(mean_val), 1e-9)
    significant = abs(trend) > threshold

    return float(trend), significant


# =====================================================================
# 4. Step detection (từ ankle_y signal)
# =====================================================================

@dataclass
class StepEvent:
    """1 bước chạy."""
    step_index: int
    contact_frame: int          # frame tiếp đất (đỉnh dưới của ankle_y)
    liftoff_frame: int          # frame rời đất (đỉnh trên của ankle_y)
    duration_frames: int        # số frame của bước
    contact_side: str           # 'left' | 'right' | 'unknown'
    body_x_at_contact: float = float("nan")   # tọa độ X của body center lúc tiếp đất


def detect_steps(
    ankle_y_left: np.ndarray,
    ankle_y_right: np.ndarray,
    fps: float,
    min_step_frames: int = 5,
) -> List[StepEvent]:
    """
    Phát hiện các bước chạy từ 2 tín hiệu ankle_y (trái + phải).

    Nguyên tắc:
        - Khi chân tiếp đất, ankle_y đạt đỉnh cục bộ (y ảnh hướng xuống).
        - Khi chân rời đất, ankle_y đạt đáy cục bộ.
        - Mỗi lần 1 chân tiếp đất = 1 step.

    Args:
        ankle_y_left, ankle_y_right: mảng y-coord của 2 cổ chân.
        fps: FPS video.
        min_step_frames: số frame tối thiểu cho 1 bước (lọc nhiễu).

    Returns:
        Danh sách StepEvent, sắp theo thời gian.
    """
    left = np.asarray(ankle_y_left, dtype=np.float64)
    right = np.asarray(ankle_y_right, dtype=np.float64)
    n = min(len(left), len(right))
    if n < min_step_frames * 2:
        return []

    # Gộp: mỗi frame chọn chân nào tiếp đất hơn (y lớn hơn = thấp hơn)
    # -> tín hiệu "ai đang chạm đất" theo thời gian
    both = np.stack([left[:n], right[:n]], axis=0)
    # Frame nào chân trái thấp hơn -> left đang tiếp đất
    # Frame nào chân phải thấp hơn -> right đang tiếp đất
    sides = np.where(left[:n] > right[:n], 0, 1)  # 0=left, 1=right

    # Phát hiện chuyển tiếp side (bước chân đổi bên)
    steps: List[StepEvent] = []
    current_side = sides[0]
    step_start = 0

    for i in range(1, n):
        if sides[i] != current_side:
            duration = i - step_start
            if duration >= min_step_frames:
                # Contact = frame có ankle_y lớn nhất (chân chạm đất sâu nhất)
                # trong đoạn [step_start, i-1]
                if current_side == 0:
                    seg = left[step_start:i]
                    contact_offset = int(np.nanargmax(seg)) if not np.all(np.isnan(seg)) else 0
                else:
                    seg = right[step_start:i]
                    contact_offset = int(np.nanargmax(seg)) if not np.all(np.isnan(seg)) else 0
                contact_frame = step_start + contact_offset
                liftoff_frame = step_start

                steps.append(StepEvent(
                    step_index=len(steps),
                    contact_frame=contact_frame,
                    liftoff_frame=liftoff_frame,
                    duration_frames=duration,
                    contact_side="left" if current_side == 0 else "right",
                ))
            current_side = sides[i]
            step_start = i

    return steps


# =====================================================================
# 5. Cadence
# =====================================================================

def compute_cadence(steps: Sequence[StepEvent], fps: float) -> float:
    """
    Tính cadence (steps per minute) từ danh sách bước.

    cadence_spm = 60 * n_steps / duration_s
    """
    if len(steps) < 2 or fps <= 0:
        return float("nan")

    first_frame = steps[0].contact_frame
    last_frame = steps[-1].contact_frame
    duration_s = (last_frame - first_frame) / fps
    if duration_s <= 1e-6:
        return float("nan")

    # Số bước = số contact - 1 (khoảng thời gian giữa các contact)
    n_intervals = len(steps) - 1
    # Nhân 2 vì cadence tính theo từng chân chạm đất
    # (chạy: mỗi chu kỳ 2 bước -> 2 * (n_intervals) bước/phút)
    return float(60.0 * n_intervals / duration_s)


# =====================================================================
# 6. Step length + contact position
# =====================================================================

def compute_step_lengths(
    steps: Sequence[StepEvent],
    body_x_series: np.ndarray,
    shoulder_width_px: float,
) -> List[float]:
    """
    Tính độ dài bước (normalized theo shoulder_width).

    step_length_norm = |body_x[contact_n] - body_x[contact_n-1]| / shoulder_width

    Args:
        steps: danh sách StepEvent (đã sort theo contact_frame).
        body_x_series: mảng x-coord của body center qua frame.
        shoulder_width_px: pixel (để normalize).

    Returns:
        List step lengths (normalized). Rỗng nếu < 2 steps.
    """
    if len(steps) < 2 or shoulder_width_px < 1e-6:
        return []

    body_x = np.asarray(body_x_series, dtype=np.float64)
    lengths: List[float] = []

    for i in range(1, len(steps)):
        f_prev = steps[i - 1].contact_frame
        f_curr = steps[i].contact_frame
        if f_prev < 0 or f_curr < 0 or f_prev >= len(body_x) or f_curr >= len(body_x):
            continue
        dx = body_x[f_curr] - body_x[f_prev]
        lengths.append(abs(float(dx)) / shoulder_width_px)

    return lengths


def compute_contact_positions(
    steps: Sequence[StepEvent],
    body_x_series: np.ndarray,
    shoulder_width_px: float,
) -> List[float]:
    """
    Tính độ lệch ngang của vị trí tiếp đất so với body center, normalized.

    lateral_offset_norm[i] = |x_contact[i] - body_center_x[i]| / shoulder_width

    Contact position lấy x-coord của ankle tại contact_frame (chưa có trong
    StepEvent -> cần caller truyền thêm ankle_x_series).

    NOTE: hiện hàm chỉ trả về offset so với body center. Muốn dùng chính xác
    cần thêm ankle_x_series. Tạm dùng body_x làm proxy nếu caller chưa có.
    """
    if not steps or shoulder_width_px < 1e-6:
        return []

    body_x = np.asarray(body_x_series, dtype=np.float64)
    offsets: List[float] = []

    for s in steps:
        f = s.contact_frame
        if f < 0 or f >= len(body_x):
            continue
        # Tạm dùng body_x làm proxy. Caller có thể override bằng cách
        # truyền ankle_x riêng (mở rộng signature sau nếu cần).
        offset = abs(body_x[f] - s.body_x_at_contact) if not np.isnan(s.body_x_at_contact) else 0.0
        offsets.append(float(offset) / shoulder_width_px)

    return offsets


# =====================================================================
# 7. High-level helpers cho drive_phase
# =====================================================================

@dataclass
class DrivePhaseTrendResult:
    """Kết quả tính 4 trend features cho drive_phase."""
    step_length_trend: float = float("nan")
    step_length_significant: bool = False
    cadence_trend: float = float("nan")
    cadence_significant: bool = False
    contact_convergence: float = float("nan")
    contact_significant: bool = False
    trunk_trend: float = float("nan")
    trunk_significant: bool = False
    n_steps: int = 0
    used_savgol: bool = False


def compute_drive_phase_trends(
    ankle_y_left: np.ndarray,
    ankle_y_right: np.ndarray,
    body_x_series: np.ndarray,
    trunk_angle_series: np.ndarray,
    shoulder_width_px: float,
    fps: float,
    max_steps: int = 8,
) -> DrivePhaseTrendResult:
    """
    Tính tất cả trend features cho drive_phase.

    Pipeline (theo spec 7.7):
        1. fill NaN (gap <= 5 frame)
        2. smooth mỗi tín hiệu
        3. detect steps
        4. tính các feature + slope trên tín hiệu ĐÃ MỊN

    Args:
        ankle_y_left, ankle_y_right: y-coord cổ chân qua các frame drive_phase.
        body_x_series: x-coord body center qua các frame.
        trunk_angle_series: góc thân vertical qua các frame.
        shoulder_width_px: để normalize step length + contact offset.
        fps: FPS video.
        max_steps: giới hạn số bước đầu dùng tính trend (spec: 5-8 bước).

    Returns:
        DrivePhaseTrendResult.
    """
    result = DrivePhaseTrendResult()
    result.used_savgol = _HAS_SCIPY and fps < 60

    # --- B1: fill NaN ---
    ankle_l_filled, _ = fill_nan_gaps(ankle_y_left, max_gap=5)
    ankle_r_filled, _ = fill_nan_gaps(ankle_y_right, max_gap=5)
    body_x_filled, _ = fill_nan_gaps(body_x_series, max_gap=5)
    trunk_filled, _ = fill_nan_gaps(trunk_angle_series, max_gap=5)

    # --- B2: smooth ---
    ankle_l_smooth = smooth_signal(ankle_l_filled, fps)
    ankle_r_smooth = smooth_signal(ankle_r_filled, fps)
    body_x_smooth = smooth_signal(body_x_filled, fps)
    trunk_smooth = smooth_signal(trunk_filled, fps)

    # --- B3: detect steps ---
    steps = detect_steps(ankle_l_smooth, ankle_r_smooth, fps)
    if not steps:
        return result

    # Giới hạn số bước đầu (spec: 5-8 bước)
    steps_for_trend = steps[:max_steps]
    result.n_steps = len(steps_for_trend)
    if result.n_steps < 2:
        return result

    # --- B4: step length trend ---
    step_lengths = compute_step_lengths(steps_for_trend, body_x_smooth, shoulder_width_px)
    if len(step_lengths) >= 2:
        trend, sig = compute_trend_percent(step_lengths, noise_factor=0.5)
        result.step_length_trend = trend
        result.step_length_significant = sig

    # --- B5: cadence trend ---
    # Chia chuỗi thành 2 nửa, cadence từng nửa, trend = (cadence_2 - cadence_1) / cadence_1
    if len(steps_for_trend) >= 4:
        half = len(steps_for_trend) // 2
        c1 = compute_cadence(steps_for_trend[:half + 1], fps)
        c2 = compute_cadence(steps_for_trend[half:], fps)
        if not np.isnan(c1) and c1 > 1e-6 and not np.isnan(c2):
            trend = (c2 - c1) / c1
            result.cadence_trend = float(trend)
            result.cadence_significant = abs(trend) > 0.03

    # --- B6: contact position convergence ---
    contact_offsets = compute_contact_positions(
        steps_for_trend, body_x_smooth, shoulder_width_px
    )
    if len(contact_offsets) >= 2:
        slope_result = compute_slope(np.array(contact_offsets), noise_factor=0.5)
        result.contact_convergence = slope_result.slope
        result.contact_significant = slope_result.significant

    # --- B7: trunk angle vertical trend ---
    slope_trunk = compute_slope(trunk_smooth, noise_factor=0.5)
    result.trunk_trend = slope_trunk.slope
    result.trunk_significant = slope_trunk.significant

    return result


# =====================================================================
# SELF-TEST
# =====================================================================

if __name__ == "__main__":
    print("=== Self-test temporal_utils ===")

    # Test 1: fill_nan_gaps
    sig = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    filled, mask = fill_nan_gaps(sig, max_gap=5)
    assert not np.any(np.isnan(filled)), "fill_nan_gaps thất bại"
    assert mask[2] and mask[3], "mask sai"
    print(" - fill_nan_gaps(): OK")

    # Test 2: smooth_signal
    noisy = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
    smoothed = smooth_signal(noisy, fps=30.0)
    assert len(smoothed) == len(noisy), "smooth_signal thay đổi shape"
    assert np.std(smoothed) < np.std(noisy), "smooth không giảm nhiễu"
    print(" - smooth_signal(): OK")

    # Test 3: compute_slope tăng rõ
    increasing = np.linspace(0, 10, 50) + np.random.randn(50) * 0.1
    sr = compute_slope(increasing)
    assert sr.slope > 0 and sr.significant, f"slope tăng không được nhận diện: {sr}"
    print(f" - compute_slope() tăng: slope={sr.slope:.3f}, significant={sr.significant}: OK")

    # Test 4: compute_slope hằng số
    flat = np.ones(50) + np.random.randn(50) * 0.01
    sr_flat = compute_slope(flat, noise_factor=0.5)
    assert abs(sr_flat.slope) < 0.01, f"slope hằng số phải ~0: {sr_flat.slope}"
    print(f" - compute_slope() hằng số: slope={sr_flat.slope:.4f}: OK")

    # Test 5: detect_steps
    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    # 2 chân chạy ngược pha, cadence ~1.5Hz
    ankle_l = -np.sin(2 * np.pi * 1.5 * t) * 30 + 500
    ankle_r = -np.sin(2 * np.pi * 1.5 * t + np.pi) * 30 + 500
    steps = detect_steps(ankle_l, ankle_r, fps)
    assert len(steps) >= 5, f"detect_steps phải tìm >= 5 bước, nhận {len(steps)}"
    print(f" - detect_steps(): phát hiện {len(steps)} bước trong 10s: OK")

    # Test 6: compute_cadence
    cad = compute_cadence(steps, fps)
    # 1.5Hz -> 90 chu kỳ/phút -> 180 bước/phút (2 chân)
    assert 150 < cad < 200, f"cadence phải ~180spm, nhận {cad:.1f}"
    print(f" - compute_cadence(): {cad:.1f} spm: OK")

    # Test 7: compute_drive_phase_trends trên dữ liệu tổng hợp
    body_x = np.linspace(0, 500, n)
    trunk = np.linspace(30, 5, n)  # giảm dần
    sw = 80.0
    result = compute_drive_phase_trends(
        ankle_l, ankle_r, body_x, trunk, sw, fps
    )
    assert result.n_steps >= 5, f"n_steps phải >= 5, nhận {result.n_steps}"
    assert result.trunk_trend < 0, f"trunk trend phải âm, nhận {result.trunk_trend}"
    print(f" - compute_drive_phase_trends(): n_steps={result.n_steps}, "
          f"trunk_trend={result.trunk_trend:.3f}: OK")

    print("\n=== TẤT CẢ SELF-TEST PASS ===")