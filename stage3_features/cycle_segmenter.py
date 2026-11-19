"""
stage3_features/cycle_segmenter.py
====================================
Mục đích : Chuẩn hóa theo chu kỳ bước (loại bỏ ảnh hưởng tốc độ chạy khác
           nhau giữa học viên và HLV) và căn chỉnh thời gian bằng DTW để so
           sánh hai chuỗi động tác có tốc độ/độ dài khác nhau.
Input    : List[NormalizedFrame] từ normalizer.py (chỉ dùng cho pha DYNAMIC;
           pha static không cần phân chu kỳ).
Output   : List[GaitCycle] — mỗi chu kỳ đã được resample về 0-100% —
           -> đưa vào feature_extractor.py để tổng hợp đặc trưng theo pha.

Phương pháp:
    1. Tìm chu kỳ bước bằng peak detection trên tín hiệu vị trí dọc (y) của
       một mắt cá chân (foot strike ~ điểm y lớn nhất cục bộ vì trục y
       hướng xuống trong ảnh).
    2. Resample mỗi chu kỳ về đúng `num_points` mốc thời gian (0-100%) bằng
       nội suy tuyến tính -> so sánh được giữa các chu kỳ có độ dài khác
       nhau (tốc độ chạy khác nhau).
    3. DTW (Dynamic Time Warping) để căn chỉnh một chuỗi (vd. của học viên)
       theo mẫu tham chiếu (HLV) khi KHÔNG muốn resample cứng theo % mà cần
       giữ nguyên tốc độ tại từng thời điểm (dùng cho so khớp toàn đoạn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from stage3_features.normalizer import NormalizedFrame

# Cadence sinh lý hợp lý cho người chạy bộ (bước/phút, 1 chân) — dùng để
# giới hạn khoảng cách tối thiểu giữa 2 đỉnh khi tìm chu kỳ, tránh phát hiện
# nhiễu thành "chu kỳ giả".
MIN_CADENCE_SPM = 60.0
MAX_CADENCE_SPM = 130.0  # spm/1 chân; cadence tổng 2 chân gấp đôi giá trị này


@dataclass
class GaitCycle:
    """Một chu kỳ bước hoàn chỉnh (foot-strike -> foot-strike kế tiếp cùng chân)."""

    cycle_index: int
    start_frame: int
    end_frame: int
    duration_s: float
    cadence_spm: float  # bước/phút CỦA CHU KỲ NÀY (ước lượng từ duration_s)
    # tín hiệu gốc (chưa resample) theo từng tên đặc trưng -> mảng theo frame trong chu kỳ
    raw_features: Dict[str, np.ndarray] = field(default_factory=dict)
    # tín hiệu đã resample về num_points mốc (0..100% chu kỳ)
    resampled_features: Dict[str, np.ndarray] = field(default_factory=dict)


def _extract_signal(normalized: List[NormalizedFrame], key: str) -> np.ndarray:
    """Lấy 1 tín hiệu (ví dụ 'angle_knee_left') xuyên suốt chuỗi frame.
    key có 2 dạng: 'angle_<name>' hoặc 'ratio_<name>'."""
    kind, name = key.split("_", 1)
    values = []
    for nf in normalized:
        source = nf.angles if kind == "angle" else nf.ratios
        values.append(source.get(name, np.nan))
    return np.array(values, dtype=np.float64)


def detect_foot_strike_peaks(
    ankle_y_signal: np.ndarray,
    fps: float,
    min_cadence_spm: float = MIN_CADENCE_SPM,
    max_cadence_spm: float = MAX_CADENCE_SPM,
) -> np.ndarray:
    """Tìm chỉ số frame của các lần tiếp đất (foot strike) qua peak detection
    trên tín hiệu vị trí dọc của mắt cá chân.

    y pixel lớn nhất cục bộ ~ chân ở điểm thấp nhất trong khung hình ~ thời
    điểm tiếp đất. `distance` (frame) được suy ra từ max_cadence_spm để
    tránh bắt nhầm rung động nhỏ thành 1 chu kỳ riêng.
    """
    if np.all(np.isnan(ankle_y_signal)):
        return np.array([], dtype=int)

    max_step_per_min = max_cadence_spm
    min_period_s = 60.0 / max_step_per_min
    min_distance_frames = max(1, int(round(min_period_s * fps)))

    # nan gây lỗi cho find_peaks -> điền tạm bằng nội suy đơn giản chỉ để dò đỉnh
    sig = ankle_y_signal.copy()
    if np.isnan(sig).any():
        idx = np.arange(len(sig))
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            return np.array([], dtype=int)
        sig = np.interp(idx, idx[valid], sig[valid])

    peaks, _ = find_peaks(sig, distance=min_distance_frames)
    return peaks


def build_cycles_from_peaks(
    peaks: np.ndarray,
    normalized: List[NormalizedFrame],
    fps: float,
    feature_keys: Sequence[str],
    num_resample_points: int = 101,
    min_cadence_spm: float = MIN_CADENCE_SPM,
    max_cadence_spm: float = MAX_CADENCE_SPM,
) -> List[GaitCycle]:
    """Ghép các đỉnh liên tiếp thành GaitCycle, lọc bỏ chu kỳ có cadence
    ngoài khoảng sinh lý hợp lý (loại nhiễu / phát hiện sai đỉnh)."""
    cycles: List[GaitCycle] = []
    for i in range(len(peaks) - 1):
        start, end = int(peaks[i]), int(peaks[i + 1])
        duration_s = (end - start) / fps
        if duration_s <= 0:
            continue
        cadence_spm = 60.0 / duration_s
        if not (min_cadence_spm <= cadence_spm <= max_cadence_spm):
            continue  # chu kỳ bất thường -> loại, không đưa vào phân tích

        raw_features: Dict[str, np.ndarray] = {}
        resampled_features: Dict[str, np.ndarray] = {}
        for key in feature_keys:
            full_signal = _extract_signal(normalized, key)
            segment = full_signal[start : end + 1]
            raw_features[key] = segment
            resampled_features[key] = resample_to_percent(segment, num_resample_points)

        cycles.append(
            GaitCycle(
                cycle_index=len(cycles),
                start_frame=start,
                end_frame=end,
                duration_s=duration_s,
                cadence_spm=cadence_spm,
                raw_features=raw_features,
                resampled_features=resampled_features,
            )
        )
    return cycles


def resample_to_percent(signal: np.ndarray, num_points: int = 101) -> np.ndarray:
    """Nội suy tuyến tính 1 tín hiệu về `num_points` mốc đều nhau trên trục
    0-100% chu kỳ, bất kể chu kỳ gốc dài bao nhiêu frame (loại ảnh hưởng của
    tốc độ chạy — người chạy nhanh/chậm vẫn so sánh được cùng 1 trục %).
    """
    n = len(signal)
    if n == 0:
        return np.full(num_points, np.nan)
    if n == 1:
        return np.full(num_points, signal[0])

    x_original = np.linspace(0, 100, n)
    x_target = np.linspace(0, 100, num_points)

    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return np.full(num_points, np.nan)
    return np.interp(x_target, x_original[valid], signal[valid])


class GaitCycleSegmenter:
    """Điều phối toàn bộ bước phân chu kỳ cho MỘT bên chân (mặc định trái),
    trên bộ đặc trưng đã chuẩn hóa từ normalizer.py.
    """

    def __init__(
        self,
        feature_keys: Sequence[str] = (
            "angle_knee_left", "angle_knee_right",
            "angle_hip_left", "angle_hip_right",
            "angle_elbow_left", "angle_elbow_right",
            "angle_trunk",
        ),
        num_resample_points: int = 101,
    ):
        self.feature_keys = list(feature_keys)
        self.num_resample_points = num_resample_points

    def segment_from_ankle_y(
        self,
        ankle_y_signal: np.ndarray,
        normalized: List[NormalizedFrame],
        fps: float,
    ) -> List[GaitCycle]:
        peaks = detect_foot_strike_peaks(ankle_y_signal, fps)
        if len(peaks) < 2:
            return []
        return build_cycles_from_peaks(
            peaks, normalized, fps, self.feature_keys, self.num_resample_points
        )


# ---------------------------------------------------------------------------
# Dynamic Time Warping — căn chỉnh 2 chuỗi có tốc độ/độ dài khác nhau
# ---------------------------------------------------------------------------
def dtw_align(seq_a: np.ndarray, seq_b: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """DTW cổ điển (O(n*m)) cho 2 chuỗi 1D. Trả về (khoảng cách tích lũy đã
    chuẩn hóa theo độ dài đường đi, danh sách cặp chỉ số (i, j) khớp nhau).

    Dùng khi cần so khớp TOÀN ĐOẠN động tác của học viên với mẫu HLV mà
    KHÔNG muốn ép cứng về % chu kỳ (ví dụ khi số chu kỳ phát hiện được quá
    ít để tin cậy việc resample). Với hầu hết trường hợp trong hệ thống,
    resample_to_percent ở trên đã đủ dùng và đơn giản hơn; dtw_align là lựa
    chọn dự phòng/nâng cao.
    """
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # truy vết đường đi tối ưu
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda c: c[0])
    path.reverse()

    normalized_distance = float(cost[n, m] / len(path)) if path else float("inf")
    return normalized_distance, path


def dtw_similarity_score(seq_a: np.ndarray, seq_b: np.ndarray, scale: float = 1.0) -> float:
    """Chuyển khoảng cách DTW thành điểm tương đồng 0-100 (100 = giống hệt),
    dùng scale để quy đổi đơn vị (ví dụ độ, hoặc tỉ lệ theo body_scale) sao
    cho điểm số có ý nghĩa nhất quán giữa các đặc trưng khác nhau.
    """
    dist, _ = dtw_align(seq_a, seq_b)
    if not np.isfinite(dist):
        return 0.0
    normalized = dist / scale if scale > 1e-9 else dist
    return float(max(0.0, 100.0 - normalized))
