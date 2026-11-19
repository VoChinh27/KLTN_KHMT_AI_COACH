"""
core/side_mapping.py
======================
Mục đích : Xác định ĐỘNG (theo từng frame / theo phase) chân nào đang là
           "chân trước" và chân nào là "chân sau" trong các tư thế KHÔNG
           đối xứng theo giải phẫu trái/phải mà đối xứng theo TRƯỚC/SAU —
           điển hình là tư thế xuất phát thấp (Vào chỗ / Sẵn sàng), nơi
           chân thuận có thể là chân trái HAY chân phải tùy từng VĐV.

Tại sao cần file này: nếu chỉ dùng knee_left/knee_right (như normalizer.py
đang làm cho môn chạy đều), việc so sánh với khoảng chuẩn sẽ SAI khi trộn
lẫn VĐV đặt chân thuận bên trái với người đặt chân thuận bên phải —
khoảng chuẩn sẽ bị nhiễu vô nghĩa.

Nguyên tắc xác định (V2.5 — chỉ dùng heuristic khoảng cách):
    Chân trước = chân có mắt cá GẦN tâm hông hơn (Euclid 2D ankle→hip_center).
    Chỉ trả kết quả khi tỉ lệ khoảng cách đủ rõ (>= _MIN_DIST_RATIO).

    Lý do bỏ heuristic "độ cao gối":
      - Ngưỡng _MIN_KNEE_Y_DIFF trước đây ghi "normalized" nhưng so với
        pixel → luôn vượt ngưỡng, gây "bỏ phiếu" nhiễu.
      - Giả định hình học "gối trước thấp hơn gối sau" sai ở tư thế Set
        (thực tế gối sau thường thấp hơn) → tạo mâu thuẫn giả với
        heuristic khoảng cách và khiến hàm trả None.

    Default _LEFT khi NaN đã bị loại bỏ (SM2): bias 50% VĐV về phía sai.

API công khai:
    determine_front_leg(pose) -> Optional[str]
        Trả 'left' | 'right' | None.

    determine_front_leg_with_confidence(pose) -> Tuple[Optional[str], float]
        Thêm confidence ∈ [0, 1].

    majority_front_leg(poses, window=None) -> Tuple[Optional[str], float]
        Vote trên nhiều frame (SM3) — dùng ở phase-level thay vì per-frame.

    remap_left_right_to_front_back(values, front_side) -> Dict
        Không đổi so với V2.1.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.pose_schema import PoseFrame

_LEFT, _RIGHT = "left", "right"

# Ngưỡng Euclidean tối thiểu để coi 2 chân "khác biệt rõ" (pixel normalized
# khoảng 0–1 hoặc pixel tuyệt đối tùy hệ tọa độ; dùng ratio tương đối).
_MIN_DIST_RATIO = 1.15          # dist_back / dist_front >= ratio này mới rõ


def _safe_xy(kp) -> Optional[Tuple[float, float]]:
    """Trả (x, y) hoặc None nếu keypoint thiếu / NaN."""
    if kp is None:
        return None
    try:
        x, y = float(kp[0]), float(kp[1])
    except (TypeError, ValueError, IndexError):
        return None
    if x != x or y != y:  # NaN
        return None
    return x, y


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def determine_front_leg_with_confidence(
    pose: PoseFrame,
) -> Tuple[Optional[str], float]:
    """
    Xác định chân trước bằng heuristic khoảng cách Euclid 2D ankle→hip_center.

    Chân trước = chân có mắt cá GẦN tâm hông hơn. Chỉ trả kết quả khi tỉ lệ
    khoảng cách giữa 2 chân >= _MIN_DIST_RATIO (tránh đoán bừa khi 2 chân
    gần như tương đương).

    Không dùng độ cao gối: sai giả định hình học ở tư thế Set và từng bị lệch
    đơn vị (px vs chuẩn hoá) khiến heuristic luôn "bỏ phiếu".

    Returns
    -------
    (side, confidence)
        side        : 'left' | 'right' | None
        confidence  : 0.0 … 1.0 (0.0 khi không phân biệt được / thiếu data)
    """
    lh = _safe_xy(pose.get("left_hip"))
    rh = _safe_xy(pose.get("right_hip"))
    la = _safe_xy(pose.get("left_ankle"))
    ra = _safe_xy(pose.get("right_ankle"))

    # --- Thiếu hip hoặc ankle → không đoán ---
    if lh is None or rh is None or la is None or ra is None:
        return None, 0.0

    hip_center = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    dist_l = _euclid(la, hip_center)
    dist_r = _euclid(ra, hip_center)

    if dist_l <= 0.0 or dist_r <= 0.0:
        return None, 0.0

    ratio = max(dist_l, dist_r) / min(dist_l, dist_r)
    if ratio < _MIN_DIST_RATIO:
        return None, 0.0

    side = _LEFT if dist_l < dist_r else _RIGHT
    # ratio 1.15 → conf 0.30 ; ratio 1.50 → conf 1.00 ; kẹp trên ở 1.0
    confidence = min(1.0, (ratio - 1.0) / 0.5)
    return side, confidence


def determine_front_leg(pose: PoseFrame) -> Optional[str]:
    """
    Trả về 'left' | 'right' | None — chân nào đang là CHÂN TRƯỚC ở frame này.

    None khi không đủ dữ liệu hoặc tỉ lệ khoảng cách chưa đủ rõ (SM1/SM2).
    Caller (normalizer) phải xử lý None: set các feature front/back = NaN
    thay vì đoán bừa.
    """
    side, _ = determine_front_leg_with_confidence(pose)
    return side


def majority_front_leg(
    poses: Sequence[PoseFrame],
    min_confidence: float = 0.3,
) -> Tuple[Optional[str], float]:
    """
    Xác định front_leg ở mức PHASE (SM3) bằng majority vote trên nhiều frame.

    Chỉ đếm các frame có confidence >= min_confidence.
    Trả (side, vote_ratio) với vote_ratio ∈ [0, 1].
    None nếu không có frame nào đủ tin cậy hoặc hòa.
    """
    if not poses:
        return None, 0.0

    votes: List[str] = []
    for p in poses:
        side, conf = determine_front_leg_with_confidence(p)
        if side is not None and conf >= min_confidence:
            votes.append(side)

    if not votes:
        return None, 0.0

    counts = Counter(votes)
    winner, n = counts.most_common(1)[0]
    ratio = n / len(votes)
    # V2.4: hạ ngưỡng 0.60 → 0.50 vì dữ liệu 30fps noisy,
    # ratio thực tế thường ~0.50-0.60 (xem log).
    # Tie-break: nếu hòa 50/50 → chọn bên có confidence trung bình cao hơn.
    if ratio < 0.50:
        return None, ratio
    if ratio == 0.50 and len(counts) == 2:
        # Tie-break theo confidence trung bình
        sums = {side: 0.0 for side in counts}
        cnts = {side: 0 for side in counts}
        for p in poses:
            s, c = determine_front_leg_with_confidence(p)
            if s is not None and c >= min_confidence:
                sums[s] += c
                cnts[s] += 1
        if all(c > 0 for c in cnts.values()):
            avg = {s: sums[s] / cnts[s] for s in sums}
            winner = max(avg, key=avg.get)
    return winner, ratio


def remap_left_right_to_front_back(
    values: Dict[str, float], front_side: str
) -> Dict[str, float]:
    """
    Nhận dict khóa '<base>_left'/'<base>_right', trả dict MỚI chỉ chứa
    '<base>_front'/'<base>_back' tương ứng.

    Dict trả về được CỘNG THÊM vào dict gốc (xem normalizer.py), không thay
    thế — các khóa _left/_right cũ vẫn giữ nguyên cho môn đối xứng trái/phải.
    """
    if front_side not in (_LEFT, _RIGHT):
        raise ValueError(
            f"front_side phải là 'left' hoặc 'right', nhận '{front_side}'"
        )
    back_side = _RIGHT if front_side == _LEFT else _LEFT

    out: Dict[str, float] = {}
    front_suffix, back_suffix = f"_{front_side}", f"_{back_side}"
    for key, val in values.items():
        if key.endswith(front_suffix):
            base = key[: -len(front_suffix)]
            out[f"{base}_front"] = val
        elif key.endswith(back_suffix):
            base = key[: -len(back_suffix)]
            out[f"{base}_back"] = val
    return out