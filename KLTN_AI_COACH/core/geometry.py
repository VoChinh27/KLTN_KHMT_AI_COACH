"""
core/geometry.py
================
Hàm hình học dùng chung cho Giai đoạn 3 (chuẩn hóa, phân đoạn) và các giai
đoạn sau. Tách riêng để normalizer.py, cycle_segmenter.py, anomaly_detector.py
đều dùng CHUNG một công thức tính góc/khoảng cách, tránh sai lệch nhỏ giữa
các module (ví dụ quy ước chiều dương của góc).

Quy ước: hệ trục ảnh (x sang phải, y XUỐNG DƯỚI - chuẩn OpenCV/pixel).

V2.2 - GE1:
    - Thêm angle_at_joint / trunk_thigh_angle_at_hip: tính góc giải phẫu
      tại CÙNG một đỉnh (hip), tránh bug angle_between_vectors với 2 gốc
      khác nhau (mid_hip vs front_hip) làm lệch 5–15° ở tư thế crouch.
"""

from __future__ import annotations

import numpy as np


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Khoảng cách Euclid giữa 2 điểm (x, y)."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (np.asarray(p1) + np.asarray(p2)) / 2.0


def angle_at_vertex(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Góc (độ) tại `vertex` tạo bởi 2 đoạn vertex->a và vertex->c.

    Ví dụ knee_angle = angle_at_vertex(hip, knee, ankle).
    Trả về NaN nếu 1 trong 2 vector có độ dài ~0 (keypoint trùng nhau, dữ
    liệu lỗi) thay vì chia cho 0.
    """
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_at_joint(
    proximal: np.ndarray, joint: np.ndarray, distal: np.ndarray
) -> float:
    """
    Góc giải phẫu tại khớp `joint` giữa điểm proximal và distal.

    Alias rõ nghĩa của angle_at_vertex — dùng khi muốn nhấn mạnh quy ước
    proximal → joint → distal (ví dụ shoulder → hip → knee).
    """
    return angle_at_vertex(proximal, joint, distal)


def trunk_thigh_angle_at_hip(
    shoulder: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray,
) -> float:
    """
    Góc thân–đùi tại HIP (GE1 — công thức khuyến nghị).

    Vector thân : hip → shoulder
    Vector đùi  : hip → knee
    Góc tại hip giữa hai vector.

    Khác với angle_between_vectors(mid_shoulder - mid_hip, front_knee - front_hip)
    vốn dùng HAI GỐC khác nhau (mid_hip vs front_hip) → lệch 5–15° khi crouch.

    Caller (normalizer) nên truyền:
        shoulder = front_shoulder  (cùng phía với front_hip)
                   hoặc mid_shoulder nếu không có front_side
        hip      = front_hip
        knee     = front_knee
    """
    return angle_at_vertex(shoulder, hip, knee)


def angle_to_vertical(vec: np.ndarray) -> float:
    """Góc (độ) giữa vector `vec` và phương thẳng đứng hướng lên (0, -1).

    Dùng để đo độ nghiêng thân người (trunk lean) hoặc độ nghiêng camera.
    0° = vec thẳng đứng; 90° = vec nằm ngang.
    """
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return float("nan")
    vertical = np.array([0.0, -1.0])
    cos_theta = np.clip(np.dot(v, vertical) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_to_horizontal(vec: np.ndarray) -> float:
    """
    Góc (độ) giữa vector `vec` và phương ngang (1, 0).

    0° = nằm ngang; 90° = thẳng đứng.
    Hữu ích cho leg_ground_angle nếu quy ước YAML là góc với mặt đất (ngang).
    """
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return float("nan")
    horizontal = np.array([1.0, 0.0])
    cos_theta = np.clip(np.dot(v, horizontal) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Góc (độ) giữa 2 vector bất kỳ (không cần chung gốc).

    CẢNH BÁO (GE1): Khi đo góc giải phẫu tại một khớp, HÃY dùng
    angle_at_vertex / angle_at_joint / trunk_thigh_angle_at_hip với CÙNG
    một điểm làm gốc. Dùng angle_between_vectors với 2 vector xuất phát
    từ 2 điểm khác nhau (mid_hip vs front_hip) sẽ sai 5–15° ở tư thế crouch.
    """
    v1, v2 = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def signed_horizontal_lean(vec: np.ndarray) -> float:
    """Độ nghiêng có dấu theo trục ngang (âm/dương thể hiện nghiêng trái/phải
    hoặc trước/sau tùy quy ước camera), dùng cho body_lean chi tiết hơn
    angle_to_vertical (vốn luôn không âm)."""
    v = np.asarray(vec, dtype=np.float64)
    if np.linalg.norm(v) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan2(v[0], -v[1])))