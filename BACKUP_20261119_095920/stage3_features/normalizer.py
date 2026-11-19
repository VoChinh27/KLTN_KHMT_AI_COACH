"""
stage3_features/normalizer.py
===============================
Mục đích : Chuyển tọa độ thô (pixel) thành đặc trưng bất biến với khoảng
           cách camera, chiều cao, vị trí đứng của từng người.

V2.1.3
    - leg_ground_angle: góc nhọn với mặt đất, [0, 90], không phụ thuộc
      chiều chạy. 0° = nằm ngang, 90° = thẳng đứng. Dải tham chiếu vật lý
      khi kết thúc đạp là khoảng [35, 55], KHÔNG phải [110, 135].
    - Ratio ngang và dọc dùng hai thước khác nhau. Thước ngang lấy max
      (vai, hông, 0.5 * thân) để camera chếch không bóp shoulder_width
      rồi thổi ratio lên hàng chục lần.
    - Keypoint dưới ngưỡng confidence -> feature liên quan = NaN.
    - shoulder_forward có dấu theo hướng chạy suy ra từ hai mắt cá.
    - ankle_is_approx không còn nằm trong dict angles.
"""



from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.geometry import (
    angle_at_vertex,
    angle_between_vectors,
    angle_to_vertical,
    distance,
    midpoint,
)
from core.pose_schema import PoseFrame
from core.side_mapping import (
    determine_front_leg,
    remap_left_right_to_front_back,
)

logger = logging.getLogger(__name__)

MIN_BODY_SCALE_PX = 15.0
MIN_JOINT_CONF = 0.45


@dataclass
class NormalizedFrame:
    frame_idx: int
    timestamp_s: float
    body_scale: float
    angles: Dict[str, float] = field(default_factory=dict)
    ratios: Dict[str, float] = field(default_factory=dict)
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    # 0 = đo cổ chân trực tiếp (có foot_index), 1 = xấp xỉ từ ống chân.
    # Tách khỏi angles để mean/std góc không bị kéo bởi cờ 0/1.
    ankle_is_approx: float = 0.0

    def to_flat_dict(self) -> Dict:
        d = {
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "body_scale": self.body_scale,
            "is_valid": self.is_valid,
            "ankle_is_approx": self.ankle_is_approx,
        }
        d.update({f"angle_{k}": v for k, v in self.angles.items()})
        d.update({f"ratio_{k}": v for k, v in self.ratios.items()})
        return d


_REQUIRED_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "nose",
]


def _has_foot_index(joint_names: Sequence[str]) -> bool:
    return (
        "left_foot_index" in joint_names
        and "right_foot_index" in joint_names
    )


def _joint_conf(pose: PoseFrame, name: str) -> float:
    getter = getattr(pose, "get_conf", None)
    if getter is None:
        return 1.0
    try:
        value = float(getter(name))
    except Exception:
        return float("nan")
    return value


def _joints_confident(
    pose: PoseFrame,
    joints: Sequence[str],
    min_conf: float,
) -> bool:
    for name in joints:
        conf = _joint_conf(pose, name)
        if np.isnan(conf) or conf < min_conf:
            return False
    return True


def _finite_scale(value: float) -> float:
    if value is None or not np.isfinite(value) or value < 1e-6:
        return float("nan")
    return float(value)


class Normalizer:
    """Tính NormalizedFrame cho từng PoseFrame."""

    def __init__(
        self,
        min_body_scale_px: float = MIN_BODY_SCALE_PX,
        min_joint_conf: float = MIN_JOINT_CONF,
    ):
        self.min_body_scale_px = min_body_scale_px
        self.min_joint_conf = min_joint_conf

    def _angle(
        self,
        pose: PoseFrame,
        joints: Sequence[str],
        compute: Callable[[], float],
    ) -> float:
        if not _joints_confident(pose, joints, self.min_joint_conf):
            return float("nan")
        try:
            value = float(compute())
        except Exception:
            return float("nan")
        if not np.isfinite(value):
            return float("nan")
        return value

    # ------------------------------------------------------------------
    def normalize_frame(self, pose: PoseFrame, front_side_hint: Optional[str] = None,) -> NormalizedFrame:
        names = pose.joint_names
        missing = [j for j in _REQUIRED_JOINTS if j not in names]
        if missing:
            raise ValueError(f"Thiếu keypoint bắt buộc để chuẩn hóa: {missing}")

        ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
        lh, rh = pose.get("left_hip"), pose.get("right_hip")
        lk, rk = pose.get("left_knee"), pose.get("right_knee")
        la, ra = pose.get("left_ankle"), pose.get("right_ankle")
        le, re = pose.get("left_elbow"), pose.get("right_elbow")
        lw, rw = pose.get("left_wrist"), pose.get("right_wrist")
        nose = pose.get("nose")

        mid_shoulder = midpoint(ls, rs)
        mid_hip = midpoint(lh, rh)
        body_scale = distance(mid_shoulder, mid_hip)

        is_valid = True
        invalid_reason = None
        if not np.isfinite(body_scale) or body_scale < self.min_body_scale_px:
            is_valid = False
            invalid_reason = (
                f"body_scale={body_scale:.1f}px < ngưỡng "
                f"{self.min_body_scale_px}px "
                "(người quá xa camera hoặc keypoint vai/hông không đáng tin)"
            )

        angles: Dict[str, float] = {
            "elbow_left": self._angle(
                pose, ("left_shoulder", "left_elbow", "left_wrist"),
                lambda: angle_at_vertex(ls, le, lw),
            ),
            "elbow_right": self._angle(
                pose, ("right_shoulder", "right_elbow", "right_wrist"),
                lambda: angle_at_vertex(rs, re, rw),
            ),
            "knee_left": self._angle(
                pose, ("left_hip", "left_knee", "left_ankle"),
                lambda: angle_at_vertex(lh, lk, la),
            ),
            "knee_right": self._angle(
                pose, ("right_hip", "right_knee", "right_ankle"),
                lambda: angle_at_vertex(rh, rk, ra),
            ),
            "hip_left": self._angle(
                pose, ("left_shoulder", "left_hip", "left_knee"),
                lambda: angle_at_vertex(ls, lh, lk),
            ),
            "hip_right": self._angle(
                pose, ("right_shoulder", "right_hip", "right_knee"),
                lambda: angle_at_vertex(rs, rh, rk),
            ),
            "trunk": self._angle(
                pose,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                lambda: angle_to_vertical(mid_shoulder - mid_hip),
            ),
            "head": self._angle(
                pose,
                ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                lambda: self._head_angle(nose, mid_shoulder, mid_hip),
            ),
        }

        ankle_is_approx = 0.0
        if _has_foot_index(names):
            lf, rf = pose.get("left_foot_index"), pose.get("right_foot_index")
            angles["ankle_left"] = self._angle(
                pose, ("left_knee", "left_ankle", "left_foot_index"),
                lambda: angle_at_vertex(lk, la, lf),
            )
            angles["ankle_right"] = self._angle(
                pose, ("right_knee", "right_ankle", "right_foot_index"),
                lambda: angle_at_vertex(rk, ra, rf),
            )
        else:
            angles["ankle_left"] = self._angle(
                pose, ("left_knee", "left_ankle"),
                lambda: angle_to_vertical(la - lk),
            )
            angles["ankle_right"] = self._angle(
                pose, ("right_knee", "right_ankle"),
                lambda: angle_to_vertical(ra - rk),
            )
            ankle_is_approx = 1.0

        # Hai thước đo. Ngang không chỉ dùng shoulder_width: side-view làm
        # bề rộng vai co lại và thổi mọi ratio. Dọc dùng body_scale.
        shoulder_width_px = distance(ls, rs)
        hip_width_px = distance(lh, rh)
        vertical_hint = 0.5 * body_scale if np.isfinite(body_scale) else 0.0
        scale_horizontal = _finite_scale(max(
            shoulder_width_px if np.isfinite(shoulder_width_px) else 0.0,
            hip_width_px if np.isfinite(hip_width_px) else 0.0,
            vertical_hint,
        ))
        scale_vertical = _finite_scale(body_scale)

        def _hdiv(numer: float, joints: Sequence[str]) -> float:
            if not np.isfinite(scale_horizontal):
                return float("nan")
            if not _joints_confident(pose, joints, self.min_joint_conf):
                return float("nan")
            if not np.isfinite(numer):
                return float("nan")
            return float(numer / scale_horizontal)

        def _vdiv(numer: float, joints: Sequence[str]) -> float:
            if not np.isfinite(scale_vertical):
                return float("nan")
            if not _joints_confident(pose, joints, self.min_joint_conf):
                return float("nan")
            if not np.isfinite(numer):
                return float("nan")
            return float(numer / scale_vertical)

        both_shoulders = ("left_shoulder", "right_shoulder")
        both_hips = ("left_hip", "right_hip")
        both_wrists = ("left_wrist", "right_wrist")
        both_ankles = ("left_ankle", "right_ankle")

        ratios: Dict[str, float] = {
            "shoulder_width": _hdiv(shoulder_width_px, both_shoulders),
            "hip_width": _hdiv(hip_width_px, both_hips),
            "thigh_left": _hdiv(
                distance(lh, lk), ("left_hip", "left_knee"),
            ),
            "thigh_right": _hdiv(
                distance(rh, rk), ("right_hip", "right_knee"),
            ),
            "shank_left": _hdiv(
                distance(lk, la), ("left_knee", "left_ankle"),
            ),
            "shank_right": _hdiv(
                distance(rk, ra), ("right_knee", "right_ankle"),
            ),
            "wrist_width": self._hdiv_wrist_width(pose, lw, rw, scale_horizontal,),
            "hand_ground_clearance": _vdiv(
                ((la[1] + ra[1]) / 2.0 - (lw[1] + rw[1]) / 2.0),
                both_wrists + both_ankles,
            ),
            "hip_above_shoulder": _vdiv(
                (mid_shoulder[1] - mid_hip[1]),
                both_shoulders + both_hips,
            ),
        }

        front_side = front_side_hint or determine_front_leg(pose)
        if front_side is None:
            front_side = "left"
        back_side = "left" if front_side == "right" else "right"
        front_hip = rh if front_side == "right" else lh
        front_knee = rk if front_side == "right" else lk
        front_ankle = ra if front_side == "right" else la
        back_hip = lh if back_side == "left" else rh
        back_knee = lk if back_side == "left" else rk
        back_ankle = la if back_side == "left" else ra
        front_joints = (
            (f"{front_side}_hip", f"{front_side}_knee", f"{front_side}_ankle")
        )
        back_joints = (f"{back_side}_hip", f"{back_side}_knee")
        
        from core.geometry import trunk_thigh_angle_at_hip
        front_shoulder = rs if front_side == "right" else ls
        angles["trunk_thigh_angle"] = self._angle(
            pose,
            (f"{front_side}_shoulder", f"{front_side}_hip", f"{front_side}_knee"),
            lambda: trunk_thigh_angle_at_hip(front_shoulder, front_hip, front_knee),
        )

        angles.update(remap_left_right_to_front_back(angles, front_side))
        ratios.update(remap_left_right_to_front_back(ratios, front_side))

        # Dương = vai nằm phía trước cổ chân trước theo hướng hai mắt cá
        # (back -> front). Âm = vai bị tụt sau cổ chân trước.
        run_dx = float(front_ankle[0] - back_ankle[0])
        direction = 0.0 if abs(run_dx) < 1e-6 else (1.0 if run_dx > 0 else -1.0)
        signed_forward = (float(mid_shoulder[0] - front_ankle[0])) * direction
        ratios["shoulder_forward"] = _hdiv(
            signed_forward,
            both_shoulders + (f"{front_side}_ankle", f"{back_side}_ankle"),
        )

        angles["trunk_angle_vertical"] = angles["trunk"]
        angles["knee_front_angle"] = angles.get("knee_front", float("nan"))
        angles["knee_back_angle"] = angles.get("knee_back", float("nan"))

        head = angles["head"]
        if not np.isnan(head):
            angles["head_trunk_alignment"] = 180.0 - head
        else:
            angles["head_trunk_alignment"] = float("nan")

        ratios["wrist_width_ratio"] = ratios["wrist_width"]
        ratios["hip_above_shoulder_norm"] = ratios["hip_above_shoulder"]
        ratios["shoulder_forward_norm"] = ratios["shoulder_forward"]

        # Góc nhọn giữa ống chân trước và mặt đất, [0, 90].
        # abs(cos) gộp hai cụm cũ (~40° và ~140°) về cùng một phía,
        # không phụ thuộc VĐV chạy sang trái hay sang phải.
        angles["leg_ground_angle"] = self._angle(
            pose,
            front_joints[1:],
            lambda: _acute_angle_to_horizontal(front_ankle - front_knee),
        )

        angles["hip_thigh_angle"] = self._angle(
            pose,
            ("left_shoulder", "right_shoulder", "left_hip", "right_hip") + back_joints,
            lambda: _angle_between(back_knee - back_hip, mid_shoulder - mid_hip),
        )

        # hip_back / thigh_back: alias để launch tier 1-2 có key thật,
        # không còn .get() ra NaN chỉ vì tên.
        angles["hip_back"] = angles.get("hip_back", angles.get(f"hip_{back_side}", float("nan")))
        angles["thigh_back"] = angles.get(
            "thigh_back",
            self._angle(
                pose,
                back_joints,
                lambda: angle_to_vertical(back_knee - back_hip),
            ),
        )

        return NormalizedFrame(
            frame_idx=pose.frame_idx,
            timestamp_s=pose.timestamp_s,
            body_scale=float(body_scale) if np.isfinite(body_scale) else float("nan"),
            angles=angles,
            ratios=ratios,
            is_valid=is_valid,
            invalid_reason=invalid_reason,
            ankle_is_approx=ankle_is_approx,
        )

    @staticmethod
    def _head_angle(
        nose: np.ndarray,
        mid_shoulder: np.ndarray,
        mid_hip: np.ndarray,
    ) -> float:
        trunk_vec = mid_shoulder - mid_hip
        head_vec = nose - mid_shoulder
        return _angle_between(trunk_vec, head_vec)

    def normalize_sequence(
        self,
        sequence: List[PoseFrame],
        front_side_hint: Optional[str] = None,
    ) -> List[NormalizedFrame]:
        """
        V2.3 — Không còn crash khi determine_front_leg trả None.
        Dùng majority vote ở mức SEQUENCE (không phải per-frame) để có
        front_side ổn định cho cả video.
        """
        if front_side_hint is None:
            front_side_hint = self._majority_front_side(sequence)
        return [
            self.normalize_frame(pf, front_side_hint=front_side_hint)
            for pf in sequence
        ]

    @staticmethod
    def _majority_front_side(sequence: List[PoseFrame]) -> str:
        """Vote chân trước trên toàn chuỗi. Fallback 'left' nếu không đủ đồng thuận."""
        try:
            from core.side_mapping import majority_front_leg
            side, ratio = majority_front_leg(sequence, min_confidence=0.2)
            if side is not None:
                logger.info(
                    "[normalizer] front_side=%s (majority %.0f%% trên %d frame)",
                    side, ratio * 100, len(sequence),
                )
                return side
            logger.warning(
                "[normalizer] majority vote fail (ratio=%.2f). Fallback 'left' — "
                "feature front/back có thể SAI cho VĐV đặt chân phải trước.",
                ratio,
            )
        except Exception as exc:
            logger.warning("[normalizer] Lỗi majority vote: %s. Fallback 'left'.", exc)
        return "left"
    
    def _hdiv_wrist_width(
        self, pose, lw, rw, scale_horizontal,
    ) -> float:
        """Wrist width: cần cả 2 wrist để đo. Nếu 1 wrist miss (conf<0.45)
        nhưng wrist kia conf>=0.30, dùng shoulder_width làm proxy
        (đo đúng với tư thế 'tay chống rộng bằng vai')."""
        if not np.isfinite(scale_horizontal) or scale_horizontal <= 0:
            return float("nan")
        conf_l = _joint_conf(pose, "left_wrist")
        conf_r = _joint_conf(pose, "right_wrist")
        both_ok = (not np.isnan(conf_l) and conf_l >= self.min_joint_conf
                and not np.isnan(conf_r) and conf_r >= self.min_joint_conf)
        if both_ok:
            dist = distance(lw, rw)
            return float(dist / scale_horizontal) if np.isfinite(dist) else float("nan")
        # Fallback: proxy = shoulder_width (đúng cho tư thế chống tay bằng vai)
        one_ok = (not np.isnan(conf_l) and conf_l >= 0.30) or \
                (not np.isnan(conf_r) and conf_r >= 0.30)
        if one_ok:
            # shoulder_width là scale_horizontal → ratio = 1.0
            return 1.0
        return float("nan")

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_theta = np.clip(float(np.dot(a, b) / (n1 * n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _acute_angle_to_horizontal(vec: np.ndarray) -> float:
    """Góc nhỏ nhất giữa vector và trục ngang, đơn vị độ, trong [0, 90]."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return float("nan")
    cos_abs = abs(float(vec[0]) / norm)
    cos_abs = float(np.clip(cos_abs, 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_abs)))


def to_dataframe(normalized: List[NormalizedFrame]) -> pd.DataFrame:
    return pd.DataFrame([nf.to_flat_dict() for nf in normalized])