"""
core/pose_schema.py
====================
Định nghĩa schema keypoint dùng chung cho toàn bộ hệ thống (Giai đoạn 1-5).

V2.1 - Bổ sung:
    - Helper get_group_conf() để lấy confidence trung bình 1 nhóm khớp.
    - Helper get_side_joint() để map front/back.
    - Helper get_front_back_conf() cho quality flags (wrist, knee_back, rear_ankle).

V2.2 - Sửa bug schema (PS4, PS5):
    - PS5: Cache _joint_index_map trong __post_init__ → joint_index O(1).
    - PS4: get_group_conf hỗ trợ mode='min' | 'mean'.
      get_min_group_conf() cho flag conservative (bắt 1 ankle kém).
      get_front_back_conf đã có sẵn — caller nên dùng sau khi biết front_side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Danh sách keypoint chuẩn
# ---------------------------------------------------------------------------

COCO17_NAMES: List[str] = [
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

POSE33_NAMES: List[str] = COCO17_NAMES + [
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    "mouth_left", "mouth_right",
    "left_eye_inner", "right_eye_inner",
    "left_eye_outer", "right_eye_outer",
]

CONF_THRESHOLD_DEFAULT = 0.35

# ---------------------------------------------------------------------------
# Nhóm joint dùng cho quality flags (V2.1)
# ---------------------------------------------------------------------------

JOINT_GROUPS: Dict[str, List[str]] = {
    "wrist":       ["left_wrist", "right_wrist"],
    "knee":        ["left_knee", "right_knee"],
    "ankle":       ["left_ankle", "right_ankle"],
    "shoulder":    ["left_shoulder", "right_shoulder"],
    "hip":         ["left_hip", "right_hip"],
    "elbow":       ["left_elbow", "right_elbow"],
    "head":        ["nose"],
}


@dataclass
class PoseFrame:
    """Kết quả pose của MỘT frame, cho MỘT người."""

    frame_idx: int
    timestamp_s: float
    xy: np.ndarray
    conf: np.ndarray
    joint_names: Sequence[str] = field(default_factory=lambda: COCO17_NAMES)
    bbox: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    # Cache nội bộ (PS5) — không serialize
    _joint_index_map: Dict[str, int] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32)
        self.conf = np.asarray(self.conf, dtype=np.float32)
        n = len(self.joint_names)
        if self.xy.shape != (n, 2):
            raise ValueError(
                f"xy phải có shape ({n}, 2), nhận được {self.xy.shape}"
            )
        if self.conf.shape != (n,):
            raise ValueError(
                f"conf phải có shape ({n},), nhận được {self.conf.shape}"
            )
        # PS5: cache O(1) lookup
        object.__setattr__(
            self,
            "_joint_index_map",
            {name: i for i, name in enumerate(self.joint_names)},
        )

    # ------------------------------------------------------------------
    # Truy xuất cơ bản
    # ------------------------------------------------------------------
    def joint_index(self, name: str) -> int:
        """O(1) nhờ cache _joint_index_map (PS5)."""
        idx = self._joint_index_map.get(name)
        if idx is None:
            raise ValueError(
                f"Joint '{name}' không có trong joint_names "
                f"({list(self.joint_names)[:5]}...)"
            )
        return idx

    def get(self, name: str) -> np.ndarray:
        return self.xy[self.joint_index(name)]

    def get_conf(self, name: str) -> float:
        return float(self.conf[self.joint_index(name)])

    def has_joint(self, name: str) -> bool:
        return name in self._joint_index_map

    def valid_mask(self, threshold: float = CONF_THRESHOLD_DEFAULT) -> np.ndarray:
        return self.conf >= threshold

    def visible_ratio(self, threshold: float = CONF_THRESHOLD_DEFAULT) -> float:
        return float(np.mean(self.valid_mask(threshold)))

    # ------------------------------------------------------------------
    # V2.1 / V2.2 - Confidence helpers cho quality flags
    # ------------------------------------------------------------------
    def get_group_conf(
        self, group: str, mode: str = "mean"
    ) -> float:
        """
        Confidence của 1 nhóm joint.

        Parameters
        ----------
        group : 'wrist' / 'knee' / 'ankle' / 'shoulder' / 'hip' / 'elbow' / 'head'
        mode  : 'mean' (mặc định) | 'min'
                'min' conservative hơn — bắt case 1 bên kém (PS4).
                Ví dụ rear ankle conf=0.2, front=0.8 → mean=0.5 (pass),
                min=0.2 (fail) → flag đúng.

        Trả về NaN nếu không có joint nào trong nhóm.
        """
        names = JOINT_GROUPS.get(group, [])
        if not names:
            return float("nan")
        vals = [self.get_conf(n) for n in names if self.has_joint(n)]
        if not vals:
            return float("nan")
        if mode == "min":
            return float(np.min(vals))
        return float(np.mean(vals))

    def get_min_group_conf(self, group: str) -> float:
        """
        PS4: Confidence thấp nhất trong nhóm (conservative).

        Dùng cho flag rear_ankle_low_confidence / wrist_low_confidence
        khi CHƯA biết front_side — bắt được case chỉ 1 bên kém.
        """
        return self.get_group_conf(group, mode="min")

    def get_side_joint(
        self, side: str, joint_type: str
    ) -> Optional[np.ndarray]:
        """
        Lấy tọa độ của 1 joint theo phía.
        side: 'left' | 'right' | 'front' | 'back'
        joint_type: 'knee' | 'ankle' | 'hip' | 'shoulder' | 'elbow' | 'wrist'
        Với 'front'/'back': cần front_side đã được tính riêng — trả None.
        """
        if side in ("left", "right"):
            name = f"{side}_{joint_type}"
            if self.has_joint(name):
                return self.get(name)
            return None
        # front/back: cần front_side đã được tính riêng, không suy từ PoseFrame
        return None

    def get_front_back_conf(
        self, joint_type: str, front_side: str
    ) -> Dict[str, float]:
        """
        Trả về confidence cho front/back của 1 joint type.

        front_side: 'left' hoặc 'right' (từ side_mapping).

        PS4 khuyến nghị: sau khi determine_front_leg ổn định, dùng
            confs = frame.get_front_back_conf("ankle", front_side)
            flags["rear_ankle_low_confidence"] = confs["back"] < 0.4
        thay vì mean(left, right).
        """
        if front_side not in ("left", "right"):
            return {"front": float("nan"), "back": float("nan")}
        back_side = "right" if front_side == "left" else "left"
        front_name = f"{front_side}_{joint_type}"
        back_name = f"{back_side}_{joint_type}"
        return {
            "front": (
                self.get_conf(front_name)
                if self.has_joint(front_name)
                else float("nan")
            ),
            "back": (
                self.get_conf(back_name)
                if self.has_joint(back_name)
                else float("nan")
            ),
        }

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "xy": self.xy.tolist(),
            "conf": self.conf.tolist(),
            "joint_names": list(self.joint_names),
            "bbox": None if self.bbox is None else self.bbox.tolist(),
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PoseFrame":
        return cls(
            frame_idx=d["frame_idx"],
            timestamp_s=d["timestamp_s"],
            xy=np.array(d["xy"], dtype=np.float32),
            conf=np.array(d["conf"], dtype=np.float32),
            joint_names=d.get("joint_names", COCO17_NAMES),
            bbox=(
                None
                if d.get("bbox") is None
                else np.array(d["bbox"], dtype=np.float32)
            ),
            track_id=d.get("track_id"),
        )