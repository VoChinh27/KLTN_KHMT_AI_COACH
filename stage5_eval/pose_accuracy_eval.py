"""
stage5_eval/pose_accuracy_eval.py
====================================
Mục đích : Đánh giá độ chính xác POSE (không phải điểm chấm kỹ thuật) so
           với nhãn tay (ground truth do người gán thủ công trên MỘT TẬP
           CON frame — gán tay toàn bộ video là không khả thi).
Input    : - predictions: Dict[frame_idx, PoseFrame] do pose_extractor.py
             (Giai đoạn 2) sinh ra.
           - ground_truth: Dict[frame_idx, GroundTruthFrame] do người gán
             nhãn thủ công (chỉ cần có ở các frame được chọn để kiểm tra).
Output   : PoseAccuracyReport (PCK theo từng khớp + OKS trung bình) -> đưa
           vào report_builder.py để tổng hợp cùng score_validity_eval.py và
           robustness_eval.py.

2 chỉ số được dùng (bổ sung cho nhau):
    PCK (Percentage of Correct Keypoints): đơn giản, dễ diễn giải — khớp
        được coi là "đúng" nếu khoảng cách tới nhãn tay < alpha * đường
        chéo thân (torso diagonal), a la MPII benchmark.
    OKS (Object Keypoint Similarity): chuẩn COCO, có trọng số khác nhau cho
        từng loại khớp (khớp nhỏ như cổ tay yêu cầu chính xác cao hơn khớp
        lớn như hông) — phản ánh đúng bản chất bài toán hơn PCK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.pose_schema import COCO17_NAMES, PoseFrame

# Sigma chuẩn COCO cho 17 keypoint, ĐÚNG THỨ TỰ với COCO17_NAMES (xem
# core/pose_schema.py). Nguồn: cocodataset.org — dùng cho công thức OKS.
COCO_KEYPOINT_SIGMAS: Dict[str, float] = {
    "nose": 0.026, "left_eye": 0.025, "right_eye": 0.025,
    "left_ear": 0.035, "right_ear": 0.035,
    "left_shoulder": 0.079, "right_shoulder": 0.079,
    "left_elbow": 0.072, "right_elbow": 0.072,
    "left_wrist": 0.062, "right_wrist": 0.062,
    "left_hip": 0.107, "right_hip": 0.107,
    "left_knee": 0.087, "right_knee": 0.087,
    "left_ankle": 0.089, "right_ankle": 0.089,
}


@dataclass
class GroundTruthFrame:
    """Nhãn tay cho MỘT frame (chỉ cần có ở tập con frame được chọn kiểm tra)."""

    frame_idx: int
    xy: np.ndarray            # (num_joints, 2)
    visible: np.ndarray       # (num_joints,) bool — khớp có được người gán nhãn hay không
    bbox_area: Optional[float] = None  # nếu None, tự tính từ các khớp visible

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.visible = np.asarray(self.visible, dtype=bool)


@dataclass
class JointAccuracy:
    joint_name: str
    pck: float
    n_evaluated: int


@dataclass
class PoseAccuracyReport:
    per_joint_pck: List[JointAccuracy]
    overall_pck: float
    mean_oks: float
    alpha: float
    n_frames_evaluated: int
    per_frame_oks: Dict[int, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"joint": j.joint_name, "pck": j.pck, "n_evaluated": j.n_evaluated} for j in self.per_joint_pck]
        )

    def summary_dict(self) -> Dict:
        return {
            "overall_pck": self.overall_pck,
            "mean_oks": self.mean_oks,
            "alpha": self.alpha,
            "n_frames_evaluated": self.n_frames_evaluated,
        }


def _torso_diagonal(xy: np.ndarray, joint_names: Sequence[str]) -> float:
    ls = xy[list(joint_names).index("left_shoulder")]
    rh = xy[list(joint_names).index("right_hip")]
    return float(np.linalg.norm(ls - rh))


def compute_pck_frame(
    pred_xy: np.ndarray,
    gt_frame: GroundTruthFrame,
    joint_names: Sequence[str],
    alpha: float = 0.2,
) -> np.ndarray:
    """Trả về mảng bool (num_joints,) — True nếu khớp được coi là đúng
    (chỉ tính cho khớp gt_frame.visible=True, khớp khác trả về NaN)."""
    norm = _torso_diagonal(gt_frame.xy, joint_names)
    if norm < 1e-6 or np.isnan(norm):
        return np.full(len(joint_names), np.nan)
    threshold = alpha * norm
    dist = np.linalg.norm(pred_xy - gt_frame.xy, axis=1)
    result = np.where(gt_frame.visible, dist <= threshold, np.nan)
    return result


def compute_oks_frame(
    pred_xy: np.ndarray,
    gt_frame: GroundTruthFrame,
    joint_names: Sequence[str],
) -> float:
    """OKS chuẩn COCO cho 1 frame. Trả về NaN nếu không có khớp nào visible."""
    if not gt_frame.visible.any():
        return float("nan")

    if gt_frame.bbox_area is not None:
        s2 = gt_frame.bbox_area
    else:
        visible_xy = gt_frame.xy[gt_frame.visible]
        w = visible_xy[:, 0].max() - visible_xy[:, 0].min()
        h = visible_xy[:, 1].max() - visible_xy[:, 1].min()
        s2 = max(w * h, 1.0)

    dist_sq = np.sum((pred_xy - gt_frame.xy) ** 2, axis=1)
    sigmas = np.array([COCO_KEYPOINT_SIGMAS.get(name, 0.08) for name in joint_names])
    k = 2 * sigmas
    per_joint_oks = np.exp(-dist_sq / (2 * s2 * (k ** 2) + 1e-9))
    visible = gt_frame.visible
    if not visible.any():
        return float("nan")
    return float(np.sum(per_joint_oks * visible) / np.sum(visible))


class PoseAccuracyEvaluator:
    def __init__(self, joint_names: Sequence[str] = COCO17_NAMES, alpha: float = 0.2):
        self.joint_names = list(joint_names)
        self.alpha = alpha

    def evaluate(
        self,
        predictions: Dict[int, PoseFrame],
        ground_truth: Dict[int, GroundTruthFrame],
    ) -> PoseAccuracyReport:
        common_frames = sorted(set(predictions.keys()) & set(ground_truth.keys()))
        if not common_frames:
            raise ValueError(
                "Không có frame nào vừa có prediction vừa có ground truth — "
                "kiểm tra lại frame_idx của 2 tập dữ liệu có khớp nhau không."
            )

        per_joint_correct: Dict[str, List[bool]] = {name: [] for name in self.joint_names}
        per_frame_oks: Dict[int, float] = {}

        for frame_idx in common_frames:
            pred = predictions[frame_idx]
            gt = ground_truth[frame_idx]
            if list(pred.joint_names) != self.joint_names:
                raise ValueError(
                    f"joint_names của prediction (frame {frame_idx}) không khớp cấu hình evaluator"
                )

            pck_result = compute_pck_frame(pred.xy, gt, self.joint_names, self.alpha)
            for i, name in enumerate(self.joint_names):
                if not np.isnan(pck_result[i]):
                    per_joint_correct[name].append(bool(pck_result[i]))

            per_frame_oks[frame_idx] = compute_oks_frame(pred.xy, gt, self.joint_names)

        per_joint_pck: List[JointAccuracy] = []
        all_correct_flat: List[bool] = []
        for name in self.joint_names:
            flags = per_joint_correct[name]
            pck = float(np.mean(flags)) if flags else float("nan")
            per_joint_pck.append(JointAccuracy(joint_name=name, pck=pck, n_evaluated=len(flags)))
            all_correct_flat.extend(flags)

        overall_pck = float(np.mean(all_correct_flat)) if all_correct_flat else float("nan")
        oks_values = [v for v in per_frame_oks.values() if not np.isnan(v)]
        mean_oks = float(np.mean(oks_values)) if oks_values else float("nan")

        return PoseAccuracyReport(
            per_joint_pck=per_joint_pck,
            overall_pck=overall_pck,
            mean_oks=mean_oks,
            alpha=self.alpha,
            n_frames_evaluated=len(common_frames),
            per_frame_oks=per_frame_oks,
        )
