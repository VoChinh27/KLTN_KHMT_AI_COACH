"""
stage5_eval/robustness_eval.py
=================================
Mục đích : Đánh giá độ bền vững của điểm chấm theo góc quay, khoảng cách
           (scale), và một proxy cho nhiễu do ánh sáng/che khuất — thông
           qua thực nghiệm ablation (thay đổi từng yếu tố, giữ nguyên các
           yếu tố khác) trên chuỗi keypoint đã có.
Input    : Chuỗi PoseFrame "sạch" (đã qua Giai đoạn 2) của MỘT video, cùng
           FeatureExtractor + ReferenceModel + Scorer đã xây ở Giai đoạn 3-4.
Output   : pandas.DataFrame (1 dòng / điều kiện ablation) -> đưa vào
           report_builder.py để tổng hợp cùng 2 file eval còn lại.

GIỚI HẠN CẦN NÊU RÕ (đúng tinh thần "chủ động ghi nhận giới hạn" đã thống
nhất): module này ablation trên KEYPOINT đã trích xuất, KHÔNG phải trên
pixel ảnh gốc. Vì vậy:
    - Góc quay (view angle)  : mô phỏng được khá trực tiếp — co ngắn
      khoảng cách ngang theo cos(θ), đúng hiệu ứng hình học đã phân tích.
    - Khoảng cách/scale      : mô phỏng được trực tiếp — co giãn toàn bộ
      khung xương quanh tâm hông mỗi frame.
    - Ánh sáng/trang phục    : KHÔNG thể mô phỏng thật ở cấp keypoint; ở
      đây dùng nhiễu Gauss lên tọa độ khớp như một PROXY cho việc pose
      model kém tin cậy hơn trong điều kiện đó — không phải mô phỏng ánh
      sáng/trang phục trực tiếp. Đây là hạn chế nên nêu khi báo cáo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.pose_schema import PoseFrame
from stage3_features.feature_extractor import FeatureExtractor
from stage4_scoring.reference_model import ReferenceModel
from stage4_scoring.scorer import Scorer

Transform = Callable[[List[PoseFrame]], List[PoseFrame]]


@dataclass
class RobustnessCondition:
    name: str
    category: str  # "view_angle" | "scale" | "noise_proxy" | "baseline"
    param: float
    transform: Transform


# ---------------------------------------------------------------------------
# 1. Các phép biến đổi (transform) trên chuỗi PoseFrame
# ---------------------------------------------------------------------------
def _hip_center(pf: PoseFrame) -> np.ndarray:
    return (pf.get("left_hip") + pf.get("right_hip")) / 2.0


def apply_view_angle_distortion(sequence: List[PoseFrame], angle_deg: float) -> List[PoseFrame]:
    """Co ngắn khoảng cách NGANG quanh tâm hông của MỖI frame theo cos(θ),
    mô phỏng hiệu ứng camera lệch góc so với hướng chuyển động (đã phân
    tích: "khoảng cách ngang bị co theo cos θ")."""
    factor = float(np.cos(np.radians(angle_deg)))
    out = []
    for pf in sequence:
        pivot = _hip_center(pf)
        new_xy = pf.xy.copy()
        new_xy[:, 0] = pivot[0] + (new_xy[:, 0] - pivot[0]) * factor
        out.append(_clone_with_xy(pf, new_xy))
    return out


def apply_scale(sequence: List[PoseFrame], factor: float) -> List[PoseFrame]:
    """Co giãn toàn bộ khung xương quanh tâm hông của MỖI frame theo `factor`,
    mô phỏng việc học viên đứng xa/gần camera hơn (thay đổi body_scale)."""
    out = []
    for pf in sequence:
        pivot = _hip_center(pf)
        new_xy = pivot + (pf.xy - pivot) * factor
        out.append(_clone_with_xy(pf, new_xy))
    return out


def apply_gaussian_noise_proxy(
    sequence: List[PoseFrame], noise_std_px: float, seed: int = 0
) -> List[PoseFrame]:
    """Thêm nhiễu Gauss độc lập lên từng khớp — PROXY cho việc pose model
    kém chính xác hơn (ví dụ do ánh sáng yếu/trang phục rộng), KHÔNG PHẢI
    mô phỏng trực tiếp ánh sáng/trang phục (xem giới hạn ở đầu module)."""
    rng = np.random.default_rng(seed)
    out = []
    for pf in sequence:
        new_xy = pf.xy + rng.normal(0, noise_std_px, size=pf.xy.shape)
        out.append(_clone_with_xy(pf, new_xy))
    return out


def _clone_with_xy(pf: PoseFrame, new_xy: np.ndarray) -> PoseFrame:
    return PoseFrame(
        frame_idx=pf.frame_idx, timestamp_s=pf.timestamp_s,
        xy=new_xy.astype(np.float32), conf=pf.conf.copy(),
        joint_names=pf.joint_names, bbox=pf.bbox, track_id=pf.track_id,
    )


# ---------------------------------------------------------------------------
# 2. Bộ điều kiện mặc định (khớp với đề xuất ở phần phân tích trước:
#    ablation 0/15/30/45 độ cho góc quay)
# ---------------------------------------------------------------------------
def default_conditions(
    view_angles_deg: Sequence[float] = (0.0, 15.0, 30.0, 45.0),
    scale_factors: Sequence[float] = (0.7, 1.0, 1.5),
    noise_stds_px: Sequence[float] = (0.0, 2.0, 5.0),
) -> List[RobustnessCondition]:
    conditions: List[RobustnessCondition] = []
    for angle in view_angles_deg:
        conditions.append(RobustnessCondition(
            name=f"view_angle_{angle:g}deg", category="view_angle", param=angle,
            transform=lambda seq, a=angle: apply_view_angle_distortion(seq, a),
        ))
    for scale in scale_factors:
        conditions.append(RobustnessCondition(
            name=f"scale_{scale:g}x", category="scale", param=scale,
            transform=lambda seq, s=scale: apply_scale(seq, s),
        ))
    for noise in noise_stds_px:
        conditions.append(RobustnessCondition(
            name=f"noise_proxy_{noise:g}px", category="noise_proxy", param=noise,
            transform=lambda seq, n=noise: apply_gaussian_noise_proxy(seq, n),
        ))
    return conditions


# ---------------------------------------------------------------------------
# 3. Điều phối chạy ablation + tính điểm cho từng điều kiện
# ---------------------------------------------------------------------------
class RobustnessEvaluator:
    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        scorer: Optional[Scorer] = None,
        drift_tolerance: float = 15.0,
    ):
        self.extractor = extractor or FeatureExtractor()
        self.scorer = scorer or Scorer()
        self.drift_tolerance = drift_tolerance

    def run(
        self,
        sequence: List[PoseFrame],
        fps: float,
        phase: str,
        reference_model: ReferenceModel,
        conditions: Optional[List[RobustnessCondition]] = None,
    ) -> pd.DataFrame:
        conditions = conditions or default_conditions()
        baseline_bundle, baseline_score = self._score_condition(sequence, fps, phase, reference_model)

        rows = [{
            "condition": "baseline", "category": "baseline", "param": 0.0,
            "overall_score": baseline_score, "valid_frame_ratio": baseline_bundle.valid_frame_ratio,
            "score_drift": 0.0, "exceeds_tolerance": False, "error": None,
        }]

        for cond in conditions:
            transformed = cond.transform(sequence)
            try:
                bundle, score = self._score_condition(transformed, fps, phase, reference_model)
            except ValueError as exc:
                rows.append({
                    "condition": cond.name, "category": cond.category, "param": cond.param,
                    "overall_score": np.nan, "valid_frame_ratio": np.nan,
                    "score_drift": np.nan, "exceeds_tolerance": True, "error": str(exc),
                })
                continue

            drift = (
                abs(score - baseline_score)
                if not (np.isnan(score) or np.isnan(baseline_score)) else np.nan
            )
            rows.append({
                "condition": cond.name, "category": cond.category, "param": cond.param,
                "overall_score": score, "valid_frame_ratio": bundle.valid_frame_ratio,
                "score_drift": drift,
                "exceeds_tolerance": bool(not np.isnan(drift) and drift > self.drift_tolerance),
                "error": None,
            })

        return pd.DataFrame(rows)

    def _score_condition(self, sequence, fps, phase, reference_model):
        bundle = self.extractor.extract(sequence, fps=fps, phase=phase)
        report = self.scorer.score(bundle, reference_model)
        return bundle, report.overall_score
