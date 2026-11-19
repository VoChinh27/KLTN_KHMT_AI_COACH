"""
stage5_eval/score_validity_eval.py
=====================================
Mục đích : Đánh giá độ TIN CẬY của điểm chấm hệ thống (scorer.py, Giai đoạn
           4) bằng cách so với nhãn HLV (data_labeling.py, Giai đoạn 1):
           MAE, tương quan Spearman, và ICC (Intraclass Correlation).
Input    : List[VideoScoreRecord] — mỗi video gồm điểm hệ thống + danh sách
           điểm của (các) HLV đã chấm video đó.
Output   : ScoreValidityReport -> đưa vào report_builder.py.

Điểm quan trọng đã thống nhất ở Giai đoạn 5: "nếu có ≥2 HLV chấm thì tính
thêm độ đồng thuận GIỮA CÁC HLV làm mốc trần" — vì hệ thống không thể
"đúng hơn con người" một cách vô nghĩa; nó chỉ có thể tiệm cận mức đồng
thuận mà chính các HLV đạt được với nhau. Module này tính CẢ HAI:
    icc_system_vs_coach : hệ thống có "hòa hợp" với HLV không.
    icc_inter_coach_only: các HLV có hòa hợp với NHAU không (mốc trần).
Nếu icc_system_vs_coach vượt hoặc gần bằng icc_inter_coach_only, hệ thống
đã đạt mức tin cậy tương đương con người — một kết luận có sức thuyết phục
cao khi bảo vệ đồ án.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class VideoScoreRecord:
    video_id: str
    system_score: float
    coach_scores: List[float]  # điểm của (các) HLV đã chấm video này

    @property
    def coach_mean(self) -> float:
        return float(np.mean(self.coach_scores)) if self.coach_scores else float("nan")

    @property
    def coach_std(self) -> float:
        return float(np.std(self.coach_scores)) if len(self.coach_scores) > 1 else 0.0


@dataclass
class ScoreValidityReport:
    n_videos: int
    mae_system_vs_coach_mean: float
    rmse_system_vs_coach_mean: float
    spearman_corr: float
    spearman_pvalue: float
    icc_system_vs_coach: Optional[float]
    icc_inter_coach_only: Optional[float]
    icc_type: str
    n_raters_used_for_icc: int
    n_videos_used_for_icc: int
    per_video_table: pd.DataFrame

    def summary_dict(self) -> Dict:
        return {
            "n_videos": self.n_videos,
            "mae": self.mae_system_vs_coach_mean,
            "rmse": self.rmse_system_vs_coach_mean,
            "spearman_corr": self.spearman_corr,
            "spearman_pvalue": self.spearman_pvalue,
            "icc_system_vs_coach": self.icc_system_vs_coach,
            "icc_inter_coach_only": self.icc_inter_coach_only,
        }

    def meets_human_level_reliability(self) -> Optional[bool]:
        """True nếu ICC(hệ thống, HLV) đạt tối thiểu 90% mức đồng thuận giữa
        chính các HLV (mốc trần) — None nếu không đủ dữ liệu để so sánh."""
        if self.icc_system_vs_coach is None or self.icc_inter_coach_only is None:
            return None
        if self.icc_inter_coach_only <= 0:
            return None
        return bool(self.icc_system_vs_coach >= 0.9 * self.icc_inter_coach_only)


# ---------------------------------------------------------------------------
# ICC (Intraclass Correlation Coefficient) — cài đặt thuần numpy, công thức
# Shrout & Fleiss (1979).
# ---------------------------------------------------------------------------
def icc(ratings: np.ndarray, icc_type: str = "ICC2_1") -> float:
    """Tính ICC cho ma trận (n_subjects, n_raters), loại bỏ (listwise) các
    dòng có giá trị NaN.

    icc_type:
        "ICC2_1" (mặc định) : two-way random effects, absolute agreement —
            phù hợp khi các "rater" (kể cả hệ thống) được coi là mẫu ngẫu
            nhiên từ một quần thể rater, và ta quan tâm liệu điểm số có
            KHỚP TUYỆT ĐỐI hay không (không chỉ tương quan thứ hạng).
        "ICC3_1" : two-way mixed effects, consistency — dùng khi tập rater
            là CỐ ĐỊNH (chính là những HLV/hệ thống này, không đại diện cho
            quần thể rộng hơn) và chỉ quan tâm tính nhất quán tương đối.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    valid_rows = ~np.isnan(ratings).any(axis=1)
    ratings = ratings[valid_rows]
    n, k = ratings.shape
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")

    if icc_type == "ICC2_1":
        denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
        if abs(denom) < 1e-12:
            return float("nan")
        return float((ms_rows - ms_error) / denom)
    elif icc_type == "ICC3_1":
        denom = ms_rows + (k - 1) * ms_error
        if abs(denom) < 1e-12:
            return float("nan")
        return float((ms_rows - ms_error) / denom)
    else:
        raise ValueError("icc_type phải là 'ICC2_1' hoặc 'ICC3_1'")


def _build_rating_matrix(
    records: List[VideoScoreRecord], include_system: bool
) -> np.ndarray:
    """Chỉ dùng các video có ĐÚNG số lượng HLV bằng mode (giá trị phổ biến
    nhất) trong tập records, để có ma trận chữ nhật hợp lệ cho ICC. Video có
    số HLV khác mode sẽ bị loại (listwise), số lượng bị loại được báo cáo
    minh bạch qua n_videos_used_for_icc so với n_videos.
    """
    counts = [len(r.coach_scores) for r in records if len(r.coach_scores) > 0]
    if not counts:
        return np.empty((0, 0))
    mode_k = max(set(counts), key=counts.count)
    if mode_k < 2:
        return np.empty((0, 0))

    rows = []
    for r in records:
        if len(r.coach_scores) != mode_k:
            continue
        row = list(r.coach_scores)
        if include_system:
            row.append(r.system_score)
        rows.append(row)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))


class ScoreValidityEvaluator:
    def __init__(self, icc_type: str = "ICC2_1"):
        self.icc_type = icc_type

    def evaluate(self, records: List[VideoScoreRecord]) -> ScoreValidityReport:
        if not records:
            raise ValueError("records rỗng — không có video nào để đánh giá")

        rows = []
        for r in records:
            rows.append({
                "video_id": r.video_id,
                "system_score": r.system_score,
                "coach_mean": r.coach_mean,
                "coach_std": r.coach_std,
                "n_coaches": len(r.coach_scores),
                "abs_error": abs(r.system_score - r.coach_mean) if not np.isnan(r.coach_mean) else np.nan,
            })
        table = pd.DataFrame(rows)

        valid = table.dropna(subset=["coach_mean"])
        if valid.empty:
            raise ValueError("Không có video nào có điểm HLV hợp lệ để so sánh")

        mae = float(np.mean(valid["abs_error"]))
        rmse = float(np.sqrt(np.mean((valid["system_score"] - valid["coach_mean"]) ** 2)))
        if len(valid) >= 3:
            corr, pvalue = spearmanr(valid["system_score"], valid["coach_mean"])
            corr, pvalue = float(corr), float(pvalue)
        else:
            corr, pvalue = float("nan"), float("nan")

        matrix_with_system = _build_rating_matrix(records, include_system=True)
        matrix_coach_only = _build_rating_matrix(records, include_system=False)

        icc_with_system = icc(matrix_with_system, self.icc_type) if matrix_with_system.size else None
        icc_coach_only = icc(matrix_coach_only, self.icc_type) if matrix_coach_only.size else None
        n_raters_used = matrix_with_system.shape[1] if matrix_with_system.size else 0
        n_videos_used = matrix_with_system.shape[0] if matrix_with_system.size else 0

        return ScoreValidityReport(
            n_videos=len(records),
            mae_system_vs_coach_mean=mae,
            rmse_system_vs_coach_mean=rmse,
            spearman_corr=corr,
            spearman_pvalue=pvalue,
            icc_system_vs_coach=icc_with_system,
            icc_inter_coach_only=icc_coach_only,
            icc_type=self.icc_type,
            n_raters_used_for_icc=n_raters_used,
            n_videos_used_for_icc=n_videos_used,
            per_video_table=table,
        )
