"""
stage5_eval/report_builder.py
================================
Mục đích : Tổng hợp kết quả từ pose_accuracy_eval.py, score_validity_eval.py
           và robustness_eval.py thành MỘT báo cáo (Markdown + bảng CSV +
           biểu đồ PNG tùy chọn) — sản phẩm đầu ra cuối cùng của Giai đoạn 5,
           dùng trực tiếp khi viết chương "Đánh giá thực nghiệm" của báo cáo
           đồ án hoặc khi trình bày trước hội đồng.
Input    : PoseAccuracyReport + ScoreValidityReport + robustness DataFrame
           (từ 3 file eval).
Output   : File .md (bảng dạng Markdown) + tùy chọn các file .png (biểu đồ)
           trong 1 thư mục do người dùng chỉđịnh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stage5_eval.pose_accuracy_eval import PoseAccuracyReport
from stage5_eval.score_validity_eval import ScoreValidityReport

try:
    import matplotlib
    matplotlib.use("Agg")  # không cần màn hình hiển thị, chỉ xuất file ảnh
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _df_to_markdown(df: pd.DataFrame, float_format: str = "{:.2f}") -> str:
    """Chuyển DataFrame thành bảng Markdown. Dùng pandas.to_markdown() nếu có
    (cần gói `tabulate`), nếu không thì tự dựng bảng thủ công — không để
    thiếu 1 gói phụ làm hỏng cả báo cáo."""
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_float_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: "" if pd.isna(v) else float_format.format(v))
    try:
        return df_fmt.to_markdown(index=False)
    except ImportError:
        headers = list(df_fmt.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df_fmt.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
        return "\n".join(lines)


@dataclass
class EvaluationReport:
    sport: str
    phase: str
    pose_accuracy: Optional[PoseAccuracyReport] = None
    score_validity: Optional[ScoreValidityReport] = None
    robustness: Optional[pd.DataFrame] = None


class ReportBuilder:
    def __init__(self):
        self._report: Optional[EvaluationReport] = None

    def start(self, sport: str, phase: str) -> "ReportBuilder":
        self._report = EvaluationReport(sport=sport, phase=phase)
        return self

    def add_pose_accuracy(self, report: PoseAccuracyReport) -> "ReportBuilder":
        self._ensure_started()
        self._report.pose_accuracy = report
        return self

    def add_score_validity(self, report: ScoreValidityReport) -> "ReportBuilder":
        self._ensure_started()
        self._report.score_validity = report
        return self

    def add_robustness(self, df: pd.DataFrame) -> "ReportBuilder":
        self._ensure_started()
        self._report.robustness = df
        return self

    def _ensure_started(self) -> None:
        if self._report is None:
            raise RuntimeError("Gọi start(sport, phase) trước khi thêm kết quả eval")

    # ------------------------------------------------------------------
    def build_markdown(self) -> str:
        self._ensure_started()
        r = self._report
        lines = [f"# Báo cáo đánh giá thực nghiệm — {r.sport} ({r.phase})", ""]

        if r.pose_accuracy is not None:
            pa = r.pose_accuracy
            lines += [
                "## 1. Độ chính xác Pose Estimation",
                "",
                f"- Số frame được gán nhãn tay để đối chiếu: **{pa.n_frames_evaluated}**",
                f"- PCK tổng thể (alpha={pa.alpha}): **{pa.overall_pck * 100:.1f}%**",
                f"- OKS trung bình: **{pa.mean_oks:.3f}**",
                "",
                "### PCK theo từng khớp",
                "",
                _df_to_markdown(pa.to_dataframe().rename(columns={
                    "joint": "Khớp", "pck": "PCK", "n_evaluated": "Số frame đánh giá",
                })),
                "",
            ]
        else:
            lines += ["## 1. Độ chính xác Pose Estimation", "", "_(chưa có dữ liệu)_", ""]

        if r.score_validity is not None:
            sv = r.score_validity
            ceiling_note = ""
            meets = sv.meets_human_level_reliability()
            if meets is not None:
                ceiling_note = (
                    "- Hệ thống **đạt** mức tin cậy tương đương giữa các HLV (≥90% mốc trần)."
                    if meets else
                    "- Hệ thống **CHƯA đạt** mức tin cậy tương đương giữa các HLV (dưới 90% mốc trần)."
                )
            lines += [
                "## 2. Độ tin cậy của điểm chấm (so với nhãn HLV)",
                "",
                f"- Số video đối chiếu: **{sv.n_videos}**",
                f"- MAE: **{sv.mae_system_vs_coach_mean:.2f}** điểm | RMSE: **{sv.rmse_system_vs_coach_mean:.2f}** điểm",
                f"- Tương quan Spearman: **{sv.spearman_corr:.3f}** (p={sv.spearman_pvalue:.3f})",
                f"- ICC(hệ thống, HLV) [{sv.icc_type}]: "
                f"**{sv.icc_system_vs_coach:.3f}**" if sv.icc_system_vs_coach is not None else
                "- ICC(hệ thống, HLV): _không đủ dữ liệu_",
                f"- ICC(giữa các HLV, mốc trần): "
                f"**{sv.icc_inter_coach_only:.3f}**" if sv.icc_inter_coach_only is not None else
                "- ICC(giữa các HLV, mốc trần): _không đủ dữ liệu (cần ≥2 HLV/video)_",
            ]
            if ceiling_note:
                lines.append(ceiling_note)
            lines += [
                "",
                "### Chi tiết theo từng video",
                "",
                _df_to_markdown(sv.per_video_table.rename(columns={
                    "video_id": "Video", "system_score": "Điểm hệ thống",
                    "coach_mean": "Điểm HLV (TB)", "coach_std": "Độ lệch giữa các HLV",
                    "n_coaches": "Số HLV chấm", "abs_error": "Sai số tuyệt đối",
                })),
                "",
            ]
        else:
            lines += ["## 2. Độ tin cậy của điểm chấm (so với nhãn HLV)", "", "_(chưa có dữ liệu)_", ""]

        if r.robustness is not None:
            rb = r.robustness
            worst = rb.loc[rb["score_drift"].idxmax()] if rb["score_drift"].notna().any() else None
            lines += [
                "## 3. Độ bền vững (Robustness)",
                "",
                "_Lưu ý: ablation góc quay/khoảng cách mô phỏng trực tiếp trên keypoint; "
                "điều kiện 'noise_proxy' là proxy cho nhiễu do ánh sáng/che khuất, "
                "KHÔNG mô phỏng trực tiếp ánh sáng/trang phục._",
                "",
                _df_to_markdown(rb.drop(columns=["error"], errors="ignore").rename(columns={
                    "condition": "Điều kiện", "category": "Loại", "param": "Tham số",
                    "overall_score": "Điểm tổng", "valid_frame_ratio": "Tỉ lệ frame hợp lệ",
                    "score_drift": "Độ lệch so với baseline", "exceeds_tolerance": "Vượt ngưỡng cho phép",
                })),
                "",
            ]
            if worst is not None:
                lines.append(
                    f"- Điều kiện gây lệch điểm nhiều nhất: **{worst['condition']}** "
                    f"(lệch {worst['score_drift']:.1f} điểm so với baseline)."
                )
            lines.append("")
        else:
            lines += ["## 3. Độ bền vững (Robustness)", "", "_(chưa có dữ liệu)_", ""]

        return "\n".join(lines)

    def save_markdown(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.build_markdown(), encoding="utf-8")

    # ------------------------------------------------------------------
    def save_charts(self, output_dir: str) -> None:
        """Xuất biểu đồ PNG minh họa (yêu cầu matplotlib — nếu không có,
        bỏ qua và ghi log thay vì làm crash toàn bộ báo cáo)."""
        self._ensure_started()
        if plt is None:
            print("[report_builder] Bỏ qua sinh biểu đồ: chưa cài matplotlib.")
            return
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self._report

        if r.pose_accuracy is not None:
            df = r.pose_accuracy.to_dataframe()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df["joint"], df["pck"] * 100)
            ax.set_ylabel("PCK (%)")
            ax.set_title(f"PCK theo từng khớp (alpha={r.pose_accuracy.alpha})")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "pose_accuracy_pck.png", dpi=150)
            plt.close(fig)

        if r.score_validity is not None:
            table = r.score_validity.per_video_table
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(table["coach_mean"], table["system_score"])
            lims = [0, 100]
            ax.plot(lims, lims, "--", color="gray", linewidth=1)
            ax.set_xlabel("Điểm HLV (trung bình)")
            ax.set_ylabel("Điểm hệ thống")
            ax.set_title("Điểm hệ thống vs. điểm HLV")
            fig.tight_layout()
            fig.savefig(out_dir / "score_validity_scatter.png", dpi=150)
            plt.close(fig)

        if r.robustness is not None:
            rb = r.robustness
            fig, ax = plt.subplots(figsize=(7, 4))
            for category in rb["category"].unique():
                if category == "baseline":
                    continue
                sub = rb[rb["category"] == category].sort_values("param")
                ax.plot(sub["param"], sub["overall_score"], marker="o", label=category)
            baseline_score = rb.loc[rb["condition"] == "baseline", "overall_score"]
            if not baseline_score.empty:
                ax.axhline(float(baseline_score.iloc[0]), color="gray", linestyle="--", linewidth=1, label="baseline")
            ax.set_xlabel("Tham số điều kiện")
            ax.set_ylabel("Điểm tổng")
            ax.set_title("Độ bền vững theo từng loại điều kiện")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "robustness_ablation.png", dpi=150)
            plt.close(fig)
