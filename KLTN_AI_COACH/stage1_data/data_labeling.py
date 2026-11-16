"""
stage1_data/data_labeling.py
=============================
Mục đích : Hỗ trợ HLV gán nhãn điểm/lỗi cho video học viên (ground truth
           dùng để đánh giá độ tin cậy hệ thống ở score_validity_eval.py,
           Giai đoạn 5), và gán nhãn "đúng/sai chủ đích" ngay từ khi thu
           thập dữ liệu HLV (Giai đoạn 1, Vấn đề "chỉ có mẫu dương").
Input    : video_id/subject_id do dataset_builder.py quản lý.
Output   : File nhãn (JSON Lines) làm ground truth cho:
             - score_validity_eval.py (Giai đoạn 5): so điểm hệ thống vs HLV
             - reference_model.py (Giai đoạn 4): khoảng chuẩn từ video HLV

Thiết kế: một Annotation ứng với MỘT lượt quay (video). Vì một học viên có
thể có NHIỀU HLV chấm (đo độ đồng thuận giữa các HLV - xem ICC ở Giai đoạn
5), annotator_id là một phần khóa, không ghi đè lẫn nhau.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

VALID_ROLES = {"coach", "trainee"}
VALID_PHASES = {"static", "dynamic", "combined"}
SCORE_MIN, SCORE_MAX = 0, 100


class LabelValidationError(ValueError):
    """Ném ra khi một nhãn không hợp lệ (điểm ngoài khoảng, thiếu trường...)."""


@dataclass
class Annotation:
    """Một bản ghi nhãn của HLV cho một video."""

    video_id: str
    subject_id: str
    subject_role: str  # "coach" | "trainee"
    sport: str  # ví dụ "running"
    annotator_id: str  # HLV nào chấm (để tính ICC/Kappa giữa các HLV)
    phase: str  # "static" | "dynamic" | "combined"
    overall_score: float  # 0-100
    phase_scores: Dict[str, float] = field(default_factory=dict)  # vd {"knee_lift": 78}
    error_tags: List[str] = field(default_factory=list)  # vd ["knee_lift_low", "body_lean_excessive"]
    is_deliberately_incorrect: bool = False  # đánh dấu mẫu "sai chủ đích" (Giai đoạn 1)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        errors: List[str] = []
        if self.subject_role not in VALID_ROLES:
            errors.append(f"subject_role phải thuộc {VALID_ROLES}, nhận '{self.subject_role}'")
        if self.phase not in VALID_PHASES:
            errors.append(f"phase phải thuộc {VALID_PHASES}, nhận '{self.phase}'")
        if not (SCORE_MIN <= self.overall_score <= SCORE_MAX):
            errors.append(f"overall_score phải trong [{SCORE_MIN},{SCORE_MAX}], nhận {self.overall_score}")
        for k, v in self.phase_scores.items():
            if not (SCORE_MIN <= v <= SCORE_MAX):
                errors.append(f"phase_scores['{k}'] phải trong [{SCORE_MIN},{SCORE_MAX}], nhận {v}")
        if not self.video_id or not self.subject_id or not self.annotator_id:
            errors.append("video_id, subject_id, annotator_id không được để trống")
        if errors:
            raise LabelValidationError("; ".join(errors))

    def to_dict(self) -> Dict:
        return asdict(self)


class LabelStore:
    """Lưu trữ nhãn dạng JSON Lines (mỗi dòng 1 Annotation) — dễ append,
    dễ đọc theo streaming, và dễ diff bằng git so với 1 file JSON khổng lồ.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, annotation: Annotation, allow_duplicate: bool = False) -> None:
        """Thêm 1 nhãn mới. Mặc định chặn trùng (video_id, annotator_id, phase)
        để tránh 1 HLV vô tình chấm lặp lại cùng 1 video/pha."""
        annotation.validate()
        if not allow_duplicate:
            for existing in self.iter_all():
                if (
                    existing.video_id == annotation.video_id
                    and existing.annotator_id == annotation.annotator_id
                    and existing.phase == annotation.phase
                ):
                    raise LabelValidationError(
                        f"Nhãn đã tồn tại cho video_id={annotation.video_id}, "
                        f"annotator_id={annotation.annotator_id}, phase={annotation.phase}. "
                        "Dùng allow_duplicate=True nếu muốn ghi đè/thêm bản mới."
                    )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")

    def iter_all(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield Annotation(**json.loads(line))

    def to_dataframe(self) -> pd.DataFrame:
        rows = [a.to_dict() for a in self.iter_all()]
        return pd.DataFrame(rows)

    def query(
        self,
        video_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        annotator_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[Annotation]:
        results = []
        for a in self.iter_all():
            if video_id and a.video_id != video_id:
                continue
            if subject_id and a.subject_id != subject_id:
                continue
            if annotator_id and a.annotator_id != annotator_id:
                continue
            if phase and a.phase != phase:
                continue
            results.append(a)
        return results

    def inter_annotator_agreement(self, video_id: str, phase: str) -> Optional[float]:
        """Tính độ lệch chuẩn giữa các HLV cho cùng 1 video/pha - một chỉ số
        thô ban đầu để phát hiện bất đồng lớn giữa các HLV trước khi tính
        ICC/Kappa đầy đủ ở score_validity_eval.py (Giai đoạn 5)."""
        scores = [a.overall_score for a in self.query(video_id=video_id, phase=phase)]
        if len(scores) < 2:
            return None
        return float(pd.Series(scores).std())


# ---------------------------------------------------------------------------
# Tiện ích nhập nhãn nhanh qua dòng lệnh (CLI tối giản cho HLV)
# ---------------------------------------------------------------------------
def label_from_cli(store: LabelStore) -> Annotation:  # pragma: no cover - I/O tương tác
    """Hỏi lần lượt các trường qua input() — dùng khi HLV chấm trực tiếp,
    không cần giao diện đồ họa riêng."""
    video_id = input("Video ID: ").strip()
    subject_id = input("Subject ID: ").strip()
    subject_role = input("Vai trò (coach/trainee): ").strip()
    sport = input("Môn thể thao: ").strip()
    annotator_id = input("Mã HLV chấm: ").strip()
    phase = input("Giai đoạn (static/dynamic/combined): ").strip()
    overall_score = float(input("Điểm tổng (0-100): ").strip())
    tags_raw = input("Lỗi kỹ thuật (phân cách bởi dấu phẩy, để trống nếu không có): ").strip()
    error_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    deliberate = input("Đây có phải mẫu sai chủ đích? (y/n): ").strip().lower() == "y"
    notes = input("Ghi chú thêm: ").strip()

    annotation = Annotation(
        video_id=video_id,
        subject_id=subject_id,
        subject_role=subject_role,
        sport=sport,
        annotator_id=annotator_id,
        phase=phase,
        overall_score=overall_score,
        error_tags=error_tags,
        is_deliberately_incorrect=deliberate,
        notes=notes,
    )
    store.add(annotation)
    return annotation
