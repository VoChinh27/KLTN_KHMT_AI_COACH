"""
stage1_data/dataset_builder.py
===============================
Mục đích : Tổ chức dữ liệu thô (video HLV + video học viên + nhãn) thành
           một manifest có cấu trúc, chia HLV/học viên, mẫu đúng/sai, và
           chia tập train/test THEO NGƯỜI (leave-one-subject-out) để tránh
           rò rỉ dữ liệu giữa các giai đoạn sau (đã nhấn mạnh ở Giai đoạn 5).
Input    : Thư mục video thô (đặt tên theo quy ước, xem `scan_raw_videos`)
           + LabelStore (từ data_labeling.py).
Output   : Manifest (pandas DataFrame + file CSV/JSON) gồm các cột:
           video_path, video_id, subject_id, subject_role, sport, split,
           is_deliberately_incorrect, overall_score
           -> là "nguồn sự thật" duy nhất cho pose_extractor.py và các
           bước eval ở Giai đoạn 5 tham chiếu tới.

Quy ước đặt tên file được kỳ vọng (có thể tùy biến qua regex khi cần):
    <subject_role>_<subject_id>_<sport>_<video_id>.mp4
    ví dụ: coach_HLV01_running_v001.mp4
           trainee_SV042_running_v017.mp4
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from stage1_data.data_labeling import LabelStore

_FILENAME_PATTERN = re.compile(
    r"^(?P<role>coach|trainee)_(?P<subject_id>[A-Za-z0-9]+)_(?P<sport>[A-Za-z0-9]+)_(?P<video_id>[A-Za-z0-9]+)\.(mp4|mov|avi)$",
    re.IGNORECASE,
)


@dataclass
class DatasetBuilderConfig:
    raw_video_dir: str
    label_store_path: str
    manifest_out_path: str
    test_ratio: float = 0.2
    random_seed: int = 42


def scan_raw_videos(raw_video_dir: str) -> pd.DataFrame:
    """Quét thư mục video thô, parse metadata từ TÊN FILE theo quy ước.

    File không khớp quy ước sẽ bị bỏ qua và liệt kê ra cảnh báo, thay vì
    làm crash toàn bộ quá trình xây dựng manifest.
    """
    rows: List[Dict] = []
    skipped: List[str] = []
    for path in sorted(Path(raw_video_dir).glob("*")):
        if not path.is_file():
            continue
        m = _FILENAME_PATTERN.match(path.name)
        if not m:
            skipped.append(path.name)
            continue
        rows.append(
            {
                "video_path": str(path),
                "video_id": m.group("video_id"),
                "subject_id": m.group("subject_id"),
                "subject_role": m.group("role").lower(),
                "sport": m.group("sport").lower(),
            }
        )
    if skipped:
        print(
            f"[dataset_builder] Bỏ qua {len(skipped)} file không đúng quy ước đặt tên: "
            f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
        )
    return pd.DataFrame(rows)


def merge_with_labels(videos_df: pd.DataFrame, label_store: LabelStore) -> pd.DataFrame:
    """Ghép thông tin nhãn (điểm tổng trung bình giữa các HLV, cờ mẫu sai
    chủ đích) vào manifest video. Video CHƯA có nhãn vẫn được giữ lại (cột
    điểm sẽ là NaN) để không âm thầm làm mất dữ liệu — nhưng được đánh dấu
    is_labeled=False để lọc riêng khi cần.
    """
    labels_df = label_store.to_dataframe()
    if labels_df.empty:
        merged = videos_df.copy()
        merged["overall_score"] = np.nan
        merged["is_deliberately_incorrect"] = False
        merged["is_labeled"] = False
        return merged

    agg = (
        labels_df.groupby("video_id")
        .agg(
            overall_score=("overall_score", "mean"),
            is_deliberately_incorrect=("is_deliberately_incorrect", "max"),
            num_annotators=("annotator_id", "nunique"),
        )
        .reset_index()
    )
    merged = videos_df.merge(agg, on="video_id", how="left")
    merged["is_labeled"] = merged["overall_score"].notna()
    return merged


def split_leave_one_subject_out(
    manifest: pd.DataFrame, random_seed: int = 42
) -> Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Generator chia tập theo kiểu leave-one-subject-out: mỗi lượt giữ lại
    ĐÚNG 1 subject_id (học viên) làm test, phần còn lại làm train. Tránh
    tuyệt đối việc cùng 1 người vừa xuất hiện ở train vừa ở test — nguyên
    tắc bắt buộc đã nêu ở Giai đoạn 5 (score_validity_eval / robustness_eval).

    Chỉ áp dụng leave-one-out trên subject_role == "trainee"; video của HLV
    (coach) luôn ở trong train vì đóng vai trò xây dựng khoảng chuẩn
    (reference_model.py), không phải đối tượng được đánh giá.
    """
    trainees = sorted(manifest.loc[manifest.subject_role == "trainee", "subject_id"].unique())
    coach_rows = manifest[manifest.subject_role == "coach"]

    for held_out in trainees:
        test_df = manifest[(manifest.subject_role == "trainee") & (manifest.subject_id == held_out)]
        train_df = pd.concat(
            [
                coach_rows,
                manifest[(manifest.subject_role == "trainee") & (manifest.subject_id != held_out)],
            ],
            ignore_index=True,
        )
        yield held_out, train_df, test_df


def split_train_test_by_subject(
    manifest: pd.DataFrame, test_ratio: float = 0.2, random_seed: int = 42
) -> pd.DataFrame:
    """Gán cột 'split' (train/test) MỘT LẦN theo subject_id (không phải theo
    frame/video riêng lẻ), dùng cho báo cáo tổng thể khi không cần lặp đủ
    leave-one-subject-out (ví dụ khi số lượng trainee còn ít trong giai đoạn
    đầu của đồ án).
    """
    manifest = manifest.copy()
    rng = np.random.default_rng(random_seed)

    trainees = sorted(manifest.loc[manifest.subject_role == "trainee", "subject_id"].unique())
    n_test = max(1, int(round(len(trainees) * test_ratio))) if trainees else 0
    test_subjects = set(rng.choice(trainees, size=n_test, replace=False)) if n_test else set()

    def _assign(row):
        if row["subject_role"] == "coach":
            return "train"  # video HLV luôn dùng để xây khoảng chuẩn
        return "test" if row["subject_id"] in test_subjects else "train"

    manifest["split"] = manifest.apply(_assign, axis=1)
    return manifest


class DatasetBuilder:
    """Điều phối toàn bộ bước xây dựng manifest, từ quét video thô đến xuất
    file manifest cuối cùng cho pose_extractor.py (Giai đoạn 2) sử dụng.
    """

    def __init__(self, config: DatasetBuilderConfig):
        self.config = config
        self.label_store = LabelStore(config.label_store_path)

    def build(self) -> pd.DataFrame:
        videos_df = scan_raw_videos(self.config.raw_video_dir)
        if videos_df.empty:
            raise FileNotFoundError(
                f"Không tìm thấy video hợp lệ trong {self.config.raw_video_dir}. "
                "Kiểm tra lại quy ước đặt tên file."
            )
        merged = merge_with_labels(videos_df, self.label_store)
        manifest = split_train_test_by_subject(
            merged, test_ratio=self.config.test_ratio, random_seed=self.config.random_seed
        )
        self._validate_manifest(manifest)
        return manifest

    @staticmethod
    def _validate_manifest(manifest: pd.DataFrame) -> None:
        """Kiểm tra bất biến quan trọng nhất: không subject nào vừa ở train
        vừa ở test (rò rỉ dữ liệu) — nếu vi phạm thì raise ngay lập tức."""
        by_subject_splits = manifest.groupby("subject_id")["split"].nunique()
        leaked = by_subject_splits[by_subject_splits > 1]
        if not leaked.empty:
            raise AssertionError(
                f"Rò rỉ dữ liệu: các subject_id sau xuất hiện ở CẢ train và test: "
                f"{list(leaked.index)}"
            )

    def save_manifest(self, manifest: pd.DataFrame) -> None:
        out_path = Path(self.config.manifest_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".json":
            manifest.to_json(out_path, orient="records", force_ascii=False, indent=2)
        else:
            manifest.to_csv(out_path, index=False)

    def summary(self, manifest: pd.DataFrame) -> Dict:
        return {
            "total_videos": int(len(manifest)),
            "num_coach_videos": int((manifest.subject_role == "coach").sum()),
            "num_trainee_videos": int((manifest.subject_role == "trainee").sum()),
            "num_trainee_subjects": int(manifest.loc[manifest.subject_role == "trainee", "subject_id"].nunique()),
            "num_labeled": int(manifest.get("is_labeled", pd.Series(dtype=bool)).sum()),
            "num_deliberately_incorrect": int(manifest.get("is_deliberately_incorrect", pd.Series(dtype=bool)).sum()),
            "split_counts": manifest["split"].value_counts().to_dict(),
        }
