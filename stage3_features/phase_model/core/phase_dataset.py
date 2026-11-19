"""
stage3_features/phase_dataset.py
==================================
Mục đích : MỘT cửa duy nhất nạp (pose cache + nhãn pha) của từng clip cho
           huấn luyện, đánh giá và tự động đánh dấu.

Tự chứa: có bản đọc pose cache riêng (hỗ trợ cả định dạng list lẫn dict
{"fps","joint_names","frames":[...]}) nên KHÔNG cần import
stage5_eval/loso_cv_eval.py (tránh kéo theo phụ thuộc không liên quan).

Quy ước:
    - pose cache : <pose_cache_dir>/<stem>.json   (stem = tên video không đuôi)
    - nhãn       : <phases_dir>/<stem>.phases.json
    - Nhóm       : stem bắt đầu 'coach_' -> 'coach'; 'trainee_' -> 'trainee'.
    - Người      : mặc định lấy từ stem (coach_HLV01_... -> 'HLV01'). Nếu một
                   người có nhiều clip với tên khác nhau, khai báo trong
                   <data>/phase_groups.json: {"<stem>": "<person_id>"} — dùng
                   để chia leave-one-PERSON-out đúng (tránh rò rỉ giữa clip).
    - Frame placeholder (pose không detect, xy = NaN) được GIỮ NGUYÊN trong
      danh sách pose để vị trí khớp frame_idx.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.phase_schema import PhaseSet, Rep, load_phases
from core.pose_schema import COCO17_NAMES, PoseFrame
from stage3_features.phase_model.core.phase_labels import IGNORE_LABEL, rep_to_frame_labels

_PERSON_RE = re.compile(r"^(?:coach|trainee)_([A-Za-z]+\d+)", re.IGNORECASE)


def group_of(stem: str) -> str:
    s = stem.lower()
    if s.startswith("coach_"):
        return "coach"
    if s.startswith("trainee_"):
        return "trainee"
    return "unknown"


def person_of(stem: str, groups_map: Optional[Dict[str, str]] = None) -> str:
    if groups_map and stem in groups_map:
        return str(groups_map[stem])
    m = _PERSON_RE.match(stem)
    return f"{group_of(stem)}:{m.group(1)}" if m else stem


def read_pose_cache(path: str | Path) -> Tuple[List[PoseFrame], float]:
    """Đọc pose cache -> (poses, fps). fps = NaN nếu không xác định được."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        frames = d
        names = (frames[0].get("joint_names") if frames else None) or COCO17_NAMES
    elif isinstance(d, dict) and "frames" in d:
        frames = d["frames"]
        names = d.get("joint_names") or COCO17_NAMES
    else:
        raise ValueError(f"Định dạng pose cache không hợp lệ: {path}")
    poses = [
        PoseFrame(
            frame_idx=int(fr["frame_idx"]), timestamp_s=float(fr["timestamp_s"]),
            xy=np.asarray(fr["xy"], dtype=np.float32),
            conf=np.asarray(fr["conf"], dtype=np.float32), joint_names=names,
        )
        for fr in frames
    ]
    fps = d.get("fps") if isinstance(d, dict) else None
    if not fps and len(poses) >= 2:
        dts = np.diff([p.timestamp_s for p in poses])
        dts = dts[dts > 0]
        fps = 1.0 / float(np.median(dts)) if len(dts) else float("nan")
    return poses, float(fps) if fps else float("nan")


@dataclass
class ClipData:
    stem: str
    poses: List[PoseFrame]
    fps: float
    phaseset: PhaseSet
    group: str
    person: str
    front_side_override: Optional[str] = None
    _labels: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def rep(self) -> Rep:
        return self.phaseset.reps[0]

    @property
    def total_frames(self) -> int:
        return self.phaseset.total_frames

    @property
    def labels(self) -> np.ndarray:
        """Nhãn 6 lớp cho MỌI frame 0..total_frames-1 (IGNORE ở khe hở)."""
        if self._labels is None:
            self._labels = rep_to_frame_labels(self.rep, self.total_frames)
        return self._labels

    def pose_labels(self) -> np.ndarray:
        """Nhãn tại đúng frame_idx của từng pose (độ dài = len(poses))."""
        lab = self.labels
        idx = np.array([p.frame_idx for p in self.poses], dtype=int)
        out = np.full(len(idx), IGNORE_LABEL, dtype=np.int8)
        ok = (idx >= 0) & (idx < len(lab))
        out[ok] = lab[idx[ok]]
        return out

    def nan_ratio(self) -> float:
        if not self.poses:
            return 1.0
        return float(np.mean([not np.isfinite(p.xy).all() for p in self.poses]))


def load_groups_map(path: str | Path | None) -> Dict[str, str]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f).items()}
    return {}


def list_clips(phases_dir: str | Path, pose_cache_dir: str | Path
               ) -> Tuple[List[str], List[str], List[str]]:
    """-> (stems có đủ cả hai, stems thiếu pose cache, stems có cache nhưng thiếu nhãn)."""
    phases_dir, cache_dir = Path(phases_dir), Path(pose_cache_dir)
    with_phases = {p.name[:-len(".phases.json")] for p in phases_dir.glob("*.phases.json")}
    with_cache = {p.stem for p in cache_dir.glob("*.json") if p.stem != "video_meta"}
    ok = sorted(with_phases & with_cache)
    return ok, sorted(with_phases - with_cache), sorted(with_cache - with_phases)


def load_clip(stem: str, phases_dir: str | Path, pose_cache_dir: str | Path,
              groups_map: Optional[Dict[str, str]] = None) -> ClipData:
    ps = load_phases(Path(phases_dir) / f"{stem}.phases.json")
    poses, fps = read_pose_cache(Path(pose_cache_dir) / f"{stem}.json")
    if not np.isfinite(fps) or fps <= 0:
        fps = float(ps.fps)
    override = ps.extra.get("front_leg_override")
    return ClipData(
        stem=stem, poses=poses, fps=float(fps), phaseset=ps, group=group_of(stem),
        person=person_of(stem, groups_map),
        front_side_override=override if override in ("left", "right") else None,
    )


def load_all(phases_dir: str | Path, pose_cache_dir: str | Path,
             groups_map: Optional[Dict[str, str]] = None,
             group: str = "all") -> List[ClipData]:
    stems, _, _ = list_clips(phases_dir, pose_cache_dir)
    clips = [load_clip(s, phases_dir, pose_cache_dir, groups_map) for s in stems]
    return [c for c in clips if group == "all" or c.group == group]
