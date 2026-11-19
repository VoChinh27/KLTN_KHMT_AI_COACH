"""
stage5_eval/loso_cv_eval.py
=============================
Mục đích : Đánh giá độ tin cậy của hệ thống chấm điểm bằng Leave-One-Subject-
           Out Cross-Validation (LOSO-CV), nhằm LOẠI BỎ data leakage:
           mỗi HLV được chấm bằng khoảng chuẩn học từ CÁC HLV KHÁC, không ai
           "tự chấm mình".

Hai quy trình được chạy song song để so sánh:
    NAIVE : reference build từ TẤT CẢ clip -> chấm lại chính các clip đó
            (circular, là mốc "có leakage").
    LOSO  : với mỗi HLV (fold), reference build chỉ từ clip của các HLV còn
            lại -> chấm clip của HLV bị giữ ra (holdout).

Output (thư mục `output_dir`, mặc định output/loso_cv/):
    loso_cv_per_hlv.csv, loso_cv_comparison.csv, loso_cv_summary.json,
    loso_cv_report.md, 3 biểu đồ PNG (nếu có matplotlib), run_info.txt.

Chạy CLI:
    python -m stage5_eval.loso_cv_eval --sport sprint_start

----------------------------------------------------------------------------
ĐIỂM TÍCH HỢP (đọc trước khi dùng)
----------------------------------------------------------------------------
Module này dựa trên các API ĐÃ BIẾT của dự án: FeatureExtractor.extract(),
FeatureBundle, ReferenceModelBuilder (build/build_fixed), FixedRangeSpec,
SportPlugin và Scorer. Một phần phụ thuộc vào mã mà module này KHÔNG được
phép giả định cứng nên được tách thành "đường nối" (seam):

  `PoseLoader` : đọc/trích xuất pose cho 1 clip -> (List[PoseFrame], fps).
        Mặc định `CachedPoseLoader` đọc cache JSON theo định dạng ghi ở
        `write_pose_cache()`. Khi KHÔNG có cache, truyền `--pose-extractor
        "pkg.module:ham"` (hàm nhận Path video, trả (List[PoseFrame], fps) đã
        qua tracker + signal_smoother) — thường là một wrapper mỏng quanh
        core.pipeline.Pipeline.

CHẤM ĐIỂM: dùng CHÍNH `stage4_scoring.scorer.Scorer` của pipeline, dựng qua
`stage4_scoring.scorer_factory.build_scorer(plugin, phase, fps)` (single source
of truth) nên điểm LOSO-CV khớp pipeline chính. Tên pha đưa vào Scorer
(SCORER_PHASE) phải trùng `bundle.phase` và `reference.phase`; riêng `vao_cho`
dùng PHASE_INTO_POSITION nên khoảng chuẩn nằm ở `reference.phase_ranges`.

Phần đọc YAML dùng yaml.safe_load thô (giống cách sprint_start.yaml đã nêu),
vì core.sport_interface chưa đọc `fixed_reference_ranges`/`phase_weights`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from core.pose_schema import PoseFrame
from core.sport_interface import SportPlugin, load_sport_plugin
from stage3_features.feature_extractor import (
    FeatureBundle, FeatureExtractor, PHASE_DRIVE, PHASE_DYNAMIC, PHASE_INTO_POSITION,
    PHASE_SPRINT_START, PHASE_STATIC,
)
from stage4_scoring.reference_model import (
    FixedRangeSpec, ReferenceModel, ReferenceModelBuilder,
)
from stage4_scoring.scorer_factory import build_scorer

logger = logging.getLogger("loso_cv_eval")

# ============================================================================
# Hằng số
# ============================================================================
EVAL_VAO_CHO = "vao_cho"
EVAL_STATIC = "static"
EVAL_SPRINT = "sprint_start"
EVAL_DRIVE = "drive_phase"
ALL_EVAL_PHASES = (EVAL_VAO_CHO, EVAL_STATIC, EVAL_SPRINT, EVAL_DRIVE)
DEFAULT_PHASES_TO_EVALUATE = (EVAL_VAO_CHO, EVAL_STATIC, EVAL_SPRINT)

# Tên pha "đánh giá" -> tên pha ENGINE (dùng cho FeatureExtractor.extract()).
# vao_cho dùng nhánh PHASE_INTO_POSITION (bundle.phase == "vao_cho"), tách khỏi
# static để khoảng chuẩn đi vào `phase_ranges` đúng như pipeline chính.
ENGINE_PHASE = {
    EVAL_VAO_CHO: PHASE_INTO_POSITION,
    EVAL_STATIC: PHASE_STATIC,
    EVAL_SPRINT: PHASE_SPRINT_START,
    EVAL_DRIVE: PHASE_DYNAMIC,
}
# Tên pha "đánh giá" -> tên pha SCORER/REFERENCE (khớp plugin.phases()). Dùng để
# build ReferenceModel và dựng Scorer; phải trùng bundle.phase khi chấm.
# Với vao_cho / static / sprint_start trùng ENGINE_PHASE; drive_phase khác (chấm
# bằng nhánh drive_phase, không phải dynamic).
SCORER_PHASE = {
    EVAL_VAO_CHO: PHASE_INTO_POSITION,
    EVAL_STATIC: PHASE_STATIC,
    EVAL_SPRINT: PHASE_SPRINT_START,
    EVAL_DRIVE: PHASE_DRIVE,
}
# Pha học khoảng chuẩn từ dữ liệu (có nguy cơ leakage) vs pha dùng YAML fixed.
STATISTICAL_PHASES = (EVAL_VAO_CHO, EVAL_STATIC)
FIXED_PHASES = (EVAL_SPRINT, EVAL_DRIVE)

DEFAULT_PHASE_WEIGHTS = {EVAL_VAO_CHO: 0.25, EVAL_STATIC: 0.30, EVAL_SPRINT: 0.25, EVAL_DRIVE: 0.0}

# Alias tên pha trong file *.phases.json -> tên pha đánh giá chuẩn.
PHASE_ALIASES = {
    "vao_cho": EVAL_VAO_CHO, "into_position": EVAL_VAO_CHO, "on_your_marks": EVAL_VAO_CHO,
    "static": EVAL_STATIC, "san_sang": EVAL_STATIC, "set": EVAL_STATIC,
    "sprint_start": EVAL_SPRINT, "bat_xuat_phat": EVAL_SPRINT, "go": EVAL_SPRINT, "launch": EVAL_SPRINT,
    "drive_phase": EVAL_DRIVE, "drive": EVAL_DRIVE,
}

_SUMMARY_ATTR = {
    PHASE_INTO_POSITION: "static_summary",
    PHASE_STATIC: "static_summary",
    PHASE_SPRINT_START: "sprint_start_summary",
    PHASE_DYNAMIC: "dynamic_summary",
}
# Khóa = tên pha REFERENCE (SCORER_PHASE); khớp ReferenceModelBuilder._assign_ranges.
_RANGE_ATTR = {
    PHASE_INTO_POSITION: "phase_ranges",
    PHASE_STATIC: "static_ranges",
    PHASE_DYNAMIC: "dynamic_ranges",
    PHASE_SPRINT_START: "sprint_start_ranges",
    PHASE_DRIVE: "drive_phase_ranges",
}

CLIP_NAME_RE = re.compile(r"^(?:coach|trainee)_([A-Za-z0-9]+)_running_(.+)$", re.IGNORECASE)

DEFAULT_PASS_THRESHOLDS = {
    "overall_mean_min": 75.0,
    "overall_std_max": 12.0,
    "leakage_gap_max": 10.0,
    "min_hlv_mean_min": 60.0,
}
LOW_REFERENCE_CLIPS = 3  # reference set < ngưỡng này -> cờ low_reference_sample


# ============================================================================
# Kiểu dữ liệu
# ============================================================================
@dataclass
class ClipRecord:
    clip_id: str
    hlv_id: str
    video_path: Path
    phases_path: Path


@dataclass
class ClipData:
    """Pose + FeatureBundle theo từng pha của 1 clip (tính MỘT LẦN, dùng lại
    cho cả naive lẫn mọi fold LOSO)."""

    record: ClipRecord
    fold_id: str
    fps: float = float("nan")
    bundles: Dict[str, Optional[FeatureBundle]] = field(default_factory=dict)
    phase_errors: Dict[str, str] = field(default_factory=dict)
    load_error: Optional[str] = None

    @property
    def clip_id(self) -> str:
        return self.record.clip_id


@dataclass
class ClipResult:
    clip_id: str
    fold_id: str
    overall: float
    phase_scores: Dict[str, float]
    skip_reasons: Dict[str, str]
    is_valid: bool
    invalid_reason: Optional[str] = None


# ============================================================================
# Giao diện "đường nối" (seam) với phần còn lại của dự án
# ============================================================================
class PoseLoader(Protocol):
    def load(self, record: ClipRecord) -> Tuple[List[PoseFrame], float]: ...


PoseExtractorFn = Callable[[Path], Tuple[List[PoseFrame], float]]


# ----------------------------------------------------------------------------
# Pose cache
# ----------------------------------------------------------------------------
def write_pose_cache(path: Path, poses: Sequence[PoseFrame], fps: float) -> None:
    """Định dạng cache: {"fps", "joint_names", "frames":[{frame_idx, timestamp_s,
    xy:[[x,y]...], conf:[...]}]}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": float(fps),
        "joint_names": list(poses[0].joint_names) if poses else [],
        "frames": [
            {
                "frame_idx": int(pf.frame_idx),
                "timestamp_s": float(pf.timestamp_s),
                "xy": np.asarray(pf.xy, dtype=float).tolist(),
                "conf": np.asarray(pf.conf, dtype=float).tolist(),
            }
            for pf in poses
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def read_pose_cache(path: Path) -> Tuple[List[PoseFrame], float]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Hỗ trợ cả 2 định dạng: List (từ step2_extract_pose) và Dict (của module này)
    if isinstance(d, list):
        frames = d
        names = frames[0].get("joint_names", []) if frames else []
    elif isinstance(d, dict) and "frames" in d:
        frames = d["frames"]
        names = d.get("joint_names", [])
    else:
        raise ValueError("Định dạng pose cache không hợp lệ (không phải list hoặc dict có 'frames')")

    poses = [
        PoseFrame(
            frame_idx=int(fr["frame_idx"]),
            timestamp_s=float(fr["timestamp_s"]),
            xy=np.asarray(fr["xy"], dtype=np.float32),
            conf=np.asarray(fr["conf"], dtype=np.float32),
            joint_names=names,
        )
        for fr in frames
    ]
    
    # Tính fps (nếu file Dict thì lấy từ key fps, nếu file List thì tính từ timestamp)
    fps = None
    if isinstance(d, dict):
        fps = d.get("fps")
    if not fps and len(poses) >= 2:
        dts = np.diff([p.timestamp_s for p in poses])
        dts = dts[dts > 0]
        fps = 1.0 / float(np.median(dts)) if len(dts) else float("nan")
        
    return poses, float(fps) if fps else float("nan")


class CachedPoseLoader:
    """Dùng cache nếu có; nếu không, gọi `extractor` (nếu được cung cấp) rồi
    ghi cache để lần chạy sau nhanh hơn."""

    def __init__(self, cache_dir: Path, extractor: Optional[PoseExtractorFn] = None,
                 write_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.extractor = extractor
        self.write_cache = write_cache

    def load(self, record: ClipRecord) -> Tuple[List[PoseFrame], float]:
        cache_path = self.cache_dir / f"{record.video_path.stem}.json"
        if cache_path.exists():
            poses, fps = read_pose_cache(cache_path)
        elif self.extractor is not None:
            poses, fps = self.extractor(record.video_path)
            if self.write_cache and poses:
                write_pose_cache(cache_path, poses, fps)
        else:
            raise FileNotFoundError(
                f"không có pose cache '{cache_path}' và chưa cấu hình pose extractor "
                "(dùng --pose-extractor pkg.module:ham)"
            )
        if not poses:
            raise ValueError("pose rỗng")
        if not (fps and np.isfinite(fps) and fps > 0):
            raise ValueError(f"fps không hợp lệ: {fps}")
        return poses, float(fps)


# ----------------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------------
class SportYaml:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"không tìm thấy YAML: {self.path}")
        raw_bytes = self.path.read_bytes()
        self.md5 = hashlib.md5(raw_bytes).hexdigest()
        self.raw: Dict[str, Any] = yaml.safe_load(raw_bytes.decode("utf-8")) or {}

    def fixed_specs(self, eval_phase: str) -> List[FixedRangeSpec]:
        block = (self.raw.get("fixed_reference_ranges") or {}).get(eval_phase) or []
        specs = []
        for r in block:
            specs.append(FixedRangeSpec(
                feature_key=r["feature_key"], min_value=float(r["min"]), max_value=float(r["max"]),
                unit=str(r.get("unit", "")), source_note=str(r.get("source_note", "")),
            ))
        return specs

    def phase_weights(self) -> Dict[str, float]:
        pw = dict(DEFAULT_PHASE_WEIGHTS)
        for k, v in (self.raw.get("phase_weights") or {}).items():
            pw[PHASE_ALIASES.get(str(k), str(k))] = float(v)
        return pw


# ----------------------------------------------------------------------------
# Phases file + phát hiện clip
# ----------------------------------------------------------------------------
def read_front_leg_override(phases_path) -> Optional[str]:
    """Đọc field 'front_leg_override' từ phases.json. None nếu không có."""
    import json
    from pathlib import Path
    with open(Path(phases_path), encoding="utf-8") as f:
        data = json.load(f)
    override = data.get("front_leg_override")
    if override not in ("left", "right"):
        return None
    return override

def parse_phase_ranges(path: Path) -> Dict[str, Tuple[int, int]]:
    """Đọc *.phases.json -> {eval_phase: (start_frame, end_frame)} (frame_idx
    gốc, đóng cả 2 đầu). Chấp nhận: {"phases": {...}} hoặc {...} ở gốc, giá
    trị dạng {"start_frame","end_frame"} / {"start","end"} / [a, b]; hoặc
    "phases" là list các {"name"/"phase", start, end}."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    body = d.get("phases", d) if isinstance(d, dict) else d

    def _range(v: Any) -> Optional[Tuple[int, int]]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
        if isinstance(v, dict):
            a = v.get("start_frame", v.get("start"))
            b = v.get("end_frame", v.get("end"))
            if a is not None and b is not None:
                return int(a), int(b)
        return None

    out: Dict[str, Tuple[int, int]] = {}
    items: List[Tuple[str, Any]] = []
    if isinstance(body, dict):
        items = list(body.items())
    elif isinstance(body, list):
        items = [(str(it.get("name", it.get("phase", ""))), it) for it in body if isinstance(it, dict)]
    for name, v in items:
        canon = PHASE_ALIASES.get(str(name).strip().lower())
        rng = _range(v)
        if canon and rng and rng[1] >= rng[0]:
            out[canon] = rng
    return out


def discover_clips(raw_dir: Path, phases_dir: Path, warnings: List[str]) -> List[ClipRecord]:
    raw_dir, phases_dir = Path(raw_dir), Path(phases_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"không thấy thư mục video HLV: {raw_dir}")
    records: List[ClipRecord] = []
    for video in sorted(raw_dir.glob("*.mp4")):
        m = CLIP_NAME_RE.match(video.stem)
        if not m:
            warnings.append(f"[bỏ qua] '{video.name}' không khớp pattern coach_<HLV_ID>_running_<clip_ID>.mp4")
            continue
        phases_path = phases_dir / f"{video.stem}.phases.json"
        if not phases_path.exists():
            warnings.append(f"[loại] '{video.name}' thiếu file phases: {phases_path}")
            continue
        records.append(ClipRecord(video.stem, m.group(1).upper(), video, phases_path))
    return records


# ============================================================================
# Thống kê tiện ích
# ============================================================================
def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def _mean(values: Sequence[float]) -> float:
    a = _clean(values)
    return float(a.mean()) if len(a) else float("nan")


def _std(values: Sequence[float]) -> float:
    a = _clean(values)
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def _min(values: Sequence[float]) -> float:
    a = _clean(values)
    return float(a.min()) if len(a) else float("nan")


def _max(values: Sequence[float]) -> float:
    a = _clean(values)
    return float(a.max()) if len(a) else float("nan")


def _fmt(x: Any, nd: int = 1) -> str:
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


# ============================================================================
# Evaluator
# ============================================================================
class LosoCvEvaluator:
    def __init__(
        self,
        sport: str = "sprint_start",
        phases_to_evaluate: Optional[Sequence[str]] = None,
        min_clips_per_subject: int = 1,
        holdout_strategy: str = "by_hlv",
        output_dir: str = "output/loso_cv",
        *,
        raw_video_dir: str = "data/raw_videos/COACH",
        phases_dir: str = "data/phases",
        pose_cache_dir: str = "data/pose_cache",
        config_dir: str = "config/sport_configs",
        statistical_method: str = "median_iqr",
        pass_thresholds: Optional[Dict[str, float]] = None,
        pose_loader: Optional[PoseLoader] = None,
        pose_extractor: Optional[PoseExtractorFn] = None,
        write_pose_cache_on_extract: bool = True,
        plugin: Optional[SportPlugin] = None,
        extractor: Optional[FeatureExtractor] = None,
        make_plots: bool = True,
        supplement_with_fixed: bool = True,
        log_drive_phase: bool = False,
        dominance_warn_ratio: float = 0.25,
        min_scored_phases: int = 2,
        min_phase_frames: int = 5,
    ):
        phases = tuple(phases_to_evaluate or DEFAULT_PHASES_TO_EVALUATE)
        bad = [p for p in phases if p not in ALL_EVAL_PHASES]
        if bad:
            raise ValueError(f"phases_to_evaluate chứa pha không hợp lệ {bad}; hợp lệ: {ALL_EVAL_PHASES}")
        if holdout_strategy not in ("by_hlv", "by_clip"):
            raise ValueError("holdout_strategy phải là 'by_hlv' hoặc 'by_clip'")
        if min_clips_per_subject < 1:
            raise ValueError("min_clips_per_subject phải >= 1")

        self.sport = sport
        self.phases_to_evaluate = phases
        self.min_clips_per_subject = min_clips_per_subject
        self.holdout_strategy = holdout_strategy
        self.output_dir = Path(output_dir)
        self.raw_video_dir = Path(raw_video_dir)
        self.phases_dir = Path(phases_dir)
        self.pose_cache_dir = Path(pose_cache_dir)
        self.config_dir = Path(config_dir)
        self.config_path = self.config_dir / f"{sport}.yaml"
        self.statistical_method = statistical_method
        self.thresholds = {**DEFAULT_PASS_THRESHOLDS, **(pass_thresholds or {})}
        self.make_plots = make_plots
        self.supplement_with_fixed = supplement_with_fixed
        self.dominance_warn_ratio = dominance_warn_ratio
        self.min_scored_phases = min_scored_phases
        self.min_phase_frames = min_phase_frames

        self.yaml = SportYaml(self.config_path)
        self.phase_weights = self.yaml.phase_weights()
        self.score_phases: Tuple[str, ...] = phases + (
            (EVAL_DRIVE,) if log_drive_phase and EVAL_DRIVE not in phases else ()
        )
        self.builder = ReferenceModelBuilder(method=statistical_method)
        self.pose_loader: PoseLoader = pose_loader or CachedPoseLoader(
            self.pose_cache_dir, pose_extractor, write_pose_cache_on_extract)
        # SportPlugin: nguồn duy nhất cho feature_weights / feature_specs /
        # tolerance / level_weights khi dựng Scorer (xem build_scorer).
        self.plugin: SportPlugin = plugin or load_sport_plugin(str(self.config_dir), sport)
        self._extractor = extractor
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        clips = self._prepare_data()
        self._validate_folds(clips)

        naive_results, naive_flags = self._run_naive(clips)
        loso_results, fold_flags = self._run_loso(clips)

        per_hlv = self._aggregate_per_hlv(clips, loso_results, fold_flags)
        comparison = self._build_comparison(naive_results, loso_results)
        summary = self._build_summary(clips, loso_results, naive_results, per_hlv,
                                      fold_flags, naive_flags)
        self._write_outputs(per_hlv, comparison, summary, loso_results, time.time() - t0)
        return summary

    # ------------------------------------------------------------------
    # Bước 1 — chuẩn bị dữ liệu
    # ------------------------------------------------------------------
    def _prepare_data(self) -> List[ClipData]:
        records = discover_clips(self.raw_video_dir, self.phases_dir, self.warnings)

        counts: Dict[str, int] = {}
        for r in records:
            counts[r.hlv_id] = counts.get(r.hlv_id, 0) + 1
        kept: List[ClipRecord] = []
        for r in records:
            if counts[r.hlv_id] < self.min_clips_per_subject:
                continue
            kept.append(r)
        for hlv, n in counts.items():
            if n < self.min_clips_per_subject:
                self.warnings.append(
                    f"[loại] {hlv} chỉ có {n} clip < min_clips_per_subject={self.min_clips_per_subject}")
        if not kept:
            raise ValueError("Không có clip hợp lệ nào để đánh giá (kiểm tra thư mục video/phases).")

        clips: List[ClipData] = []
        for i, rec in enumerate(kept, 1):
            logger.info("Chuẩn bị clip %d/%d: %s", i, len(kept), rec.clip_id)
            fold_id = rec.hlv_id if self.holdout_strategy == "by_hlv" else rec.clip_id
            clips.append(self._load_and_extract(rec, fold_id))
        return clips

    def _get_extractor(self) -> FeatureExtractor:
        if self._extractor is None:
            # Nếu là sprint_start, phải dùng ngưỡng riêng để không loại nhầm tư thế ngồi xổm
            if self.sport == "sprint_start":
                from stage3_features.anomaly_detector import sprint_start_thresholds
                self._extractor = FeatureExtractor(
                    anomaly_thresholds=sprint_start_thresholds()
                )
            else:
                self._extractor = FeatureExtractor()
        return self._extractor

    def _load_and_extract(self, rec: ClipRecord, fold_id: str) -> ClipData:
        cd = ClipData(record=rec, fold_id=fold_id)
        try:
            poses, fps = self.pose_loader.load(rec)
            cd.fps = fps
            ranges = parse_phase_ranges(rec.phases_path)
        except Exception as e:  # noqa: BLE001 — mọi lỗi I/O/định dạng đều chỉ loại clip này
            cd.load_error = f"{type(e).__name__}: {e}"
            self.warnings.append(f"[lỗi tải] {rec.clip_id}: {cd.load_error}")
            for p in self.score_phases:
                cd.bundles[p] = None
                cd.phase_errors[p] = "clip_load_failed"
            return cd

        extractor = self._get_extractor()
        for phase in self.score_phases:
            cd.bundles[phase] = None
            if phase not in ranges:
                cd.phase_errors[phase] = "missing_phase_marker"
                continue
            a, b = ranges[phase]
            sub = [pf for pf in poses if a <= pf.frame_idx <= b]
            if len(sub) < self.min_phase_frames:
                cd.phase_errors[phase] = f"insufficient_data ({len(sub)} frame)"
                continue
            engine = ENGINE_PHASE[phase]
            try:
                override = read_front_leg_override(rec.phases_path)
                bundle = extractor.extract(
                    sub, fps=fps, phase=engine, front_side_hint=override,
                )
            except Exception as e:  # noqa: BLE001
                cd.phase_errors[phase] = f"extract_failed: {type(e).__name__}: {e}"
                continue
            if getattr(bundle, _SUMMARY_ATTR[engine]) is None:
                cd.phase_errors[phase] = "no_valid_segment"
                continue
            cd.bundles[phase] = bundle
        return cd

    def _validate_folds(self, clips: List[ClipData]) -> None:
        folds = {c.fold_id for c in clips}
        if len(folds) < 2:
            raise ValueError(
                f"LOSO-CV cần ít nhất 2 fold (hiện có {len(folds)}: {sorted(folds)}) — "
                "không thể tạo reference set tách rời holdout.")

    # ------------------------------------------------------------------
    # Xây reference theo tập clip
    # ------------------------------------------------------------------
    def _build_reference_models(
        self, ref_clips: List[ClipData]
    ) -> Tuple[Dict[str, Optional[ReferenceModel]], Dict[str, Dict[str, Any]]]:
        models: Dict[str, Optional[ReferenceModel]] = {}
        flags: Dict[str, Dict[str, Any]] = {}
        for phase in self.score_phases:
            ref_phase = SCORER_PHASE[phase]
            specs = self.yaml.fixed_specs(phase)
            info: Dict[str, Any] = {
                "n_reference_clips": 0, "low_reference_sample": False,
                "reference_fallback": False, "fallback_reason": None, "source": None,
            }
            model: Optional[ReferenceModel] = None

            if phase in STATISTICAL_PHASES:
                bundles = [c.bundles.get(phase) for c in ref_clips if c.bundles.get(phase) is not None]
                info["n_reference_clips"] = len(bundles)
                info["low_reference_sample"] = len(bundles) < LOW_REFERENCE_CLIPS
                try:
                    if not bundles:
                        raise ValueError("reference set không có clip hợp lệ cho pha này")
                    model = self.builder.build(self.sport, ref_phase, bundles)
                    info["source"] = "statistical"
                    if self.supplement_with_fixed and specs and self._supplement(model, ref_phase, specs):
                        info["source"] = "statistical+fixed_supplement"
                except ValueError as e:
                    info["reference_fallback"] = True
                    info["fallback_reason"] = str(e)
                    if specs:
                        model = self.builder.build_fixed(self.sport, ref_phase, specs)
                        info["source"] = "fixed_fallback"
                    else:
                        info["fallback_reason"] += " (và YAML không có fixed_reference_ranges để fallback)"
                        info["source"] = "unavailable"
            else:
                if specs:
                    model = self.builder.build_fixed(self.sport, ref_phase, specs)
                    info["source"] = "fixed"
                else:
                    info["source"] = "unavailable"
                    info["fallback_reason"] = "YAML không có fixed_reference_ranges cho pha này"
            models[phase] = model
            flags[phase] = info
        return models, flags

    @staticmethod
    def _supplement(model: ReferenceModel, ref_phase: str, specs: List[FixedRangeSpec]) -> bool:
        """Bổ sung các khóa mà reference thống kê KHÔNG có (build_static chỉ
        gộp các cột `angle_*`, nên các TỈ LỆ như hip_above_shoulder/wrist_width
        sẽ thiếu) bằng khoảng cố định theo giáo trình. Không ghi đè khóa đã có
        từ thống kê, và khoảng giáo trình không phụ thuộc dữ liệu holdout nên
        không gây leakage."""
        target: Dict[str, Any] = getattr(model, _RANGE_ATTR[ref_phase])
        added = False
        for spec in specs:
            if spec.feature_key not in target:
                target[spec.feature_key] = spec.to_reference_range()
                added = True
        if added:
            model.method = "mixed"
        return added

    # ------------------------------------------------------------------
    # Chấm điểm
    # ------------------------------------------------------------------
    def _aggregate_overall(self, phase_scores: Dict[str, float]) -> Tuple[float, Optional[str]]:
        weighted = {p: self.phase_weights.get(p, 0.0) for p in self.phases_to_evaluate}
        weighted = {p: w for p, w in weighted.items() if w > 0}
        scored = {p: phase_scores.get(p, float("nan")) for p in weighted}
        scored = {p: s for p, s in scored.items() if np.isfinite(s)}
        required = min(self.min_scored_phases, len(weighted))
        if len(scored) < required or not scored:
            return float("nan"), f"chỉ chấm được {len(scored)}/{len(weighted)} pha có trọng số (cần >= {required})"
        wsum = sum(weighted[p] for p in scored)
        return sum(weighted[p] * scored[p] for p in scored) / wsum, None

    def _score_clip(self, cd: ClipData, models: Dict[str, Optional[ReferenceModel]]) -> ClipResult:
        phase_scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        for phase in self.score_phases:
            bundle, model = cd.bundles.get(phase), models.get(phase)
            if bundle is None:
                phase_scores[phase] = float("nan")
                reasons[phase] = cd.phase_errors.get(phase, "no_bundle")
            elif model is None:
                phase_scores[phase] = float("nan")
                reasons[phase] = "no_reference"
            else:
                try:
                    scorer = build_scorer(self.plugin, SCORER_PHASE[phase], int(round(cd.fps)))
                    report = scorer.score(bundle, model)
                    score = float(report.overall_score)
                    phase_scores[phase] = score
                    if not np.isfinite(score):
                        reasons[phase] = "no_scorable_feature"
                except Exception as e:  # noqa: BLE001
                    phase_scores[phase] = float("nan")
                    reasons[phase] = f"score_failed: {type(e).__name__}: {e}"
        overall, why = self._aggregate_overall(phase_scores)
        return ClipResult(cd.clip_id, cd.fold_id, overall, phase_scores, reasons,
                          bool(np.isfinite(overall)), why)

    # ------------------------------------------------------------------
    # Bước 2 — Naive
    # ------------------------------------------------------------------
    def _run_naive(self, clips: List[ClipData]):
        models, flags = self._build_reference_models(clips)
        return [self._score_clip(c, models) for c in clips], flags

    # ------------------------------------------------------------------
    # Bước 3 — LOSO
    # ------------------------------------------------------------------
    def _run_loso(self, clips: List[ClipData]):
        folds: Dict[str, List[ClipData]] = {}
        for c in clips:
            folds.setdefault(c.fold_id, []).append(c)

        results: List[ClipResult] = []
        fold_flags: Dict[str, Dict[str, Any]] = {}
        total = len(clips)
        for i, (fold_id, holdout) in enumerate(sorted(folds.items()), 1):
            hold_ids = {c.clip_id for c in holdout}
            reference = [c for c in clips if c.clip_id not in hold_ids]
            self._assert_no_leakage(reference, holdout)
            logger.info("Fold %d/%d: holdout=%s (%d clip), reference=%d clip",
                        i, len(folds), fold_id, len(holdout), len(reference))

            models, flags = self._build_reference_models(reference)
            dominance = len(holdout) / total
            if dominance > self.dominance_warn_ratio:
                msg = (f"{fold_id} chiếm {dominance:.0%} dữ liệu, khi làm holdout, "
                       "reference bị mất cân bằng nhẹ.")
                self.warnings.append(f"[dominance] {msg}")
            fold_flags[fold_id] = {
                "n_holdout_clips": len(holdout), "n_reference_clips_total": len(reference),
                "holdout_dominance": dominance,
                "dominance_warning": dominance > self.dominance_warn_ratio,
                "phases": flags,
            }
            results.extend(self._score_clip(c, models) for c in holdout)
        return results, fold_flags

    def _assert_no_leakage(self, reference: List[ClipData], holdout: List[ClipData]) -> None:
        ref_ids = {c.clip_id for c in reference}
        hold_ids = {c.clip_id for c in holdout}
        overlap = ref_ids & hold_ids
        if overlap:
            raise ValueError(f"LEAKAGE: clip xuất hiện ở cả reference và holdout: {sorted(overlap)}")
        if self.holdout_strategy == "by_hlv":
            hlv_overlap = {c.record.hlv_id for c in reference} & {c.record.hlv_id for c in holdout}
            if hlv_overlap:
                raise ValueError(f"LEAKAGE: HLV xuất hiện ở cả reference và holdout: {sorted(hlv_overlap)}")

    # ------------------------------------------------------------------
    # Bước 4 — Tổng hợp
    # ------------------------------------------------------------------
    def _aggregate_per_hlv(self, clips, loso_results, fold_flags) -> pd.DataFrame:
        rows = []
        n_by_fold: Dict[str, int] = {}
        for c in clips:
            n_by_fold[c.fold_id] = n_by_fold.get(c.fold_id, 0) + 1
        for fold_id in sorted(n_by_fold):
            rs = [r for r in loso_results if r.fold_id == fold_id]
            valid = [r for r in rs if r.is_valid]
            overall = [r.overall for r in valid]
            phase_means = {
                p: round(_mean([r.phase_scores.get(p, float("nan")) for r in valid]), 2)
                for p in self.score_phases
            }
            phase_means = {p: v for p, v in phase_means.items() if np.isfinite(v)}
            flag_bits = []
            ff = fold_flags[fold_id]
            if ff["dominance_warning"]:
                flag_bits.append("holdout_dominance")
            for p, info in ff["phases"].items():
                if info["low_reference_sample"]:
                    flag_bits.append(f"low_reference_sample:{p}")
                if info["reference_fallback"]:
                    flag_bits.append(f"reference_fallback:{p}")
            rows.append({
                "hlv_id": fold_id,
                "n_clips": n_by_fold[fold_id],
                "mean_score": _mean(overall),
                "std_score": _std(overall),
                "min_score": _min(overall),
                "max_score": _max(overall),
                "phase_scores": json.dumps(phase_means, ensure_ascii=False),
                "n_valid": len(valid),
                "holdout_dominance": ff["holdout_dominance"],
                "flags": ";".join(flag_bits),
            })
        return pd.DataFrame(rows)

    def _build_comparison(self, naive_results, loso_results) -> pd.DataFrame:
        folds = sorted({r.fold_id for r in loso_results})
        rows = []
        for f in folds:
            nv = [r.overall for r in naive_results if r.fold_id == f and r.is_valid]
            lv = [r.overall for r in loso_results if r.fold_id == f and r.is_valid]
            n_m, l_m = _mean(nv), _mean(lv)
            rows.append({"hlv_id": f, "naive_mean": n_m, "loso_mean": l_m, "delta": n_m - l_m})
        return pd.DataFrame(rows)

    def _evaluate_criteria(self, overall_mean, overall_std, leakage_gap, min_hlv_mean) -> Dict[str, Any]:
        th = self.thresholds
        defs = {
            "overall_mean": (overall_mean, th["overall_mean_min"], ">="),
            "overall_std": (overall_std, th["overall_std_max"], "<="),
            "leakage_gap": (leakage_gap, th["leakage_gap_max"], "<="),
            "min_hlv_mean": (min_hlv_mean, th["min_hlv_mean_min"], ">="),
        }
        out: Dict[str, Any] = {}
        for name, (val, thr, op) in defs.items():
            ok = bool(np.isfinite(val) and (val >= thr if op == ">=" else val <= thr))
            out[name] = {"value": val, "threshold": thr, "operator": op, "passed": ok}
        out["all_passed"] = all(v["passed"] for v in out.values())
        return out

    @staticmethod
    def _verdict(overall_mean: float, all_passed: bool) -> str:
        if not np.isfinite(overall_mean):
            return "Không đủ dữ liệu hợp lệ để kết luận."
        if all_passed:
            return "Hệ thống tổng quát hóa tốt, sẵn sàng cho triển khai thực tế."
        if overall_mean < 70:
            return "Cần xem lại reference ranges hoặc thu thập thêm HLV."
        if overall_mean < 75:
            return "Chấp nhận được cho pilot, cần thêm dữ liệu để cải thiện."
        return ("Điểm trung bình đạt ngưỡng nhưng chưa đạt đủ các tiêu chí ổn định/tổng quát hóa "
                "(xem chi tiết từng chỉ số); cần rà soát trước khi triển khai.")

    def _build_summary(self, clips, loso_results, naive_results, per_hlv, fold_flags, naive_flags):
        valid = [r for r in loso_results if r.is_valid]
        valid_ids = {r.clip_id for r in valid}
        naive_paired = [r.overall for r in naive_results if r.clip_id in valid_ids and r.is_valid]

        overall_scores = [r.overall for r in valid]
        overall_mean, overall_std = _mean(overall_scores), _std(overall_scores)
        naive_mean = _mean(naive_paired)
        leakage_gap = naive_mean - overall_mean
        min_hlv_mean = _min(per_hlv["mean_score"].tolist()) if len(per_hlv) else float("nan")

        phase_means = {p: _mean([r.phase_scores.get(p) for r in valid]) for p in self.score_phases}
        phase_stats = {
            p: {
                "mean": phase_means[p],
                "std": _std([r.phase_scores.get(p) for r in valid]),
                "n_scored": int(len(_clean([r.phase_scores.get(p) for r in valid]))),
                "reference_type": "statistical" if p in STATISTICAL_PHASES else "fixed",
            } for p in self.score_phases
        }
        criteria = self._evaluate_criteria(overall_mean, overall_std, leakage_gap, min_hlv_mean)
        invalid = [{"clip_id": r.clip_id, "hlv_id": r.fold_id, "reason": r.invalid_reason,
                    "phase_skip_reasons": r.skip_reasons}
                   for r in loso_results if not r.is_valid]
        return {
            "sport": self.sport,
            "holdout_strategy": self.holdout_strategy,
            "phases_evaluated": list(self.phases_to_evaluate),
            "n_subjects": int(len(per_hlv)),
            "n_clips": len(clips),
            "n_valid": len(valid),
            "n_total": len(clips),
            "n_invalid": len(invalid),
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "overall_min": _min(overall_scores),
            "overall_max": _max(overall_scores),
            "naive_mean": naive_mean,
            "leakage_gap": leakage_gap,
            "min_hlv_mean": min_hlv_mean,
            "phase_means": phase_means,
            "phase_stats": phase_stats,
            "pass_criteria": criteria,
            "verdict": self._verdict(overall_mean, criteria["all_passed"]),
            "invalid_clips": invalid,
            "fold_flags": fold_flags,
            "naive_reference_flags": naive_flags,
            "phase_weights": {p: self.phase_weights.get(p, 0.0) for p in self.phases_to_evaluate},
            "notes": {
                "naive_mean_definition": "trung bình điểm NAIVE trên đúng tập clip hợp lệ của LOSO (ghép cặp)",
                "fixed_phase_leakage": "sprint_start/drive_phase dùng khoảng chuẩn cố định từ YAML nên không bị leakage",
            },
        }

    # ------------------------------------------------------------------
    # Bước 5 — Xuất file
    # ------------------------------------------------------------------
    def _write_outputs(self, per_hlv, comparison, summary, loso_results, elapsed_s) -> None:
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        per_hlv.round(3).to_csv(out / "loso_cv_per_hlv.csv", index=False, encoding="utf-8")
        comparison.round(3).to_csv(out / "loso_cv_comparison.csv", index=False, encoding="utf-8")
        with open(out / "loso_cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
        (out / "loso_cv_report.md").write_text(
            self._build_markdown(per_hlv, comparison, summary), encoding="utf-8")

        if self.make_plots:
            self._make_plots(per_hlv, comparison, loso_results)
        self._write_run_info(summary, elapsed_s)
        logger.info("Đã ghi kết quả vào %s", out.resolve())

    def _write_run_info(self, summary, elapsed_s: float) -> None:
        cfg = {
            "sport": self.sport, "holdout_strategy": self.holdout_strategy,
            "phases_to_evaluate": list(self.phases_to_evaluate),
            "min_clips_per_subject": self.min_clips_per_subject,
            "statistical_method": self.statistical_method,
            "supplement_with_fixed": self.supplement_with_fixed,
            "pass_thresholds": self.thresholds, "phase_weights": summary["phase_weights"],
            "scorer": "stage4_scoring.scorer.Scorer via build_scorer", "raw_video_dir": str(self.raw_video_dir),
            "phases_dir": str(self.phases_dir), "pose_cache_dir": str(self.pose_cache_dir),
        }
        lines = [
            f"timestamp      : {datetime.now().isoformat(timespec='seconds')}",
            f"elapsed_s      : {elapsed_s:.1f}",
            f"yaml_path      : {self.config_path}",
            f"yaml_md5       : {self.yaml.md5}",
            f"python         : {platform.python_version()}",
            f"numpy / pandas : {np.__version__} / {pd.__version__}",
            f"n_clips / valid: {summary['n_clips']} / {summary['n_valid']}",
            "config         : " + json.dumps(cfg, ensure_ascii=False),
            f"warnings ({len(self.warnings)}):",
        ]
        lines += [f"  - {w}" for w in self.warnings]
        (self.output_dir / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def _build_markdown(self, per_hlv: pd.DataFrame, comparison: pd.DataFrame, s: Dict[str, Any]) -> str:
        pc = s["pass_criteria"]
        mark = lambda ok: "✅ Đạt" if ok else "❌ Không đạt"  # noqa: E731
        L: List[str] = ["# Đánh giá LOSO-CV", ""]

        # 1. Phương pháp
        unit = "HLV" if self.holdout_strategy == "by_hlv" else "clip"
        L += [
            "## 1. Phương pháp", "",
            f"Đánh giá Leave-One-Subject-Out Cross-Validation cho môn `{self.sport}` trên "
            f"**{s['n_clips']} clip** của **{s['n_subjects']} {unit}** (chiến lược holdout: `{self.holdout_strategy}`). "
            f"Với mỗi {unit}, toàn bộ clip của {unit} đó bị giữ ra khỏi tập reference; khoảng chuẩn được xây "
            f"lại từ các clip còn lại rồi dùng để chấm điểm clip bị giữ ra. Đối chứng (Naive): reference xây từ "
            "toàn bộ clip rồi chấm lại chính các clip đó (có data leakage).", "",
            f"- Các pha tính điểm tổng: {', '.join(self.phases_to_evaluate)} "
            f"(trọng số: {json.dumps(s['phase_weights'], ensure_ascii=False)}).",
            "- Pha `vao_cho`/`static`: khoảng chuẩn học thống kê từ reference set "
            f"(`{self.statistical_method}`); pha `sprint_start`/`drive_phase`: khoảng cố định theo YAML.",
            "- Điểm tổng tự chuẩn hóa theo các pha chấm được; clip chấm được ít hơn "
            f"{min(self.min_scored_phases, len([p for p in self.phases_to_evaluate if self.phase_weights.get(p, 0) > 0]))} "
            "pha có trọng số bị đánh dấu không hợp lệ và loại khỏi mean/std.", "",
        ]
        if self.holdout_strategy == "by_clip":
            L += ["> ⚠️ `by_clip`: các clip khác của cùng HLV vẫn nằm trong reference — còn rò rỉ phong cách "
                  "theo HLV. Chỉ dùng để tham khảo; chuẩn khoa học là `by_hlv`.", ""]

        # 2. Kết quả tổng hợp
        L += [
            "## 2. Kết quả tổng hợp", "",
            f"Số clip hợp lệ: **{s['n_valid']} / {s['n_total']}**.", "",
            "| Chỉ số | Giá trị | Ngưỡng | Kết quả |", "|---|---|---|---|",
            f"| overall_mean | {_fmt(pc['overall_mean']['value'])} | ≥ {pc['overall_mean']['threshold']:g} | {mark(pc['overall_mean']['passed'])} |",
            f"| overall_std | {_fmt(pc['overall_std']['value'])} | ≤ {pc['overall_std']['threshold']:g} | {mark(pc['overall_std']['passed'])} |",
            f"| leakage_gap | {_fmt(pc['leakage_gap']['value'])} | ≤ {pc['leakage_gap']['threshold']:g} | {mark(pc['leakage_gap']['passed'])} |",
            f"| min_hlv_mean | {_fmt(pc['min_hlv_mean']['value'])} | ≥ {pc['min_hlv_mean']['threshold']:g} | {mark(pc['min_hlv_mean']['passed'])} |",
            "", f"Khoảng điểm LOSO: min {_fmt(s['overall_min'])} – max {_fmt(s['overall_max'])}.", "",
        ]

        # 3. Theo HLV
        L += ["## 3. Kết quả theo HLV", "",
              "| HLV | Số clip | Hợp lệ | Mean | Std | Min | Max | Cờ |", "|---|---|---|---|---|---|---|---|"]
        for _, r in per_hlv.iterrows():
            L.append(f"| {r['hlv_id']} | {int(r['n_clips'])} | {int(r['n_valid'])} | {_fmt(r['mean_score'])} | "
                     f"{_fmt(r['std_score'])} | {_fmt(r['min_score'])} | {_fmt(r['max_score'])} | {r['flags'] or '—'} |")
        L.append("")
        dom = [(f, v) for f, v in s["fold_flags"].items() if v["dominance_warning"]]
        for f, v in dom:
            L.append(f"> ⚠️ {f} chiếm {v['holdout_dominance']:.0%} dữ liệu, khi làm holdout, reference bị mất "
                     "cân bằng nhẹ. Để khắc phục, cần thu thập thêm clip từ các HLV khác.")
        if dom:
            L.append("")
        if s["invalid_clips"]:
            L += ["Clip không hợp lệ (không tính vào mean/std):", ""]
            for ic in s["invalid_clips"]:
                L.append(f"- `{ic['clip_id']}` ({ic['hlv_id']}): {ic['reason']}")
            L.append("")

        # 4. Naive vs LOSO
        L += ["## 4. So sánh Naive vs LOSO (chứng minh data leakage)", "",
              "| HLV | Naive | LOSO | Δ (Naive − LOSO) |", "|---|---|---|---|"]
        for _, r in comparison.iterrows():
            L.append(f"| {r['hlv_id']} | {_fmt(r['naive_mean'])} | {_fmt(r['loso_mean'])} | {_fmt(r['delta'])} |")
        n_pos = int((comparison["delta"] > 0).sum()) if len(comparison) else 0
        L += ["", f"Naive mean = **{_fmt(s['naive_mean'])}**, LOSO mean = **{_fmt(s['overall_mean'])}**, "
              f"leakage_gap = **{_fmt(s['leakage_gap'])}**. Có {n_pos}/{len(comparison)} {unit} có Δ > 0 "
              "(LOSO khó hơn Naive, đúng kỳ vọng).", ""]
        if np.isfinite(s["leakage_gap"]) and s["leakage_gap"] <= 0:
            L += ["> Lưu ý: leakage_gap ≤ 0 — không thấy bằng chứng leakage trong dữ liệu này, "
                  "có thể do các pha cố định (không học từ dữ liệu) chiếm ưu thế hoặc cỡ mẫu nhỏ.", ""]

        # 5. Phase yếu
        L += ["## 5. Phân tích phase yếu", "",
              "| Pha | Loại reference | Mean | Std | Số clip chấm được |", "|---|---|---|---|---|"]
        ordered = sorted(s["phase_stats"].items(),
                         key=lambda kv: (np.isnan(kv[1]["mean"]), kv[1]["mean"]))
        for p, st in ordered:
            L.append(f"| {p} | {st['reference_type']} | {_fmt(st['mean'])} | {_fmt(st['std'])} | {st['n_scored']} |")
        weak = [p for p, st in ordered if np.isfinite(st["mean"]) and p in self.phases_to_evaluate]
        if weak:
            L += ["", f"Pha có điểm trung bình thấp nhất: **{weak[0]}** ({_fmt(s['phase_stats'][weak[0]]['mean'])})."]
        low_ref = sorted({(f, p) for f, v in s["fold_flags"].items() for p, i in v["phases"].items()
                          if i["low_reference_sample"]})
        fb = sorted({(f, p) for f, v in s["fold_flags"].items() for p, i in v["phases"].items()
                     if i["reference_fallback"]})
        if low_ref:
            L += ["", f"- Cờ `low_reference_sample` (reference < {LOW_REFERENCE_CLIPS} clip): "
                  + ", ".join(f"{f}/{p}" for f, p in low_ref)]
        if fb:
            L += ["", "- Cờ `reference_fallback` (dùng khoảng chuẩn YAML thay cho thống kê): "
                  + ", ".join(f"{f}/{p}" for f, p in fb)]
        L.append("")

        # 6. Kết luận
        L += ["## 6. Kết luận", "", f"**{s['verdict']}**", "", "**Giới hạn cần nêu:**", "",
              f"1. Cỡ mẫu nhỏ ({s['n_subjects']} {unit}; chuẩn khoa học thường ≥ 30): điểm có thể dao động "
              "mạnh giữa các fold, nhất là khi HLV chiếm nhiều clip làm holdout.",
              "2. LOSO-CV chỉ đánh giá HLV vs HLV, chưa trả lời việc hệ thống chấm có đúng cho học viên hay không — "
              "cần kiểm định bổ sung trên ≥ 10 clip học viên có ≥ 2 HLV chấm tay (tính ICC).",
              "3. LOSO-CV chỉ thực sự cross-validate các pha dùng statistical reference (`vao_cho`, `static`); "
              "các pha còn lại dùng reference cố định từ YAML nên không bị ảnh hưởng bởi leakage.", ""]
        return "\n".join(L)

    # ------------------------------------------------------------------
    # Biểu đồ
    # ------------------------------------------------------------------
    def _make_plots(self, per_hlv, comparison, loso_results) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.warnings.append("[plot] matplotlib không khả dụng — bỏ qua biểu đồ PNG")
            return
        out = self.output_dir

        # 1) Scatter Naive vs LOSO
        comp = comparison.dropna(subset=["naive_mean", "loso_mean"])
        if len(comp):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(comp["naive_mean"], comp["loso_mean"], s=45)
            for _, r in comp.iterrows():
                ax.annotate(r["hlv_id"], (r["naive_mean"], r["loso_mean"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            lo = float(min(comp["naive_mean"].min(), comp["loso_mean"].min())) - 3
            hi = float(max(comp["naive_mean"].max(), comp["loso_mean"].max())) + 3
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Naive (có leakage)"); ax.set_ylabel("LOSO-CV")
            ax.set_title("LOSO vs Naive — mỗi điểm = 1 HLV"); ax.legend()
            fig.tight_layout(); fig.savefig(out / "loso_vs_naive_scatter.png", dpi=150); plt.close(fig)

        # 2) Bar mean ± std
        bar = per_hlv.dropna(subset=["mean_score"]).sort_values("mean_score", ascending=False)
        if len(bar):
            fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(bar) + 2), 4.5))
            ax.bar(bar["hlv_id"], bar["mean_score"], yerr=bar["std_score"].fillna(0.0), capsize=3)
            ax.axhline(self.thresholds["overall_mean_min"], color="r", ls="--", lw=1,
                       label=f"ngưỡng {self.thresholds['overall_mean_min']:g}")
            ax.set_ylabel("Điểm LOSO (mean ± std)"); ax.set_title("Điểm LOSO theo HLV")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); ax.legend()
            fig.tight_layout(); fig.savefig(out / "loso_per_hlv_bar.png", dpi=150); plt.close(fig)

        # 3) Boxplot theo phase
        data, labels = [], []
        for p in self.score_phases:
            vals = _clean([r.phase_scores.get(p) for r in loso_results if r.is_valid])
            if len(vals):
                data.append(vals); labels.append(p)
        if data:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.boxplot(data, showmeans=True)
            ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
            ax.set_ylabel("Điểm phase (LOSO)"); ax.set_title("Phân bố điểm theo phase")
            fig.tight_layout(); fig.savefig(out / "loso_phase_breakdown.png", dpi=150); plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def _load_callable(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("--pose-extractor phải có dạng 'package.module:function'")
    mod_name, fn_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), fn_name)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Đánh giá LOSO-CV cho hệ thống chấm điểm kỹ thuật.")
    p.add_argument("--sport", default="sprint_start")
    p.add_argument("--phases", nargs="+", default=list(DEFAULT_PHASES_TO_EVALUATE),
                   choices=ALL_EVAL_PHASES)
    p.add_argument("--holdout-strategy", default="by_hlv", choices=("by_hlv", "by_clip"))
    p.add_argument("--min-clips-per-subject", type=int, default=1)
    p.add_argument("--output-dir", default="output/loso_cv")
    p.add_argument("--raw-video-dir", default="data/raw_videos/COACH")
    p.add_argument("--phases-dir", default="data/phases")
    p.add_argument("--pose-cache-dir", default="data/pose_cache")
    p.add_argument("--config-dir", default="config/sport_configs")
    p.add_argument("--statistical-method", default="median_iqr", choices=("median_iqr", "mean_std"))
    p.add_argument("--pose-extractor", default=None,
                   help="'pkg.module:ham' — hàm(Path video) -> (List[PoseFrame], fps), dùng khi thiếu cache")
    p.add_argument("--no-write-cache", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-fixed-supplement", action="store_true",
                   help="không bổ sung khoảng cố định YAML cho các khóa thiếu trong reference thống kê")
    p.add_argument("--log-drive-phase", action="store_true")
    p.add_argument("--fail-on-criteria", action="store_true",
                   help="thoát mã 1 nếu không đạt đủ 4 tiêu chí pass")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    evaluator = LosoCvEvaluator(
        sport=args.sport, phases_to_evaluate=args.phases,
        min_clips_per_subject=args.min_clips_per_subject,
        holdout_strategy=args.holdout_strategy, output_dir=args.output_dir,
        raw_video_dir=args.raw_video_dir, phases_dir=args.phases_dir,
        pose_cache_dir=args.pose_cache_dir, config_dir=args.config_dir,
        statistical_method=args.statistical_method,
        pose_extractor=_load_callable(args.pose_extractor) if args.pose_extractor else None,
        write_pose_cache_on_extract=not args.no_write_cache,
        make_plots=not args.no_plots, supplement_with_fixed=not args.no_fixed_supplement,
        log_drive_phase=args.log_drive_phase,
    )
    summary = evaluator.run()
    print(f"LOSO-CV: mean={_fmt(summary['overall_mean'])} std={_fmt(summary['overall_std'])} "
          f"naive={_fmt(summary['naive_mean'])} gap={_fmt(summary['leakage_gap'])} "
          f"valid={summary['n_valid']}/{summary['n_total']}")
    print("Kết luận:", summary["verdict"])
    print("Kết quả:", Path(args.output_dir).resolve())
    return 1 if args.fail_on_criteria and not summary["pass_criteria"]["all_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())