"""
core/pipeline.py
==================
Mục đích : Điều phối TOÀN BỘ luồng 5 giai đoạn cho MỘT lượt quay.

V2.1 - Bổ sung:
    - Pipeline full sprint_start: 1 video -> 4 sub-phase -> 1 điểm tổng.
    - Quality gate trước scoring (không chấm nếu dữ liệu kém).
    - SprintStartFullResult chứa 4 sub-phase + overall + quality flags.
    - Backward compat với process_tracked_sequence() cũ (chấm 1 phase).

V2.1.1 - Hoàn thiện:
    - _get_scorer() truyền fps_actual + tolerance_map cho Scorer.
    - _make_sub_bundle() hỗ trợ PHASE_DRIVE.
    - _check_sub_phase_quality() thêm gate cho drive_phase.

V2.1.2 - Đồng bộ với Scorer V2.1.1 (drive_phase chấm riêng, slope).

V2.1.3 - Hỗ trợ file phases đánh dấu thủ công (*.phases.json).

V2.1.4 - SỬA LỖI chọn nhầm file phases (ví dụ video HLV012 bị nạp file của HLV01):
    NGUYÊN NHÂN: bản cũ quét toàn bộ data/phases/ rồi chọn file có độ dài
    frame "gần giống nhất" với video -> các video có độ dài tương đương bị
    bốc nhầm file của nhau, dẫn tới cắt sai đoạn và điểm số vô nghĩa.

    CÁCH SỬA:
        1. Khớp file theo TÊN video (stem): "<video_stem>.phases.json"
           (khớp CHÍNH XÁC, không dùng prefix/startswith để tránh
           HLV01 khớp nhầm HLV012).
        2. video_id được truyền tường minh qua tham số (không lưu vào self),
           nên không rò rỉ trạng thái giữa các lần chạy.
        3. Hỗ trợ truyền thẳng đường dẫn file phases (phases_path).
        4. Mọi file phases đều được kiểm tra hợp lệ (start < end, nằm trong
           dải frame của video). File không hợp lệ -> bỏ qua, KHÔNG dùng.
        5. Khớp mờ theo độ dài frame bị TẮT mặc định (config
           `phases_fuzzy_fallback: true` để bật), có dung sai chặt và từ chối
           khi có nhiều ứng viên ngang nhau.
        6. Không tìm được file phù hợp -> fallback auto-segmentation,
           kết quả ghi rõ nguồn phases (SprintStartFullResult.phase_source).

V2.1.5 - SỬA LỖI tracker bắt nhầm vật thể (ghế đá, logo Kinemaster...):
    - Trước khi đưa vào track_and_filter, mỗi frame chỉ giữ lại 1 detection
      có BBox LỚN NHẤT (_keep_largest_person). Bộ lọc nằm trong
      _extract_tracked() nên áp dụng cho CẢ process_video() và
      process_sprint_start_video() (app_demo.py, render_overlay.py...).
    - Có thể tắt bằng config `select_largest_person: false`.
    - Giả định bbox có dạng (x1, y1, x2, y2) như trong render_overlay.py.

V2.2 - Sửa bug pipeline (PL1, PL2, PL5, PL6):
    - PL1: _validate_windows_in_range nhận smoothed_frame_set, kiểm tra
      TỪNG window.start/end có thực sự tồn tại trong smoothed (tránh crash
      khi SignalSmoother drop frame giữa chừng — root cause HLV08_v0013).
    - PL2: _try_load_manual_phases trả thêm status rõ ràng
      (ok / missing / invalid_content / out_of_range) → phase_source
      phân biệt được "không có file" vs "có file nhưng hỏng".
    - PL5: quality gate sprint_start đổi sang đếm PRIMARY NaN ≥ 3
      thay vì n_nan/total > 0.5.
    - PL6: SprintStartFullResult thêm n_phases_scored; overall = NaN
      khi n_phases_scored < 3 để tránh misleading.

Refactor - Scorer factory:
    - _get_scorer() chỉ còn ủy quyền cho stage4_scoring.scorer_factory.build_scorer
      (single source of truth; stage5_eval/loso_cv_eval.py dùng chung hàm này
      nên điểm LOSO-CV khớp pipeline chính).

CẤU HÌNH (system_config, đều tùy chọn):
    select_largest_person          : chỉ giữ người có BBox lớn nhất mỗi frame
                                     (mặc định True)
    phases_dir                     : thư mục chứa *.phases.json (mặc định data/phases)
    use_manual_phases              : bật/tắt dùng phases thủ công (mặc định True)
    phases_fuzzy_fallback          : cho phép khớp mờ theo độ dài (mặc định False)
    phases_fuzzy_tolerance_frames  : dung sai 2 đầu khi khớp mờ (mặc định 5)
    phases_range_tolerance_frames  : dung sai khi kiểm tra phases nằm trong
                                     dải frame của video (mặc định 2)

GIỚI HẠN CÒN LẠI:
    - Nếu YAML không khai reference range cho drive_phase, các trend feature
      sẽ vào unscored_features (fail-safe có chủ đích).
    - drive_phase yêu cầu num_steps >= 2 mới chấm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from core.pose_schema import PoseFrame
from core.sport_interface import SportPlugin, load_sport_plugin
from stage1_data.capture_guide import CaptureThresholds
from stage2_pose.pose_extractor import PoseExtractor, build_pose_extractor
from stage2_pose.signal_smoother import SignalSmoother
from stage2_pose.tracker import track_and_filter
from stage3_features.anomaly_detector import sprint_start_thresholds
from stage3_features.feature_extractor import (
    FeatureBundle,
    FeatureExtractor,
    PHASE_DRIVE,
    PHASE_INTO_POSITION,
    PHASE_SPRINT_START,
    PHASE_STATIC,
)
from stage4_scoring.feedback_generator import FeedbackGenerator, FeedbackReport
from stage4_scoring.reference_model import (
    FixedRangeSpec,
    ReferenceModel,
    ReferenceModelBuilder,
)
from stage4_scoring.scorer import ScoreReport, Scorer
from stage4_scoring.scorer_factory import build_scorer

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)

PHASES_FILE_SUFFIX = ".phases.json"
PhaseWindows = Dict[str, Tuple[int, int]]

# Trạng thái nạp phases (PL2)
PHASE_STATUS_OK = "ok"
PHASE_STATUS_MISSING = "missing"
PHASE_STATUS_INVALID = "invalid_content"
PHASE_STATUS_OUT_OF_RANGE = "out_of_range"
PHASE_STATUS_DISABLED = "disabled"


# =====================================================================
# RESULT DATACLASS
# =====================================================================

@dataclass
class PipelineResult:
    phase: str
    bundle: FeatureBundle
    score_report: ScoreReport
    feedback: FeedbackReport


@dataclass
class SubPhaseResult:
    """Kết quả chấm 1 sub-phase trong pipeline full."""
    phase: str
    result: Optional[PipelineResult] = None
    is_scored: bool = False
    skip_reason: str = ""           # lý do không chấm (quality gate)
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    def has_score(self) -> bool:
        return self.is_scored and self.result is not None


@dataclass
class SprintStartFullResult:
    sport: str
    sub_phase_results: Dict[str, SubPhaseResult]
    sub_phase_order: List[str]
    overall_score: float
    quality_flags: Dict[str, bool]
    score_confidence: str = "medium"    # high / medium / low
    # Nguồn mốc phase: "auto" | "manual:<path>" | "manual-fuzzy:<path>"
    # kèm chú thích nếu file tồn tại nhưng bị loại (PL2).
    phase_source: str = "auto"
    # Số phase thực sự được chấm (PL6)
    n_phases_scored: int = 0
    needs_review: bool = False

    def render_text(self) -> str:
        lines = [
            "=" * 70,
            "KẾT QUẢ CHẤM XUẤT PHÁT THẤP - ĐẠP LAO",
            "=" * 70,
            f"Điểm tổng: {self.overall_score:.0f}/100  "
            f"(độ tin cậy: {self.score_confidence}, "
            f"{self.n_phases_scored}/{len(self.sub_phase_order)} phase)",
            f"Nguồn phases: {self.phase_source}",
            f"Cần duyệt: {self.needs_review}",
            "",
        ]
        for phase in self.sub_phase_order:
            sp = self.sub_phase_results.get(phase)
            if sp is None:
                continue
            if sp.has_score():
                # V2.4: phase có weight = 0 không tính vào tổng → ghi rõ
                note = ""
                if phase == PHASE_DRIVE:
                    note = "  [chỉ tham khảo — không tính vào điểm tổng]"
                lines.append(f"[{phase}] {sp.result.feedback.summary_line}{note}")
            else:
                lines.append(f"[{phase}] Bỏ qua — {sp.skip_reason}")
        active_flags = [k for k, v in self.quality_flags.items() if v]
        if active_flags:
            lines.append("")
            lines.append(
                "Cảnh báo chất lượng dữ liệu: " + ", ".join(active_flags)
            )
        return "\n".join(lines)


# =====================================================================
# PIPELINE
# =====================================================================

class Pipeline:
    def __init__(
        self,
        sport_plugin: SportPlugin,
        system_config: Dict,
        reference_models: Optional[Dict[str, ReferenceModel]] = None,
    ):
        self.plugin = sport_plugin
        self.config = system_config
        self.reference_models: Dict[str, ReferenceModel] = reference_models or {}

        if sport_plugin.sport_name == "sprint_start":
            self.extractor = FeatureExtractor(
                anomaly_thresholds=sprint_start_thresholds()
            )
        else:
            self.extractor = FeatureExtractor()

        self.smoother = SignalSmoother(
            conf_threshold=system_config.get("min_keypoint_confidence", 0.35)
        )
        self._pose_extractor: Optional[PoseExtractor] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        system_config_path: str,
        sport: str,
        load_reference_models: bool = True,
    ) -> "Pipeline":
        with open(system_config_path, "r", encoding="utf-8") as f:
            system_config = yaml.safe_load(f)

        raw_dir = Path(system_config["sport_config_dir"])
        candidate_dirs = [
            raw_dir,
            Path(system_config_path).parent.parent / raw_dir,
        ]

        plugin = None
        last_error: Optional[Exception] = None
        for d in candidate_dirs:
            try:
                plugin = load_sport_plugin(str(d), sport)
                break
            except FileNotFoundError as exc:
                last_error = exc
        if plugin is None:
            raise last_error or FileNotFoundError(
                f"Không nạp được cấu hình môn '{sport}'"
            )

        # Đưa khối phase_segmenter từ cấu hình môn vào resolver; giá trị ở
        # system_config được ưu tiên nếu người vận hành có override riêng.
        sport_config = {}
        for directory in candidate_dirs:
            config_path = directory / f"{sport}.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as sf:
                    sport_config = yaml.safe_load(sf) or {}
                break
        system_config = dict(system_config)
        system_config["phase_segmenter"] = {
            **(sport_config.get("phase_segmenter") or {}),
            **(system_config.get("phase_segmenter") or {}),
        }

        reference_models: Dict[str, ReferenceModel] = {}
        if load_reference_models:
            ref_dir = Path(
                system_config.get("reference_model_dir", "data/reference_models")
            )
            for phase in plugin.phases():
                ref_path = ref_dir / f"{sport}_{phase}.json"
                if ref_path.exists():
                    reference_models[phase] = ReferenceModel.load(str(ref_path))
                else:
                    fixed_specs_raw = []
                    if hasattr(plugin, "fixed_reference_ranges"):
                        raw = plugin.fixed_reference_ranges(phase)
                        for r in raw:
                            if hasattr(r, "feature_key"):
                                fixed_specs_raw.append({
                                    "feature_key": r.feature_key,
                                    "min": r.min,
                                    "max": r.max,
                                    "unit": r.unit,
                                    "source_note": r.source_note,
                                })
                            else:
                                fixed_specs_raw.append(r)
                    if fixed_specs_raw:
                        specs = [
                            FixedRangeSpec(
                                feature_key=r["feature_key"],
                                min_value=float(r["min"]),
                                max_value=float(r["max"]),
                                unit=r.get("unit", ""),
                                source_note=r.get("source_note", ""),
                            )
                            for r in fixed_specs_raw
                        ]
                        try:
                            reference_models[phase] = ReferenceModelBuilder(
                                method=plugin.reference_method()
                            ).build_fixed(sport, phase, specs)
                        except ValueError as exc:
                            import warnings
                            warnings.warn(
                                f"Bỏ qua build_fixed cho phase '{phase}': {exc}",
                                RuntimeWarning,
                            )
        return cls(plugin, system_config, reference_models)

    # ------------------------------------------------------------------
    def set_reference_model(self, phase: str, model: ReferenceModel) -> None:
        self.reference_models[phase] = model

    def get_capture_thresholds(self) -> CaptureThresholds:
        return self.plugin.capture_thresholds()

    # ------------------------------------------------------------------
    def _get_scorer(self, phase: str, fps_actual: int = 60) -> Scorer:
        """
        Khởi tạo Scorer cho 1 phase. Toàn bộ logic dựng Scorer nằm ở
        stage4_scoring.scorer_factory.build_scorer (single source of truth),
        dùng chung với stage5_eval/loso_cv_eval.py.
        """
        return build_scorer(self.plugin, phase, fps_actual)

    def _get_feedback_generator(self, phase: str) -> FeedbackGenerator:
        thresholds = self.config.get("feedback_thresholds", {})
        overrides = self.plugin.feedback_advice_overrides(phase)
        # Đảm bảo key là tuple (feature, direction) nếu plugin trả string
        if overrides and isinstance(next(iter(overrides.keys()), None), str):
            parsed: Dict[tuple, str] = {}
            for k, v in overrides.items():
                if isinstance(k, str) and ":" in k:
                    feat, direction = k.rsplit(":", 1)
                    parsed[(feat.strip(), direction.strip())] = v
                elif isinstance(k, tuple):
                    parsed[k] = v
            overrides = parsed
        return FeedbackGenerator(
            good_threshold=thresholds.get("good", 85.0),
            warn_threshold=thresholds.get("warn", 60.0),
            advice_overrides=overrides,
        )

    # ==================================================================
    # LOAD FILE PHASES ĐÁNH DẤU THỦ CÔNG (V2.1.4 + V2.2 PL1/PL2)
    # ==================================================================
    @staticmethod
    def _normalize_video_id(video_id: Optional[str]) -> str:
        """
        Chuẩn hóa video_id thành stem thuần:
            'a/b/coach_HLV012.mp4'            -> 'coach_HLV012'
            'coach_HLV012.phases.json'        -> 'coach_HLV012'
        """
        if not video_id:
            return ""
        name = Path(str(video_id)).name
        if name.lower().endswith(PHASES_FILE_SUFFIX):
            return name[: -len(PHASES_FILE_SUFFIX)]
        return Path(name).stem

    @staticmethod
    def _read_phases_file(path: Path) -> Optional[PhaseWindows]:
        """
        Đọc 1 file phases. Trả về {phase: (start, end)} hoặc None nếu file
        hỏng / thiếu trường / có phase với start >= end.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            phases_raw = data.get("phases", {})
            if not isinstance(phases_raw, dict) or not phases_raw:
                logger.warning(
                    "[pipeline] File phases rỗng/sai cấu trúc: %s", path
                )
                return None

            windows: PhaseWindows = {}
            for name, spec in phases_raw.items():
                if isinstance(spec, dict) and spec.get("absent"):
                    continue
                if not isinstance(spec, dict):
                    logger.warning("[pipeline] File %s: phase '%s' has invalid format.", path, name)
                    return None
                start = int(spec["start_frame"])
                end = int(spec["end_frame"])
                if start > end:
                    logger.warning(
                        "[pipeline] File %s: phase '%s' có start (%d) >= end (%d)",
                        path.name, name, start, end,
                    )
                    return None
                windows[name] = (start, end)
            if not windows:
                logger.warning("[pipeline] File phases không có pha hiện diện: %s", path)
                return None
            return windows
        except Exception as exc:
            logger.warning("[pipeline] Lỗi đọc file phases %s: %s", path, exc)
            return None

    def _validate_windows_in_range(
        self,
        windows: PhaseWindows,
        first_frame: int,
        last_frame: int,
        source: str,
        smoothed_frame_set: Optional[Set[int]] = None,
    ) -> bool:
        """
        Kiểm tra mọi mốc phase nằm trong dải frame của chuỗi pose hiện tại
        (có dung sai nhỏ) VÀ (PL1) từng start/end thực sự tồn tại trong
        smoothed nếu smoothed_frame_set được truyền.

        Nếu smoothed drop frame giữa chừng (confidence thấp), boundary check
        cũ vẫn PASS nhưng lookup frame_idx sẽ fail → crash HLV08_v0013.
        """
        tol = int(self.config.get("phases_range_tolerance_frames", 2))
        min_start = min(s for s, _ in windows.values())
        max_end = max(e for _, e in windows.values())
        if min_start < first_frame - tol or max_end > last_frame + tol:
            logger.warning(
                "[pipeline] Bỏ qua phases '%s': khoảng %d→%d nằm ngoài dải "
                "frame của video (%d→%d).",
                source, min_start, max_end, first_frame, last_frame,
            )
            return False

        # PL1: kiểm tra từng endpoint có thực sự trong smoothed
        if smoothed_frame_set is not None:
            missing: List[str] = []
            for phase_name, (start, end) in windows.items():
                if start not in smoothed_frame_set:
                    missing.append(f"{phase_name}.start={start}")
                if end not in smoothed_frame_set:
                    missing.append(f"{phase_name}.end={end}")
            if missing:
                logger.warning(
                    "[pipeline] Bỏ qua phases '%s': frame không tồn tại trong "
                    "smoothed (bị smoother drop): %s.",
                    source, ", ".join(missing),
                )
                return False
        return True

    def _find_phases_file_for_video(
        self, phases_dir: Path, video_stem: str
    ) -> Optional[Path]:
        """
        Tìm file '<video_stem>.phases.json' theo khớp CHÍNH XÁC tên
        (không phân biệt hoa/thường). KHÔNG dùng prefix: HLV01 != HLV012.
        """
        exact = phases_dir / f"{video_stem}{PHASES_FILE_SUFFIX}"
        if exact.exists():
            return exact

        target = f"{video_stem}{PHASES_FILE_SUFFIX}".lower()
        matches = [
            p for p in phases_dir.glob(f"*{PHASES_FILE_SUFFIX}")
            if p.name.lower() == target
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "[pipeline] Có nhiều file trùng tên (khác hoa/thường) cho '%s': %s "
                "-> không dùng file nào.",
                video_stem, [m.name for m in matches],
            )
        return None

    def _fuzzy_match_phases(
        self,
        phases_dir: Path,
        first_frame: int,
        last_frame: int,
        smoothed_frame_set: Optional[Set[int]] = None,
    ) -> Optional[Tuple[Path, PhaseWindows]]:
        """
        Khớp mờ theo độ dài frame — CHỈ chạy khi config
        `phases_fuzzy_fallback: true`. Dung sai chặt và từ chối nếu có
        nhiều ứng viên ngang điểm (tránh bốc nhầm).
        """
        tol = int(self.config.get("phases_fuzzy_tolerance_frames", 5))
        candidates: List[Tuple[int, Path, PhaseWindows]] = []

        for path in phases_dir.glob(f"*{PHASES_FILE_SUFFIX}"):
            windows = self._read_phases_file(path)
            if windows is None:
                continue
            min_start = min(s for s, _ in windows.values())
            max_end = max(e for _, e in windows.values())
            d_start = abs(min_start - first_frame)
            d_end = abs(max_end - last_frame)
            if d_start > tol or d_end > tol:
                continue
            if not self._validate_windows_in_range(
                windows, first_frame, last_frame, path.name,
                smoothed_frame_set=smoothed_frame_set,
            ):
                continue
            candidates.append((d_start + d_end, path, windows))

        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            logger.warning(
                "[pipeline] Khớp mờ không rõ ràng (nhiều file ngang điểm: %s) "
                "-> từ chối, dùng auto-segmentation.",
                [c[1].name for c in candidates[:3]],
            )
            return None
        return candidates[0][1], candidates[0][2]

    def _try_load_manual_phases(
        self,
        smoothed: List[PoseFrame],
        video_id: Optional[str] = None,
        phases_path: Optional[str] = None,
    ) -> Tuple[Optional[PhaseWindows], str]:
        """
        Nạp phases đánh dấu thủ công theo thứ tự ưu tiên:
            1. phases_path được truyền tường minh.
            2. File '<video_stem>.phases.json' khớp CHÍNH XÁC tên video.
            3. (Tùy chọn, mặc định TẮT) khớp mờ theo độ dài frame.

        Trả về (windows, phase_source_string).
        phase_source luôn ghi rõ trạng thái (PL2):
            "auto"                              — không có / tắt
            "auto (file X missing)"             — không tìm thấy
            "auto (file X invalid_content: …)"  — file hỏng
            "auto (file X out_of_range: …)"     — lệch frame / drop
            "manual:<path>"                     — dùng thành công
            "manual-fuzzy:<path>"               — khớp mờ

        windows=None → dùng auto-segmentation.
        """
        if not self.config.get("use_manual_phases", True):
            return None, "auto"

        if not smoothed:
            return None, "auto"

        first_frame = smoothed[0].frame_idx
        last_frame = smoothed[-1].frame_idx
        # PL1: tập frame_idx thực sự có trong smoothed
        smoothed_frame_set: Set[int] = {f.frame_idx for f in smoothed}

        def _src_auto(path_name: str, status: str, detail: str = "") -> str:
            extra = f": {detail}" if detail else ""
            return f"auto (file '{path_name}' {status}{extra})"

        # --- 1. Đường dẫn tường minh ---
        if phases_path:
            path = Path(phases_path)
            if not path.exists():
                logger.warning(
                    "[pipeline] phases_path không tồn tại: %s", path
                )
                return None, _src_auto(path.name, PHASE_STATUS_MISSING)
            windows = self._read_phases_file(path)
            if windows is None:
                return None, _src_auto(path.name, PHASE_STATUS_INVALID)
            if not self._validate_windows_in_range(
                windows, first_frame, last_frame, path.name,
                smoothed_frame_set=smoothed_frame_set,
            ):
                return None, _src_auto(
                    path.name, PHASE_STATUS_OUT_OF_RANGE,
                    f"boundary {first_frame}→{last_frame}",
                )
            logger.info(
                "[pipeline] Dùng phases thủ công (chỉ định): %s", path
            )
            return windows, f"manual:{path}"

        phases_dir = Path(self.config.get("phases_dir", "data/phases"))
        if not phases_dir.exists():
            logger.info(
                "[pipeline] Thư mục phases không tồn tại: %s", phases_dir
            )
            return None, "auto"

        # --- 2. Khớp chính xác theo tên video ---
        video_stem = self._normalize_video_id(video_id)
        if video_stem:
            path = self._find_phases_file_for_video(phases_dir, video_stem)
            if path is not None:
                windows = self._read_phases_file(path)
                if windows is None:
                    logger.warning(
                        "[pipeline] File phases '%s' nội dung không hợp lệ "
                        "-> dùng auto-segmentation.", path.name,
                    )
                    return None, _src_auto(path.name, PHASE_STATUS_INVALID)
                if not self._validate_windows_in_range(
                    windows, first_frame, last_frame, path.name,
                    smoothed_frame_set=smoothed_frame_set,
                ):
                    logger.warning(
                        "[pipeline] File phases '%s' out_of_range / frame "
                        "bị smoother drop -> dùng auto-segmentation.",
                        path.name,
                    )
                    return None, _src_auto(
                        path.name, PHASE_STATUS_OUT_OF_RANGE,
                        f"video frames {first_frame}→{last_frame}",
                    )
                logger.info("[pipeline] Dùng phases thủ công: %s", path)
                return windows, f"manual:{path}"

            logger.info(
                "[pipeline] Không có file '%s%s' trong %s",
                video_stem, PHASES_FILE_SUFFIX, phases_dir,
            )
            return None, _src_auto(
                f"{video_stem}{PHASES_FILE_SUFFIX}", PHASE_STATUS_MISSING
            )

        # --- 3. Khớp mờ (chỉ khi được bật rõ ràng) ---
        if self.config.get("phases_fuzzy_fallback", False):
            found = self._fuzzy_match_phases(
                phases_dir, first_frame, last_frame,
                smoothed_frame_set=smoothed_frame_set,
            )
            if found is not None:
                path, windows = found
                logger.warning(
                    "[pipeline] Dùng phases theo KHỚP MỜ độ dài frame: %s "
                    "(nên đặt tên file theo tên video để tránh nhầm).", path,
                )
                return windows, f"manual-fuzzy:{path}"

        return None, "auto"

    # ==================================================================
    # CORE - chấm 1 phase (backward compat)
    # ==================================================================
    def process_tracked_sequence(
        self,
        sequence: List[PoseFrame],
        fps: float,
        phase: str,
        reference_model: Optional[ReferenceModel] = None,
    ) -> PipelineResult:
        if phase not in self.plugin.phases():
            raise ValueError(
                f"Môn '{self.plugin.sport_name}' không có pha '{phase}' "
                f"(các pha hợp lệ: {self.plugin.phases()})"
            )
        ref = reference_model or self.reference_models.get(phase)
        if ref is None:
            raise ValueError(
                f"Chưa có ReferenceModel cho pha '{phase}'. Hãy build trước "
                f"hoặc truyền trực tiếp qua tham số reference_model."
            )

        smoothed = self.smoother.process(sequence)
        bundle = self.extractor.extract(smoothed, fps=fps, phase=phase)
        scorer = self._get_scorer(phase, fps_actual=int(round(fps)))
        score_report = scorer.score(bundle, ref)
        feedback = self._get_feedback_generator(phase).generate(
            score_report, bundle=bundle
        )
        return PipelineResult(
            phase=phase,
            bundle=bundle,
            score_report=score_report,
            feedback=feedback,
        )

    # ==================================================================
    # V2.1 - PIPELINE FULL SPRINT_START (4 sub-phase)
    # ==================================================================
    def process_sprint_start_sequence(
        self,
        sequence: List[PoseFrame],
        fps: float,
        video_id: Optional[str] = None,
        phases_path: Optional[str] = None,
    ) -> SprintStartFullResult:
        """
        Chấm đầy đủ 4 sub-phase cho 1 lượt sprint_start.

        Args:
            sequence    : chuỗi pose đã tracking.
            fps         : fps video.
            video_id    : tên/đường dẫn video (dùng stem để tìm file
                          '<stem>.phases.json'). Nên luôn truyền khi có.
            phases_path : đường dẫn file phases chỉ định trực tiếp
                          (ưu tiên cao nhất).

        Ưu tiên nguồn phases:
            1. phases_path (nếu truyền).
            2. File phases khớp CHÍNH XÁC tên video.
            3. Auto-segmentation (fallback).
        Quality gate: sub-phase nào fail quality -> skip, không chấm.
        """
        if self.plugin.sport_name != "sprint_start":
            raise ValueError(
                "process_sprint_start_sequence chỉ dùng cho sport='sprint_start'"
            )

        smoothed = self.smoother.process(sequence)

        # --- Extract full bundle (ưu tiên manual phases) ---
        manual_phases, manual_source = self._try_load_manual_phases(
            smoothed, video_id=video_id, phases_path=phases_path
        )
        from core.phase_resolver import resolve_phases

        def _legacy_heuristic(poses, actual_fps):
            # Chỉ dùng heuristic cũ làm nguồn cuối và luôn yêu cầu review.
            result = self.extractor.extract_sprint_start_full(poses, fps=actual_fps)
            for attr in ("phase_windows", "windows", "boundaries"):
                value = getattr(result, attr, None)
                if value is not None:
                    return value
            # Một số bản FeatureBundle giữ ranh giới trong summaries.
            summaries = getattr(result, "sub_phases", None)
            if isinstance(summaries, dict):
                out = {}
                for name, item in summaries.items():
                    if isinstance(item, dict):
                        start = item.get("start_frame", item.get("start"))
                        end = item.get("end_frame", item.get("end"))
                    else:
                        start = getattr(item, "start_frame", getattr(item, "start", None))
                        end = getattr(item, "end_frame", getattr(item, "end", None))
                    if start is not None and end is not None:
                        out[name] = (int(start), int(end))
                return out or None
            return None

        phase_cfg = dict(self.config)
        windows, phase_source, needs_review = resolve_phases(
            video_id=video_id or "",
            poses=smoothed,
            fps=fps,
            cfg=phase_cfg,
            manual_windows=manual_phases,
            manual_source=manual_source,
            heuristic_detector=_legacy_heuristic,
        )
        if needs_review or not windows:
            order = [PHASE_INTO_POSITION, PHASE_STATIC, PHASE_SPRINT_START, PHASE_DRIVE]
            skipped = {
                phase: SubPhaseResult(
                    phase=phase, is_scored=False,
                    skip_reason="Cần người dùng review và xác nhận ranh giới pha",
                    quality_flags={"needs_review": True},
                ) for phase in order
            }
            return SprintStartFullResult(
                sport=self.plugin.sport_name,
                sub_phase_results=skipped,
                sub_phase_order=order,
                overall_score=float("nan"),
                quality_flags={"needs_review": True},
                score_confidence="low",
                phase_source=phase_source,
                n_phases_scored=0,
                needs_review=True,
            )
        bundle = self.extractor.extract_sprint_start_from_windows(
            smoothed, fps=fps, windows=windows
        )
        if bundle is None:
            order = [PHASE_INTO_POSITION, PHASE_STATIC, PHASE_SPRINT_START, PHASE_DRIVE]
            skipped = {
                phase: SubPhaseResult(
                    phase=phase, is_scored=False,
                    skip_reason="Không có biên pha đáng tin cậy; cần duyệt thủ công",
                    quality_flags={"needs_review": True},
                ) for phase in order
            }
            return SprintStartFullResult(
                sport=self.plugin.sport_name,
                sub_phase_results=skipped,
                sub_phase_order=order,
                overall_score=float("nan"),
                quality_flags={"needs_review": True},
                score_confidence="low",
                phase_source=phase_source,
                n_phases_scored=0,
            )

        # --- Quality gate config ---
        quality_gate_cfg = {}
        if hasattr(self.plugin, "quality_gates"):
            quality_gate_cfg = self.plugin.quality_gates()

        # --- Chấm từng sub-phase ---
        sub_phase_results: Dict[str, SubPhaseResult] = {}
        all_quality_flags: Dict[str, bool] = dict(bundle.quality_flags)
        fps_int = int(round(fps))

        for phase in bundle.sub_phase_order:
            summary = bundle.sub_phases.get(phase)
            if summary is None:
                sub_phase_results[phase] = SubPhaseResult(
                    phase=phase,
                    is_scored=False,
                    skip_reason="Không có summary",
                )
                continue

            skip_reason = self._check_sub_phase_quality(
                phase, summary, quality_gate_cfg
            )
            if skip_reason:
                sub_phase_results[phase] = SubPhaseResult(
                    phase=phase,
                    is_scored=False,
                    skip_reason=skip_reason,
                    quality_flags=dict(bundle.quality_flags),
                )
                continue

            ref = self.reference_models.get(phase)
            if ref is None:
                sub_phase_results[phase] = SubPhaseResult(
                    phase=phase,
                    is_scored=False,
                    skip_reason=f"Chưa có ReferenceModel cho phase '{phase}'",
                )
                continue

            sub_bundle = self._make_sub_bundle(bundle, phase, summary)

            try:
                scorer = self._get_scorer(phase, fps_actual=fps_int)
                score_report = scorer.score(sub_bundle, ref)
                feedback = self._get_feedback_generator(phase).generate(
                    score_report, bundle=sub_bundle
                )
                _phase_is_scored = bool(getattr(score_report, "is_scored", True))
                _phase_skip_reason = (
                    "" if _phase_is_scored
                    else "; ".join(getattr(score_report, "notes", []) or [])
                )
                sub_phase_results[phase] = SubPhaseResult(
                    phase=phase,
                    result=PipelineResult(
                        phase=phase,
                        bundle=sub_bundle,
                        score_report=score_report,
                        feedback=feedback,
                    ),
                    is_scored=_phase_is_scored,
                    skip_reason=_phase_skip_reason,
                    quality_flags=dict(bundle.quality_flags),
                )
            except Exception as exc:
                sub_phase_results[phase] = SubPhaseResult(
                    phase=phase,
                    is_scored=False,
                    skip_reason=f"Lỗi chấm: {exc}",
                )

        # --- Tính overall + n_phases_scored (PL6) ---
        # --- Tính overall + n_phases_scored (PL6) ---
        # Chỉ đếm các phase có trọng số > 0 (ví dụ drive_phase weight=0 không tính).
        # Lý do: nếu đếm cả phase weight=0, một clip chỉ còn Tĩnh + Chạy lao
        # (weight 0.40) vẫn qua ngưỡng "≥2 phase" nhưng thực chất chỉ 40% trọng số.
        _phase_weights_map = (
            self.plugin.phase_weights()
            if hasattr(self.plugin, "phase_weights")
            else {}
        )
        n_scored = sum(
            1
            for phase, sp in sub_phase_results.items()
            if sp.is_scored and _phase_weights_map.get(phase, 1.0) > 0
        )
        overall = self._aggregate_overall(sub_phase_results)

        # PL6: nếu < 2 phase được chấm → overall = NaN, confidence = low.
        # Lý do: 2 phase có weight từ 0.60 đến 0.70 tùy phase bị bỏ
        # (0.70 nếu bỏ vao_cho hoặc sprint_start; 0.60 nếu bỏ static),
        # đủ để đánh giá tổng hợp; chỉ khi <2 phase (weight <0.50) mới không đủ cơ sở.
        # Khi chỉ có 2 phase (mất 1 phase có weight), hạ confidence xuống "low".

        # PL6: nếu < 2 phase được chấm → overall = NaN, confidence = low.
        # Lý do: 2 phase có weight (static 0.40 + sprint_start 0.30) = 0.70 trọng số,
        # đủ để đánh giá tổng hợp; chỉ khi <2 phase (weight <0.50) mới không đủ cơ sở.
        # Khi chỉ có 2 phase (mất 1 phase có weight), hạ confidence xuống "low".
        if n_scored < 2:
            overall = float("nan")
            confidence = "low"
        elif n_scored >= 3 and not all_quality_flags.get("low_fps", False):
            confidence = "high"
        elif n_scored == 2:
            confidence = "low"     # bị mất phase → cảnh báo rõ
        else:
            confidence = "medium"

        return SprintStartFullResult(
            sport=self.plugin.sport_name,
            sub_phase_results=sub_phase_results,
            sub_phase_order=list(bundle.sub_phase_order),
            overall_score=overall,
            quality_flags=all_quality_flags,
            score_confidence=confidence,
            phase_source=phase_source,
            n_phases_scored=n_scored,
        )

    # ------------------------------------------------------------------
    def _check_sub_phase_quality(
        self,
        phase: str,
        summary,
        gate_cfg: Dict,
    ) -> str:
        """
        Trả về lý do skip (string) hoặc '' nếu OK.
        Áp dụng cho cả 4 sub-phase, bao gồm drive_phase.

        V2.2 (PL5): với sprint_start, skip khi số PRIMARY NaN ≥ 3
        thay vì n_nan/total > 0.5 (tránh skip/không skip sai sau khi
        thêm boolean features).
        """
        if hasattr(summary, "window"):
            duration_s = summary.window.duration_s
            if duration_s < 0.3:
                return f"Sub-phase quá ngắn ({duration_s:.2f}s)"

        if phase == PHASE_DRIVE:
            n_steps = getattr(summary, "num_steps", 0)
            if n_steps < 2:
                return f"Không đủ bước để chấm trend ({n_steps} bước)"

        if hasattr(summary, "feature_values"):
            values = summary.feature_values
            if not values:
                return "Không có feature nào được trích xuất"

            nan_keys = [
                k for k, v in values.items()
                if isinstance(v, float) and v != v
            ]
            total = len(values)

            if phase == PHASE_SPRINT_START:
                # V2.4: đọc level từ feature_metadata của summary.
                # KHÔNG dùng gate_cfg["primary_features"] vì YAML không khai.
                primary_keys = set()
                meta = getattr(summary, "feature_metadata", None)
                if isinstance(meta, dict) and meta:
                    primary_keys = {
                        k for k, m in meta.items()
                        if isinstance(m, dict) and m.get("level") == "PRIMARY"
                    }
                if not primary_keys:
                    # Fallback theo catalog nếu summary thiếu metadata
                    from stage3_features.feature_extractor import (
                        SUBPHASE_FEATURE_CATALOG,
                    )
                    catalog = SUBPHASE_FEATURE_CATALOG.get(phase, {})
                    primary_keys = {
                        k for k, spec in catalog.items()
                        if spec.get("level") == "PRIMARY"
                    }
                n_primary_nan = sum(1 for k in nan_keys if k in primary_keys)
                if n_primary_nan >= 3:
                    return (
                        f"Không đủ dữ liệu PRIMARY "
                        f"({n_primary_nan} PRIMARY NaN: "
                        f"{', '.join(k for k in nan_keys if k in primary_keys)})"
                    )

        return ""

    # ------------------------------------------------------------------
    def _make_sub_bundle(
        self,
        full_bundle: FeatureBundle,
        phase: str,
        summary,
    ) -> FeatureBundle:
        """
        Tạo FeatureBundle mini chứa thông tin của 1 sub-phase, để tái sử dụng
        scorer + feedback generator hiện có (không cần viết lại logic chấm).

        Mapping:
            - vao_cho / static -> StaticPhaseSummary (feature_mean = values)
            - sprint_start     -> SprintStartPhaseSummary (event_features)
            - drive_phase      -> StaticPhaseSummary (feature_mean = 4 trend)
              Scorer._score_drive_phase() sẽ đọc từ static_summary.feature_mean
              và chấm với default_scoring_type="slope".
        """
        sub = FeatureBundle(
            phase=phase,
            fps=full_bundle.fps,
            total_frames=full_bundle.total_frames,
            valid_frame_ratio=full_bundle.valid_frame_ratio,
            state_summary=full_bundle.state_summary,
            normalized_df=full_bundle.normalized_df,
            quality_flags=full_bundle.quality_flags,
        )

        if phase in (PHASE_INTO_POSITION, PHASE_STATIC):
            from stage3_features.feature_extractor import StaticPhaseSummary
            sub.static_summary = StaticPhaseSummary(
                segment_start_frame=summary.window.start_frame,
                segment_end_frame=summary.window.end_frame,
                duration_s=summary.window.duration_s,
                angle_mean={},
                angle_std={},
                ratio_mean={},
                ratio_std={},
                feature_mean=dict(summary.feature_values),
                feature_std=dict(getattr(summary, "feature_std", {})),
            )

        elif phase == PHASE_SPRINT_START:
            from stage3_features.feature_extractor import SprintStartPhaseSummary
            sub.sprint_start_summary = SprintStartPhaseSummary(
                segment_start_frame=summary.window.start_frame,
                segment_end_frame=summary.window.end_frame,
                duration_s=summary.window.duration_s,
                launch_frame_idx=summary.launch_frame_idx,
                launch_timestamp_s=summary.launch_timestamp_s,
                peak_acceleration=summary.peak_acceleration,
                angles_at_launch={},
                trunk_angle_series=[],
                event_features=dict(summary.feature_values),
                launch_detection_method=summary.launch_detection_method,
                launch_confidence=summary.launch_confidence,
            )

        elif phase == PHASE_DRIVE:
            # DrivePhaseSummary chỉ có feature_values (4 trend) + num_steps.
            # Map sang static_summary để Scorer._score_drive_phase() đọc được
            # qua feature_mean.
            from stage3_features.feature_extractor import StaticPhaseSummary
            sub.static_summary = StaticPhaseSummary(
                segment_start_frame=summary.window.start_frame,
                segment_end_frame=summary.window.end_frame,
                duration_s=summary.window.duration_s,
                angle_mean={},
                angle_std={},
                ratio_mean={},
                ratio_std={},
                feature_mean=dict(summary.feature_values),
                feature_std={},
            )

        return sub

    # ------------------------------------------------------------------
    def _aggregate_overall(
        self, sub_phase_results: Dict[str, SubPhaseResult]
    ) -> float:
        """
        Tính điểm tổng từ các sub-phase, có trọng số phase_weights từ YAML.
        Chỉ tính trên các sub-phase đã được chấm thành công.
        """
        phase_weights = (
            self.plugin.phase_weights()
            if hasattr(self.plugin, "phase_weights")
            else {}
        )
        weighted = []
        for phase, sp in sub_phase_results.items():
            if not sp.has_score():
                continue
            score = sp.result.score_report.overall_score
            if score != score:  # NaN check
                continue
            w = phase_weights.get(phase, 1.0)
            if w > 0:
                weighted.append((score, w))
        if not weighted:
            return float("nan")
        total = sum(w for _, w in weighted)
        return (
            sum(s * w for s, w in weighted) / total
            if total > 0 else float("nan")
        )

    # ==================================================================
    # POSE EXTRACTOR + VIDEO
    # ==================================================================
    def _get_pose_extractor(self) -> PoseExtractor:
        if self._pose_extractor is None:
            backend = self.config.get("pose_backend", "mediapipe")
            kwargs = self.config.get("pose_backend_kwargs", {}) or {}
            self._pose_extractor = build_pose_extractor(backend=backend, **kwargs)
        return self._pose_extractor

    def _read_video_meta(self, video_path: str) -> Tuple[float, int, int]:
        """Đọc (fps, width, height) từ video; ném lỗi nếu không mở được."""
        if cv2 is None:
            raise RuntimeError(
                "Cần cài opencv-python: pip install opencv-python-headless"
            )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.get(
                "fallback_fps", 30.0
            )
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return fps, width, height

    @staticmethod
    def _bbox_area(det) -> float:
        """
        Diện tích bbox (x1, y1, x2, y2) của 1 detection.
        Trả về 0.0 nếu không có bbox / bbox hỏng / bbox suy biến.
        """
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            return 0.0
        try:
            x1, y1, x2, y2 = (float(v) for v in list(bbox)[:4])
        except (TypeError, ValueError):
            return 0.0
        w, h = x2 - x1, y2 - y1
        if w != w or h != h or w <= 0 or h <= 0:  # NaN hoặc không dương
            return 0.0
        return w * h

    @classmethod
    def _keep_largest_person(cls, per_frame_detections: list) -> list:
        """
        Mỗi frame chỉ giữ lại 1 detection có BBox lớn nhất (VĐV chính),
        loại các vật thể/người nhỏ ở nền (ghế đá, logo, khán giả...).

        - Frame không có detection -> giữ list rỗng (giữ nguyên chỉ số frame).
        - Phần tử frame không lặp được -> giữ nguyên, không đụng tới.
        """
        filtered: list = []
        n_nonempty = 0
        n_without_bbox = 0

        for dets in per_frame_detections:
            if not dets:
                filtered.append([])
                continue
            try:
                candidates = list(dets)
            except TypeError:
                filtered.append(dets)
                continue
            if not candidates:
                filtered.append([])
                continue

            n_nonempty += 1
            areas = [cls._bbox_area(d) for d in candidates]
            if max(areas) <= 0.0:
                n_without_bbox += 1
            best = candidates[
                max(range(len(candidates)), key=areas.__getitem__)
            ]
            filtered.append([best])

        if n_nonempty and n_without_bbox == n_nonempty:
            logger.warning(
                "[pipeline] Không detection nào có bbox hợp lệ -> bộ lọc "
                "'người lớn nhất' không có tác dụng (kiểm tra trường .bbox "
                "của pose extractor)."
            )
        elif n_without_bbox:
            logger.info(
                "[pipeline] %d/%d frame không có bbox hợp lệ khi lọc người lớn nhất.",
                n_without_bbox, n_nonempty,
            )
        return filtered

    def _extract_tracked(self, video_path: str, width: int, height: int):
        pose_extractor = self._get_pose_extractor()
        per_frame_detections = list(pose_extractor.extract_video(video_path))

        # Lọc: chỉ giữ người có BBox lớn nhất mỗi frame (tránh bắt nhầm ghế/logo)
        if self.config.get("select_largest_person", True):
            per_frame_detections = self._keep_largest_person(
                per_frame_detections
            )

        tracked = track_and_filter(
            per_frame_detections, frame_shape=(height, width)
        )
        if not tracked:
            raise ValueError(
                f"Không phát hiện được người nào trong video: {video_path}"
            )
        return tracked

    def process_video(
        self,
        video_path: str,
        phase: str,
        reference_model: Optional[ReferenceModel] = None,
    ) -> PipelineResult:
        fps, width, height = self._read_video_meta(video_path)
        tracked = self._extract_tracked(video_path, width, height)
        return self.process_tracked_sequence(
            tracked, fps=fps, phase=phase, reference_model=reference_model
        )

    def process_sprint_start_video(
        self,
        video_path: str,
        phases_path: Optional[str] = None,
    ) -> SprintStartFullResult:
        """
        Chấm full sprint_start từ video.
        video_id = stem của video -> tìm '<stem>.phases.json' khớp chính xác.
        Có thể truyền phases_path để chỉ định thẳng file phases.
        """
        fps, width, height = self._read_video_meta(video_path)
        tracked = self._extract_tracked(video_path, width, height)
        return self.process_sprint_start_sequence(
            tracked,
            fps=fps,
            video_id=Path(video_path).stem,
            phases_path=phases_path,
        )
