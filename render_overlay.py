"""
render_overlay.py  —  V2.3
==================================
Xuất video MP4 overlay hiển thị TỪNG GIAI ĐOẠN của kỹ thuật xuất phát thấp.

CHANGELOG V2.3 (đồng bộ với core/pipeline.py V2.1.5):
    - FIX LỖI CHÍNH: truyền video_id (stem của video) vào
      process_sprint_start_sequence() -> pipeline tìm đúng file
      '<video_stem>.phases.json' thay vì rơi về auto-segmentation.
    - Dùng chung Pipeline._extract_tracked() để trích pose: bộ lọc "người có
      BBox lớn nhất" nằm một chỗ duy nhất (không còn nhân bản code, và
      render_overlay / app_demo luôn cho cùng một kết quả).
    - Chỉ khởi tạo Pipeline MỘT lần (trích pose + chấm điểm dùng chung).
    - Hiển thị NGUỒN PHASES (manual / auto / ước lượng) trên panel và in ra
      console để phát hiện ngay trường hợp dùng sai phases.
    - Thêm tham số --phases để chỉ định thẳng file phases.
    - Panel + timeline luôn được vẽ (kể cả frame mất pose, có cảnh báo),
      tránh nhấp nháy.
    - Gộp toàn bộ chữ của 1 frame vào 1 lần chuyển đổi PIL (nhanh hơn nhiều
      so với chuyển ảnh cho từng dòng chữ).
    - Không còn biến global _BACKEND_NAME (dùng RenderContext).
    - Fallback chia phase theo % dựa trên frame_idx thật, có gắn nhãn rõ
      "ước lượng"; kiểm tra VideoWriter mở được; in traceback khi lỗi.

CHANGELOG V2.2:
    - phase_at_frame() trả PHASE_NONE cho frame ngoài mọi phase.
    - draw_panel() hiển thị đúng "0 frames" cho phase bị skip.
    - Format reference_spread an toàn khi NaN.
    - Skeleton không tô đỏ khớp focus khi đang ở PHASE_NONE.

Chạy:
    python -m render_overlay --video "data/raw_videos/coach_HLV04_running_v006.mp4" \
        --out "output/verify/overlay.mp4"

LƯU Ý: pose cache cũ (tạo trước khi có bộ lọc người lớn nhất) có thể chứa
nhầm đối tượng (ghế, logo...). Nếu nghi ngờ, xóa file trong data/pose_cache/
để trích xuất lại.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from core.pose_schema import PoseFrame
from core.pipeline import Pipeline, SprintStartFullResult
from stage3_features.feature_extractor import (
    PHASE_DRIVE,
    PHASE_INTO_POSITION,
    PHASE_SPRINT_START,
    PHASE_STATIC,
)


# =====================================================================
# Cấu hình hiển thị
# =====================================================================

PHASE_NONE = "none"   # frame ngoài mọi phase

PHASE_ORDER_TIMELINE = [
    PHASE_INTO_POSITION, PHASE_STATIC, PHASE_SPRINT_START, PHASE_DRIVE,
]

PHASE_COLORS = {
    PHASE_INTO_POSITION: (180, 180, 180),
    PHASE_STATIC:        (0, 220, 255),
    PHASE_SPRINT_START:  (0, 140, 255),
    PHASE_DRIVE:         (80, 220, 80),
    PHASE_NONE:          (150, 150, 150),
}

PHASE_LABELS_VI = {
    PHASE_INTO_POSITION: "VÀO CHỖ (Into position)",
    PHASE_STATIC:        "SẴN SÀNG (Set)",
    PHASE_SPRINT_START:  "BỘC PHÁT (Explosive start)",
    PHASE_DRIVE:         "CHẠY LAO (Drive phase)",
    PHASE_NONE:          "NGOÀI VÙNG ĐÁNH DẤU",
}

PHASE_SHORT_LABELS = {
    PHASE_INTO_POSITION: "vao_cho",
    PHASE_STATIC:        "static",
    PHASE_SPRINT_START:  "sprint_start",
    PHASE_DRIVE:         "drive_phase",
}

SKELETON_EDGES_COCO17: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SKELETON_EDGES_POSE33: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

PHASE_FOCUS_JOINTS = {
    PHASE_INTO_POSITION: {
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_wrist", "right_wrist",
    },
    PHASE_STATIC: {
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
    },
    PHASE_SPRINT_START: {
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_wrist", "right_wrist",
    },
    PHASE_DRIVE: {
        "left_knee", "right_knee", "left_hip", "right_hip",
    },
    PHASE_NONE: set(),
}

PHASE_HINTS = {
    PHASE_INTO_POSITION: [
        "• Hai tay rộng bằng vai, chống đất",
        "• Gối sau chạm đất, gối trước gập ~100°",
    ],
    PHASE_STATIC: [
        "• Hông cao hơn vai 10-20cm",
        "• Gối trước ~90°, gối sau 120-140°",
    ],
    PHASE_SPRINT_START: [
        "• Chân sau rời bàn đạp trước",
        "• Thân nghiêng 25-30° so với phương thẳng đứng",
    ],
    PHASE_DRIVE: [
        "• Độ dài bước tăng dần",
        "• Thân dựng thẳng dần",
    ],
}

PhaseWindows = Dict[str, Tuple[int, int]]
TextItem = Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]


@dataclass(frozen=True)
class RenderContext:
    """Thông tin dùng chung cho mọi frame khi render."""
    fps: float
    total_frames: int
    backend_name: str
    phase_source: str                      # auto / manual:<path> / fallback
    phases: PhaseWindows
    result: Optional[SprintStartFullResult]


# =====================================================================
# Font tiếng Việt + vẽ chữ (gộp theo frame)
# =====================================================================

_FONT_CACHE: Dict[int, object] = {}
_FONT_CANDIDATES = (
    "arial.ttf", "segoeui.ttf", "tahoma.ttf", "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
)


def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    if _HAS_PIL:
        for name in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(name, size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_texts_bgr(img_bgr: np.ndarray, items: List[TextItem]) -> np.ndarray:
    """
    Vẽ NHIỀU dòng chữ chỉ với 1 lần chuyển đổi BGR<->RGB (nhanh hơn nhiều
    so với chuyển ảnh cho từng dòng). items: (text, (x, y), font_size, bgr).
    """
    if not items:
        return img_bgr
    if _HAS_PIL:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for text, pos, size, color_bgr in items:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(pos, text, font=_get_font(size), fill=color_rgb)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for text, pos, size, color_bgr in items:
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
        cv2.putText(
            img_bgr, ascii_text, (pos[0], pos[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
            size / 32.0, color_bgr, 1, cv2.LINE_AA,
        )
    return img_bgr


# =====================================================================
# Pose cache I/O
# =====================================================================

def load_pose_cache(path: str) -> List[PoseFrame]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Pose cache không hỗ trợ format: {type(raw)}")
    frames: List[PoseFrame] = []
    for item in items:
        try:
            frames.append(PoseFrame.from_dict(item))
        except Exception:
            continue
    return frames


def save_pose_cache(path: str, frames: List[PoseFrame]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([pf.to_dict() for pf in frames], f, ensure_ascii=False)


# =====================================================================
# Trích pose (dùng chung logic với Pipeline)
# =====================================================================

def extract_poses(
    pipeline: Pipeline,
    video_path: str,
) -> List[PoseFrame]:
    """
    Trích pose bằng Pipeline._extract_tracked(): đã bao gồm bộ lọc
    'người có BBox lớn nhất' + tracker, giống hệt đường chạy của app_demo.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return pipeline._extract_tracked(video_path, width, height)


# =====================================================================
# Phân đoạn + chấm điểm
# =====================================================================

def _fallback_phases(frames: List[PoseFrame]) -> PhaseWindows:
    """Chia phase theo % độ dài (chỉ là ƯỚC LƯỢNG để hiển thị)."""
    first = frames[0].frame_idx
    last = frames[-1].frame_idx
    span = max(last - first, 1)

    def at(ratio: float) -> int:
        return first + int(span * ratio)

    return {
        PHASE_INTO_POSITION: (at(0.00), at(0.15)),
        PHASE_STATIC:        (at(0.15), at(0.30)),
        PHASE_SPRINT_START:  (at(0.30), at(0.35)),
        PHASE_DRIVE:         (at(0.35), last),
    }

def _load_raw_windows(
    phases_path: Optional[str], video_id: Optional[str]
) -> PhaseWindows:
    """Đọc ranh giới pha thô từ file phases, để tô màu cả pha không được chấm điểm."""
    import json

    candidates = []
    if phases_path:
        candidates.append(Path(phases_path))
    if video_id:
        candidates.append(Path("data/phases") / f"{video_id}.phases.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out: PhaseWindows = {}
            for name, spec in (data.get("phases") or {}).items():
                if isinstance(spec, dict) and not spec.get("absent"):
                    out[name] = (int(spec["start_frame"]), int(spec["end_frame"]))
            if out:
                return out
        except Exception as exc:
            print(f"[render_overlay] Không đọc được {path}: {exc}")
    return {}


def detect_and_score(
    frames: List[PoseFrame],
    fps: float,
    pipeline: Optional[Pipeline],
    video_id: Optional[str] = None,
    phases_path: Optional[str] = None,
) -> Tuple[PhaseWindows, Optional[SprintStartFullResult], str]:
    """
    Trả về (phases, result, phase_source).

    video_id    : stem của video -> pipeline tìm '<stem>.phases.json'.
    phases_path : chỉ định thẳng file phases (ưu tiên cao nhất).
    phase_source: 'auto' | 'manual:<path>' | 'fallback' (chia % ước lượng).
    """
    fallback = _fallback_phases(frames)

    if pipeline is None:
        print("[render_overlay] Không có pipeline -> dùng phases ước lượng.")
        return fallback, None, "fallback"

    try:
        result = pipeline.process_sprint_start_sequence(
            frames, fps=fps, video_id=video_id, phases_path=phases_path,
        )
    except Exception as exc:
        print(f"[render_overlay] Lỗi chấm điểm: {exc}. Dùng phases ước lượng.")
        traceback.print_exc()
        return fallback, None, "fallback"

    raw_windows = _load_raw_windows(phases_path, video_id)

    phases: PhaseWindows = {}
    for ph, sp in result.sub_phase_results.items():
        if not sp.has_score():
            # Pha không chấm điểm vẫn được tô màu theo ranh giới trong file phases
            phases[ph] = raw_windows.get(ph, (0, 0))
            continue

        bundle = sp.result.bundle
        summary = None
        for attr in ("static_summary", "sprint_start_summary"):
            obj = getattr(bundle, attr, None)
            if obj is not None and hasattr(obj, "segment_start_frame"):
                summary = obj
                break
        if summary is not None:
            phases[ph] = (summary.segment_start_frame, summary.segment_end_frame)
        else:
            print(f"[render_overlay] Cảnh báo: phase '{ph}' đã chấm nhưng "
                  f"không lấy được mốc frame -> coi như không hiển thị.")
            phases[ph] = (0, 0)

    if not phases:
        print("[render_overlay] Không có sub-phase nào chấm được. Dùng ước lượng.")
        return fallback, result, "fallback"

    return phases, result, getattr(result, "phase_source", "auto")


def phase_at_frame(frame_idx: int, phases: PhaseWindows) -> str:
    """Trả PHASE_NONE nếu frame nằm ngoài mọi phase (không fallback sai)."""
    for phase in (PHASE_SPRINT_START, PHASE_STATIC, PHASE_INTO_POSITION, PHASE_DRIVE):
        if phase not in phases:
            continue
        s, e = phases[phase]
        if s != e and s <= frame_idx <= e:
            return phase
    return PHASE_NONE


def _phase_source_label(source: str) -> str:
    if source.startswith("manual-fuzzy"):
        return "thủ công (khớp mờ!)"
    if source.startswith("manual"):
        return "thủ công"
    if source == "auto":
        return "tự động"
    return "ước lượng %"


# =====================================================================
# Vẽ
# =====================================================================

def _resolve_edges(joint_names: List[str]) -> List[Tuple[int, int]]:
    if len(joint_names) >= 33:
        return SKELETON_EDGES_POSE33
    return SKELETON_EDGES_COCO17


def draw_skeleton(
    frame: np.ndarray,
    pf: PoseFrame,
    phase: str,
    focus_joints: set,
) -> None:
    joint_names = list(pf.joint_names)
    edges = _resolve_edges(joint_names)
    base_color = PHASE_COLORS.get(phase, (200, 200, 200))
    focus_color = (0, 0, 255)
    conf_thr = 0.3
    n_joints = min(len(pf.xy), len(pf.conf))

    for j1, j2 in edges:
        if j1 >= n_joints or j2 >= n_joints:
            continue
        if pf.conf[j1] < conf_thr or pf.conf[j2] < conf_thr:
            continue
        if np.isnan(pf.xy[j1]).any() or np.isnan(pf.xy[j2]).any():
            continue
        pt1 = (int(pf.xy[j1][0]), int(pf.xy[j1][1]))
        pt2 = (int(pf.xy[j2][0]), int(pf.xy[j2][1]))
        n1 = joint_names[j1] if j1 < len(joint_names) else ""
        n2 = joint_names[j2] if j2 < len(joint_names) else ""
        edge_color = focus_color if (n1 in focus_joints and n2 in focus_joints) else base_color
        cv2.line(frame, pt1, pt2, edge_color, 3, cv2.LINE_AA)

    for j in range(n_joints):
        pt, c = pf.xy[j], pf.conf[j]
        if c < conf_thr or np.isnan(pt).any():
            continue
        name = joint_names[j] if j < len(joint_names) else ""
        in_focus = name in focus_joints
        radius = 6 if in_focus else 4
        color = focus_color if in_focus else (255, 255, 255)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1, cv2.LINE_AA)


def _fmt_spread(value) -> str:
    """Format an toàn khi NaN/None."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(v):
        return ""
    return f" ± {v:.1f}"


def draw_panel(
    frame: np.ndarray,
    texts: List[TextItem],
    ctx: RenderContext,
    frame_idx: int,
    phase: str,
    has_pose: bool,
) -> None:
    """Vẽ nền + viền panel lên frame; chữ được gom vào `texts` để vẽ 1 lần."""
    panel_w, panel_h = 620, 320
    x0, y0 = 15, 15
    phase_color = PHASE_COLORS.get(phase, (200, 200, 200))
    phase_window = ctx.phases.get(phase, (0, 0))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), phase_color, 2)
    cv2.line(frame, (x0 + 10, y0 + 62), (x0 + panel_w - 10, y0 + 62), (100, 100, 100), 1)

    header = (f"Frame {frame_idx}  |  {ctx.fps:.0f} FPS  |  {ctx.backend_name}"
              f"  |  Phases: {_phase_source_label(ctx.phase_source)}")
    texts.append((header, (x0 + 10, y0 + 8), 15, (220, 220, 220)))
    texts.append((PHASE_LABELS_VI.get(phase, phase), (x0 + 10, y0 + 30), 20, phase_color))

    y = y0 + 74
    spr = ctx.result.sub_phase_results.get(phase) if ctx.result is not None else None

    if phase == PHASE_NONE:
        texts.append(("Frame nằm ngoài vùng đánh dấu 4 sub-phase.",
                      (x0 + 10, y), 14, (200, 200, 200)))
        texts.append(("Chỉ hiển thị skeleton, không chấm điểm.",
                      (x0 + 10, y + 22), 13, (180, 180, 180)))

    elif spr is not None:
        if spr.has_score():
            sr = spr.result.score_report
            texts.append((
                f"Điểm phase: {sr.overall_score:.0f}/100  "
                f"(PRIMARY fail: {sr.n_primary_failed}, SECONDARY: {sr.n_secondary_failed})",
                (x0 + 10, y), 15, (255, 255, 200),
            ))
            y += 24

            fails = sorted(
                [fs for fs in sr.feature_scores
                 if not fs.is_in_range and not fs.is_info_only],
                key=lambda fs: fs.score_0_100,
            )[:4]
            for fs in fails:
                line = (f"  × {fs.feature_key}: {fs.student_value:.1f}"
                        f"  (chuẩn {fs.reference_center:.1f}"
                        f"{_fmt_spread(fs.reference_spread)})")
                texts.append((line, (x0 + 10, y), 13, (200, 180, 255)))
                y += 20
            if not fails:
                texts.append(("  OK - Tất cả feature đều đạt",
                              (x0 + 10, y), 13, (200, 255, 200)))
        else:
            reason = spr.skip_reason or "Không chấm được"
            texts.append((f"Bỏ qua: {reason}", (x0 + 10, y), 14, (200, 200, 120)))

    else:
        for hint in PHASE_HINTS.get(phase, []):
            texts.append((hint, (x0 + 10, y), 13, (220, 220, 220)))
            y += 20

    if not has_pose:
        texts.append(("⚠ Không có pose ở frame này",
                      (x0 + 10, y0 + panel_h - 46), 13, (0, 165, 255)))

    s, e = phase_window
    n_frames = (e - s + 1) if e > s else 0
    texts.append((
        f"Sub-phase window: frame {s} → {e}  ({n_frames} frames)",
        (x0 + 10, y0 + panel_h - 24), 12, (180, 180, 180),
    ))


def draw_timeline(
    frame: np.ndarray,
    texts: List[TextItem],
    ctx: RenderContext,
    frame_idx: int,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 42
    y_bar = h - bar_h - 10
    x_bar0 = 15
    x_bar1 = w - 15
    bar_w = x_bar1 - x_bar0
    denom = max(ctx.total_frames - 1, 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_bar0, y_bar), (x_bar1, y_bar + bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for phase in PHASE_ORDER_TIMELINE:
        if phase not in ctx.phases:
            continue
        s, e = ctx.phases[phase]
        if e <= s:
            continue
        x1 = x_bar0 + int(bar_w * min(s / denom, 1.0))
        x2 = x_bar0 + int(bar_w * min(e / denom, 1.0))
        cv2.rectangle(
            frame, (x1, y_bar + 6), (x2, y_bar + bar_h - 6),
            PHASE_COLORS[phase], -1,
        )
        if x2 - x1 > 60:
            texts.append((PHASE_SHORT_LABELS[phase], (x1 + 6, y_bar + 12), 12, (30, 30, 30)))

    x_cur = x_bar0 + int(bar_w * min(frame_idx / denom, 1.0))
    cv2.line(frame, (x_cur, y_bar), (x_cur, y_bar + bar_h), (255, 255, 255), 3)


# =====================================================================
# Main render
# =====================================================================

def render_overlay(
    video_path: str,
    pose_json: Optional[str],
    out_path: str,
    system_config_path: str = "config/system_config.yaml",
    phases_path: Optional[str] = None,
) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    video_id = Path(video_path).stem
    use_cache = bool(pose_json) and Path(pose_json).exists()

    # --- Khởi tạo Pipeline MỘT lần (dùng cho cả trích pose lẫn chấm điểm) ---
    pipeline: Optional[Pipeline] = None
    try:
        pipeline = Pipeline.from_config(system_config_path, sport="sprint_start")
    except Exception as exc:
        print(f"[render_overlay] Lỗi load pipeline: {exc}")
        if not use_cache:
            raise   # không có cache và không có pipeline -> không thể trích pose

    # --- Lấy pose ---
    print("[render_overlay] Đang load pose...")
    if use_cache:
        frames = load_pose_cache(pose_json)
        backend_name = "cache"
        print(f"[render_overlay] Đã load {len(frames)} frame từ pose cache.")
    else:
        backend_name = pipeline.config.get("pose_backend", "unknown")
        print(f"[render_overlay] Không có pose cache — chạy pose extraction "
              f"(backend: {backend_name})...")
        frames = extract_poses(pipeline, video_path)
        print(f"[render_overlay] Đã trích xuất {len(frames)} frame.")

        cache_path = Path("data/pose_cache") / f"{video_id}.json"
        save_pose_cache(str(cache_path), frames)
        print(f"[render_overlay] Đã lưu pose cache: {cache_path}")

    if not frames:
        raise ValueError("Không có pose frame nào để render.")

    # --- Mở video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Phân đoạn + chấm điểm ---
    print(f"[render_overlay] Đang phân đoạn + chấm điểm "
          f"(video_id={video_id}, fps={fps:.1f})...")
    phases, result, phase_source = detect_and_score(
        frames, fps, pipeline, video_id=video_id, phases_path=phases_path,
    )
    print(f"[render_overlay] Nguồn phases: {phase_source}")
    for ph, (s, e) in phases.items():
        print(f"  - {ph}: frame {s} → {e} ({(e - s + 1) if e > s else 0} frames)")
    if result is not None:
        print(f"[render_overlay] Điểm tổng: {result.overall_score:.0f}/100 "
              f"(độ tin cậy: {result.score_confidence})")
    if phase_source == "auto":
        print("[render_overlay] ⚠ Đang dùng phases TỰ ĐỘNG. Nếu bạn đã đánh dấu "
              f"thủ công, kiểm tra file '{video_id}.phases.json' trong thư mục phases.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (frames[-1].frame_idx + 1)
    ctx = RenderContext(
        fps=fps,
        total_frames=total_frames,
        backend_name=backend_name,
        phase_source=phase_source,
        phases=phases,
        result=result,
    )

    # --- Render ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được file video output: {out_path}")

    frame_idx_to_pf = {pf.frame_idx: pf for pf in frames}
    print(f"[render_overlay] Bắt đầu render {total_frames} frame...")

    cur_frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            phase = phase_at_frame(cur_frame_idx, phases)
            pf = frame_idx_to_pf.get(cur_frame_idx)
            if pf is not None:
                draw_skeleton(frame, pf, phase, PHASE_FOCUS_JOINTS.get(phase, set()))

            texts: List[TextItem] = []
            draw_panel(frame, texts, ctx, cur_frame_idx, phase, has_pose=pf is not None)
            draw_timeline(frame, texts, ctx, cur_frame_idx)
            frame = draw_texts_bgr(frame, texts)

            writer.write(frame)
            cur_frame_idx += 1
            if cur_frame_idx % 100 == 0:
                print(f"  ... đã render {cur_frame_idx}/{total_frames} frame")
    finally:
        cap.release()
        writer.release()

    print(f"[render_overlay] ✅ Đã xuất video: {out_path}")


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    # Bật log INFO để thấy thông báo chọn file phases của core/pipeline.py
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Render video overlay 4 sub-phase sprint_start (RTMPose ONNX).",
    )
    parser.add_argument("--video", required=True, help="Đường dẫn video mp4 gốc")
    parser.add_argument("--pose-json", default=None, help="(Tùy chọn) pose cache JSON")
    parser.add_argument("--phases", default=None,
                        help="(Tùy chọn) chỉ định thẳng file *.phases.json")
    parser.add_argument("--out", default="output/verify/sprint_start_overlay.mp4",
                        help="Đường dẫn video output")
    parser.add_argument("--system-config", default="config/system_config.yaml",
                        help="Đường dẫn system_config.yaml")
    args = parser.parse_args()

    render_overlay(
        video_path=args.video,
        pose_json=args.pose_json,
        out_path=args.out,
        system_config_path=args.system_config,
        phases_path=args.phases,
    )


if __name__ == "__main__":
    main()