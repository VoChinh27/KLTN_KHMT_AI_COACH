"""
tools/mark_phases.py
====================
Công cụ đánh dấu thủ công 4 sub-phase (vao_cho / static / sprint_start / drive_phase)
trên video HLV demo — HLV xem video và ghi lại mốc chuyển phase chính xác
đến từng frame, thay vì để pipeline tự đoán.

CÁCH DÙNG
---------
1) Chạy KHÔNG tham số -> hiện MENU để chọn:
       [1] Một video          (nhập đường dẫn file video)
       [2] Cả thư mục video   (nhập đường dẫn thư mục; lần lượt xử lý từng video)

       python tools/mark_phases.py

2) Hoặc truyền thẳng tham số:
       python tools/mark_phases.py --video "data/raw_videos/coach_HLV04_running_v006.mp4"
       python tools/mark_phases.py --folder "data/raw_videos"
       python tools/mark_phases.py --folder "data/raw_videos" --recursive --only-new

   Mặc định, nếu file mốc cũ đã tồn tại thì được TỰ ĐỘNG nạp lại để chỉnh sửa
   (thêm --fresh nếu muốn bắt đầu lại từ đầu).

PHÍM TẮT
--------
  PHÁT / TUA
    SPACE             : Phát / Tạm dừng
    ← → | A D | J L   : Lùi / tiến 1 frame
    ↑ ↓ | ] [         : Tiến / lùi 10 frame
    < >  (, .) | PgUp/PgDn : Lùi / tiến 1 giây
    Home / End        : Về đầu / cuối video
    + / -             : Tăng / giảm tốc độ phát (0.1x … 4x)
    CHUỘT             : Click / kéo trên TIMELINE để tua nhanh

  ĐÁNH DẤU (đánh dấu tại frame đang hiển thị)
    1 : BẮT ĐẦU "VÀO CHỖ"        2 : BẮT ĐẦU "SẴN SÀNG"
    3 : BẮT ĐẦU "BỘC PHÁT"       4 : BẮT ĐẦU "CHẠY LAO"
    5 : KẾT THÚC (END) của video
    CHUỘT: click vào dòng pha ở bảng = đánh dấu; chuột phải = xóa mốc đó

  XÓA
    Shift+1..5  hoặc  C rồi 1..5 : Xóa mốc tương ứng
    X                            : Xóa TẤT CẢ mốc (có thể hoàn tác)

  NHẢY ĐẾN MỐC
    6 7 8 9 : nhảy tới mốc 1..4        0 : nhảy tới END

  HOÀN TÁC
    U hoặc Ctrl+Z : Hoàn tác        R hoặc Ctrl+Y : Làm lại

  LƯU / CHUYỂN VIDEO / THOÁT
    S : Lưu file JSON (hợp lệ mới lưu) rồi TỰ CHUYỂN sang video kế tiếp
    N : Video kế tiếp (không lưu)       B : Video trước
    Q : Thoát (nếu có thay đổi chưa lưu sẽ hỏi lại — bấm Q lần nữa để xác nhận)

ĐẦU RA (KHÔNG ĐỔI so với bản cũ)
--------------------------------
  data/phases/<video_stem>.phases.json
  {
    "video": "<tên file>", "fps": 30.0, "total_frames": 450,
    "phases": {
      "vao_cho":      {"start_frame": 0,   "end_frame": 119},
      "static":       {"start_frame": 120, "end_frame": 239},
      "sprint_start": {"start_frame": 240, "end_frame": 259},
      "drive_phase":  {"start_frame": 260, "end_frame": 449}
    },
    "marked_by": "manual"
  }

YÊU CẦU: opencv-python + numpy. Khuyến nghị cài thêm Pillow (pip install pillow)
để hiển thị tiếng Việt CÓ DẤU trên cửa sổ (cv2.putText không vẽ được chữ có dấu);
không có Pillow chương trình tự chuyển sang hiển thị không dấu.
"""
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # Pillow chỉ dùng để vẽ chữ Unicode (tiếng Việt có dấu)
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


# =====================================================================
# Hằng số
# =====================================================================

PHASE_ORDER = ["vao_cho", "static", "sprint_start", "drive_phase"]
PHASE_LABELS_VI = {
    "vao_cho":      "VÀO CHỖ",
    "static":       "SẴN SÀNG",
    "sprint_start": "BỘC PHÁT",
    "drive_phase":  "CHẠY LAO",
}
PHASE_HINTS_VI = {
    "vao_cho":      "đứng, chống tay xuống đất",
    "static":       "nâng hông, giữ tư thế chờ",
    "sprint_start": "đạp bàn đạp, rời đất",
    "drive_phase":  "chạy lao tăng tốc",
}
PHASE_COLORS = {  # BGR
    "vao_cho":      (180, 180, 180),   # xám
    "static":       (0, 220, 255),     # vàng
    "sprint_start": (0, 140, 255),     # cam
    "drive_phase":  (80, 220, 80),     # xanh lá
}
PHASE_KEY_INDEX = {"vao_cho": 1, "static": 2, "sprint_start": 3, "drive_phase": 4}
PHASE_BY_KEY = {"1": "vao_cho", "2": "static", "3": "sprint_start", "4": "drive_phase"}
REQUIRED_PHASES = ("static", "sprint_start", "drive_phase")  # vao_cho mặc định = frame 0

END_KEY = "5"
END_LABEL = "KẾT THÚC"
END_HINT = "hết video / hết chạy lao"
COLOR_END = (255, 0, 200)          # magenta
COLOR_PLAYHEAD = (0, 0, 255)       # đỏ
COLOR_PANEL_BG = (30, 30, 30)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg", ".flv", ".mts", ".m2ts"}
DEFAULT_OUT_DIR = Path("data/phases")

WINDOW_NAME = "Mark Sprint Start Phases"
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_IDX = 3

# Bố cục cửa sổ (px)
CANVAS_W = 1120
VIDEO_H = 560
PANEL_Y = VIDEO_H
PANEL_H = 300
CANVAS_H = VIDEO_H + PANEL_H
TOP_BAR_H = 36
BOTTOM_BANNER_H = 40
TL_X0, TL_X1 = 20, CANVAS_W - 20
TL_Y0, TL_H = PANEL_Y + 24, 34
INFO_Y = PANEL_Y + 64
TABLE_X0, TABLE_X1 = 20, 745
TABLE_Y, ROW_H = PANEL_Y + 92, 27
HELP_X, HELP_Y, HELP_DY = 757, PANEL_Y + 92, 17
GUIDE_Y = PANEL_Y + 236
TOAST_Y = PANEL_Y + 262

# Bộ nhớ: preload toàn bộ video (nén JPEG) để tua chính xác từng frame
MAX_PRELOAD_FRAMES = 6000
LAZY_CACHE_FRAMES = 96

TOAST_COLORS = {
    "ok": (90, 220, 90),
    "warn": (0, 200, 255),
    "error": (80, 80, 255),
    "info": (230, 230, 230),
}

HELP_LINES = [
    "SPACE phát/dừng   +/- tốc độ",
    "← →  A D  J L : ±1 frame",
    "↑ ↓  [ ] : ±10 frame",
    "< >  PgUp PgDn : ±1 giây   Home End",
    "1-4 : BẮT ĐẦU pha    5 : KẾT THÚC",
    "Shift+1..5 / C+1..5 : xóa   X : xóa hết",
    "6-9, 0 : nhảy tới mốc    U/R : hoàn tác/làm lại",
    "S lưu   N/B video tiếp/trước   Q thoát",
]


# =====================================================================
# Tiện ích
# =====================================================================

def fmt_time(frame: int, fps: float) -> str:
    return f"{frame / max(fps, 1e-6):.2f}s"


def clean_path(raw: str) -> str:
    """Bỏ khoảng trắng + dấu nháy (khi kéo-thả / 'Copy as path' trên Windows)."""
    s = raw.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return os.path.expanduser(s)


def natural_key(s: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_videos(folder: Path, recursive: bool = False) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    vids = [
        p for p in it
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("._")
    ]
    return sorted(vids, key=lambda p: natural_key(str(p.relative_to(folder))))


def build_out_paths(videos: List[Path], out_dir: Path) -> Dict[Path, Path]:
    """<out_dir>/<stem>.phases.json. Nếu 2 video trùng stem (khi quét đệ quy)
    thì thêm tên thư mục cha để KHÔNG ghi đè lẫn nhau."""
    stems: Dict[str, int] = {}
    for v in videos:
        stems[v.stem] = stems.get(v.stem, 0) + 1
    out: Dict[Path, Path] = {}
    for v in videos:
        name = v.stem if stems[v.stem] == 1 else f"{v.parent.name}__{v.stem}"
        out[v] = out_dir / f"{name}.phases.json"
    return out


# =====================================================================
# MarkerState — mốc + undo/redo
# =====================================================================

class MarkerState:
    """Lưu 4 mốc START + 1 mốc END và lịch sử để hoàn tác / làm lại."""

    MAX_HISTORY = 50

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.marks: Dict[str, Optional[int]] = {p: None for p in PHASE_ORDER}
        self.end_frame: Optional[int] = None
        self.version = 0  # tăng mỗi khi đổi -> UI biết cần vẽ lại
        self._undo: Deque = deque(maxlen=self.MAX_HISTORY)
        self._redo: Deque = deque(maxlen=self.MAX_HISTORY)
        self._saved = self._snapshot()

    # --- snapshot ---
    def _snapshot(self):
        return (tuple(self.marks[p] for p in PHASE_ORDER), self.end_frame)

    def _restore(self, snap) -> None:
        starts, end = snap
        self.marks = {p: s for p, s in zip(PHASE_ORDER, starts)}
        self.end_frame = end
        self.version += 1

    def _commit(self, mutate) -> bool:
        before = self._snapshot()
        mutate()
        if self._snapshot() == before:
            return False
        self._undo.append(before)
        self._redo.clear()
        self.version += 1
        return True

    # --- thao tác ---
    def set_mark(self, phase: str, frame_idx: int) -> bool:
        return self._commit(lambda: self.marks.__setitem__(phase, int(frame_idx)))

    def clear_mark(self, phase: str) -> bool:
        return self._commit(lambda: self.marks.__setitem__(phase, None))

    def set_end(self, frame_idx: int) -> bool:
        return self._commit(lambda: setattr(self, "end_frame", int(frame_idx)))

    def clear_end(self) -> bool:
        return self._commit(lambda: setattr(self, "end_frame", None))

    def clear_all(self) -> bool:
        def _do():
            for p in PHASE_ORDER:
                self.marks[p] = None
            self.end_frame = None
        return self._commit(_do)

    def undo(self) -> bool:
        if not self._undo:
            return False
        snap = self._undo.pop()
        self._redo.append(self._snapshot())
        self._restore(snap)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        snap = self._redo.pop()
        self._undo.append(self._snapshot())
        self._restore(snap)
        return True

    # --- trạng thái lưu ---
    def mark_saved(self) -> None:
        self._saved = self._snapshot()
        self.version += 1

    @property
    def is_dirty(self) -> bool:
        return self._snapshot() != self._saved

    def has_any_mark(self) -> bool:
        return any(m is not None for m in self.marks.values()) or self.end_frame is not None

    # --- nạp từ JSON cũ ---
    def load(self, phases: Dict[str, Dict]) -> None:
        last = self.total_frames - 1
        for p in PHASE_ORDER:
            if p in phases and "start_frame" in phases[p]:
                self.marks[p] = max(0, min(last, int(phases[p]["start_frame"])))
        if "drive_phase" in phases and "end_frame" in phases["drive_phase"]:
            end = max(0, min(last, int(phases["drive_phase"]["end_frame"])))
            # end == frame cuối chính là mặc định -> coi như chưa đặt END
            self.end_frame = None if end >= last else end
        self.version += 1
        self.mark_saved()

    # --- kiểm tra ---
    def missing_required(self) -> List[str]:
        return [p for p in REQUIRED_PHASES if self.marks[p] is None]

    def validation_errors(self) -> List[str]:
        """Lỗi NHẤT QUÁN giữa các mốc đã đặt (không tính mốc còn thiếu)."""
        errors: List[str] = []
        last = self.total_frames - 1
        chain = [(p, self.marks[p]) for p in PHASE_ORDER if self.marks[p] is not None]

        for p, f in chain:
            if f < 0 or f > last:
                errors.append(f"Mốc '{PHASE_LABELS_VI[p]}' (frame {f}) nằm ngoài video [0..{last}]")

        for (p1, f1), (p2, f2) in zip(chain, chain[1:]):
            if f1 >= f2:
                errors.append(
                    f"Mốc '{PHASE_LABELS_VI[p1]}' (frame {f1}) phải NHỎ HƠN mốc '{PHASE_LABELS_VI[p2]}' (frame {f2})"
                )

        if self.marks["vao_cho"] is None and chain and chain[0][1] == 0:
            errors.append(
                f"'{PHASE_LABELS_VI[chain[0][0]]}' đang ở frame 0 nên không còn chỗ cho 'VÀO CHỖ' — "
                "hãy đặt mốc sau frame 0"
            )

        if self.end_frame is not None:
            for p, f in chain:
                if f > self.end_frame:
                    errors.append(f"Mốc '{PHASE_LABELS_VI[p]}' (frame {f}) vượt quá KẾT THÚC (frame {self.end_frame})")
        return errors

    def build_result(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Mốc start -> {start_frame, end_frame} (định dạng giống bản cũ)."""
        if self.missing_required() or self.validation_errors():
            return None
        starts = {p: (self.marks[p] if self.marks[p] is not None else 0) for p in PHASE_ORDER}
        end = self.end_frame if self.end_frame is not None else self.total_frames - 1
        phases: Dict[str, Dict[str, int]] = {}
        for i, p in enumerate(PHASE_ORDER):
            start = starts[p]
            if i + 1 < len(PHASE_ORDER):
                end_ = max(start, starts[PHASE_ORDER[i + 1]] - 1)
            else:
                end_ = end
            phases[p] = {"start_frame": int(start), "end_frame": int(end_)}
        return phases


def compute_segments(state: MarkerState, total: int) -> List[Tuple[str, int, int, bool]]:
    """Các đoạn pha để vẽ timeline: (phase, start, end, is_implicit). Chỉ gồm các
    mốc đúng thứ tự; 'vao_cho' chưa đánh dấu được coi là bắt đầu ở frame 0."""
    last = total - 1
    starts = dict(state.marks)
    implicit_vao = False
    others = [starts[p] for p in PHASE_ORDER[1:] if starts[p] is not None]
    if starts["vao_cho"] is None and others and min(others) > 0:
        starts["vao_cho"] = 0
        implicit_vao = True

    chain: List[Tuple[str, int]] = []
    prev = -1
    for p in PHASE_ORDER:
        s = starts[p]
        if s is None or s <= prev:
            continue
        chain.append((p, s))
        prev = s

    end_pos = state.end_frame if state.end_frame is not None else last
    segs = []
    for i, (p, s) in enumerate(chain):
        e = chain[i + 1][1] - 1 if i + 1 < len(chain) else end_pos
        e = min(e, end_pos)
        if s > end_pos or e < s:
            continue
        segs.append((p, s, e, implicit_vao and p == "vao_cho"))
    return segs


def guidance(state: MarkerState) -> Tuple[str, Tuple[int, int, int]]:
    errs = state.validation_errors()
    if errs:
        return ("Lỗi: " + errs[0], TOAST_COLORS["error"])
    missing = state.missing_required()
    if missing:
        p = missing[0]
        return (
            f"Bước tiếp theo: tua tới frame BẮT ĐẦU pha {PHASE_LABELS_VI[p]} ({PHASE_HINTS_VI[p]}) rồi bấm [{PHASE_KEY_INDEX[p]}]",
            TOAST_COLORS["warn"],
        )
    if state.is_dirty:
        return ("Đã đủ mốc hợp lệ — bấm S để LƯU", TOAST_COLORS["ok"])
    return ("Mốc đã khớp file đã lưu — bấm N sang video kế tiếp hoặc chỉnh tiếp", TOAST_COLORS["ok"])


# =====================================================================
# Đọc video — preload nén JPEG để tua CHÍNH XÁC từng frame, mượt
# =====================================================================

class FrameSource:
    """Đọc toàn bộ video 1 lần (nén JPEG trong RAM, đã thu nhỏ vừa khung hiển thị).
    Nhờ vậy tua tới/lui, kéo timeline đều tức thì và index frame luôn đúng
    (cv2 seek theo CAP_PROP_POS_FRAMES có thể lệch với một số codec).
    Video quá dài (> MAX_PRELOAD_FRAMES) chuyển sang chế độ đọc-theo-yêu-cầu."""

    def __init__(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {path}")
        self.path = path
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps_guessed = not fps or not np.isfinite(fps) or fps <= 0 or fps > 1000
        self.fps = 30.0 if self.fps_guessed else float(fps)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._cap: Optional[cv2.VideoCapture] = cap
        self._jpeg: List[bytes] = []
        self._lazy = reported > MAX_PRELOAD_FRAMES
        self._lazy_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._next_pos = 0
        self._last_idx = -1
        self._last_img: Optional[np.ndarray] = None

        if self._lazy:
            self.total = reported
            print(f"[mark_phases] Video dài ({reported} frame) -> đọc theo yêu cầu; "
                  "tua có thể chậm hơn và lệch ±1 frame tùy codec.")
        else:
            self._preload(reported)
            self._cap.release()
            self._cap = None
            self.total = len(self._jpeg)
            if self.total == 0:
                raise ValueError(f"Không đọc được frame nào từ video: {path}")
            if reported > 0 and reported != self.total:
                print(f"[mark_phases] Lưu ý: container báo {reported} frame nhưng thực tế đọc được "
                      f"{self.total} -> dùng {self.total}.")

    @staticmethod
    def _fit(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(CANVAS_W / w, VIDEO_H / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if (nw, nh) == (w, h):
            return frame
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    def _preload(self, reported: int) -> None:
        assert self._cap is not None
        hard_cap = MAX_PRELOAD_FRAMES + 600
        t0 = time.time()
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            ok2, buf = cv2.imencode(".jpg", self._fit(frame), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok2:
                break
            self._jpeg.append(buf.tobytes())
            n = len(self._jpeg)
            if reported > 0 and n % 40 == 0:
                print(f"\r[mark_phases] Đang tải video vào bộ nhớ… {min(100, n * 100 // reported):3d}%", end="", flush=True)
            if n >= hard_cap:
                print(f"\n[mark_phases] Cảnh báo: dừng ở {n} frame (quá dài).")
                break
        print(f"\r[mark_phases] Đã tải {len(self._jpeg)} frame trong {time.time() - t0:.1f}s.            ")

    def get(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(self.total - 1, int(idx)))
        if not self._lazy:
            if idx != self._last_idx:
                self._last_img = cv2.imdecode(np.frombuffer(self._jpeg[idx], np.uint8), cv2.IMREAD_COLOR)
                self._last_idx = idx
            return self._last_img

        if idx in self._lazy_cache:
            self._lazy_cache.move_to_end(idx)
            return self._lazy_cache[idx]
        assert self._cap is not None
        if idx != self._next_pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return self._last_img
        self._next_pos = idx + 1
        img = self._fit(frame)
        self._lazy_cache[idx] = img
        while len(self._lazy_cache) > LAZY_CACHE_FRAMES:
            self._lazy_cache.popitem(last=False)
        self._last_img = img
        return img

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._jpeg = []
        self._lazy_cache.clear()
        self._last_img = None


# =====================================================================
# Vẽ chữ Unicode (tiếng Việt có dấu)
# =====================================================================

TextItem = Tuple[int, int, str, int, Tuple[int, int, int], bool]  # x, y, text, size, color(BGR), bold

_FONT_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf", "arial.ttf",
]
_FONT_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/tahomabd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "DejaVuSans-Bold.ttf", "arialbd.ttf",
]
_ASCII_REPLACE = {"←": "<-", "→": "->", "↑": "Up", "↓": "Dn", "·": "-", "…": "...", "±": "+/-", "—": "-", "–": "-"}


def fold_ascii(text: str) -> str:
    for k, v in _ASCII_REPLACE.items():
        text = text.replace(k, v)
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class TextRenderer:
    def __init__(self):
        self.use_pil = False
        self._reg: Optional[str] = None
        self._bold: Optional[str] = None
        self._fonts: Dict[Tuple[int, bool], "ImageFont.FreeTypeFont"] = {}
        if not _HAS_PIL:
            print("[mark_phases] Chưa cài Pillow -> hiển thị KHÔNG DẤU. "
                  "Cài bằng: pip install pillow (để có tiếng Việt có dấu).")
            return
        self._reg = self._find(_FONT_REGULAR)
        self._bold = self._find(_FONT_BOLD) or self._reg
        if self._reg is None:
            print("[mark_phases] Không tìm thấy font Unicode -> hiển thị KHÔNG DẤU.")
            return
        self.use_pil = True

    @staticmethod
    def _find(candidates: List[str]) -> Optional[str]:
        for path in candidates:
            try:
                ImageFont.truetype(path, 12)
                return path
            except (OSError, ValueError):
                continue
        return None

    def _font(self, size: int, bold: bool):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype((self._bold if bold else self._reg), size)
        return self._fonts[key]

    def measure(self, text: str, size: int, bold: bool = False) -> int:
        if self.use_pil:
            return int(self._font(size, bold).getlength(text))
        (w, _), _ = cv2.getTextSize(fold_ascii(text), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 1)
        return int(w)

    def draw_all(self, canvas: np.ndarray, items: List[TextItem]) -> None:
        if not items:
            return
        if self.use_pil:
            # Mảng BGR được coi như RGB; màu truyền vào cũng là BGR -> khớp nhau, khỏi đổi kênh.
            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            for x, y, text, size, color, bold in items:
                draw.text((x, y), text, font=self._font(size, bold), fill=color)
            canvas[:] = np.asarray(img)
        else:
            for x, y, text, size, color, bold in items:
                cv2.putText(canvas, fold_ascii(text), (x, y + int(size * 0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color,
                            2 if (bold and size >= 16) else 1, cv2.LINE_AA)


# =====================================================================
# Renderer — dựng toàn bộ giao diện
# =====================================================================

@dataclass
class RenderContext:
    frame_img: Optional[np.ndarray]
    video_name: str
    video_pos: Tuple[int, int]      # (số thứ tự từ 1, tổng)
    cur: int
    total: int
    fps: float
    playing: bool
    speed: float
    state: MarkerState
    toast: Optional[Tuple[str, str]]  # (text, kind)
    clear_armed: bool


class Renderer:
    def __init__(self):
        self.text = TextRenderer()

    @staticmethod
    def _darken(canvas, y0, y1, factor=0.35):
        canvas[y0:y1] = (canvas[y0:y1].astype(np.float32) * factor).astype(np.uint8)

    def render(self, c: RenderContext) -> np.ndarray:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        canvas[PANEL_Y:] = COLOR_PANEL_BG
        texts: List[TextItem] = []
        T = self.text
        state, total, fps, cur = c.state, c.total, c.fps, c.cur
        denom = max(1, total - 1)

        segs = compute_segments(state, total)
        cur_seg = next((s for s in segs if s[1] <= cur <= s[2]), None)

        # ---------------- Video ----------------
        img = c.frame_img
        if img is not None:
            h, w = img.shape[:2]
            w, h = min(w, CANVAS_W), min(h, VIDEO_H)
            x0, y0 = (CANVAS_W - w) // 2, (VIDEO_H - h) // 2
            canvas[y0:y0 + h, x0:x0 + w] = img[:h, :w]
        else:
            texts.append((CANVAS_W // 2 - 120, VIDEO_H // 2, "Không đọc được frame này", 20, (200, 200, 200), True))

        self._darken(canvas, 0, TOP_BAR_H)
        self._darken(canvas, VIDEO_H - BOTTOM_BANNER_H, VIDEO_H)
        if cur_seg:
            cv2.rectangle(canvas, (0, 0), (CANVAS_W - 1, VIDEO_H - 1), PHASE_COLORS[cur_seg[0]], 4)

        # Thanh trên: tên video + trạng thái lưu
        name = c.video_name if len(c.video_name) <= 70 else c.video_name[:67] + "…"
        texts.append((14, 8, f"[{c.video_pos[0]}/{c.video_pos[1]}]  {name}", 17, (255, 255, 255), True))
        if state.is_dirty:
            st_txt, st_col = "● CHƯA LƯU", TOAST_COLORS["warn"]
        elif state.has_any_mark():
            st_txt, st_col = "✓ ĐÃ LƯU", TOAST_COLORS["ok"]
        else:
            st_txt, st_col = "CHƯA ĐÁNH DẤU", (170, 170, 170)
        texts.append((CANVAS_W - 14 - T.measure(st_txt, 15, True), 10, st_txt, 15, st_col, True))

        # Banner dưới: pha hiện tại
        by = VIDEO_H - BOTTOM_BANNER_H
        if cur_seg:
            p = cur_seg[0]
            cv2.rectangle(canvas, (0, by), (10, VIDEO_H), PHASE_COLORS[p], -1)
            label = PHASE_LABELS_VI[p] + ("  (mặc định từ frame 0)" if cur_seg[3] else "")
            texts.append((22, by + 8, label, 20, PHASE_COLORS[p], True))
            texts.append((22 + T.measure(label, 20, True) + 16, by + 12, PHASE_HINTS_VI[p], 15, (210, 210, 210), False))
        elif not state.has_any_mark():
            texts.append((22, by + 10, "Chưa đánh dấu — tua tới đầu mỗi pha rồi bấm phím 1-4", 17, (190, 190, 190), False))
        else:
            texts.append((22, by + 10, "Frame này nằm ngoài vùng pha đã đánh dấu", 17, (190, 190, 190), False))

        # ---------------- Timeline ----------------
        y0, y1 = TL_Y0, TL_Y0 + TL_H

        def fx(f: int) -> int:
            return TL_X0 + int(round((TL_X1 - TL_X0) * f / denom))

        cv2.rectangle(canvas, (TL_X0, y0), (TL_X1, y1), (60, 60, 60), -1)
        for p, s, e, _imp in segs:
            xa, xb = fx(s), min(TL_X1, fx(e + 1))
            xb = max(xb, xa + 2)
            cv2.rectangle(canvas, (xa, y0 + 3), (xb, y1 - 3), PHASE_COLORS[p], -1)
            if xb - xa > 80:
                texts.append((xa + 6, y0 + 9, PHASE_LABELS_VI[p], 13, (25, 25, 25), True))
        # vạch giây
        duration = total / max(fps, 1e-6)
        step_s = 1 if duration <= 90 else (5 if duration <= 600 else 30)
        for t in range(0, int(duration) + 1, step_s):
            f = int(round(t * fps))
            if f < total:
                cv2.line(canvas, (fx(f), y1), (fx(f), y1 + 5), (130, 130, 130), 1)
        # mốc
        for p in PHASE_ORDER:
            m = state.marks[p]
            if m is not None and 0 <= m < total:
                x = fx(m)
                cv2.line(canvas, (x, y0 - 4), (x, y1 + 2), (255, 255, 255), 2)
                texts.append((x - 4, y0 - 22, str(PHASE_KEY_INDEX[p]), 13, PHASE_COLORS[p], True))
        if state.end_frame is not None and 0 <= state.end_frame < total:
            x = min(TL_X1, fx(state.end_frame + 1))
            cv2.line(canvas, (x, y0 - 4), (x, y1 + 2), COLOR_END, 2)
            texts.append((x - 4, y0 - 22, END_KEY, 13, COLOR_END, True))
        # đầu phát
        xc = fx(cur)
        cv2.line(canvas, (xc, y0 - 8), (xc, y1 + 8), COLOR_PLAYHEAD, 3)
        cv2.fillPoly(canvas, [np.array([(xc - 6, y0 - 15), (xc + 6, y0 - 15), (xc, y0 - 6)], np.int32)], COLOR_PLAYHEAD)

        # ---------------- Dòng thông tin ----------------
        status = "ĐANG PHÁT" if c.playing else "TẠM DỪNG"
        info = (f"Frame {cur:>5d} / {total - 1}   ·   {fmt_time(cur, fps)} / {total / max(fps, 1e-6):.2f}s"
                f"   ·   {fps:.2f} FPS   ·   Tốc độ {c.speed:g}x   ·   ")
        texts.append((TL_X0, INFO_Y, info, 15, (235, 235, 235), False))
        texts.append((TL_X0 + T.measure(info, 15), INFO_Y, status, 15,
                      TOAST_COLORS["ok"] if c.playing else TOAST_COLORS["warn"], True))
        if cur_seg:
            ptxt = f"Pha hiện tại: {PHASE_LABELS_VI[cur_seg[0]]}"
            pcol = PHASE_COLORS[cur_seg[0]]
        else:
            ptxt, pcol = "Pha hiện tại: —", (150, 150, 150)
        texts.append((TL_X1 - T.measure(ptxt, 16, True), INFO_Y - 1, ptxt, 16, pcol, True))

        # ---------------- Bảng mốc ----------------
        rows = [(p, str(PHASE_KEY_INDEX[p])) for p in PHASE_ORDER] + [("end", END_KEY)]
        for i, (p, key) in enumerate(rows):
            y = TABLE_Y + i * ROW_H
            color = COLOR_END if p == "end" else PHASE_COLORS[p]
            if cur_seg is not None and cur_seg[0] == p:
                cv2.rectangle(canvas, (TABLE_X0, y), (TABLE_X1, y + ROW_H - 3),
                              tuple(int(v * 0.30) for v in color), -1)
                cv2.rectangle(canvas, (TABLE_X0, y), (TABLE_X1, y + ROW_H - 3), color, 1)
            cv2.rectangle(canvas, (TABLE_X0 + 8, y + 6), (TABLE_X0 + 22, y + 20), color, -1)

            label = END_LABEL if p == "end" else PHASE_LABELS_VI[p]
            hint = END_HINT if p == "end" else PHASE_HINTS_VI[p]
            texts.append((TABLE_X0 + 32, y + 4, f"[{key}]", 15, color, True))
            texts.append((TABLE_X0 + 68, y + 4, label, 15, (255, 255, 255), True))
            texts.append((TABLE_X0 + 190, y + 6, hint, 12, (165, 165, 165), False))

            mark = state.end_frame if p == "end" else state.marks[p]
            if mark is not None:
                stxt, scol = f"frame {mark}  ·  {fmt_time(mark, fps)}", (255, 255, 255)
            elif p == "vao_cho":
                stxt, scol = "mặc định: frame 0", (150, 150, 150)
            elif p == "end":
                stxt, scol = f"mặc định: frame cuối ({total - 1})", (150, 150, 150)
            else:
                stxt, scol = "chưa đánh dấu", (80, 120, 255)
            texts.append((TABLE_X0 + 480, y + 4, stxt, 14, scol, mark is not None))

        # ---------------- Phím tắt ----------------
        for i, line in enumerate(HELP_LINES):
            texts.append((HELP_X, HELP_Y + i * HELP_DY, line, 12, (190, 190, 190), False))

        # ---------------- Hướng dẫn bước tiếp ----------------
        gtxt, gcol = guidance(state)
        texts.append((TL_X0, GUIDE_Y, gtxt, 15, gcol, True))

        # ---------------- Toast ----------------
        cv2.rectangle(canvas, (TL_X0, TOAST_Y), (TL_X1, TOAST_Y + 28), (45, 45, 45), -1)
        if c.clear_armed:
            tt, tk = "XÓA MỐC: bấm phím 1..5 để chọn mốc cần xóa (phím khác = hủy)", "warn"
        elif c.toast:
            tt, tk = c.toast
        else:
            tt, tk = "Mẹo: click/kéo trên timeline để tua · click dòng pha để đánh dấu · chuột phải để xóa mốc", "info"
        if c.toast or c.clear_armed:
            cv2.rectangle(canvas, (TL_X0, TOAST_Y), (TL_X0 + 5, TOAST_Y + 28), TOAST_COLORS[tk], -1)
        texts.append((TL_X0 + 14, TOAST_Y + 6, tt, 14, TOAST_COLORS[tk] if (c.toast or c.clear_armed) else (150, 150, 150), False))

        T.draw_all(canvas, texts)
        return canvas


# =====================================================================
# Bàn phím / chuột
# =====================================================================

_SPECIAL_KEYS = {
    # Windows
    2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",
    2359296: "home", 2293760: "end", 2162688: "pgup", 2228224: "pgdn",
    # Linux (GTK/X11)
    65361: "left", 65363: "right", 65362: "up", 65364: "down",
    65360: "home", 65367: "end", 65365: "pgup", 65366: "pgdn",
    # Linux (Qt)
    16777234: "left", 16777236: "right", 16777235: "up", 16777237: "down",
    16777232: "home", 16777233: "end", 16777238: "pgup", 16777239: "pgdn",
    # macOS
    63234: "left", 63235: "right", 63232: "up", 63233: "down",
    63273: "home", 63275: "end", 63276: "pgup", 63277: "pgdn",
}


def decode_key(k: int) -> Optional[str]:
    """Mã phím thô (waitKeyEx) -> token: 'left', 'undo', hoặc 1 ký tự in được."""
    if k == -1:
        return None
    if k in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[k]
    if k == 26:
        return "undo"
    if k == 25:
        return "redo"
    if k == 27:
        return "esc"
    if 32 <= k < 127:
        ch = chr(k)
        return ch.lower() if ch.isalpha() else ch
    return None


class MouseController:
    """Click/kéo timeline để tua; click dòng pha để đánh dấu; chuột phải để xóa."""

    def __init__(self):
        self.total = 1
        self.dragging = False
        self.events: List[Tuple[str, object]] = []

    def _frame_at(self, x: int) -> int:
        ratio = (min(max(x, TL_X0), TL_X1) - TL_X0) / max(1, TL_X1 - TL_X0)
        return int(round(ratio * max(0, self.total - 1)))

    @staticmethod
    def _in_timeline(x: int, y: int) -> bool:
        return TL_X0 - 8 <= x <= TL_X1 + 8 and TL_Y0 - 26 <= y <= TL_Y0 + TL_H + 12

    @staticmethod
    def _row_at(x: int, y: int) -> Optional[int]:
        if TABLE_X0 <= x <= TABLE_X1 and TABLE_Y <= y < TABLE_Y + ROW_H * 5:
            return (y - TABLE_Y) // ROW_H
        return None

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._in_timeline(x, y):
                self.dragging = True
                self.events.append(("seek", self._frame_at(x)))
            else:
                row = self._row_at(x, y)
                if row is not None:
                    self.events.append(("mark", str(row + 1)))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.events.append(("seek", self._frame_at(x)))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            row = self._row_at(x, y)
            if row is not None:
                self.events.append(("clear", str(row + 1)))


# =====================================================================
# VideoMarker — xử lý 1 video
# =====================================================================

class Action(Enum):
    SAVED = "saved"
    NEXT = "next"
    PREV = "prev"
    QUIT = "quit"


@dataclass
class _Toast:
    text: str
    kind: str
    expire: float
    ident: int


class VideoMarker:
    def __init__(self, source: FrameSource, video_path: Path, out_path: Path,
                 renderer: Renderer, pos: Tuple[int, int], fresh: bool = False):
        self.src = source
        self.video_path = video_path
        self.out_path = out_path
        self.renderer = renderer
        self.pos = pos  # (index 0-based, total_videos)
        self.total = source.total
        self.fps = source.fps
        self.state = MarkerState(self.total)
        self.mouse = MouseController()
        self.mouse.total = self.total

        self.cur = 0
        self.playing = False
        self.speed_idx = DEFAULT_SPEED_IDX
        self._accum = 0.0
        self._last_tick = time.perf_counter()

        self.clear_armed = False
        self._toast: Optional[_Toast] = None
        self._toast_seq = 0
        self._confirm: Optional[str] = None
        self._confirm_until = 0.0

        if not fresh:
            self._load_existing()

    # ---------------- nạp / lưu ----------------
    def _load_existing(self) -> None:
        if not self.out_path.exists():
            return
        try:
            with open(self.out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("video") != self.video_path.name:
                print(f"[mark_phases] File {self.out_path} thuộc video khác ('{data.get('video')}'), bỏ qua.")
                return
            self.state.load(data.get("phases", {}))
            print(f"[mark_phases] Đã nạp mốc cũ từ: {self.out_path}")
            if data.get("total_frames") not in (None, self.total):
                print(f"  ! Cảnh báo: file cũ ghi {data.get('total_frames')} frame, video hiện có {self.total} frame.")
            for p in PHASE_ORDER:
                if self.state.marks[p] is not None:
                    print(f"  - {p:<13s}: {self.state.marks[p]}")
            if self.state.end_frame is not None:
                print(f"  - end_frame   : {self.state.end_frame}")
        except Exception as exc:  # noqa: BLE001
            print(f"[mark_phases] Không nạp được file cũ: {exc}")

    def _save(self) -> Optional[Action]:
        problems = [f"Chưa đánh dấu pha {PHASE_LABELS_VI[p]} (phím {PHASE_KEY_INDEX[p]})"
                    for p in self.state.missing_required()] + self.state.validation_errors()
        if problems:
            print("  [SAVE FAIL] Chưa thể lưu:")
            for e in problems:
                print(f"    - {e}")
            self._set_toast("Không lưu được: " + problems[0], "error", 6.0)
            return None
        phases = self.state.build_result()
        assert phases is not None
        payload = {
            "video": self.video_path.name,
            "fps": float(self.fps),
            "total_frames": int(self.total),
            "phases": phases,
            "marked_by": "manual",
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_name(self.out_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.out_path)  # ghi nguyên tử, không để lại file hỏng
        self.state.mark_saved()
        print(f"  [SAVED] {self.out_path}")
        for p, w in phases.items():
            print(f"    {p:<13s}: frame {w['start_frame']:>5d} → {w['end_frame']:>5d} "
                  f"({w['end_frame'] - w['start_frame'] + 1} frames)")
        return Action.SAVED

    # ---------------- toast / xác nhận ----------------
    def _set_toast(self, text: str, kind: str = "info", secs: float = 3.5) -> None:
        self._toast_seq += 1
        self._toast = _Toast(text, kind, time.perf_counter() + secs, self._toast_seq)

    def _active_toast(self, now: float) -> Optional[_Toast]:
        if self._toast and now > self._toast.expire:
            self._toast = None
        return self._toast

    def _confirm_discard(self, name: str, key_hint: str) -> bool:
        if not self.state.is_dirty:
            return True
        now = time.perf_counter()
        if self._confirm == name and now < self._confirm_until:
            return True
        self._confirm, self._confirm_until = name, now + 4.0
        self._set_toast(f"Có thay đổi CHƯA LƯU! Bấm {key_hint} lần nữa để bỏ thay đổi, hoặc S để lưu.", "warn", 4.0)
        return False

    # ---------------- điều hướng frame ----------------
    def _goto(self, frame: int, pause: bool = True) -> None:
        self.cur = max(0, min(self.total - 1, int(frame)))
        if pause:
            self.playing = False

    def _toggle_play(self) -> None:
        if not self.playing and self.cur >= self.total - 1:
            self.cur = 0
        self.playing = not self.playing
        self._accum = 0.0
        self._last_tick = time.perf_counter()

    def _advance_playback(self, now: float) -> None:
        dt = now - self._last_tick
        self._last_tick = now
        if not self.playing:
            self._accum = 0.0
            return
        self._accum += dt * self.fps * SPEEDS[self.speed_idx]
        step = int(self._accum)
        if step >= 1:
            self._accum -= step
            new = self.cur + min(step, 30)
            if new >= self.total - 1:
                new, self.playing = self.total - 1, False
            self.cur = new

    # ---------------- đánh dấu ----------------
    def _mark(self, key: str) -> None:
        f = self.cur
        if key == END_KEY:
            changed = self.state.set_end(f)
            name = END_LABEL
        else:
            phase = PHASE_BY_KEY[key]
            changed = self.state.set_mark(phase, f)
            name = f"BẮT ĐẦU {PHASE_LABELS_VI[phase]}"
        msg = f"{name} = frame {f} (t={fmt_time(f, self.fps)})"
        if not changed:
            self._set_toast(f"Mốc đã có sẵn: {msg}", "info")
            return
        print(f"  [MARK] {msg}")
        errs = self.state.validation_errors()
        if errs:
            self._set_toast(f"Đã đánh dấu {msg} — nhưng: {errs[0]}", "warn", 5.0)
        else:
            self._set_toast(f"Đã đánh dấu {msg}", "ok")

    def _clear(self, key: str) -> None:
        if key == END_KEY:
            ok, name = self.state.clear_end(), END_LABEL
        else:
            phase = PHASE_BY_KEY[key]
            ok, name = self.state.clear_mark(phase), PHASE_LABELS_VI[phase]
        if ok:
            print(f"  [CLEAR] Đã xóa mốc {name}.")
            self._set_toast(f"Đã xóa mốc {name} (U / Ctrl+Z để hoàn tác)", "info")
        else:
            self._set_toast(f"Mốc {name} đang trống", "info")

    def _jump_to_mark(self, key: str) -> None:
        mapping = {"6": "vao_cho", "7": "static", "8": "sprint_start", "9": "drive_phase"}
        if key == "0":
            f, name = self.state.end_frame, END_LABEL
        else:
            p = mapping[key]
            f, name = self.state.marks[p], PHASE_LABELS_VI[p]
        if f is None:
            self._set_toast(f"Chưa có mốc {name}", "warn")
        else:
            self._goto(f)
            self._set_toast(f"Nhảy tới mốc {name} (frame {f})", "info", 2.0)

    # ---------------- xử lý sự kiện ----------------
    def _process_mouse(self) -> None:
        events, self.mouse.events = self.mouse.events, []
        # gộp seek liên tiếp: chỉ lấy cái cuối
        last_seek = None
        for kind, val in events:
            if kind == "seek":
                last_seek = int(val)  # type: ignore[arg-type]
            elif kind == "mark":
                self._mark(str(val))
            elif kind == "clear":
                self._clear(str(val))
        if last_seek is not None:
            self._goto(last_seek)

    def _handle_key(self, key: Optional[str]) -> Optional[Action]:
        if key is None:
            return None

        if self.clear_armed:
            self.clear_armed = False
            if key in ("1", "2", "3", "4", "5"):
                self._clear(key)
            else:
                self._set_toast("Đã hủy chế độ xóa mốc", "info", 1.5)
            return None

        step_s = max(1, int(round(self.fps)))
        if key == " ":
            self._toggle_play()
        elif key in ("left", "a", "j"):
            self._goto(self.cur - 1)
        elif key in ("right", "d", "l"):
            self._goto(self.cur + 1)
        elif key in ("up", "]"):
            self._goto(self.cur + 10)
        elif key in ("down", "["):
            self._goto(self.cur - 10)
        elif key in ("pgup", ",", "<"):
            self._goto(self.cur - step_s)
        elif key in ("pgdn", ".", ">"):
            self._goto(self.cur + step_s)
        elif key == "home":
            self._goto(0)
        elif key == "end":
            self._goto(self.total - 1)
        elif key in ("+", "="):
            self.speed_idx = min(len(SPEEDS) - 1, self.speed_idx + 1)
            self._set_toast(f"Tốc độ phát: {SPEEDS[self.speed_idx]:g}x", "info", 1.5)
        elif key in ("-", "_"):
            self.speed_idx = max(0, self.speed_idx - 1)
            self._set_toast(f"Tốc độ phát: {SPEEDS[self.speed_idx]:g}x", "info", 1.5)
        elif key in ("1", "2", "3", "4", "5"):
            self._mark(key)
        elif key in ("!", "@", "#", "$", "%"):
            self._clear({"!": "1", "@": "2", "#": "3", "$": "4", "%": "5"}[key])
        elif key == "c":
            self.clear_armed = True
        elif key == "x":
            if self.state.clear_all():
                print("  [CLEAR ALL] Đã xóa tất cả mốc.")
                self._set_toast("Đã xóa TẤT CẢ mốc (U / Ctrl+Z để hoàn tác)", "warn")
            else:
                self._set_toast("Không có mốc nào để xóa", "info")
        elif key in ("6", "7", "8", "9", "0"):
            self._jump_to_mark(key)
        elif key in ("u", "undo"):
            self._set_toast("Đã hoàn tác" if self.state.undo() else "Không có gì để hoàn tác",
                            "info", 1.8)
        elif key in ("r", "redo"):
            self._set_toast("Đã làm lại" if self.state.redo() else "Không có gì để làm lại",
                            "info", 1.8)
        elif key == "s":
            return self._save()
        elif key == "n":
            if self.pos[0] >= self.pos[1] - 1:
                self._set_toast("Đây là video cuối cùng", "info")
            elif self._confirm_discard("next", "N"):
                return Action.NEXT
        elif key == "b":
            if self.pos[0] <= 0:
                self._set_toast("Đây là video đầu tiên", "info")
            elif self._confirm_discard("prev", "B"):
                return Action.PREV
        elif key == "q":
            if self._confirm_discard("quit", "Q"):
                print("[mark_phases] Thoát.")
                return Action.QUIT
        return None

    # ---------------- vòng lặp chính ----------------
    def run(self) -> Action:
        print(f"\n[mark_phases] Video [{self.pos[0] + 1}/{self.pos[1]}]: {self.video_path}")
        print(f"[mark_phases] FPS: {self.fps:.2f}{' (không đọc được, dùng mặc định)' if self.src.fps_guessed else ''}"
              f" | Frames: {self.total} | Duration: {self.total / max(self.fps, 1e-6):.2f}s"
              f" | {self.src.width}x{self.src.height}")
        print(f"[mark_phases] File mốc: {self.out_path}")

        cv2.setMouseCallback(WINDOW_NAME, self.mouse.on_mouse)
        self._set_toast("Sẵn sàng. Tua tới đầu mỗi pha rồi bấm phím 1-4 (xem bảng phím tắt bên phải).", "info", 6.0)

        last_sig = None
        last_draw = 0.0
        shown = 0
        self._last_tick = time.perf_counter()

        while True:
            now = time.perf_counter()
            self._advance_playback(now)
            self._process_mouse()
            toast = self._active_toast(now)
            sig = (self.cur, self.state.version, self.state.is_dirty, self.playing,
                   self.speed_idx, self.clear_armed, toast.ident if toast else None)
            if sig != last_sig or now - last_draw > 1.0:
                ctx = RenderContext(
                    frame_img=self.src.get(self.cur),
                    video_name=self.video_path.name,
                    video_pos=(self.pos[0] + 1, self.pos[1]),
                    cur=self.cur, total=self.total, fps=self.fps,
                    playing=self.playing, speed=SPEEDS[self.speed_idx],
                    state=self.state,
                    toast=(toast.text, toast.kind) if toast else None,
                    clear_armed=self.clear_armed,
                )
                cv2.imshow(WINDOW_NAME, self.renderer.render(ctx))
                last_sig, last_draw = sig, now
                shown += 1

            key = cv2.waitKeyEx(1 if self.playing else 15)
            action = self._handle_key(decode_key(key))
            if action is not None:
                return action

            if shown > 3:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("[mark_phases] Cửa sổ đã đóng.")
                        return Action.QUIT
                except cv2.error:
                    return Action.QUIT


# =====================================================================
# Phiên làm việc: 1 video hoặc nhiều video
# =====================================================================

def run_session(videos: List[Path], out_for: Dict[Path, Path], fresh: bool = False) -> Dict[str, int]:
    if not videos:
        print("[mark_phases] Không có video nào để xử lý.")
        return {"saved": 0, "skipped": 0, "failed": 0}

    renderer = Renderer()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    saved: set = set()
    failed: set = set()
    n = len(videos)
    idx, last_dir = 0, 1

    def step(i: int, d: int) -> int:
        j = i + d
        while 0 <= j < n and j in failed:
            j += d
        return j

    try:
        while 0 <= idx < n:
            vp = videos[idx]
            try:
                source = FrameSource(str(vp))
            except Exception as exc:  # noqa: BLE001
                print(f"[mark_phases] ✗ Bỏ qua '{vp.name}': {exc}")
                failed.add(idx)
                nxt = step(idx, last_dir)
                idx = nxt if nxt >= 0 else step(idx, 1)
                continue

            marker = VideoMarker(source, vp, out_for[vp], renderer, (idx, n), fresh=fresh)
            try:
                action = marker.run()
            finally:
                source.close()

            if action is Action.QUIT:
                break
            if action is Action.SAVED:
                saved.add(idx)
                last_dir = 1
                idx = step(idx, 1)
            elif action is Action.NEXT:
                last_dir = 1
                idx = step(idx, 1)
            elif action is Action.PREV:
                last_dir = -1
                j = step(idx, -1)
                idx = j if j >= 0 else idx
    except KeyboardInterrupt:
        print("\n[mark_phases] Đã ngắt (Ctrl+C).")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    summary = {"saved": len(saved), "skipped": n - len(saved) - len(failed), "failed": len(failed)}
    print(f"\n[mark_phases] Tổng kết: đã lưu {summary['saved']}/{n} video"
          f" | chưa lưu {summary['skipped']} | lỗi mở {summary['failed']}")
    return summary


# =====================================================================
# Giao diện dòng lệnh: menu 2 lựa chọn
# =====================================================================

def _ask(prompt: str) -> Optional[str]:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def mode_single(video_arg: Optional[str], out_arg: Optional[str], out_dir: Path, fresh: bool) -> Optional[bool]:
    """Trả về True nếu chạy xong, False nếu lỗi, None nếu người dùng quay lại menu."""
    raw = video_arg
    while True:
        if raw is None:
            raw = _ask("\nNhập đường dẫn FILE video (Enter để quay lại): ")
            if not raw:
                return None
        path = Path(clean_path(raw))
        if path.is_dir():
            ans = _ask(f"'{path}' là THƯ MỤC. Chuyển sang chế độ xử lý cả thư mục? [Y/n]: ")
            if ans is None:
                return None
            if ans.lower() in ("", "y", "yes", "c", "co", "có"):
                return mode_folder(str(path), out_dir, False, False, fresh)
            raw = None
            continue
        if not path.is_file():
            print(f"✗ Không tìm thấy file: {path}")
            if video_arg is not None:
                return False
            raw = None
            continue
        break

    out_path = Path(clean_path(out_arg)) if out_arg else out_dir / f"{path.stem}.phases.json"
    run_session([path], {path: out_path}, fresh=fresh)
    return True


def mode_folder(folder_arg: Optional[str], out_dir: Path, recursive: Optional[bool],
                only_new: Optional[bool], fresh: bool) -> Optional[bool]:
    raw = folder_arg
    interactive = folder_arg is None
    while True:
        if raw is None:
            raw = _ask("\nNhập đường dẫn THƯ MỤC chứa video (Enter để quay lại): ")
            if not raw:
                return None
        folder = Path(clean_path(raw))
        if not folder.is_dir():
            print(f"✗ Không tìm thấy thư mục: {folder}")
            if not interactive:
                return False
            raw = None
            continue

        if recursive is None:
            ans = _ask("Quét cả thư mục con? [y/N]: ")
            if ans is None:
                return None
            use_recursive = ans.lower() in ("y", "yes", "c", "co", "có")
        else:
            use_recursive = recursive

        videos = list_videos(folder, use_recursive)
        if not videos:
            print(f"✗ Không có video nào ({', '.join(sorted(VIDEO_EXTENSIONS))}) trong: {folder}")
            if not interactive:
                return False
            raw, recursive = None, None
            continue
        break

    out_for = build_out_paths(videos, out_dir)
    if len({p for p in out_for.values()}) != len(videos):  # không thể xảy ra, phòng thủ
        print("✗ Trùng tên file đầu ra.")
        return False
    if any(v.stem != out_for[v].name.replace(".phases.json", "") for v in videos):
        print("! Có video trùng tên ở các thư mục con khác nhau -> file mốc được thêm tiền tố tên thư mục cha.")

    print(f"\nTìm thấy {len(videos)} video trong '{folder}':")
    done_flags = {v: out_for[v].exists() for v in videos}
    for i, v in enumerate(videos, 1):
        print(f"  {i:>3d}. [{'✓ đã có mốc' if done_flags[v] else '  chưa có   '}] {v.relative_to(folder)}")
    n_done = sum(done_flags.values())
    print(f"→ {n_done} đã có file mốc, {len(videos) - n_done} chưa có.")

    if only_new is None:
        print("\nChọn cách xử lý:")
        print("  [1] Tất cả video (video đã có mốc sẽ được nạp lại để chỉnh sửa)   <- mặc định")
        print("  [2] Chỉ các video CHƯA có file mốc")
        print("  [0] Quay lại")
        ans = _ask("Lựa chọn [1/2/0]: ")
        if ans is None or ans == "0":
            return None
        only_new = ans == "2"

    todo = [v for v in videos if not (only_new and done_flags[v])]
    if not todo:
        print("Tất cả video đã có file mốc — không còn gì để xử lý (bỏ --only-new để chỉnh sửa lại).")
        return True

    print(f"\nBắt đầu xử lý {len(todo)} video. Trong cửa sổ: S = lưu & sang video kế, N/B = tiếp/trước, Q = thoát.")
    run_session(todo, out_for, fresh=fresh)
    return True


def interactive_menu(out_dir: Path, fresh: bool) -> None:
    while True:
        print("\n" + "=" * 62)
        print("  CÔNG CỤ ĐÁNH DẤU PHASE — XUẤT PHÁT THẤP")
        print("=" * 62)
        print("  [1] Xử lý MỘT video            (nhập đường dẫn file)")
        print("  [2] Xử lý CẢ THƯ MỤC video     (nhập đường dẫn thư mục)")
        print("  [0] Thoát")
        choice = _ask("Chọn [1/2/0]: ")
        if choice is None or choice == "0":
            print("Tạm biệt!")
            return
        if choice == "1":
            mode_single(None, None, out_dir, fresh)
        elif choice == "2":
            mode_folder(None, out_dir, None, None, fresh)
        else:
            print("✗ Lựa chọn không hợp lệ.")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(
        description="Công cụ đánh dấu 4 sub-phase sprint start trên video HLV (1 video hoặc cả thư mục).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python tools/mark_phases.py                       # hiện menu chọn\n"
            "  python tools/mark_phases.py --video data/raw_videos/coach.mp4\n"
            "  python tools/mark_phases.py --folder data/raw_videos --only-new\n"
        ),
    )
    parser.add_argument("--video", help="Đường dẫn 1 file video")
    parser.add_argument("--folder", help="Đường dẫn thư mục chứa nhiều video")
    parser.add_argument("--out", default=None,
                        help="File phases JSON (CHỈ dùng với --video). Mặc định: data/phases/<video_stem>.phases.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Thư mục lưu file phases JSON (mặc định: data/phases)")
    parser.add_argument("--recursive", action="store_true", help="Với --folder: quét cả thư mục con")
    parser.add_argument("--only-new", action="store_true", help="Với --folder: bỏ qua video đã có file mốc")
    parser.add_argument("--fresh", action="store_true", help="Không nạp mốc cũ, bắt đầu từ trống")
    parser.add_argument("--resume", action="store_true",
                        help="(Giữ để tương thích) mốc cũ luôn được tự động nạp nếu có")
    args = parser.parse_args()

    if args.video and args.folder:
        parser.error("Chỉ chọn MỘT trong --video hoặc --folder.")
    if args.out and not args.video:
        parser.error("--out chỉ dùng được cùng --video (với --folder hãy dùng --out-dir).")

    out_dir = Path(clean_path(args.out_dir))

    if args.video:
        ok = mode_single(args.video, args.out, out_dir, args.fresh)
        return 0 if ok else 2
    if args.folder:
        ok = mode_folder(args.folder, out_dir, args.recursive, args.only_new, args.fresh)
        return 0 if ok else 2

    interactive_menu(out_dir, args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
