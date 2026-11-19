"""Thư mục phiên + sao chép video tên duy nhất (tránh resolve_phases dùng nhầm nhãn cũ trùng stem)."""
import shutil
from datetime import datetime
from pathlib import Path

from gui_app import constants as C


def create_session_dir(student_code: str) -> Path:
    safe = "".join(ch for ch in student_code if ch.isalnum() or ch in "-_") or "HV"
    d = C.SESSIONS_DIR / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "snapshots").mkdir()
    return d


def copy_video_unique(src: Path, session_dir: Path) -> Path:
    dst = session_dir / f"input_{session_dir.name}{src.suffix.lower()}"
    shutil.copy2(src, dst)
    return dst
