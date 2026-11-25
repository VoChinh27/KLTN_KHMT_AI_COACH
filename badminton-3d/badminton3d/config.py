"""Nạp cấu hình YAML (config/badminton3d.yaml); có thể đổi bằng biến môi trường BADMINTON3D_CONFIG hoặc tham số."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "badminton3d.yaml"


def load_config(path: Optional[str] = None) -> dict:
    p = Path(path) if path else Path(os.environ.get("BADMINTON3D_CONFIG", DEFAULT_CONFIG))
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy file cấu hình: {p}")
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(p)
    return cfg


def resolve(path: str) -> Path:
    """Đường dẫn tương đối trong cấu hình tính từ thư mục gốc dự án."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
