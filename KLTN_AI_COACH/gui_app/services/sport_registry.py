"""Quét config/sport_configs/*.yaml -> trạng thái thẻ môn + phiên bản mô hình pha."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from gui_app import constants as C


@dataclass
class SportInfo:
    code: str
    display: str
    ready: bool
    badge: str                      # "Sẵn sàng" | "Nháp" | "Chưa có mô hình"
    yaml_path: Path | None = None


class SportRegistry:
    def __init__(self, config_dir: Path, model_dir: Path):
        self.config_dir, self.model_dir = Path(config_dir), Path(model_dir)

    def has_phase_model(self) -> bool:
        return (self.model_dir / "registry.json").exists()

    def phase_model_versions(self) -> List[str]:
        p = self.model_dir / "registry.json"
        if not p.exists():
            return []
        try:
            return list((json.loads(p.read_text(encoding="utf-8")).get("versions") or {}).keys())
        except Exception:
            return []

    def current_phase_model(self) -> str:
        p = self.model_dir / "registry.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("current") or ""
        except Exception:
            return ""

    def scan(self) -> List[SportInfo]:
        out: List[SportInfo] = []
        for p in sorted(self.config_dir.glob("*.yaml")):          # không đệ quy -> bỏ _backup/
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            code = cfg.get("sport", p.stem)
            display = {v: k for k, v in C.SPORT_ALIAS.items()}.get(code, cfg.get("display_name") or code)
            ready = code == "sprint_start" and self.has_phase_model()   # engine chỉ chấm đủ sprint_start
            out.append(SportInfo(code, display, ready, "Sẵn sàng" if ready else "Nháp", p))
        have = {s.display for s in out}
        for name in C.COMING_SOON:
            if name not in have:
                out.append(SportInfo(name, name, False, "Chưa có mô hình"))
        return out
