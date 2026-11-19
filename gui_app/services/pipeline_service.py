"""Giữ MỘT Pipeline (nạp lười). Tách bước: đọc meta -> trích pose -> đề xuất pha -> chấm điểm."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2

from gui_app import constants as C


class PipelineService:
    def __init__(self, system_config: Path, phase_model_dir: Path):
        self.system_config, self.phase_model_dir = Path(system_config), Path(phase_model_dir)
        self._pipeline = None
        self._segmenter = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from core.pipeline import Pipeline
            self._pipeline = Pipeline.from_config(str(self.system_config), sport="sprint_start")
        return self._pipeline

    @property
    def segmenter(self):
        if self._segmenter is None:       # FileNotFoundError nếu chưa có mô hình -> chế độ thủ công (D10)
            from stage3_features.phase_model.inference.phase_segmenter import PhaseSegmenter
            self._segmenter = PhaseSegmenter.from_registry(self.phase_model_dir)
        return self._segmenter

    def reload_segmenter(self):
        self._segmenter = None

    def describe(self) -> str:
        backend, device = "?", "?"
        try:
            import yaml
            cfg = yaml.safe_load(self.system_config.read_text(encoding="utf-8")) or {}
            backend = cfg.get("pose_backend", "?")
            kw = cfg.get("pose_backend_kwargs", {}) or {}
            device = kw.get("rtmpose_device") or kw.get("yolo_device") or "cpu"
        except Exception:
            pass
        reg = C.PHASE_MODEL_DIR / "registry.json"
        ver = "chưa có"
        if reg.exists():
            import json
            try:
                ver = json.loads(reg.read_text(encoding="utf-8")).get("current") or "chưa có"
            except Exception:
                pass
        return f"Pose: {backend} · {device}   |   Mô hình phân pha: {ver}"

    @staticmethod
    def read_meta(path: str) -> dict:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video: {path}")
        meta = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        if meta["frames"] <= 0:
            raise ValueError("Video không có khung hình")
        return meta

    def extract_poses(self, video: str, width: int, height: int) -> list:
        return self.pipeline._extract_tracked(video, width, height)   # ValueError nếu không có người (D09)

    def propose_phases(self, frames: list, fps: float, video_id: str):
        from core.phase_resolver import _pose_array
        return self.segmenter.propose(_pose_array(frames), float(fps), video_id=video_id)

    def score(self, frames: list, fps: float, video_id: str, phases_path: str):
        # luôn truyền phases_path (ưu tiên cao nhất, tránh nạp nhầm nhãn cũ trùng tên video)
        return self.pipeline.process_sprint_start_sequence(
            frames, fps=fps, video_id=video_id, phases_path=phases_path)
