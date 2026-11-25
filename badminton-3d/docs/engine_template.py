"""Mẫu engine cho môn thể thao mới (xem docs/MULTISPORT.md). Sao chép file này, đổi tên lớp, rồi khai báo trong config/sports.yaml:

    engine: "my_sport_engine:MySportEngine"      # module phải import được (đặt cạnh app_ui.py hoặc trong PYTHONPATH)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


class MySportEngine:
    name = "my-sport"
    supported_actions = ("forehand",)                    # các bài engine này chấm được

    # (tuỳ chọn) form nhanh: giao diện tự vẽ từ dữ liệu này
    quick_settings = [{"title": "Điểm", "items": [{"keys": ["scoring", "pass_score"], "label": "Điểm đạt", "kind": "float", "min": 0.0, "max": 100.0, "step": 1.0}]}]

    def analyze(self, sport, video: Path, action: str, hand: str = "auto", force: bool = False) -> dict:
        """Chấm một video. Ghi kết quả vào sport.output/<tên video>/ và trả report đúng cấu trúc report.json."""
        out = Path(sport.output) / Path(video).stem
        out.mkdir(parents=True, exist_ok=True)
        # TODO: trích khung xương, tính chỉ số, so với sport.reference_file(action), chấm điểm...
        report = {"status": "SCORED", "exercise": action, "duration_s": 0.0, "warnings": ["Engine mẫu: chưa tính điểm thật."],
                  "score": {"overall": 0.0, "grade": "Chưa đạt", "errors": []}}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        return report

    def build_reference(self, sport, action: str, videos: Sequence[Path]) -> Tuple[int, str]:
        """Dựng chuẩn từ 1–3 video HLV và ghi sport.reference_file(action)."""
        sport.reference_file(action).write_text(json.dumps({"built_from": [Path(v).name for v in videos], "n_videos": len(videos)}), encoding="utf-8")
        return 0, "Đã dựng chuẩn (mẫu)."

    # (tuỳ chọn) def reference_rows(self, sport, action) -> (thông tin, danh sách dòng)
    # (tuỳ chọn) def record(self, sport, action, source, role, hand, max_seconds) -> (mã, log)
    # (tuỳ chọn) def run_all(self, sport) -> (mã, log)
