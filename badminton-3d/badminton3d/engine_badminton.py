"""Engine cầu lông: bọc lõi badminton3d (runner, cli, record, reference) theo hợp đồng engine của giao diện đa môn (badminton3d/sports.py).

Muốn thêm môn khác: viết một lớp có cùng các thành phần dưới đây rồi khai báo `engine: "module:Lớp"` trong config/sports.yaml (docs/MULTISPORT.md).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _dirs(sport) -> List[str]:
    return ["--input", str(sport.input), "--output", str(sport.output), "--reference", str(sport.reference), "--config", str(sport.config)]


class BadmintonEngine:
    name = "badminton3d"
    supported_actions = ("ready", "serve")

    # Các ngưỡng hay chỉnh nhất, mô tả bằng dữ liệu để giao diện tự vẽ form (không hard-code trong app_ui.py)
    quick_settings = [
        {"title": "Vung xả đà quá dài", "help": "Sau cú chạm tay vợt vẫn lên cao hơn vai: không phạm luật nhưng cầu dễ ra ngoài.", "items": [
            {"keys": ["faults", "follow_through", "enabled"], "label": "Bật kiểm tra", "kind": "bool"},
            {"keys": ["faults", "follow_through", "warn_m"], "label": "Ngưỡng nhẹ (m)", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"keys": ["faults", "follow_through", "high_m"], "label": "Ngưỡng nặng (m)", "kind": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            {"keys": ["faults", "follow_through", "penalty_warn"], "label": "Trừ điểm (nhẹ)", "kind": "float", "min": 0.0, "max": 50.0, "step": 1.0},
            {"keys": ["faults", "follow_through", "penalty_high"], "label": "Trừ điểm (nặng)", "kind": "float", "min": 0.0, "max": 50.0, "step": 1.0}]},
        {"title": "Nhấc gót khi phát", "help": "Số đo gót của MediaPipe chưa chính xác: mặc định 0,30 để không báo nhầm HLV; hạ ≈ 0,15 sẽ bắt được trainee_serve_01 nhưng báo nhầm thêm HLV.", "items": [
            {"keys": ["faults", "heel_lift", "enabled"], "label": "Bật kiểm tra", "kind": "bool"},
            {"keys": ["faults", "heel_lift", "threshold"], "label": "Ngưỡng nhấc gót (heel_lift.threshold)", "kind": "slider", "min": 0.05, "max": 0.60, "step": 0.01},
            {"keys": ["faults", "heel_lift", "penalty"], "label": "Trừ điểm (nhấc gót)", "kind": "float", "min": 0.0, "max": 50.0, "step": 1.0}]},
        {"title": "Giới hạn điểm khi có lỗi nghiêm trọng", "help": "Một góc quan trọng lệch quá xa chuẩn thì điểm tối đa bằng giá trị này.", "items": [
            {"keys": ["scoring", "critical_cap"], "label": "Điểm tối đa khi có lỗi nghiêm trọng", "kind": "int_or_null", "min": 0, "max": 100, "default": 69}]},
    ]

    def analyze(self, sport, video: Path, action: str, hand: str = "auto", force: bool = False) -> dict:
        from badminton3d.config import load_config
        from badminton3d.runner import process_video
        cfg = load_config(str(sport.config))
        process_video(str(video), cfg, str(sport.output), str(sport.reference), None if hand in (None, "auto") else hand, force_pose=force)
        p = Path(sport.output) / Path(video).stem / "report.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def build_reference(self, sport, action: str, videos: Sequence[Path]) -> Tuple[int, str]:
        from badminton3d.ui_backend import run_cli
        return run_cli(["build-reference", action, *[str(v) for v in videos], *_dirs(sport)], timeout=1800)

    def record(self, sport, action: str, source: str, role: str, hand: str, max_seconds: int) -> Tuple[int, str]:
        from badminton3d.ui_backend import run_cli
        args = ["record", action, "--source", str(source), "--role", role, "--max-seconds", str(int(max_seconds)), *_dirs(sport)]
        if hand in ("left", "right"):
            args += ["--racket-hand", hand]
        return run_cli(args, timeout=int(max_seconds) + 900)

    def run_all(self, sport) -> Tuple[int, str]:
        from badminton3d.ui_backend import run_cli
        return run_cli(["run", *_dirs(sport)], timeout=7200)

    def reference_rows(self, sport, action: str) -> Tuple[Optional[dict], List[dict]]:
        from badminton3d.ui_backend import fmt_val
        p = sport.reference_file(action)
        if not p.is_file():
            return None, []
        ref = json.loads(p.read_text(encoding="utf-8"))
        pv = {"chuan_bi": "Chuẩn bị", "rut_vot": "Rút vợt", "danh_cau": "Chạm cầu", "ket_thuc": "Kết thúc"}
        groups = [("", ref["features"])] if "features" in ref else list(ref.get("phases", {}).items())
        rows = []
        for ph, feats in groups:
            for k, f in feats.items():
                row = {"Góc / đặc trưng": f.get("label", k), "Mức": f.get("level", ""), "Giá trị chuẩn": fmt_val(f["center"], f["unit"]), "Dung sai ±": fmt_val(f["tol"], f["unit"]),
                       "Khoảng chấp nhận": f"{fmt_val(f['center'] - f['tol'], f['unit'])} → {fmt_val(f['center'] + f['tol'], f['unit'])}"}
                if ph:
                    row = {"Pha": pv.get(ph, ph), **row}
                rows.append(row)
        return {"built_from": ref.get("built_from", []), "n_videos": ref.get("n_videos", 0), "built_on": ref.get("built_on", "")}, rows
