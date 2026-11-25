"""Chạy 1 video từ Input -> Output/<video>/ (pose, chuẩn, chấm điểm, biểu đồ, báo cáo, overlay)."""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import pandas as pd

from badminton3d import charts
from badminton3d.analyze import Context, analyze_context, build_context, exercise_of, load_annotation, make_warnings, role_of
from badminton3d.config import resolve
from badminton3d.pose_io import extract_pose, load_pose, save_pose
from badminton3d.reference import load_reference
from badminton3d.render import render_overlay
from badminton3d.report import build_html, to_py, write_angles_csv


def get_pose(video_path: str, cfg: dict, out_dir: str, force: bool = False, roi=None) -> pd.DataFrame:
    """Pose được lưu ở Output/<video>/pose.csv; chạy lại thì dùng bản đã lưu (--force-pose để trích lại)."""
    cache = os.path.join(out_dir, "pose.csv")
    if os.path.isfile(cache) and not force:
        return load_pose(cache)
    df = extract_pose(video_path, str(resolve(cfg["pose"]["model_path"])), cfg["pose"], roi=roi)
    save_pose(df, cache)
    return df


def try_reference(reference_dir: str, exercise: str) -> Optional[dict]:
    p = os.path.join(reference_dir, f"{exercise}.json")
    return load_reference(p) if os.path.isfile(p) else None


def process_video(video_path: str, cfg: dict, out_root: str, reference_dir: str, racket_hand: Optional[str] = None, render: bool = True,
                  force_pose: bool = False, reference: Optional[dict] = None, exercise: Optional[str] = None) -> Tuple[dict, Optional[Context]]:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    ex = exercise or exercise_of(stem)
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    if ex is None:
        res = {"video": stem, "status": "UNKNOWN_EXERCISE", "message": "Tên file phải chứa 'ready' hoặc 'serve' (vd. trainee_serve_01.mp4)."}
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        return res, None
    ann = load_annotation(video_path)
    hand = ann.get("racket_hand") or racket_hand or cfg["camera_protocol"]["racket_hand"]
    df = get_pose(video_path, cfg, out_dir, force_pose, ann.get("person_roi"))
    ctx = build_context(df, cfg, ex, hand, ann.get("serve_windows_s"))
    ref = reference if reference is not None else try_reference(reference_dir, ex)
    res = analyze_context(ctx, cfg, ref, stem)
    res["warnings"] = make_warnings(res, ctx, ref, cfg)
    res["annotation_used"] = bool(ann)
    if "n_people" in df:                                              # nhiều người trong khung: nói rõ đã chọn người nào
        multi = float((df["n_people"] > 1).mean() * 100)
        if multi >= 10:
            res["warnings"].append(f"Có nhiều người trong khung ({multi:.0f}% frame). Hệ thống chọn người mục tiêu theo '{cfg['pose'].get('select', 'largest')}' "
                                   "và bám theo; nếu chọn sai người, khai báo vùng người tập bằng \"person_roi\": [x0, y0, x1, y1] (tỉ lệ 0–1) trong Input/<video>.json và chạy lại với --force-pose.")
        sw = int(df["switches"].max()) if "switches" in df else 0
        if sw:
            res["warnings"].append(f"Phải chọn lại người mục tiêu {sw} lần (mất dấu người): kiểm tra overlay xem có bám nhầm người khác không.")
    try:
        from badminton3d.videoqc import video_qc
        res["video_qc"] = video_qc(video_path, ctx, cfg)
        res["warnings"].extend(res["video_qc"]["warnings"])
    except Exception as exc:                                            # kiểm tra chất lượng không được làm hỏng việc chấm điểm
        res["video_qc"] = {"error": str(exc)}
    paths = {"report_json": "report.json", "report_html": "report.html", "dashboard": "dashboard.png", "angles_timeline": "angles_timeline.png",
             "quality": "quality.png", "angles_csv": "angles.csv", "pose_csv": "pose.csv"}
    charts.dashboard(ctx, res, cfg, os.path.join(out_dir, "dashboard.png"))
    charts.angles_timeline(ctx, ref, res, cfg, os.path.join(out_dir, "angles_timeline.png"))
    charts.quality_chart(res, cfg, os.path.join(out_dir, "quality.png"))
    write_angles_csv(ctx, os.path.join(out_dir, "angles.csv"))
    if render:
        render_overlay(video_path, ctx, res, cfg, os.path.join(out_dir, "overlay.mp4"))
        paths["overlay"] = "overlay.mp4"
    res["files"] = paths
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(build_html(res, cfg, out_dir))
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(to_py(res), fh, ensure_ascii=False, indent=2)
    return res, ctx
