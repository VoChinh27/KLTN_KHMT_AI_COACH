"""Giao diện dòng lệnh:  python -m badminton3d <lệnh>   (chạy từ thư mục gốc dự án; Input/ → Output/)"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from badminton3d import __version__
from badminton3d.analyze import exercise_of
from badminton3d.batch import build_reference_cmd, list_videos, run_batch, run_coverage, run_evaluation
from badminton3d.config import load_config
from badminton3d.runner import process_video


def _common(p):
    p.add_argument("--input", default="Input", help="Thư mục video đầu vào")
    p.add_argument("--output", default="Output", help="Thư mục kết quả")
    p.add_argument("--reference", default="Reference", help="Thư mục chuẩn HLV (ready.json, serve.json)")
    p.add_argument("--config", default=None, help="File cấu hình YAML (mặc định config/badminton3d.yaml)")
    p.add_argument("--racket-hand", choices=["auto", "left", "right"], default=None, help="Tay vợt (mặc định lấy trong cấu hình)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="badminton3d", description="Chấm điểm tư thế cầu lông (ready / serve) bằng MediaPipe 3D so với chuẩn HLV",
                                 epilog="Ví dụ:\n  python -m badminton3d build-reference ready Input/coach_ready_01.mp4 Input/coach_ready_04.mp4\n"
                                        "  python -m badminton3d run\n  python -m badminton3d analyze Input/trainee_serve_01.mp4",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"badminton3d {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="Chạy TẤT CẢ video trong Input/ -> Output/ (báo cáo từng video + tổng hợp + biểu đồ)")
    _common(p)
    p.add_argument("--no-video", action="store_true", help="Không xuất video overlay (nhanh hơn)")
    p.add_argument("--force-pose", action="store_true", help="Trích lại pose, bỏ qua bản đã lưu")
    p.set_defaults(fn=_run)

    p = sub.add_parser("analyze", help="Chạy MỘT video")
    p.add_argument("video")
    _common(p)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--force-pose", action="store_true")
    p.set_defaults(fn=_analyze)

    p = sub.add_parser("build-reference", help="Dựng chuẩn từ 1–3 video HLV tập ĐÚNG")
    p.add_argument("exercise", choices=["ready", "serve"])
    p.add_argument("videos", nargs="+")
    _common(p)
    p.set_defaults(fn=_build_ref)

    p = sub.add_parser("coverage", help="Đo độ phủ: %%Valid cũ -> Valid theo từng góc -> sau xử lý mất khớp")
    _common(p)
    p.set_defaults(fn=_coverage)

    p = sub.add_parser("evaluate", help="Điểm không lạc quan (bỏ-một-video) + biểu đồ tổng hợp")
    _common(p)
    p.set_defaults(fn=_evaluate)

    p = sub.add_parser("record", help="QUAY VIDEO TRỰC TIẾP từ camera, dừng quay là chấm điểm ngay (Output y như `analyze`)")
    p.add_argument("exercise", choices=["ready", "serve"])
    p.add_argument("--source", default="0", help="Số thứ tự webcam (0, 1...), địa chỉ camera điện thoại (http://.../video) hoặc đường dẫn file để giả lập")
    p.add_argument("--role", default="trainee", choices=["trainee", "coach"], help="Vai trò, quyết định tên file (mặc định trainee)")
    p.add_argument("--no-preview", action="store_true", help="Không mở cửa sổ xem trước: quay ngay và dừng khi hết nguồn / quá --max-seconds")
    p.add_argument("--max-seconds", type=float, default=180.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-analyze", action="store_true", help="Chỉ quay và lưu, chưa chấm")
    _common(p)
    p.set_defaults(fn=_record)

    p = sub.add_parser("check", help="Kiểm tra chất lượng video (FPS, rung máy, kích thước người, nhiễu pose) mà không chấm điểm")
    p.add_argument("video")
    _common(p)
    p.set_defaults(fn=_check)

    p = sub.add_parser("config", help="In file cấu hình đang dùng")
    p.add_argument("--config", default=None)
    p.set_defaults(fn=_config)
    return ap


def _run(a) -> int:
    cfg = load_config(a.config)
    res = run_batch(a.input, cfg, a.output, a.reference, a.racket_hand, not a.no_video, a.force_pose)
    ok = sum(1 for r in res if r.get("status") == "SCORED")
    print(f"\nXong: {ok}/{len(res)} video được chấm điểm. Mở {os.path.join(a.output, '_tong_hop', 'index.html')} để xem tổng hợp.")
    return 0


def _analyze(a) -> int:
    cfg = load_config(a.config)
    res, _ = process_video(a.video, cfg, a.output, a.reference, a.racket_hand, not a.no_video, a.force_pose)
    sc = res.get("score")
    print(f"[{res['status']}] {res['video']} -> {os.path.join(a.output, res['video'], 'report.html')}")
    if sc and sc.get("overall") is not None:
        print(f"  Điểm: {sc['overall']:.1f}/100 ({sc['grade']})")
        for e in sc["errors"][:3]:
            print(f"  - {e['advice']}  (+{e['potential_gain']:.1f} điểm)")
    elif res.get("message"):
        print("  " + res["message"])
    for w in res.get("warnings", []):
        print("  ⚠ " + w)
    return 0


def _build_ref(a) -> int:
    cfg = load_config(a.config)
    for v in a.videos:
        if exercise_of(v) != a.exercise:
            print(f"Lỗi: {v} không phải video bài '{a.exercise}'", file=sys.stderr)
            return 1
    ref = build_reference_cmd(a.exercise, a.videos, cfg, a.output, a.reference, a.racket_hand)
    print(f"Đã dựng chuẩn '{a.exercise}' từ {ref['n_videos']} video -> {os.path.join(a.reference, a.exercise + '.json')}")
    return 0


def _coverage(a) -> int:
    cfg = load_config(a.config)
    det = run_coverage(list_videos(a.input), cfg, a.output, os.path.join(a.output, "_tong_hop"), a.racket_hand)
    from badminton3d.coverage import summarize
    print(summarize(det).to_string(index=False))
    return 0


def _evaluate(a) -> int:
    cfg = load_config(a.config)
    ev = run_evaluation(list_videos(a.input), cfg, a.output, a.reference, os.path.join(a.output, "_tong_hop"), a.racket_hand)
    print(ev.to_string(index=False))
    return 0


def _record(a) -> int:
    from badminton3d.record import record
    cfg = load_config(a.config)
    res = record(a.exercise, cfg, source=a.source, role=a.role, input_dir=a.input, output_dir=a.output, reference_dir=a.reference, racket_hand=a.racket_hand,
                 width=a.width, height=a.height, fps=a.fps, preview=not a.no_preview, max_seconds=a.max_seconds, analyze=not a.no_analyze)
    return 0 if res is not None else 1


def _check(a) -> int:
    from badminton3d.analyze import build_context, load_annotation
    from badminton3d.runner import get_pose
    from badminton3d.videoqc import video_qc
    cfg = load_config(a.config)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    ex = exercise_of(stem) or "serve"
    ann = load_annotation(a.video)
    out = os.path.join(a.output, stem)
    ctx = build_context(get_pose(a.video, cfg, out, False, ann.get("person_roi")), cfg, ex, ann.get("racket_hand") or "auto", None)
    q = video_qc(a.video, ctx, cfg)
    for k, v in q.items():
        if k != "warnings":
            print(f"  {k}: {v}")
    print("Cảnh báo:" if q["warnings"] else "Video đạt các kiểm tra chất lượng.")
    for w in q["warnings"]:
        print("  - " + w)
    return 0


def _config(a) -> int:
    from badminton3d.config import DEFAULT_CONFIG
    print(open(a.config or DEFAULT_CONFIG, encoding="utf-8").read())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1
