"""
demo/app_demo.py
==================
Mục đích : Chạy demo trực quan — gọi core/pipeline.py rồi hiển thị kết quả.

V2.2 - Tích hợp render_overlay:
    - Mode video + sprint_start: xuất video overlay 4 sub-phase (skeleton,
      panel điểm, timeline) bằng render_overlay.py khi có --output-video /
      --output, hoặc bật nhanh bằng --overlay (đường dẫn mặc định
      output/verify/<video_stem>_overlay.mp4).
    - Thêm --pose-json (dùng pose cache, bỏ qua bước trích pose chậm) và
      --phases (chỉ định thẳng file *.phases.json) — được chuyển tiếp cho
      render_overlay.
    - Sửa lỗi V2.1: --output có giá trị mặc định nên --output-video luôn
      khác None ở mode video; nay --output chỉ có giá trị mặc định ở mode
      synthetic, nên "có truyền hay không" được phân biệt đúng.

V2.1 - Bổ sung:
    - Mode 'sprint-full': cham 4 sub-phase sprint_start tu 1 video.
    - Output args: --output-report, --output-json, --output-video.
    - Tu dong chon mode 'sprint-full' khi --sport sprint_start (khong can --phase).

Chạy thử:
    python -m demo.app_demo --mode synthetic --output demo_output.mp4
    python -m demo.app_demo --mode video --sport sprint_start --video <path>
    python -m demo.app_demo --mode video --sport sprint_start --video <path> --overlay
    python -m demo.app_demo --mode video --sport sprint_start --video <path> \\
        --output-video output/verify/overlay.mp4 --pose-json data/pose_cache/<stem>.json \\
        --phases data/phases/<stem>.phases.json --output-json report.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.pipeline import Pipeline, PipelineResult, SprintStartFullResult
from core.pose_schema import COCO17_NAMES, PoseFrame
from core.sport_interface import YamlSportPlugin
from stage3_features.feature_extractor import (
    PHASE_DYNAMIC,
    FeatureExtractor,
    bundle_to_summary_dict,
)
from stage4_scoring.reference_model import ReferenceModelBuilder

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

SKELETON_EDGES: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]

COLOR_SKELETON = (60, 220, 60)
COLOR_JOINT = (255, 255, 255)
COLOR_TEXT_GOOD = (60, 200, 60)
COLOR_TEXT_WARN = (0, 200, 255)
COLOR_TEXT_BAD = (60, 60, 230)

DEFAULT_SYNTHETIC_VIDEO = "demo_output.mp4"
DEFAULT_OVERLAY_DIR = "output/verify"


# ---------------------------------------------------------------------------
# Sinh du lieu tong hop
# ---------------------------------------------------------------------------
def _rot(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([np.sin(a), np.cos(a)])


def _make_frame(
    frame_idx: int, t: float, forward_x: float,
    knee_angle_left: float, knee_angle_right: float,
    thigh_angle_left: float = 0.0, thigh_angle_right: float = 0.0,
    trunk_lean_deg: float = 0.0,
    arm_swing_left_deg: float = 0.0, arm_swing_right_deg: float = 0.0,
) -> PoseFrame:
    torso_len, shoulder_w, hip_w = 150.0, 80.0, 60.0
    thigh_len, shank_len = 90.0, 90.0
    upper_arm_len, forearm_len = 60.0, 55.0

    mid_hip = np.array([400.0 + forward_x, 500.0])
    mid_shoulder = mid_hip - torso_len * _rot(trunk_lean_deg)
    left_shoulder = mid_shoulder + np.array([-shoulder_w / 2, 0.0])
    right_shoulder = mid_shoulder + np.array([shoulder_w / 2, 0.0])
    left_hip = mid_hip + np.array([-hip_w / 2, 0.0])
    right_hip = mid_hip + np.array([hip_w / 2, 0.0])
    nose = mid_shoulder + np.array([0.0, -40.0])

    def leg(hip_pt, thigh_angle_deg, knee_angle_deg):
        knee_flex = 180.0 - knee_angle_deg
        knee_pt = hip_pt + thigh_len * _rot(thigh_angle_deg)
        ankle_pt = knee_pt + shank_len * _rot(thigh_angle_deg + knee_flex)
        return knee_pt, ankle_pt

    def arm(shoulder_pt, swing_deg, elbow_flex_deg):
        elbow_pt = shoulder_pt + upper_arm_len * _rot(swing_deg)
        wrist_pt = elbow_pt + forearm_len * _rot(swing_deg + elbow_flex_deg)
        return elbow_pt, wrist_pt

    left_knee, left_ankle = leg(left_hip, thigh_angle_left, knee_angle_left)
    right_knee, right_ankle = leg(right_hip, thigh_angle_right, knee_angle_right)
    left_elbow, left_wrist = arm(
        left_shoulder, arm_swing_left_deg,
        30.0 + 15.0 * np.cos(np.radians(arm_swing_left_deg))
    )
    right_elbow, right_wrist = arm(
        right_shoulder, arm_swing_right_deg,
        30.0 + 15.0 * np.cos(np.radians(arm_swing_right_deg))
    )

    coords = {
        "nose": nose, "left_eye": nose, "right_eye": nose,
        "left_ear": nose, "right_ear": nose,
        "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
        "left_elbow": left_elbow, "right_elbow": right_elbow,
        "left_wrist": left_wrist, "right_wrist": right_wrist,
        "left_hip": left_hip, "right_hip": right_hip,
        "left_knee": left_knee, "right_knee": right_knee,
        "left_ankle": left_ankle, "right_ankle": right_ankle,
    }
    xy = np.stack([coords[n] for n in COCO17_NAMES], axis=0).astype(np.float32)
    conf = np.ones(len(COCO17_NAMES), dtype=np.float32) * 0.95
    return PoseFrame(
        frame_idx=frame_idx, timestamp_s=t, xy=xy, conf=conf,
        joint_names=COCO17_NAMES,
    )


def _make_dynamic_sequence(
    n_frames: int, fps: float, cadence_hz: float,
    knee_amp_left: float, knee_amp_right: float,
    trunk_lean_bias: float = 0.0, forward_speed: float = 220.0,
) -> List[PoseFrame]:
    seq = []
    for i in range(n_frames):
        t = i / fps
        thigh_l = 22.0 * np.sin(2 * np.pi * cadence_hz * t)
        thigh_r = 22.0 * np.sin(2 * np.pi * cadence_hz * t + np.pi)
        knee_l = 180.0 - (50.0 - knee_amp_left * np.cos(2 * np.pi * cadence_hz * t))
        knee_r = 180.0 - (50.0 - knee_amp_right * np.cos(2 * np.pi * cadence_hz * t + np.pi))
        trunk = trunk_lean_bias + 4.0 * np.sin(2 * np.pi * cadence_hz * t)
        arm_l = 15.0 * np.sin(2 * np.pi * cadence_hz * t + np.pi)
        arm_r = 15.0 * np.sin(2 * np.pi * cadence_hz * t)
        seq.append(
            _make_frame(i, t, forward_speed * t, knee_l, knee_r,
                        thigh_l, thigh_r, trunk, arm_l, arm_r)
        )
    return seq


def build_synthetic_demo(fps: float = 30.0):
    coach_sequences = [
        _make_dynamic_sequence(90, fps, cadence, 40.0, 40.0)
        for cadence in (1.4, 1.45, 1.5, 1.55)
    ]
    trainee_sequence = _make_dynamic_sequence(
        90, fps, cadence_hz=1.48, knee_amp_left=39.0, knee_amp_right=18.0,
    )
    return coach_sequences, trainee_sequence


# ---------------------------------------------------------------------------
# Overlay (synthetic) — skeleton đơn giản trên canvas trống
# ---------------------------------------------------------------------------
def _fit_transform(
    sequence: List[PoseFrame], canvas_size: Tuple[int, int]
) -> Tuple[float, np.ndarray]:
    all_xy = np.concatenate([pf.xy for pf in sequence], axis=0)
    min_xy, max_xy = all_xy.min(axis=0), all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    w, h = canvas_size
    margin = 0.85
    scale = min((w * margin) / span[0], (h * margin) / span[1])
    offset = np.array([w / 2, h / 2]) - scale * (min_xy + max_xy) / 2.0
    return float(scale), offset


def _status_color(symbol: str):
    return {
        "✓": COLOR_TEXT_GOOD,
        "⚠": COLOR_TEXT_WARN,
        "✗": COLOR_TEXT_BAD,
    }.get(symbol, COLOR_TEXT_WARN)


def render_demo_video(
    sequence: List[PoseFrame],
    result: PipelineResult,
    fps: float,
    output_path: str,
    canvas_size: Tuple[int, int] = (720, 1000),
    max_feedback_lines: int = 4,
) -> None:
    if cv2 is None:
        raise RuntimeError("Cần cài opencv-python: pip install opencv-python-headless")

    w, h = canvas_size
    scale, offset = _fit_transform(sequence, canvas_size)
    joint_names = list(sequence[0].joint_names)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được file video output: {output_path}")
    feedback = result.feedback
    top_items = feedback.items[:max_feedback_lines]

    for pf in sequence:
        canvas = np.full((h, w, 3), 24, dtype=np.uint8)
        pts = (pf.xy * scale + offset).astype(int)

        for a, b in SKELETON_EDGES:
            pa, pb = pts[joint_names.index(a)], pts[joint_names.index(b)]
            cv2.line(canvas, tuple(pa), tuple(pb), COLOR_SKELETON, 3)
        for p in pts:
            cv2.circle(canvas, tuple(p), 5, COLOR_JOINT, -1)

        cv2.putText(canvas, feedback.summary_line, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y = 75
        for item in top_items:
            color = _status_color(item.status_symbol)
            line = f"{item.status_symbol} {item.display_name}"
            cv2.putText(canvas, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 28

        writer.write(canvas)

    writer.release()


# ---------------------------------------------------------------------------
# Overlay (video thật, sprint_start) — dùng render_overlay.py
# ---------------------------------------------------------------------------
def _resolve_overlay_path(args: argparse.Namespace) -> Optional[str]:
    """Đường dẫn video overlay cho mode video, hoặc None nếu không yêu cầu.

    Ưu tiên: --output-video / --output (người dùng chỉ định) > --overlay
    (đường dẫn mặc định theo tên video) > None.
    """
    if args.output_video:
        return args.output_video
    if args.overlay:
        return str(Path(DEFAULT_OVERLAY_DIR) / f"{Path(args.video).stem}_overlay.mp4")
    return None


def export_sprint_start_overlay(args: argparse.Namespace, out_path: str) -> None:
    """Xuất video overlay 4 sub-phase bằng render_overlay.render_overlay().

    render_overlay tự khởi tạo Pipeline, trích pose (hoặc đọc --pose-json),
    chấm điểm theo phases (manual '<stem>.phases.json' hoặc auto) rồi vẽ
    skeleton + panel + timeline lên từng frame của video gốc.
    """
    try:
        from render_overlay import render_overlay
    except ImportError as exc:
        raise RuntimeError(
            "Không import được render_overlay (cần chạy từ thư mục gốc dự án, "
            "ví dụ `python -m demo.app_demo ...`, và đã cài opencv-python): "
            f"{exc}"
        ) from exc

    if args.pose_json and not Path(args.pose_json).exists():
        print(f"[app_demo] ⚠ --pose-json '{args.pose_json}' không tồn tại — "
              "render_overlay sẽ trích pose lại từ video (chậm hơn).")
    if args.phases and not Path(args.phases).exists():
        print(f"[app_demo] ⚠ --phases '{args.phases}' không tồn tại — "
              "pipeline sẽ tự tìm file theo tên video hoặc phân đoạn tự động.")

    print(f"[app_demo] Đang xuất video overlay sprint_start → {out_path}")
    render_overlay(
        video_path=args.video,
        pose_json=args.pose_json,
        out_path=out_path,
        system_config_path=args.system_config,
        phases_path=args.phases,
    )
    print(f"[app_demo] Đã ghi video overlay tại: {out_path}")


def _accepts_kwarg(fn, name: str) -> bool:
    """True nếu `fn` nhận tham số tên `name` (hoặc có **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------
def _feature_scores_payload(score_report) -> List[Dict[str, Any]]:
    return [
        {
            "feature_key": fs.feature_key,
            "student_value": fs.student_value,
            "reference_center": fs.reference_center,
            "score": fs.score_0_100,
            "level": fs.level,
            "scoring_type": fs.scoring_type,
            "is_in_range": fs.is_in_range,
        }
        for fs in score_report.feature_scores
    ]


def _save_json_report(result: Any, path: str) -> None:
    """Luu report dang JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, SprintStartFullResult):
        payload = {
            "sport": result.sport,
            "overall_score": float(result.overall_score),
            "score_confidence": result.score_confidence,
            "quality_flags": result.quality_flags,
            "sub_phase_order": result.sub_phase_order,
            "sub_phases": {},
        }
        for phase, sp in result.sub_phase_results.items():
            sp_data: Dict[str, Any] = {
                "is_scored": sp.is_scored,
                "skip_reason": sp.skip_reason,
            }
            if sp.has_score():
                sp_data["score"] = float(sp.result.score_report.overall_score)
                sp_data["summary"] = sp.result.feedback.summary_line
                sp_data["feature_scores"] = _feature_scores_payload(sp.result.score_report)
            payload["sub_phases"][phase] = sp_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    elif isinstance(result, PipelineResult):
        payload = {
            "phase": result.phase,
            "score": float(result.score_report.overall_score),
            "summary": result.feedback.summary_line,
            "feature_scores": _feature_scores_payload(result.score_report),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_text_report(text: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------
def run_synthetic_demo(args: argparse.Namespace) -> None:
    fps = 30.0
    coach_sequences, trainee_sequence = build_synthetic_demo(fps=fps)

    plugin = YamlSportPlugin.from_yaml(
        str(Path(args.sport_config_dir) / f"{args.sport}.yaml")
    )
    extractor = FeatureExtractor()
    coach_bundles = [
        extractor.extract(seq, fps=fps, phase=args.phase)
        for seq in coach_sequences
    ]
    reference_model = ReferenceModelBuilder(
        method=plugin.reference_method()
    ).build(sport=args.sport, phase=args.phase, bundles=coach_bundles)

    pipeline = Pipeline(
        sport_plugin=plugin, system_config=_minimal_system_config()
    )
    pipeline.set_reference_model(args.phase, reference_model)

    result = pipeline.process_tracked_sequence(
        trainee_sequence, fps=fps, phase=args.phase
    )
    text = result.feedback.render_text()
    print(text)

    if args.output_video:
        render_demo_video(
            trainee_sequence, result, fps=fps, output_path=args.output_video
        )
        print(f"\n[app_demo] Đã ghi video overlay tại: {args.output_video}")

    if args.output_report:
        _save_text_report(text, args.output_report)
        print(f"[app_demo] Đã ghi text report tại: {args.output_report}")

    if args.output_json:
        _save_json_report(result, args.output_json)
        print(f"[app_demo] Đã ghi JSON report tại: {args.output_json}")


# ---------------------------------------------------------------------------
# Video demo
# ---------------------------------------------------------------------------
def run_video_demo(args: argparse.Namespace) -> None:
    if not Path(args.video).exists():
        raise SystemExit(f"Không tìm thấy video: {args.video}")

    overlay_path = _resolve_overlay_path(args)

    # Sprint_start co mode full 4 sub-phase
    if args.sport == "sprint_start":
        pipeline = Pipeline.from_config(args.system_config, sport=args.sport)

        # Chuyển --phases cho pipeline nếu API hỗ trợ, để báo cáo văn bản và
        # video overlay dùng CÙNG một nguồn phases (tránh lệch kết quả).
        call_kwargs: Dict[str, Any] = {}
        if args.phases:
            if _accepts_kwarg(pipeline.process_sprint_start_video, "phases_path"):
                call_kwargs["phases_path"] = args.phases
            else:
                print("[app_demo] ⚠ process_sprint_start_video() chưa nhận phases_path — "
                      "báo cáo văn bản dùng phases tự tìm theo tên video; --phases chỉ "
                      "áp dụng cho video overlay.")
        full_result = pipeline.process_sprint_start_video(args.video, **call_kwargs)
        text = full_result.render_text()
        print(text)

        # In chi tiet tung sub-phase
        print()
        for phase, sp in full_result.sub_phase_results.items():
            if sp.has_score():
                print("=" * 60)
                print(f"[{phase}]")
                print(sp.result.feedback.render_text())

        # Lưu báo cáo TRƯỚC khi render overlay: nếu render lỗi (thiếu codec,
        # hết VRAM...) thì báo cáo vẫn được giữ lại.
        if args.output_report:
            _save_text_report(text, args.output_report)
            print(f"\n[app_demo] Đã ghi text report tại: {args.output_report}")

        if args.output_json:
            _save_json_report(full_result, args.output_json)
            print(f"[app_demo] Đã ghi JSON report tại: {args.output_json}")

        if overlay_path:
            # Giải phóng pipeline đã dùng để chấm điểm: render_overlay tự tạo
            # Pipeline riêng (tránh giữ 2 bản model pose trên GPU cùng lúc).
            del pipeline
            try:
                export_sprint_start_overlay(args, overlay_path)
            except Exception as exc:
                traceback.print_exc()
                raise SystemExit(f"[app_demo] Xuất video overlay thất bại: {exc}")

        return

    # Các sport khác: cham 1 phase
    if not args.phase:
        raise SystemExit(f"Sport '{args.sport}' cần --phase (ví dụ static/dynamic).")

    pipeline = Pipeline.from_config(args.system_config, sport=args.sport)
    if args.phase not in pipeline.reference_models:
        raise SystemExit(
            f"Chưa có ReferenceModel cho pha '{args.phase}'. Hãy build trước, "
            f"hoặc dùng --mode synthetic để demo nhanh."
        )
    result = pipeline.process_video(args.video, phase=args.phase)
    text = result.feedback.render_text()
    print(text)

    if args.output_report:
        _save_text_report(text, args.output_report)
        print(f"[app_demo] Đã ghi text report tại: {args.output_report}")
    if args.output_json:
        _save_json_report(result, args.output_json)
        print(f"[app_demo] Đã ghi JSON report tại: {args.output_json}")
    if overlay_path:
        print(f"[app_demo] Video overlay trên video thật hiện chỉ hỗ trợ sport "
              f"'sprint_start' (sport hiện tại: '{args.sport}') — bỏ qua.")


def _minimal_system_config() -> dict:
    return {
        "pose_backend": "rtmpose",
        "pose_backend_kwargs": {
            "yolo_weights": "yolo11n-pose.pt",
            "yolo_conf": 0.25,
            "yolo_imgsz": 640,
            "yolo_device": "cuda:0",
            "tracker_yaml": "bytetrack.yaml",
            "rtmpose_onnx": "weights/rtmpose-m.onnx",
            "rtmpose_device": "cuda",
            "rtmpose_bbox_pad": 0.10,
        },
        "min_keypoint_confidence": 0.35,
        "feedback_thresholds": {"good": 85.0, "warn": 60.0},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo trực quan AI Personal Sports Coach"
    )
    parser.add_argument("--mode", choices=["synthetic", "video"], default="synthetic")
    parser.add_argument("--sport", default="running")
    parser.add_argument(
        "--phase", default=None,
        help=(
            "Pha cần chấm. Với sprint_start có thể bỏ qua để dùng chế độ "
            "full 4 sub-phase."
        ),
    )
    parser.add_argument("--sport-config-dir", default="config/sport_configs")
    parser.add_argument("--system-config", default="config/system_config.yaml")
    parser.add_argument("--video", help="Đường dẫn video (bắt buộc nếu --mode video)")

    # Output args
    parser.add_argument(
        "--output", default=None,
        help=(f"Đường dẫn video overlay (tương thích ngược, đồng nghĩa --output-video). "
             f"Mặc định '{DEFAULT_SYNTHETIC_VIDEO}' ở mode synthetic; ở mode video "
             "chỉ xuất khi được truyền."),
    )
    parser.add_argument("--output-video", default=None,
                        help="Đường dẫn video overlay output (.mp4)")
    parser.add_argument("--output-report", default=None,
                        help="Đường dẫn text report (.txt)")
    parser.add_argument("--output-json", default=None,
                        help="Đường dẫn JSON report (.json)")

    # Overlay cho video thật (sprint_start) — chuyển tiếp cho render_overlay
    parser.add_argument(
        "--overlay", action="store_true",
        help=(f"Mode video + sprint_start: xuất video overlay tới "
              f"{DEFAULT_OVERLAY_DIR}/<video_stem>_overlay.mp4 "
              "(bỏ qua nếu đã truyền --output-video/--output)."),
    )
    parser.add_argument(
        "--pose-json", default=None,
        help="(Tùy chọn) pose cache JSON để overlay không phải trích pose lại",
    )
    parser.add_argument(
        "--phases", default=None,
        help="(Tùy chọn) chỉ định thẳng file *.phases.json cho sprint_start",
    )

    args = parser.parse_args()

    # Backward compat: --output == --output-video. Chỉ áp mặc định cho
    # synthetic; ở mode video, không truyền thì KHÔNG xuất overlay.
    args.output_video = args.output_video or args.output
    if args.mode == "synthetic" and args.output_video is None:
        args.output_video = DEFAULT_SYNTHETIC_VIDEO

    # Auto phase cho synthetic
    if args.mode == "synthetic" and args.phase is None:
        args.phase = PHASE_DYNAMIC

    if args.mode == "synthetic":
        run_synthetic_demo(args)
    else:
        if not args.video:
            raise SystemExit("--mode video cần kèm --video <đường dẫn>")
        run_video_demo(args)


if __name__ == "__main__":
    main()