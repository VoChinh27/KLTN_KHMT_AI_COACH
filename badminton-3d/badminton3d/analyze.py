"""Điều phối 1 video: pose -> làm sạch -> tách đoạn cần chấm -> đặc trưng -> so với chuẩn HLV -> kết quả."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from badminton3d.angles import ready_features, serve_features
from badminton3d.cleaning import CleanPose, clean_pose
from badminton3d.coverage import legacy_valid_pct
from badminton3d.features import ready_units, serve_setup, serve_units
from badminton3d.pose_io import PoseData
from badminton3d.scoring import score_ready, score_serve
from badminton3d.signals import view_yaw_deg
from badminton3d.segment import ReadySegmentation, ServeSegmentation, segment_ready, segment_serve, serve_from_windows


def exercise_of(name: str) -> Optional[str]:
    m = re.search(r"(ready|serve)", os.path.basename(name), re.I)
    return m.group(1).lower() if m else None


def role_of(name: str) -> str:
    m = re.match(r"(coach|trainee)_", os.path.basename(name), re.I)
    return m.group(1).lower() if m else "unknown"


def load_annotation(video_path: str) -> dict:
    """Đánh dấu tay (tuỳ chọn): <video>.json  ví dụ {"serve_windows_s": [[3.0, 4.5]], "racket_hand": "right"}"""
    for suf in (".json", ".events.json"):
        p = os.path.splitext(video_path)[0] + suf
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


@dataclass
class Context:
    """Mọi thứ cần để vẽ biểu đồ / overlay mà không tính lại."""
    pose: PoseData
    clean: CleanPose
    exercise: str
    ready_seg: Optional[ReadySegmentation] = None
    serve_seg: Optional[ServeSegmentation] = None
    feats: Dict[str, dict] = field(default_factory=dict)
    units: List[dict] = field(default_factory=list)
    arm: Optional[dict] = None
    reach: Optional[np.ndarray] = None


def build_context(df: pd.DataFrame, cfg: dict, exercise: str, racket_hand: str = "auto", serve_windows: Optional[list] = None) -> Context:
    pose = PoseData.from_df(df)
    clean = clean_pose(pose, cfg)
    ctx = Context(pose, clean, exercise)
    if exercise == "ready":
        ctx.ready_seg = segment_ready(clean, cfg)
        ctx.feats = ready_features(clean, cfg)
        ctx.units = ready_units(clean, ctx.ready_seg, cfg)
    else:
        arm, reach, feats = serve_setup(clean, cfg, racket_hand)
        ctx.arm, ctx.reach, ctx.feats = arm, reach, feats
        ctx.serve_seg = serve_from_windows(clean, cfg, [tuple(w) for w in serve_windows], reach) if serve_windows else segment_serve(clean, cfg, reach)
        if ctx.serve_seg.reps:                       # chọn lại tay theo đúng các frame được chấm (ổn định hơn chọn theo cả video)
            mask = ~ctx.serve_seg.not_scored
            arm, reach, feats = serve_setup(clean, cfg, racket_hand, mask)
            ctx.arm, ctx.reach, ctx.feats = arm, reach, feats
        ctx.units = serve_units(clean, ctx.serve_seg, ctx.feats, cfg)
    return ctx


def analyze_context(ctx: Context, cfg: dict, reference: Optional[dict], video_name: str) -> dict:
    p, c = ctx.pose, ctx.clean
    q = cfg["quality"]
    res = {"video": video_name, "role": role_of(video_name), "exercise": ctx.exercise, "fps": p.fps, "n_frames": p.n, "duration_s": p.n / p.fps}
    # ---- chất lượng dữ liệu
    fcov = {n: round(100 * float(np.mean(~np.isnan(f["value"]))), 1) for n, f in ctx.feats.items()}
    subs = {n: round(100 * float(np.nanmean(f["substituted"][~np.isnan(f["value"])])) if (~np.isnan(f["value"])).any() else 0.0, 1) for n, f in ctx.feats.items()}
    res["quality"] = {"legacy_valid_pct": round(legacy_valid_pct(p, q["legacy_valid_conf"]), 1), "feature_valid_pct": fcov, "feature_substituted_pct": subs,
                      "joint_valid_pct": {j: round(100 * float(np.mean(c.valid[j])), 1) for j in c.valid}, "view_yaw_deg": round(view_yaw_deg(c), 1),
                      "interpolated_frames": {j: int(c.interpolated[j].sum()) for j in c.interpolated if c.interpolated[j].any()}}
    if ctx.arm:
        res["quality"]["arm"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ctx.arm.items() if k != "scores"}
        res["quality"]["arm"]["scores"] = {k: round(v, 3) for k, v in ctx.arm["scores"].items()}
    # ---- tách đoạn
    if ctx.exercise == "ready":
        sg = ctx.ready_seg
        names = {v: k for k, v in __import__("badminton3d.segment", fromlist=["READY_STATES"]).READY_STATES.items()}
        counts = {names[k]: round(float(np.mean(sg.state == k)) * 100, 1) for k in names}
        res["segmentation"] = {"episodes": [[int(a), int(b), round((b - a + 1) / p.fps, 2)] for a, b in sg.episodes], "state_pct": counts,
                               "scored_seconds": round(sum(b - a + 1 for a, b in sg.episodes) / p.fps, 2)}
    else:
        sg = ctx.serve_seg
        res["segmentation"] = {"reps": [{"prep_start": r.prep_start, "swing_start": r.swing_start, "peak": r.peak, "end": r.end, "peak_activity_mps": round(r.peak_activity, 2),
                                         "phases": {k: [int(a), int(b)] for k, (a, b) in r.phases.items()}, "truncated": r.truncated, "no_prep": r.no_prep} for r in sg.reps],
                               "not_scored_pct": round(float(np.mean(sg.not_scored)) * 100, 1)}
    # ---- chấm điểm
    if reference is None:
        res["status"], res["score"] = "NO_REFERENCE", None
        return res
    res["reference"] = {"built_from": reference["built_from"], "n_videos": reference["n_videos"], "in_sample": video_name in reference["built_from"]}
    if not ctx.units:
        res["status"], res["score"] = "NO_UNITS", None
        res["message"] = ("Không có đoạn nào đã về tư thế sẵn sàng để chấm (đang di chuyển liên tục hoặc thiếu dữ liệu khớp)." if ctx.exercise == "ready"
                          else "Không phát hiện cú phát. Nếu video có phát cầu: quay từ phía TAY VỢT (tay vợt phải là tay gần camera) hoặc đánh dấu tay cú phát trong file <video>.json.")
        return res
    sc = score_ready(ctx.units, reference, cfg) if ctx.exercise == "ready" else score_serve(ctx.units, reference, cfg)
    if ctx.exercise == "serve" and ctx.serve_seg.reps and ctx.arm:
        from badminton3d.faults import serve_faults, apply_faults
        fl, mt = serve_faults(ctx, cfg)
        sc = apply_faults(sc, fl, mt, cfg)
        res["fault_metrics"] = mt
    res["status"] = "SCORED" if sc["overall"] is not None else "INSUFFICIENT_DATA"
    res["score"] = sc
    return res


def make_warnings(res: dict, ctx: Context, reference: Optional[dict], cfg: dict) -> List[str]:
    """Cảnh báo giúp người đọc biết nên tin điểm ở mức nào."""
    w = []
    if reference:
        if reference["n_videos"] < 3:
            w.append(f"Chuẩn HLV chỉ dựng từ {reference['n_videos']} video: dung sai đang dựa vào mức tối thiểu trong cấu hình, nên điểm mang tính tham khảo.")
        if res.get("reference", {}).get("in_sample"):
            w.append("Video này nằm trong tập video dựng chuẩn: điểm mang tính lạc quan (không phải đánh giá độc lập).")
    q = res["quality"]
    if ctx.exercise == "serve" and ctx.arm:
        if ctx.arm["mode"] == "best_visible":
            w.append(f"Chỉ một tay thấy rõ nên đo bằng tay '{ctx.arm['side']}' (tay camera thấy rõ nhất), chưa chắc là tay vợt. Quay từ phía TAY VỢT hoặc khai báo racket_hand để đo đúng tay vợt.")
        elif ctx.arm["mode"] == "most_active_arm":
            w.append(f"Cả hai tay đều thấy rõ; chọn tay hoạt động nhiều nhất ('{ctx.arm['side']}') làm tay vợt. Có thể khai báo racket_hand để chắc chắn.")
        elif ctx.arm["mode"] == "best_visible_substitute":
            w.append(f"Tay vợt khai báo bị che; đo thay bằng tay '{ctx.arm['side']}' (tay camera thấy rõ nhất).")
        if reference and reference.get("meta", {}).get("arm_side"):
            sides = set(reference["meta"]["arm_side"].values())
            if ctx.arm["side"] not in sides:
                w.append(f"Tay được đo ('{ctx.arm['side']}') khác tay dùng để dựng chuẩn ({', '.join(sorted(sides))}): so sánh có thể không cùng loại.")
    if reference and reference.get("meta", {}).get("view_yaw_deg"):
        yaws = list(reference["meta"]["view_yaw_deg"].values())
        mine = q.get("view_yaw_deg")
        ref_yaw = float(np.median(yaws))
        if mine is not None and mine == mine and abs(mine - ref_yaw) > cfg["quality"].get("view_mismatch_deg", 30):
            w.append(f"Góc quay lệch chuẩn: video này {mine:.0f}° so với {ref_yaw:.0f}° của video dựng chuẩn (0° = chính diện, 90° = ngang). "
                     "Góc 3D của MediaPipe thay đổi theo góc quay nên điểm kém tin cậy — nên quay cùng góc với video HLV.")
    for n, v in q["feature_valid_pct"].items():
        if v < 60:
            w.append(f"Góc '{cfg['features'][ctx.exercise][n]['label']}' chỉ đo được {v:.0f}% số frame.")
    sc = res.get("score")
    if sc and ctx.exercise == "ready":
        for u in sc["units"]:
            if u["score"] is not None and u["score"] < 35:
                w.append(f"Đoạn {u['id']} lệch rất xa chuẩn ({u['score']:.0f} điểm): có thể đang đứng nghỉ hoặc chưa vào tư thế sẵn sàng, không hẳn là sai kỹ thuật.")
    if sc:
        for u in sc["units"]:
            if u.get("critical") and u.get("score") is not None:
                lab = {f: ctx_label(ctx, f) for f in u["critical"]}
                w.append(f"{'Đoạn' if ctx.exercise == 'ready' else 'Lần phát'} {u['id']}: góc quan trọng lệch rất xa chuẩn ({', '.join(lab.values())}) nên điểm bị giới hạn tối đa {cfg['scoring']['critical_cap']:.0f}.")
    if sc and ctx.exercise == "serve":
        if ctx.pose.foot is None:
            w.append("File pose.csv cũ chưa có gót/mũi chân nên chưa kiểm tra được nhấc gót; chạy lại với --force-pose để trích lại.")
        for u in sc["units"]:
            for f in u.get("faults", []):
                w.append(f"Lần phát {u['id']}: {f['kind'].lower()} - {f['label']} (trừ {f['penalty']:.0f} điểm).")
            if u.get("truncated"):
                w.append(f"Lần phát {u['id']}: video kết thúc trước khi cú phát ổn định lại; pha kết thúc có thể chưa đầy đủ.")
            if u.get("no_prep"):
                w.append(f"Lần phát {u['id']}: clip cắt sát, không có đoạn đứng yên trước cú vung nên pha 'Chuẩn bị' chỉ là 0,3 giây trước cú vung (kém tin cậy hơn).")
    return w


def ctx_label(ctx: "Context", feature: str) -> str:
    """Tên tiếng Việt của một đặc trưng (cho cảnh báo)."""
    return {"knee_mean": "góc gối", "trunk_lean": "độ nghiêng thân", "stance_ratio": "độ rộng chân", "elbow_arm": "góc khuỷu tay", "wrist_height": "độ cao cổ tay",
            "hip_mean": "góc hông", "shoulder_arm": "góc nâng tay", "wrist_reach": "tầm với", "elbow_mean": "góc khuỷu", "shoulder_mean": "góc nâng tay"}.get(feature, feature)
