"""ĐO ĐỘ PHỦ (Valid theo từng góc) và hiệu quả của từng bước xử lý mất khớp.
   B0  chỉ số %Valid cũ: 10 khớp cùng lúc có visibility > 0.3 trong cùng một frame
   S1  Valid theo từng góc (visibility >= ngưỡng cho đúng các khớp của góc đó)
   S2  S1 + nội suy khoảng mất ngắn (<= gap_interp_max_s)
   S3  S2 + thay thế: bên còn lại (ready) / cánh tay camera thấy rõ nhất (serve)"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from badminton3d.angles import choose_arm, raw_angles, ready_features, serve_features
from badminton3d.cleaning import clean_pose
from badminton3d.pose_io import PoseData

LEGACY_JOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
                 "left_wrist", "right_wrist", "left_ankle", "right_ankle"]


def legacy_valid_pct(pose: PoseData, conf: float) -> float:
    ok = pose.detected & np.all([np.nan_to_num(pose.vis[j]) > conf for j in LEGACY_JOINTS], axis=0)
    return 100.0 * float(ok.mean())


def coverage_for_video(pose: PoseData, exercise: str, cfg: dict, racket_hand: str = "auto") -> List[dict]:
    rows = []
    add = lambda stage, name, pct: rows.append({"video": pose.video, "exercise": exercise, "stage": stage, "name": name, "pct": round(pct, 1)})
    add("B0", "10 khớp cùng lúc (cũ)", legacy_valid_pct(pose, cfg["quality"]["legacy_valid_conf"]))
    pct = lambda val: 100.0 * float(np.mean(~np.isnan(val)))
    for stage, interp in (("S1", False), ("S2", True)):
        clean = clean_pose(pose, cfg, interpolate=interp, smooth=False)
        for name, (val, rel) in raw_angles(clean, exercise).items():
            add(stage, name, pct(val))
    clean = clean_pose(pose, cfg, interpolate=True, smooth=False)
    if exercise == "ready":
        feats = ready_features(clean, cfg)
    else:
        arm = choose_arm(clean, racket_hand, cfg["quality"]["min_reliability_to_score"])
        feats = serve_features(clean, cfg, arm)
    for name, f in feats.items():
        add("S3", name, pct(f["value"]))
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    g = detail.groupby(["video", "exercise", "stage"])["pct"].mean().unstack("stage")
    return g.reset_index()[["video", "exercise", "B0", "S1", "S2", "S3"]].round(1)
