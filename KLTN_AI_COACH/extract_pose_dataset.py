"""
extract_pose_dataset.py
=======================
Trích xuất DỮ LIỆU POSE ĐẦY ĐỦ từ nhiều video (RTMPose-m ONNX + YOLO/ByteTrack) và GOM
mỗi loại dữ liệu thành MỘT file chung cho toàn bộ video.

Đầu ra (OUTPUT_DIR/_aggregated/):
    keypoints_long.csv            [A] 1 dòng / keypoint / frame  (nguồn dữ liệu gốc, đầy đủ nhất)
    keypoints_wide_<model>.csv    [B] 1 dòng / frame, mỗi model 1 file (schema keypoint khác nhau)
    metrics_frame_level.csv       [C] 1 dòng / frame: góc, tỉ lệ, front/back, vận tốc/gia tốc COM, cờ chất lượng
    summary_video_level.csv       [D] 1 dòng / (video, model): metadata + thống kê + static/launch/cadence + QC
    run_manifest.json             cấu hình, phiên bản thư viện, trạng thái từng video, lỗi

Khóa nối giữa các file: (video_id, model, frame).

Quy ước metric (BÁM THEO stage3_features/normalizer.py để số liệu khớp pipeline):
    - Tên cột: angle_<tên> (độ), ratio_<tên> (chia cho body_scale = |vai_giữa - hông_giữa|).
    - angle_trunk: góc so với phương thẳng đứng (0° = đứng thẳng, 90° = nằm ngang).
    - *_front / *_back được CỘNG THÊM bên cạnh *_left / *_right (chân trước = chân có
      |x mắt cá - x tâm hông| nhỏ hơn, như core/side_mapping.py). Heuristic này CHỈ có
      nghĩa ở tư thế xuất phát; khi chạy đều chân trước/sau đổi liên tục - đừng dùng.
    - Keypoint có conf < ngưỡng bị che thành NaN trước khi tính metric (file A vẫn giữ giá
      trị thô + conf, không mất dữ liệu).
    - Cột hậu tố _sm = đã nội suy khe hở ngắn + làm mượt Savitzky-Golay (chỉ dùng phân
      tích/cadence; cột không _sm là dữ liệu thô).

Chạy:
    python extract_pose_dataset.py --input <thư mục video> --output <thư mục kết quả>
    python extract_pose_dataset.py --help
Chạy lại sẽ BỎ QUA video đã xong (resume); dùng --force để xử lý lại.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# =========================================================
#                       CẤU HÌNH
# =========================================================
@dataclass
class Config:
    input_dir: str = r"C:\Users\FPTSHOP\Desktop\KLTN_AI_COACH\data\raw_videos"
    output_dir: str = r"C:\Users\FPTSHOP\Desktop\KLTN_AI_COACH\test_posemodal"
    keyword: str = "coach"
    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    models: Tuple[str, ...] = ("rtmpose",)          # ← ĐỔI: bỏ yolo/mediapipe
    yolo_weights: str = "yolo11n-pose.pt"           # ← ĐỔI: yolov8m → yolo11n
    yolo_imgsz: int = 640                           # ← ĐỔI: 960 → 640
    rtmpose_onnx: str = "weights/rtmpose-m.onnx"    # ← THÊM
    rtmpose_device: str = "cuda"                    # ← THÊM
    yolo_device: str = "cuda:0"                     # ← THÊM
    mp_model_complexity: int = 2                    # giữ để backward compat
    conf_threshold: float = 0.3
    write_video: bool = True
    export_wide: bool = True
    force: bool = False
    clean_parts: bool = False


# Tham số phân tích (cùng bậc độ lớn với anomaly_detector.py)
MIN_BODY_SCALE_PX = 15.0
STATIC_VEL_MAX = 0.6          # body_scale/s; dưới ngưỡng = đứng yên
STATIC_MIN_S = 0.5            # đoạn tĩnh tối thiểu (giây)
SMOOTH_WINDOW_S = 0.2         # cửa sổ Savitzky-Golay (giây)
MAX_INTERP_GAP_FRAMES = 5     # chỉ nội suy khe hở NaN ngắn hơn mức này
PEAK_PROMINENCE_DEG = 15.0    # độ nổi bật đỉnh góc gối khi đếm cadence
PEAK_MIN_DISTANCE_S = 0.25
QC_MIN_PCT_DETECTED = 0.80
QC_MIN_PCT_VALID = 0.60
QC_FPS_MISMATCH = 0.05

# =========================================================
#                 SCHEMA KEYPOINT / SKELETON
# =========================================================
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
POSE33_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
MODEL_NAMES = {"rtmpose": COCO17_NAMES, "yolo": COCO17_NAMES, "mediapipe": POSE33_NAMES}
COCO17_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                   (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

REQUIRED_JOINTS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_elbow", "right_elbow",
    # "left_wrist", "right_wrist",  # ← BỎ vì tay đánh sau lưng hay che
    # "nose",                        # ← BỎ vì cúi đầu khi xuất phát hay che
]
MAIN_JOINTS = REQUIRED_JOINTS[:-1]  # hoặc chỉnh theo logic của bạn

# Đặc trưng then chốt cho tư thế "Sẵn sàng" (static) và thời điểm bật (launch)
KEY_FEATURES = [
    "angle_knee_front", "angle_knee_back", "angle_hip_front", "angle_hip_back",
    "angle_elbow_left", "angle_elbow_right", "angle_trunk",
    "ratio_hip_above_shoulder", "ratio_wrist_width", "ratio_shoulder_width",
    "ratio_hand_ground_clearance", "trunk_lean_signed_deg", "hip_height_rel_ankle",
]


# =========================================================
#                      HÌNH HỌC
# =========================================================
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Góc tại đỉnh b tạo bởi a-b-c (độ). NaN nếu thiếu điểm/suy biến."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if not np.isfinite(n1 * n2) or n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _angle_to_vertical(v: np.ndarray) -> float:
    """0° = thẳng đứng, 90° = nằm ngang (không phân biệt chiều)."""
    if not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def determine_front_leg(P: np.ndarray, idx: Dict[str, int]) -> Optional[str]:
    hip_cx = (P[idx["left_hip"]][0] + P[idx["right_hip"]][0]) / 2.0
    dl = abs(P[idx["left_ankle"]][0] - hip_cx)
    dr = abs(P[idx["right_ankle"]][0] - hip_cx)
    if not (np.isfinite(dl) and np.isfinite(dr)):
        return None
    return "left" if dl <= dr else "right"


def remap_front_back(values: Dict[str, float], front: Optional[str]) -> Dict[str, float]:
    """Thêm khóa *_front/*_back từ *_left/*_right (NaN nếu chưa xác định được chân trước)."""
    nan = float("nan")
    bases: List[str] = []
    for key in values:
        for side in ("left", "right"):
            if key.endswith(f"_{side}"):
                b = key[: -len(side) - 1]
                if b not in bases:
                    bases.append(b)
    # Thứ tự cột CỐ ĐỊNH: luôn <base>_front rồi <base>_back, bất kể chân nào đang là chân trước
    out: Dict[str, float] = {}
    for b in bases:
        l, r = values.get(f"{b}_left", nan), values.get(f"{b}_right", nan)
        if front == "left":
            f, k = l, r
        elif front == "right":
            f, k = r, l
        else:
            f = k = nan
        out[f"{b}_front"] = f
        out[f"{b}_back"] = k
    return out


# =========================================================
#                  METRIC THEO TỪNG FRAME
# =========================================================
def compute_frame_metrics(kps: np.ndarray, idx: Dict[str, int], thr: float) -> Dict[str, float]:
    """kps: (K,3) [x_px, y_px, conf]. Keypoint conf<thr -> NaN."""
    P = kps[:, :2].astype(np.float64).copy()
    conf = kps[:, 2]
    P[~(conf >= thr)] = np.nan
    g = lambda n: P[idx[n]]

    ls, rs, lh, rh = g("left_shoulder"), g("right_shoulder"), g("left_hip"), g("right_hip")
    lk, rk, la, ra = g("left_knee"), g("right_knee"), g("left_ankle"), g("right_ankle")
    le, re, lw, rw, nose = g("left_elbow"), g("right_elbow"), g("left_wrist"), g("right_wrist"), g("nose")

    mid_s, mid_h = (ls + rs) / 2.0, (lh + rh) / 2.0
    trunk_vec = mid_s - mid_h
    body_scale = float(np.linalg.norm(trunk_vec))
    scale = body_scale if np.isfinite(body_scale) and body_scale >= 1e-6 else float("nan")

    head_vec = nose - mid_s
    n1, n2 = np.linalg.norm(trunk_vec), np.linalg.norm(head_vec)
    head_angle = (float(np.degrees(np.arccos(np.clip(np.dot(trunk_vec, head_vec) / (n1 * n2), -1, 1))))
                  if np.isfinite(n1 * n2) and n1 > 1e-6 and n2 > 1e-6 else float("nan"))

    angles = {
        "elbow_left": _angle(ls, le, lw), "elbow_right": _angle(rs, re, rw),
        "knee_left": _angle(lh, lk, la), "knee_right": _angle(rh, rk, ra),
        "hip_left": _angle(ls, lh, lk), "hip_right": _angle(rs, rh, rk),
        "trunk": _angle_to_vertical(trunk_vec), "head": head_angle,
    }
    if "left_foot_index" in idx:  # MediaPipe: đo trực tiếp
        angles["ankle_left"] = _angle(lk, la, g("left_foot_index"))
        angles["ankle_right"] = _angle(rk, ra, g("right_foot_index"))
        angles["ankle_is_approx"] = 0.0
    else:                         # YOLO: xấp xỉ, độ tin cậy thấp
        angles["ankle_left"] = _angle_to_vertical(la - lk)
        angles["ankle_right"] = _angle_to_vertical(ra - rk)
        angles["ankle_is_approx"] = 1.0

    ratios = {
        "shoulder_width": _dist(ls, rs) / scale, "hip_width": _dist(lh, rh) / scale,
        "thigh_left": _dist(lh, lk) / scale, "thigh_right": _dist(rh, rk) / scale,
        "shank_left": _dist(lk, la) / scale, "shank_right": _dist(rk, ra) / scale,
        "wrist_width": _dist(lw, rw) / scale,
        # ~0: tay ngang mức mắt cá (chống đất); dương lớn: tay nhấc cao
        "hand_ground_clearance": ((la[1] + ra[1]) / 2.0 - (lw[1] + rw[1]) / 2.0) / scale,
        # dương: hông cao hơn vai (y ảnh hướng xuống)
        "hip_above_shoulder": (mid_s[1] - mid_h[1]) / scale,
    }

    front = determine_front_leg(P, idx)
    angles.update(remap_front_back(angles, front))
    ratios.update(remap_front_back(ratios, front))

    main_ids = [idx[n] for n in MAIN_JOINTS]
    main_conf = conf[main_ids]
    finite_conf = main_conf[np.isfinite(main_conf)]
    required_ok = all(np.all(np.isfinite(g(n))) for n in REQUIRED_JOINTS)
    is_valid = bool(required_ok and np.isfinite(body_scale) and body_scale >= MIN_BODY_SCALE_PX)
    reason = ""
    if not is_valid:
        reason = "thiếu keypoint bắt buộc" if not required_ok else f"body_scale={body_scale:.1f}px < {MIN_BODY_SCALE_PX}"

    kl, kr = angles["knee_left"], angles["knee_right"]
    out: Dict[str, float] = {
        "body_scale_px": body_scale,
        "com_x_px": float(((mid_s + mid_h) / 2.0)[0]), "com_y_px": float(((mid_s + mid_h) / 2.0)[1]),
        "trunk_dx_px": float(trunk_vec[0]), "trunk_dy_px": float(trunk_vec[1]),
        "head_dx_px": float((nose - mid_h)[0]),
        "hip_height_rel_ankle": float(((la[1] + ra[1]) / 2.0 - mid_h[1]) / scale),
        "sym_index_knee": (float(abs(kl - kr) / ((kl + kr) / 2.0))
                           if np.isfinite(kl) and np.isfinite(kr) and (kl + kr) > 1e-6 else float("nan")),
        "front_leg": front or "",
        "visible_ratio": float(np.mean(main_conf >= thr)),
        "conf_mean": float(np.mean(finite_conf)) if len(finite_conf) else float("nan"),
        "is_valid": is_valid, "invalid_reason": reason,
    }
    out.update({f"angle_{k}": v for k, v in angles.items()})
    out.update({f"ratio_{k}": v for k, v in ratios.items()})
    return out


# =========================================================
#            LÀM MƯỢT + ĐỘNG HỌC (HẬU XỬ LÝ THEO VIDEO)
# =========================================================
def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    n, isn, i = len(x), np.isnan(x), 0
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1], x[j]])
            i = j
        else:
            i += 1
    return x


def smooth_nan_aware(x: np.ndarray, window: int, poly: int = 2,
                     max_gap: int = MAX_INTERP_GAP_FRAMES) -> np.ndarray:
    """Nội suy khe hở ngắn rồi Savitzky-Golay trên từng đoạn liên tục; khe hở dài giữ NaN."""
    y = _fill_short_gaps(np.asarray(x, dtype=np.float64), max_gap)
    out = np.full_like(y, np.nan)
    finite, n, i = np.isfinite(y), len(y), 0
    while i < n:
        if finite[i]:
            j = i
            while j < n and finite[j]:
                j += 1
            seg = y[i:j]
            w = min(window, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            out[i:j] = savgol_filter(seg, w, poly, mode="interp") if w >= max(3, poly + 2) else seg
            i = j
        else:
            i += 1
    return out


def postprocess_metrics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    n = len(df)
    win = max(5, int(round(SMOOTH_WINDOW_S * fps)) | 1)
    dt = 1.0 / fps

    angle_cols = [c for c in df.columns if c.startswith("angle_") and c != "angle_ankle_is_approx"]
    sm = {f"{c}_sm": smooth_nan_aware(df[c].to_numpy(float), win) for c in angle_cols}

    # Động học COM chuẩn hóa theo body_scale (bất biến khoảng cách camera)
    cx = smooth_nan_aware(df["com_x_px"].to_numpy(float), win)
    cy = smooth_nan_aware(df["com_y_px"].to_numpy(float), win)
    bs = smooth_nan_aware(df["body_scale_px"].to_numpy(float), win)
    if n >= 3:
        vel_px = np.hypot(np.gradient(cx) / dt, np.gradient(cy) / dt)
        vel = vel_px / np.where(bs > 1e-6, bs, np.nan)
        acc = np.abs(np.gradient(vel) / dt)
    else:
        vel = acc = np.full(n, np.nan)
    sm["com_vel_norm"], sm["com_acc_norm"] = vel, acc

    # Góc thân CÓ DẤU: dương = nghiêng về PHÍA MẶT HƯỚNG (hướng suy ra từ vị trí mũi so với hông)
    head_dx = df["head_dx_px"].to_numpy(float)
    facing = float(np.sign(np.nanmedian(head_dx))) if np.any(np.isfinite(head_dx)) else 1.0
    facing = facing if facing != 0 else 1.0
    sm["facing_dir"] = np.full(n, facing)
    sm["trunk_lean_signed_deg"] = np.degrees(np.arctan2(
        df["trunk_dx_px"].to_numpy(float) * facing, -df["trunk_dy_px"].to_numpy(float)))

    return pd.concat([df, pd.DataFrame(sm, index=df.index)], axis=1)


# =========================================================
#                  TỔNG HỢP CẤP VIDEO (D)
# =========================================================
def _longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
    best, cur_start, best_range = 0, None, (-1, -1)
    for i, v in enumerate(list(mask) + [False]):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best:
                best, best_range = i - cur_start, (cur_start, i - 1)
            cur_start = None
    return best_range


def estimate_cadence(df: pd.DataFrame, fps: float) -> Tuple[float, int]:
    """Số bước/phút từ đỉnh (duỗi tối đa) của góc gối trái+phải đã làm mượt."""
    times, ts = [], df["timestamp"].to_numpy(float)
    for col in ("angle_knee_left_sm", "angle_knee_right_sm"):
        s = df[col].to_numpy(float)
        ok = np.isfinite(s)
        if ok.sum() < 5:
            continue
        filled = np.where(ok, s, np.nanmedian(s))
        pk, _ = find_peaks(filled, prominence=PEAK_PROMINENCE_DEG,
                           distance=max(1, int(PEAK_MIN_DISTANCE_S * fps)))
        times.extend(ts[pk[ok[pk]]])
    times = np.sort(np.asarray(times))
    if len(times) < 3 or times[-1] - times[0] <= 0:
        return float("nan"), int(len(times))
    return float(60.0 * (len(times) - 1) / (times[-1] - times[0])), int(len(times))


def build_summary(df: pd.DataFrame, meta: dict, model: str, proc_time_s: float) -> Dict:
    valid = df["is_valid"].astype(bool).to_numpy()
    detected = df["detected"].astype(bool).to_numpy()
    n = len(df)
    fps = meta["fps"]

    row: Dict = dict(meta)
    row.update({
        "model": model, "n_frames_analyzed": n,
        "n_frames_detected": int(detected.sum()), "pct_detected": float(detected.mean()) if n else float("nan"),
        "n_frames_valid": int(valid.sum()), "pct_valid": float(valid.mean()) if n else float("nan"),
        "longest_undetected_gap_s": (lambda r: (r[1] - r[0] + 1) / fps if r[0] >= 0 else 0.0)(
            _longest_true_run(~detected)),
        "n_persons_max": int(df["n_persons"].max()) if "n_persons" in df else 1,
        "processing_time_s": proc_time_s,
    })

    stat_cols = [c for c in df.columns
                 if c.startswith(("angle_", "ratio_")) and not c.endswith("_sm") and c != "angle_ankle_is_approx"]
    stat_cols += ["sym_index_knee", "trunk_lean_signed_deg", "hip_height_rel_ankle",
                  "com_vel_norm", "com_acc_norm", "conf_mean", "visible_ratio"]
    vdf = df[valid]
    for col in stat_cols:
        s = vdf[col].dropna()
        row[f"{col}_mean"] = float(s.mean()) if len(s) else float("nan")
        row[f"{col}_std"] = float(s.std()) if len(s) > 1 else float("nan")
        row[f"{col}_min"] = float(s.min()) if len(s) else float("nan")
        row[f"{col}_max"] = float(s.max()) if len(s) else float("nan")
        if col.startswith("angle_") and len(s) > 1:
            row[f"{col}_rom"] = float(np.percentile(s, 95) - np.percentile(s, 5))

    cadence, n_steps = estimate_cadence(df, fps)
    row["cadence_spm"], row["n_step_peaks"] = cadence, n_steps

    # --- Đoạn giữ tư thế tĩnh dài nhất (nguyên liệu cho khoảng chuẩn pha "static") ---
    still = valid & np.isfinite(df["com_vel_norm"].to_numpy(float)) & (df["com_vel_norm"].to_numpy(float) <= STATIC_VEL_MAX)
    s0, s1 = _longest_true_run(still)
    dur = (s1 - s0 + 1) / fps if s0 >= 0 else 0.0
    row["static_duration_s"] = dur
    if s0 >= 0 and dur >= STATIC_MIN_S:
        row["static_start_s"], row["static_end_s"] = float(df["timestamp"].iloc[s0]), float(df["timestamp"].iloc[s1])
        win = df.iloc[s0:s1 + 1]
        for col in KEY_FEATURES:
            row[f"static_{col}_mean"] = float(win[col].mean())
            row[f"static_{col}_std"] = float(win[col].std()) if len(win) > 1 else float("nan")

    # --- Thời điểm bộc phát lực (chỉ có ý nghĩa với video xuất phát) ---
    acc = df["com_acc_norm"].to_numpy(float)
    cand = np.where(valid & np.isfinite(acc), acc, -np.inf)
    if np.isfinite(cand.max()):
        lp = int(np.argmax(cand))
        mean_acc = float(np.nanmean(acc[valid])) if valid.any() else float("nan")
        row.update({
            "launch_frame": int(df["frame"].iloc[lp]), "launch_time_s": float(df["timestamp"].iloc[lp]),
            "launch_peak_acc": float(acc[lp]),
            "launch_prominence_ratio": float(acc[lp] / mean_acc) if mean_acc and mean_acc > 1e-9 else float("nan"),
        })
        for col in KEY_FEATURES:
            row[f"launch_{col}"] = float(df[col].iloc[lp])

    # --- Cờ chất lượng ---
    flags = []
    if row["pct_detected"] < QC_MIN_PCT_DETECTED:
        flags.append("LOW_DETECTION")
    if row["pct_valid"] < QC_MIN_PCT_VALID:
        flags.append("LOW_VALID_FRAMES")
    if row["n_persons_max"] > 1:
        flags.append("MULTI_PERSON")
    if meta.get("fps_assumed"):
        flags.append("FPS_ASSUMED")
    if meta.get("vfr_suspect"):
        flags.append("VFR_SUSPECT")
    if model in ("yolo", "rtmpose"):
        flags.append("ANKLE_APPROX")
    row["qc_flags"] = ";".join(flags)
    return row


# =========================================================
#                         VIDEO I/O
# =========================================================
def is_valid_frame(frame) -> bool:
    return (frame is not None and hasattr(frame, "size") and frame.size > 0
            and frame.ndim == 3 and frame.shape[0] >= 2 and frame.shape[1] >= 2)


def open_video(path: str, video_id: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps_assumed = not (fps_raw and fps_raw > 0 and np.isfinite(fps_raw))
    fps = 30.0 if fps_assumed else float(fps_raw)   # GIỮ fps thực (29.97...), không làm tròn
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError(f"Kích thước video không hợp lệ: {w}x{h}")
    meta = {
        "video_id": video_id, "path": path, "width": w, "height": h, "fps": fps, "fps_assumed": fps_assumed,
        "frames_declared": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "file_size_mb": round(os.path.getsize(path) / 1e6, 3),
        "rotation_meta": float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0),
    }
    return cap, meta


def find_videos(cfg: Config) -> List[str]:
    root = Path(cfg.input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Không phải thư mục: {root}")
    vids = [str(p) for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in cfg.video_exts
            and cfg.keyword.lower() in p.name.lower()]
    stems = [Path(v).stem for v in vids]
    dups = {s for s in stems if stems.count(s) > 1}
    if dups:
        raise ValueError(f"Trùng tên (khác đuôi) sẽ gây trùng video_id: {sorted(dups)}")
    return vids


# =========================================================
#                      BACKEND POSE
# =========================================================
@dataclass
class PoseResult:
    kps: np.ndarray                          # (K,3) x_px, y_px, conf
    bbox: np.ndarray                         # (4,) x1,y1,x2,y2
    n_persons: int = 1
    bbox_conf: float = float("nan")
    z: Optional[np.ndarray] = None           # (K,) độ sâu tương đối (MediaPipe)
    world: Optional[np.ndarray] = None       # (K,3) mét, gốc tại tâm hông (MediaPipe)
    native: object = None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class YoloBackend:
    name = "yolo"
    names = COCO17_NAMES

    def __init__(self, cfg: Config):
        from ultralytics import YOLO
        self.model = YOLO(cfg.yolo_weights)
        self.imgsz = cfg.yolo_imgsz
        self.prev_box: Optional[np.ndarray] = None

    def start_video(self, w: int, h: int) -> None:
        self.prev_box = None

    def close(self) -> None:
        pass

    def infer(self, frame) -> Optional[PoseResult]:
        res = self.model(frame, verbose=False, imgsz=self.imgsz)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        bconf = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        pick = int(np.argmax(areas))                       # lần đầu: người lớn nhất khung hình
        if self.prev_box is not None:                      # các frame sau: bám theo IoU để không nhảy người
            ious = [_iou(self.prev_box, b) for b in boxes]
            if max(ious) >= 0.1:
                pick = int(np.argmax(ious))
        self.prev_box = boxes[pick]
        return PoseResult(kps=kps[pick].astype(np.float64), bbox=boxes[pick].astype(np.float64),
                          n_persons=len(boxes), bbox_conf=float(bconf[pick]))

    @staticmethod
    def draw(frame, r: PoseResult) -> None:
        k = r.kps
        for p1, p2 in COCO17_SKELETON:
            if k[p1][2] > 0.5 and k[p2][2] > 0.5:
                cv2.line(frame, (int(k[p1][0]), int(k[p1][1])), (int(k[p2][0]), int(k[p2][1])), (0, 255, 0), 2)
        for x, y, c in k:
            if c > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


class MediaPipeBackend:
    name = "mediapipe"
    names = POSE33_NAMES

    def __init__(self, cfg: Config):
        import mediapipe as mp
        self.mp, self.cfg, self.pose = mp, cfg, None
        self.w = self.h = 0

    def start_video(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.close()
        # Tạo mới cho mỗi video để reset trạng thái tracking theo thời gian
        self.pose = self.mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=self.cfg.mp_model_complexity,
            enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def infer(self, frame) -> Optional[PoseResult]:
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        kps = np.array([[lm.x * self.w, lm.y * self.h, lm.visibility] for lm in lms], dtype=np.float64)
        z = np.array([lm.z * self.w for lm in lms], dtype=np.float64)
        world = (np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks.landmark], dtype=np.float64)
                 if res.pose_world_landmarks else None)
        vis = kps[kps[:, 2] >= 0.5]
        bbox = (np.array([vis[:, 0].min(), vis[:, 1].min(), vis[:, 0].max(), vis[:, 1].max()])
                if len(vis) else np.full(4, np.nan))
        return PoseResult(kps=kps, bbox=bbox, n_persons=1, z=z, world=world, native=res.pose_landmarks)

    def draw(self, frame, r: PoseResult) -> None:
        sol = self.mp.solutions
        sol.drawing_utils.draw_landmarks(
            frame, r.native, sol.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=sol.drawing_styles.get_default_pose_landmarks_style())


class RTMPoseBackend:
    """Wrapper RTMPoseONNXModel → PoseResult cho extract_pose_dataset.py.

    Dùng chung core/stage2_pose RTMPoseONNXModel (đã tích hợp YOLO/ByteTrack).
    """
    name = "rtmpose"
    names = COCO17_NAMES

    def __init__(self, cfg: Config):
        from stage2_pose.pose_extractor import RTMPoseONNXModel
        self.model = RTMPoseONNXModel(
            yolo_weights=cfg.yolo_weights,
            yolo_conf=0.25,
            yolo_imgsz=cfg.yolo_imgsz,
            yolo_device=cfg.yolo_device,
            tracker_yaml="bytetrack.yaml",
            rtmpose_onnx=cfg.rtmpose_onnx,
            rtmpose_device=cfg.rtmpose_device,
            rtmpose_bbox_pad=0.10,
        )
        self.conf_threshold = cfg.conf_threshold

    def start_video(self, w: int, h: int) -> None:
        # Reset tracker mỗi video
        if hasattr(self.model, "_yolo"):
            try:
                self.model._yolo.predictor = None
            except Exception:
                pass

    def close(self) -> None:
        self.model.close()

    def infer(self, frame) -> Optional[PoseResult]:
        """Chọn 1 người chính (người to nhất / track ổn định nhất)."""
        detections = self.model.infer(frame)
        if not detections:
            return None

        # Chọn người lớn nhất (diện tích bbox lớn nhất)
        areas = []
        for d in detections:
            if d.bbox is None:
                areas.append(0.0)
            else:
                x1, y1, x2, y2 = d.bbox
                areas.append((x2 - x1) * (y2 - y1))
        pick = int(np.argmax(areas))
        d = detections[pick]

        kps = np.concatenate([d.xy, d.conf[:, None]], axis=1).astype(np.float64)
        return PoseResult(
            kps=kps,
            bbox=d.bbox if d.bbox is not None else np.full(4, np.nan),
            n_persons=len(detections),
            bbox_conf=float(d.det_score),
        )

    @staticmethod
    def draw(frame, r: PoseResult) -> None:
        """Vẽ skeleton COCO17."""
        k = r.kps
        for p1, p2 in COCO17_SKELETON:
            if k[p1][2] > 0.5 and k[p2][2] > 0.5:
                cv2.line(frame, (int(k[p1][0]), int(k[p1][1])),
                         (int(k[p2][0]), int(k[p2][1])), (0, 255, 0), 2)
        for x, y, c in k:
            if c > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def make_backend(model: str, cfg: Config):
    if model == "rtmpose":
        return RTMPoseBackend(cfg)
    if model == "yolo":
        return YoloBackend(cfg)
    return MediaPipeBackend(cfg)



# =========================================================
#              XỬ LÝ 1 VIDEO x 1 MODEL -> CÁC BẢNG
# =========================================================
def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.4f")
    os.replace(tmp, path)


def process_video(backend, video_path: str, video_id: str, cfg: Config, parts_dir: Path,
                  annotated_dir: Path) -> Dict:
    names = backend.names
    idx = {n: i for i, n in enumerate(names)}
    K = len(names)
    t0 = time.time()

    cap, meta = open_video(video_path, video_id)
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    backend.start_video(w, h)

    writer = None
    if cfg.write_video:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(annotated_dir / f"{video_id}_{backend.name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            writer = None
            print("      [WARN] Không tạo được VideoWriter, bỏ qua video minh họa")

    kps_all, z_all, world_all, bbox_all = [], [], [], []
    npers, bconf, pts_ms, metric_rows = [], [], [], []
    n_bad = n_err = 0
    frame_idx = 0

    try:
        while True:
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = cap.read()
            if not ok:
                break
            result: Optional[PoseResult] = None
            if is_valid_frame(frame):
                try:
                    result = backend.infer(frame)
                except Exception as e:
                    n_err += 1
                    if n_err == 1:
                        print(f"      [WARN-{backend.name}] frame {frame_idx}: {type(e).__name__}: {e}")
            else:
                n_bad += 1
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            kps = result.kps if result is not None else np.full((K, 3), np.nan)
            kps_all.append(kps)
            z_all.append(result.z if result is not None and result.z is not None else np.full(K, np.nan))
            world_all.append(result.world if result is not None and result.world is not None
                             else np.full((K, 3), np.nan))
            bbox_all.append(result.bbox if result is not None else np.full(4, np.nan))
            npers.append(result.n_persons if result is not None else 0)
            bconf.append(result.bbox_conf if result is not None else np.nan)
            pts_ms.append(pts)
            m = compute_frame_metrics(kps, idx, cfg.conf_threshold)
            m["detected"] = result is not None
            metric_rows.append(m)

            if writer is not None:
                if result is not None:
                    backend.draw(frame, result)
                cv2.putText(frame, f"{backend.name} | {frame_idx}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        backend.close()

    T = frame_idx
    if T == 0:
        raise RuntimeError("Video không có frame nào đọc được")

    # fps ước lượng từ PTS để phát hiện video VFR (fps thay đổi)
    pts_arr = np.asarray(pts_ms, dtype=float)
    fps_pts = (T - 1) / ((pts_arr[-1] - pts_arr[0]) / 1000.0) if T > 2 and pts_arr[-1] > pts_arr[0] else float("nan")
    meta.update({
        "frames_read": T, "duration_s": T / fps, "fps_pts_est": fps_pts, "n_bad_frames": n_bad,
        "n_infer_errors": n_err, "model_conf_threshold": cfg.conf_threshold,
        "vfr_suspect": bool(np.isfinite(fps_pts) and abs(fps_pts - fps) / fps > QC_FPS_MISMATCH),
    })

    frames = np.arange(T)
    ts = frames / fps
    kps_arr = np.stack(kps_all)            # (T,K,3)
    z_arr, world_arr, bbox_arr = np.stack(z_all), np.stack(world_all), np.stack(bbox_all)

    # ---- [A] long ----
    df_A = pd.DataFrame({
        "video_id": video_id, "model": backend.name,
        "frame": np.repeat(frames, K), "timestamp": np.repeat(ts, K),
        "keypoint": np.tile(names, T),
        "x": kps_arr[:, :, 0].ravel(), "y": kps_arr[:, :, 1].ravel(), "conf": kps_arr[:, :, 2].ravel(),
        "z": z_arr.ravel(),
        "world_x": world_arr[:, :, 0].ravel(), "world_y": world_arr[:, :, 1].ravel(),
        "world_z": world_arr[:, :, 2].ravel(),
    })
    _atomic_csv(df_A, parts_dir / "keypoints_long.csv")

    # ---- [B] wide (mỗi model 1 file riêng) ----
    if cfg.export_wide:
        wide = kps_arr.reshape(T, K * 3)
        cols = [f"{n}_{c}" for n in names for c in ("x", "y", "conf")]
        df_B = pd.concat([pd.DataFrame({"video_id": video_id, "model": backend.name,
                                        "frame": frames, "timestamp": ts}),
                          pd.DataFrame(wide, columns=cols)], axis=1)
        _atomic_csv(df_B, parts_dir / "keypoints_wide.csv")

    # ---- [C] frame-level metrics ----
    df_C = pd.DataFrame(metric_rows)
    df_C.insert(0, "timestamp", ts)
    df_C.insert(0, "frame", frames)
    df_C.insert(0, "model", backend.name)
    df_C.insert(0, "video_id", video_id)
    df_C["pts_ms"] = pts_arr
    df_C["n_persons"] = npers
    df_C["bbox_conf"] = bconf
    for j, c in enumerate(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        df_C[c] = bbox_arr[:, j]
    df_C = postprocess_metrics(df_C, fps)
    _atomic_csv(df_C, parts_dir / "metrics_frame.csv")

    # ---- [D] summary: GHI CUỐI CÙNG, đóng vai trò dấu hiệu "đã xong" để resume ----
    summary = build_summary(df_C, meta, backend.name, time.time() - t0)
    _atomic_csv(pd.DataFrame([summary]), parts_dir / "summary.csv")

    print(f"   [{backend.name:<9}] {T} frames | detect {summary['pct_detected']:.0%} | "
          f"valid {summary['pct_valid']:.0%} | {T / max(time.time() - t0, 1e-3):.1f} FPS"
          + (f" | QC: {summary['qc_flags']}" if summary["qc_flags"] else ""))
    return summary


# =========================================================
#                    GOM CÁC PHẦN -> FILE CHUNG
# =========================================================
def concat_csv(parts: Sequence[Path], dest: Path) -> int:
    """Nối CSV theo kiểu streaming (không nạp hết vào RAM), kiểm tra header khớp."""
    if not parts:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    # Lượt 1: chỉ đọc header của mọi part -> cột chuẩn = union, giữ thứ tự xuất hiện đầu tiên
    headers: Dict[Path, List[str]] = {}
    union: List[str] = []
    for p in parts:
        with open(p, "r", encoding="utf-8", newline="") as f:
            cols = f.readline().rstrip("\r\n").split(",")
        headers[p] = cols
        for c in cols:
            if c not in union:
                union.append(c)

    # Lượt 2: part nào đã đúng header chuẩn -> copy thô (nhanh); part lệch -> căn lại theo cột
    # chuẩn bằng pandas (đọc từng khối, không nạp hết vào RAM). Cột thiếu sẽ để trống.
    n_realigned = 0
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        out.write(",".join(union) + "\n")
        for p in parts:
            if headers[p] == union:
                with open(p, "r", encoding="utf-8", newline="") as f:
                    f.readline()
                    shutil.copyfileobj(f, out)
            else:
                n_realigned += 1
                for chunk in pd.read_csv(p, chunksize=50_000, low_memory=False):
                    chunk.reindex(columns=union).to_csv(
                        out, header=False, index=False, float_format="%.4f")
    os.replace(tmp, dest)
    if n_realigned:
        print(f"   [aggregate] {dest.name}: đã căn lại cột cho {n_realigned}/{len(parts)} file (header lệch thứ tự/tập cột)")
    return len(parts)


def aggregate(parts_root: Path, agg_dir: Path, cfg: Config) -> None:
    agg_dir.mkdir(parents=True, exist_ok=True)
    pick = lambda fname, model=None: sorted(
        p for p in parts_root.glob(f"*/*/{fname}") if model is None or p.parent.name == model)
    concat_csv(pick("keypoints_long.csv"), agg_dir / "keypoints_long.csv")
    concat_csv(pick("metrics_frame.csv"), agg_dir / "metrics_frame_level.csv")
    if cfg.export_wide:
        for model in cfg.models:
            concat_csv(pick("keypoints_wide.csv", model), agg_dir / f"keypoints_wide_{model}.csv")
    sums = pick("summary.csv")
    if sums:
        pd.concat([pd.read_csv(p) for p in sums], ignore_index=True, sort=False).to_csv(
            agg_dir / "summary_video_level.csv", index=False, float_format="%.4f")


def lib_versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "opencv": cv2.__version__}
    for pkg in ("ultralytics", "onnxruntime", "mediapipe", "scipy"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


# =========================================================
#                           MAIN
# =========================================================
def parse_args() -> Config:
    d = Config()
    ap = argparse.ArgumentParser(description="Trích xuất pose + metric từ video, gom thành file chung")
    ap.add_argument("--input", default=d.input_dir)
    ap.add_argument("--output", default=d.output_dir)
    ap.add_argument("--keyword", default=d.keyword, help="chỉ xử lý video có từ khóa này trong tên ('' = tất cả)")
    ap.add_argument("--models", nargs="+", default=list(d.models), choices=list(MODEL_NAMES))
    ap.add_argument("--yolo-weights", default=d.yolo_weights)
    ap.add_argument("--imgsz", type=int, default=d.yolo_imgsz)
    ap.add_argument("--rtmpose-onnx", default=d.rtmpose_onnx)     # THÊM
    ap.add_argument("--rtmpose-device", default=d.rtmpose_device) # THÊM
    ap.add_argument("--yolo-device", default=d.yolo_device)       # THÊM
    ap.add_argument("--conf", type=float, default=d.conf_threshold)
    ap.add_argument("--no-video", action="store_true", help="không xuất video minh họa (nhanh hơn)")
    ap.add_argument("--no-wide", action="store_true", help="không xuất dạng wide [B] (dư thừa so với [A])")
    ap.add_argument("--force", action="store_true", help="xử lý lại cả video đã xong")
    ap.add_argument("--clean-parts", action="store_true", help="xóa _parts sau khi gom xong")
    a = ap.parse_args()
    return Config( input_dir=a.input, output_dir=a.output, keyword=a.keyword,
        models=tuple(a.models), yolo_weights=a.yolo_weights, yolo_imgsz=a.imgsz,
        rtmpose_onnx=a.rtmpose_onnx, rtmpose_device=a.rtmpose_device,
        yolo_device=a.yolo_device,
        conf_threshold=a.conf, write_video=not a.no_video, export_wide=not a.no_wide,
        force=a.force, clean_parts=a.clean_parts,
)


def main() -> int:
    cfg = parse_args()
    out_root = Path(cfg.output_dir)
    parts_root, agg_dir, ann_dir = out_root / "_parts", out_root / "_aggregated", out_root / "annotated_videos"
    out_root.mkdir(parents=True, exist_ok=True)

    videos = find_videos(cfg)
    print("=" * 70)
    print(f"Input: {cfg.input_dir}\nOutput: {cfg.output_dir}\nModels: {cfg.models} | keyword='{cfg.keyword}'")
    print(f"Tìm thấy {len(videos)} video")
    print("=" * 70)
    if not videos:
        return 1

    backends = {m: make_backend(m, cfg) for m in cfg.models}   # nạp model 1 lần cho cả batch
    manifest = {"started_at": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg),
                "versions": lib_versions(), "videos": []}
    t_start = time.time()

    try:
        for i, vp in enumerate(videos, 1):
            vid = Path(vp).stem
            print(f"[{i}/{len(videos)}] {vid}")
            for model in cfg.models:
                pdir = parts_root / vid / model
                entry = {"video_id": vid, "model": model}
                if (pdir / "summary.csv").exists() and not cfg.force:
                    entry["status"] = "skipped_done"
                    print(f"   [{model:<9}] đã xong, bỏ qua")
                else:
                    try:
                        s = process_video(backends[model], vp, vid, cfg, pdir, ann_dir)
                        entry.update(status="ok", qc_flags=s["qc_flags"], pct_detected=s["pct_detected"])
                    except Exception as e:
                        entry.update(status="failed", error=f"{type(e).__name__}: {e}")
                        print(f"   [{model:<9}] LỖI: {entry['error']}")
                        traceback.print_exc(limit=2)
                manifest["videos"].append(entry)
    finally:
        aggregate(parts_root, agg_dir, cfg)
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        agg_dir.mkdir(parents=True, exist_ok=True)
        with open(agg_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    failed = [v for v in manifest["videos"] if v["status"] == "failed"]
    print("=" * 70)
    print(f"XONG trong {manifest['elapsed_min']} phút | lỗi: {len(failed)} | kết quả: {agg_dir}")
    if cfg.clean_parts and not failed:
        shutil.rmtree(parts_root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())