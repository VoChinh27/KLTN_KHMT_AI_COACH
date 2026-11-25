"""Kiểm tra chất lượng video đầu vào: FPS, frame trùng lặp, rung máy, kích thước người trong khung, nhiễu (jitter) của pose
và tác dụng của bước làm mượt. Trả về số đo + cảnh báo tiếng Việt; không sửa video."""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def _second_diff_rms(x: np.ndarray) -> float:
    d = x[2:] - 2 * x[1:-1] + x[:-2]
    d = d[~np.isnan(d).any(axis=1)] if d.ndim == 2 else d[~np.isnan(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")


def camera_shake(video_path: str, person_boxes: np.ndarray, fps: float, max_frames: int = 400, width: int = 480) -> Dict[str, float]:
    """Ước lượng rung máy bằng luồng quang học thưa trên NỀN (bỏ vùng quanh người): dịch chuyển nền từng frame, bỏ phần trôi chậm
    (trung bình trượt 1 giây) rồi lấy RMS. Đơn vị: % bề rộng khung hình."""
    cap = cv2.VideoCapture(video_path)
    prev, tx, ty, i = None, [], [], 0
    while i < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        sc = width / fr.shape[1]
        g = cv2.cvtColor(cv2.resize(fr, (width, int(fr.shape[0] * sc))), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mask = np.full(g.shape, 255, np.uint8)
            if i < len(person_boxes) and not np.isnan(person_boxes[i]).any():
                x0, y0, x1, y1 = (person_boxes[i] * sc).astype(int)
                pad = int(0.25 * max(x1 - x0, y1 - y0))
                mask[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = 0
            p0 = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
                good = st.ravel() == 1
                if good.sum() >= 12:
                    d = (p1[good] - p0[good]).reshape(-1, 2)
                    tx.append(float(np.median(d[:, 0]) / width * 100))
                    ty.append(float(np.median(d[:, 1]) / width * 100))
                else:
                    tx.append(np.nan); ty.append(np.nan)
            else:
                tx.append(np.nan); ty.append(np.nan)
        prev, i = g, i + 1
    cap.release()
    if len(tx) < 10 or np.isnan(tx).mean() > 0.5:
        return {"shake_rms_pct": float("nan"), "shake_peak_pct": float("nan"), "frames_used": len(tx)}
    px, py = np.nan_to_num(np.cumsum(np.nan_to_num(tx))), np.nan_to_num(np.cumsum(np.nan_to_num(ty)))
    k = max(3, int(round(fps)))
    ker = np.ones(k) / k
    hp = lambda a: a - np.convolve(np.pad(a, (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")
    r = np.hypot(hp(px), hp(py))
    return {"shake_rms_pct": round(float(np.sqrt(np.mean(r ** 2))), 3), "shake_peak_pct": round(float(np.max(r)), 3), "frames_used": len(tx)}


def video_qc(video_path: str, ctx, cfg: dict) -> dict:
    q = cfg["video_qc"]
    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_tot, dup, prev = 0, 0, None
    while n_tot < 600:                                           # frame trùng lặp: video bị nội suy tăng FPS / đứt nhịp
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (96, 54))
        if prev is not None and float(np.mean(np.abs(g.astype(int) - prev.astype(int)))) < 0.05:
            dup += 1
        prev, n_tot = g, n_tot + 1
    cap.release()
    xy = ctx.pose.xy
    pts = np.stack([np.stack([xy[j][:, 0], xy[j][:, 1]], 1) for j in xy], 0)          # (J, N, 2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                           # frame không dò được người -> NaN là bình thường
        boxes = np.c_[np.nanmin(pts[:, :, 0], 0), np.nanmin(pts[:, :, 1], 0), np.nanmax(pts[:, :, 0], 0), np.nanmax(pts[:, :, 1], 0)]
    height_frac = float(np.nanmedian(boxes[:, 3] - boxes[:, 1]) / h) if h else float("nan")
    torso = float(np.nanmedian(np.linalg.norm((xy["left_shoulder"] + xy["right_shoulder"]) / 2 - (xy["left_hip"] + xy["right_hip"]) / 2, axis=1)))
    # nhiễu pose: RMS đạo hàm bậc 2 vị trí 2D (theo chiều dài thân) của các khớp, thô và sau làm mượt
    raw = np.nanmean([_second_diff_rms(xy[j]) for j in xy]) / max(torso, 1e-6)
    cl = np.nanmean([_second_diff_rms(ctx.clean.xy[j]) for j in ctx.clean.xy]) / max(torso, 1e-6)
    shake = camera_shake(video_path, boxes, fps or ctx.pose.fps)
    out = {"fps": round(fps, 2), "width": w, "height": h, "frames": int(ctx.pose.n), "duplicate_frame_pct": round(100 * dup / max(n_tot - 1, 1), 1),
           "subject_height_frac": round(height_frac, 2), "subject_height_px": int(height_frac * h),
           "jitter_raw_pct_torso": round(100 * float(raw), 2), "jitter_clean_pct_torso": round(100 * float(cl), 2),
           "jitter_reduction_pct": round(100 * (1 - float(cl) / float(raw)), 1) if raw else float("nan"),
           "smooth_window_frames": __import__("badminton3d.cleaning", fromlist=["x"]).smooth_window_frames(cfg["quality"], ctx.pose.fps), **shake}
    warn: List[str] = []
    if out["fps"] and out["fps"] < q["fps_min"]:
        warn.append(f"Video chỉ {out['fps']:.0f} FPS (< {q['fps_min']}): cú vung nhanh bị lấy mẫu thưa, nên quay từ 30 FPS trở lên (khuyến nghị 60 FPS nếu máy hỗ trợ).")
    if out["duplicate_frame_pct"] > q["dup_frame_warn_pct"]:
        warn.append(f"{out['duplicate_frame_pct']:.0f}% frame bị trùng lặp (video bị nội suy tăng FPS hoặc đứt nhịp): tốc độ/chuyển động đo được có thể thiếu chính xác.")
    s = out["shake_rms_pct"]
    if s == s and s >= q["shake_high_pct"]:
        warn.append(f"Máy rung mạnh (RMS nền {s:.2f}% bề rộng khung): nên đặt máy trên chân đế cố định; số đo tốc độ có thể sai lệch.")
    elif s == s and s >= q["shake_warn_pct"]:
        warn.append(f"Máy hơi rung (RMS nền {s:.2f}% bề rộng khung): nên dùng chân đế.")
    if out["subject_height_frac"] < q["subject_min_frac"]:
        warn.append(f"Người chỉ cao {out['subject_height_frac'] * 100:.0f}% chiều cao khung hình (~{out['subject_height_px']} px): khớp kém chính xác; quay gần hơn hoặc zoom để người chiếm ≥ 50% chiều cao khung.")
    out["warnings"] = warn
    return out
