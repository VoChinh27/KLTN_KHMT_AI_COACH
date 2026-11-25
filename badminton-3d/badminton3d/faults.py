"""Kiểm tra lỗi riêng của phát cầu (ngoài so góc với chuẩn): vung xả đà quá dài sau cú chạm, nhấc gót khi phát.

Mỗi kiểm tra có ngưỡng và điểm trừ trong config/badminton3d.yaml (mục `faults`), tắt/bật hoặc chỉnh ngưỡng không cần sửa mã.
Kết quả mỗi lần phát: danh sách lỗi (`faults`) + số đo (`metrics`) dù có vượt ngưỡng hay không, để người dùng tự hiệu chỉnh.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter


def _smooth(x: np.ndarray, vis: Optional[np.ndarray] = None, thr: float = 0.3) -> np.ndarray:
    x = np.asarray(x, float).copy()
    if vis is not None:
        x[vis < thr] = np.nan
    ok = ~np.isnan(x)
    if ok.sum() < 8:
        return x
    x = np.interp(np.arange(len(x)), np.where(ok)[0], x[ok])
    return savgol_filter(x, 7, 2)


def follow_through(ctx, rep, cfg: dict) -> Tuple[Optional[float], Tuple[int, int]]:
    """Độ cao lớn nhất của cổ tay (tay vợt) so với VAI trong lúc vung theo sau cú chạm (m; > 0 = cao hơn vai)."""
    fps, n = ctx.pose.fps, ctx.pose.n
    side = ctx.arm["side"]
    w, sh = ctx.clean.w[f"{side}_wrist"], ctx.clean.w[f"{side}_shoulder"]
    a, b = rep.peak, min(rep.end + int(round(cfg["faults"]["follow_through"].get("window_after_end_s", 0.3) * fps)), n - 1)
    above = (sh[:, 1] - w[:, 1])[a:b + 1]
    return (float(np.nanmax(above)) if np.isfinite(above).any() else None), (a, b)


def heel_lift(ctx, rep, cfg: dict) -> Tuple[Optional[float], Tuple[int, int], Optional[str]]:
    """Mức gót nhấc cao hơn mũi chân (so với lúc bàn chân phẳng nhất trong clip), chia cho chiều dài thân (trong ảnh), lấy chân tệ nhất,
    trong đoạn từ `lookback_s` trước khi bắt đầu vung tới sát cú chạm. None nếu pose cũ chưa có gót/mũi chân."""
    if ctx.pose.foot is None:
        return None, (0, 0), None
    c = cfg["faults"]["heel_lift"]
    fps, n = ctx.pose.fps, ctx.pose.n
    xy = ctx.clean.xy
    torso = np.nanmedian(np.linalg.norm((xy["left_shoulder"] + xy["right_shoulder"]) / 2 - (xy["left_hip"] + xy["right_hip"]) / 2, axis=1))
    a, b = max(0, rep.swing_start - int(round(c.get("lookback_s", 1.0) * fps))), min(rep.peak + int(round(0.05 * fps)), n - 1)
    best, foot = None, None
    for sd in ("left", "right"):
        h, t = ctx.pose.foot[f"{sd}_heel"], ctx.pose.foot[f"{sd}_foot_index"]
        d = (_smooth(t["xy"][:, 1], np.minimum(h["vis"], t["vis"])) - _smooth(h["xy"][:, 1], np.minimum(h["vis"], t["vis"]))) / max(torso, 1e-6)
        if not np.isfinite(d).any():
            continue
        v = float(np.nanmax(d[a:b + 1]) - np.nanpercentile(d, 5))
        if best is None or v > best:
            best, foot = v, ("trái" if sd == "left" else "phải")
    return best, (a, b), foot


def serve_faults(ctx, cfg: dict) -> Tuple[List[List[dict]], List[dict]]:
    """Trả (lỗi theo từng lần phát, số đo theo từng lần phát). Lỗi = dict dùng chung định dạng với bảng lỗi của chấm điểm."""
    fc = cfg["faults"]
    all_faults, all_metrics = [], []
    for rep in ctx.serve_seg.reps:
        faults, metrics = [], {}
        ft = fc["follow_through"]
        if ft.get("enabled", True) and ctx.arm:
            v, fr = follow_through(ctx, rep, cfg)
            metrics["follow_through_above_shoulder_m"] = None if v is None else round(v, 3)
            if v is not None and v > ft["warn_m"]:
                high = v >= ft["high_m"]
                pen = float(ft["penalty_high"] if high else ft["penalty_warn"])
                faults.append({"feature": "follow_through", "label": "Vung theo sau cú chạm (cổ tay so với vai)", "kind": "LỖI HIỆU QUẢ", "severity": "nặng" if high else "nhẹ",
                               "unit": "m", "value": v, "center": 0.0, "tol": float(ft["warn_m"]), "dev": v, "penalty": pen, "frames": [fr[0], fr[1]],
                               "advice": (f"Sau khi chạm cầu tay vợt vung lên {'quá đầu' if high else 'cao hơn vai'} ({v * 100:.0f} cm trên vai): "
                                          "kết thúc động tác sớm, để vợt dừng ngang/dưới ngực để cầu không bay ra ngoài.")})
        hl = fc["heel_lift"]
        if hl.get("enabled", True):
            v, fr, foot = heel_lift(ctx, rep, cfg)
            metrics["heel_lift_ratio"] = None if v is None else round(v, 3)
            if v is not None and v > hl["threshold"]:
                faults.append({"feature": "heel_lift", "label": f"Nhấc gót chân {foot} khi phát", "kind": "PHẠM LUẬT", "severity": "nặng", "unit": "x", "value": v, "center": 0.0,
                               "tol": float(hl["threshold"]), "dev": v, "penalty": float(hl["penalty"]), "frames": [fr[0], fr[1]],
                               "advice": f"Gót chân {foot} nhấc lên trong lúc phát: giữ hai bàn chân đứng yên, chạm sàn cho tới khi cầu rời vợt."})
        all_faults.append(faults)
        all_metrics.append(metrics)
    return all_faults, all_metrics


def apply_faults(sc: dict, faults: List[List[dict]], metrics: List[dict], cfg: dict) -> dict:
    """Trừ điểm từng lần phát theo lỗi, thêm lỗi vào bảng lỗi (xếp theo điểm mất), tính lại điểm cả video."""
    scored = [u for u in sc["units"] if u["score"] is not None]
    for u, fl, m in zip(sc["units"], faults, metrics):
        u["faults"], u["fault_metrics"] = fl, m
        u["penalty"] = float(sum(f["penalty"] for f in fl))
        if u["score"] is not None and u["penalty"] > 0:
            u["score_before_faults"] = u["score"]
            u["score"] = max(0.0, u["score"] - u["penalty"])
    if scored:
        sc["overall"] = float(np.mean([u["score"] for u in scored]))
        from badminton3d.scoring import grade
        sc["grade"] = grade(sc["overall"], cfg)
    n = max(len(scored), 1)
    extra: Dict[str, dict] = {}
    for u in sc["units"]:
        if u["score"] is None:
            continue
        for f in u["faults"]:
            e = extra.setdefault(f["feature"], {"where": "Sau cú chạm" if f["feature"] == "follow_through" else "Trước cú chạm", "feature": f["feature"], "label": f["label"], "unit": f["unit"],
                                                "value": 0.0, "center": f["center"], "tol": f["tol"], "dev": 0.0, "direction": "cao hơn chuẩn", "score": 0.0, "lost_points": 0.0,
                                                "potential_gain": 0.0, "advice": f["advice"], "n_units": 0, "severity": f["severity"], "kind": f["kind"]})
            e["value"] = max(e["value"], f["value"])
            e["dev"] = e["value"]
            e["lost_points"] += f["penalty"] / n
            e["potential_gain"] += f["penalty"] / n
            e["n_units"] += 1
    sc["errors"] = sorted(sc["errors"] + list(extra.values()), key=lambda e: -e["lost_points"])
    return sc
