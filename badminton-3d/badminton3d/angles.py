"""Góc 3D (MediaPipe world, mét) + Valid THEO TỪNG GÓC + xử lý mất khớp:
   (1) độ tin cậy = visibility nhỏ nhất của các khớp tạo nên góc;  (2) nội suy khoảng mất ngắn (ở cleaning.py);
   (3) bên còn lại thay bên bị mất (ready, đối xứng);  (4) cánh tay camera thấy rõ nhất (serve).
Quy ước trục MediaPipe world: x sang phải, y XUỐNG, z hướng về phía camera; gốc = giữa hai hông."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from badminton3d.cleaning import CleanPose

Angle = Tuple[np.ndarray, np.ndarray]          # (giá trị (N,), độ tin cậy (N,)); giá trị NaN và độ tin cậy 0 nếu không dùng được


def _combine_rel(clean: CleanPose, joints: List[str]) -> np.ndarray:
    return np.min(np.vstack([clean.rel[j] if True else 0 for j in joints]), axis=0)


def _valid_all(clean: CleanPose, joints: List[str]) -> np.ndarray:
    return np.all(np.vstack([clean.valid[j] for j in joints]), axis=0)


def angle3(clean: CleanPose, a: str, b: str, c: str) -> Angle:
    """Góc 3D tại khớp b giữa hai đoạn b->a và b->c (độ)."""
    ok = _valid_all(clean, [a, b, c])
    v1, v2 = clean.w[a] - clean.w[b], clean.w[c] - clean.w[b]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-9)
    val = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    rel = _combine_rel(clean, [a, b, c])
    return np.where(ok, val, np.nan), np.where(ok, rel, 0.0)


def wrist_height(clean: CleanPose, side: str) -> Angle:
    """Độ cao cổ tay so với hông CÙNG BÊN (m, dương = cao hơn hông)."""
    w, h = f"{side}_wrist", f"{side}_hip"
    ok = _valid_all(clean, [w, h])
    val = clean.w[h][:, 1] - clean.w[w][:, 1]                      # y hướng xuống
    return np.where(ok, val, np.nan), np.where(ok, _combine_rel(clean, [w, h]), 0.0)


def wrist_reach(clean: CleanPose, side: str) -> Angle:
    """Khoảng cách NGANG (mặt phẳng XZ) từ hông cùng bên tới cổ tay (m) — bất biến khi người xoay quanh trục đứng."""
    w, h = f"{side}_wrist", f"{side}_hip"
    ok = _valid_all(clean, [w, h])
    d = clean.w[w] - clean.w[h]
    return np.where(ok, np.hypot(d[:, 0], d[:, 2]), np.nan), np.where(ok, _combine_rel(clean, [w, h]), 0.0)


def trunk_lean(clean: CleanPose, strict: bool = False, sub_rel: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Góc giữa trục thân (giữa hông -> giữa vai) và phương thẳng đứng (độ). strict=True: cần đủ 2 vai + 2 hông.
    strict=False: chấp nhận thiếu 1 bên (dùng bên còn lại), độ tin cậy giảm. Trả (giá trị, độ tin cậy, đã_thay_thế)."""
    def mid(names):
        pts = np.stack([clean.w[n] for n in names]); ok = np.stack([clean.valid[n] for n in names])
        rel = np.stack([clean.rel[n] for n in names])
        cnt = ok.sum(0)
        m = np.nansum(np.where(ok[..., None], pts, 0.0), axis=0) / np.maximum(cnt, 1)[:, None]
        r = np.where(cnt > 0, np.sum(np.where(ok, rel, 0.0), axis=0) / np.maximum(cnt, 1), 0.0)
        return m, cnt, r
    ms, cs, rs = mid(["left_shoulder", "right_shoulder"])
    mh, ch, rh = mid(["left_hip", "right_hip"])
    need = 2 if strict else 1
    ok = (cs >= need) & (ch >= need)
    t = ms - mh
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (-t[:, 1]) / np.maximum(np.linalg.norm(t, axis=1), 1e-9)                # vertical = -y
    val = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    subst = ok & ((cs < 2) | (ch < 2))
    rel = np.minimum(rs, rh) * np.where(subst, sub_rel, 1.0)
    return np.where(ok, val, np.nan), np.where(ok, rel, 0.0), subst


def stance_ratio(clean: CleanPose) -> Angle:
    """Khoảng cách 2 cổ chân / bề rộng 2 vai (3D). Cần đủ 4 khớp."""
    js = ["left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    ok = _valid_all(clean, js)
    da = np.linalg.norm(clean.w["left_ankle"] - clean.w["right_ankle"], axis=1)
    ds = np.linalg.norm(clean.w["left_shoulder"] - clean.w["right_shoulder"], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = da / np.maximum(ds, 1e-6)
    return np.where(ok, val, np.nan), np.where(ok, _combine_rel(clean, js), 0.0)


def bilateral(left: Angle, right: Angle, sub_rel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trung bình hai bên; thiếu một bên -> dùng bên còn lại (ready đối xứng), độ tin cậy nhân sub_rel. Trả (giá trị, tin cậy, đã_thay_thế)."""
    (vl, rl), (vr, rr) = left, right
    okl, okr = ~np.isnan(vl), ~np.isnan(vr)
    both = okl & okr
    val = np.where(both, (np.nan_to_num(vl) + np.nan_to_num(vr)) / 2, np.where(okl, vl, np.where(okr, vr, np.nan)))
    rel = np.where(both, (rl + rr) / 2, np.where(okl, rl * sub_rel, np.where(okr, rr * sub_rel, 0.0)))
    return val, rel, (okl ^ okr)


def choose_arm(clean: CleanPose, racket_hand: str = "auto", min_rel: float = 0.4, frames: Optional[np.ndarray] = None,
               both_visible: float = 0.7) -> Dict:
    """Chọn CÁNH TAY dùng để đo (serve). Điểm của mỗi tay = độ tin cậy trung bình của (vai, khuỷu, cổ tay) trong `frames`.
      - racket_hand = 'left'|'right' và tay đó thấy đủ rõ  -> dùng tay vợt đã khai báo.
      - 'auto' và CẢ HAI tay đều thấy rõ                   -> dùng tay hoạt động nhiều nhất (nhiều khả năng là tay vợt).
      - 'auto' và chỉ MỘT tay thấy rõ                      -> dùng tay camera thấy rõ nhất (tay kia bị che).
    Chọn cố định cho cả đoạn (không đổi tay giữa chừng) để so sánh cùng loại với chuẩn."""
    sel = np.ones(clean.n, bool) if frames is None else frames
    score, path = {}, {}
    for s in ("left", "right"):
        r = np.min(np.vstack([clean.rel[f"{s}_{j}"] for j in ("shoulder", "elbow", "wrist")]), axis=0)
        score[s] = float(np.mean(r[sel])) if sel.any() else 0.0
        w = clean.w[f"{s}_wrist"]
        d = np.linalg.norm(np.diff(w, axis=0), axis=1)
        ok = sel[1:] & ~np.isnan(d)
        path[s] = float(np.sum(d[ok])) if ok.any() else 0.0
    best = max(score, key=score.get)
    info = {"scores": score, "path_m": path}
    if racket_hand in ("left", "right"):
        if score[racket_hand] >= min_rel:
            return {"side": racket_hand, "mode": "racket_arm", "substituted": False, **info}
        return {"side": best, "mode": "best_visible_substitute", "substituted": True, **info}
    if min(score.values()) >= both_visible:
        return {"side": max(path, key=path.get), "mode": "most_active_arm", "substituted": False, **info}
    return {"side": best, "mode": "best_visible", "substituted": False, **info}


# ------------------------------------------------------------------------------------------------
# Tập đặc trưng (đã xử lý mất khớp) cho từng bài
# ------------------------------------------------------------------------------------------------
def ready_features(clean: CleanPose, cfg: dict) -> Dict[str, dict]:
    sub = cfg["quality"]["substitute_reliability"]
    A = lambda a, b, c: {s: angle3(clean, f"{s}_{a}", f"{s}_{b}", f"{s}_{c}") for s in ("left", "right")}
    out = {}
    for name, (a, b, c) in {"knee_mean": ("hip", "knee", "ankle"), "hip_mean": ("shoulder", "hip", "knee"),
                            "elbow_mean": ("shoulder", "elbow", "wrist"), "shoulder_mean": ("hip", "shoulder", "elbow")}.items():
        d = A(a, b, c)
        v, r, s = bilateral(d["left"], d["right"], sub)
        out[name] = {"value": v, "rel": r, "substituted": s}
    v, r, s = trunk_lean(clean, strict=False, sub_rel=sub)
    out["trunk_lean"] = {"value": v, "rel": r, "substituted": s}
    v, r = stance_ratio(clean)
    out["stance_ratio"] = {"value": v, "rel": r, "substituted": np.zeros(clean.n, bool)}
    return out


def serve_features(clean: CleanPose, cfg: dict, arm: Dict) -> Dict[str, dict]:
    sub = cfg["quality"]["substitute_reliability"]
    s = arm["side"]
    zero = np.zeros(clean.n, bool)
    out = {}
    v, r = angle3(clean, f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist")
    out["elbow_arm"] = {"value": v, "rel": r, "substituted": zero | arm["substituted"]}
    v, r = angle3(clean, f"{s}_hip", f"{s}_shoulder", f"{s}_elbow")
    out["shoulder_arm"] = {"value": v, "rel": r, "substituted": zero | arm["substituted"]}
    v, r = wrist_height(clean, s)
    out["wrist_height"] = {"value": v, "rel": r, "substituted": zero | arm["substituted"]}
    v, r = wrist_reach(clean, s)
    out["wrist_reach"] = {"value": v, "rel": r, "substituted": zero | arm["substituted"]}
    for name, (a, b, c) in {"knee_mean": ("hip", "knee", "ankle"), "hip_mean": ("shoulder", "hip", "knee")}.items():
        d = {x: angle3(clean, f"{x}_{a}", f"{x}_{b}", f"{x}_{c}") for x in ("left", "right")}
        v, r, ss = bilateral(d["left"], d["right"], sub)
        out[name] = {"value": v, "rel": r, "substituted": ss}
    v, r, ss = trunk_lean(clean, strict=False, sub_rel=sub)
    out["trunk_lean"] = {"value": v, "rel": r, "substituted": ss}
    return out


def raw_angles(clean: CleanPose, exercise: str) -> Dict[str, Angle]:
    """Các góc 'thô' từng bên, KHÔNG thay thế — dùng để đo độ phủ trước/sau (bước S1, S2)."""
    out = {}
    for s in ("left", "right"):
        out[f"knee_{s[0].upper()}"] = angle3(clean, f"{s}_hip", f"{s}_knee", f"{s}_ankle")
        out[f"hip_{s[0].upper()}"] = angle3(clean, f"{s}_shoulder", f"{s}_hip", f"{s}_knee")
        out[f"elbow_{s[0].upper()}"] = angle3(clean, f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist")
        out[f"shoulder_{s[0].upper()}"] = angle3(clean, f"{s}_hip", f"{s}_shoulder", f"{s}_elbow")
        if exercise == "serve":
            out[f"wrist_height_{s[0].upper()}"] = wrist_height(clean, s)
            out[f"wrist_reach_{s[0].upper()}"] = wrist_reach(clean, s)
    v, r, _ = trunk_lean(clean, strict=True)
    out["trunk_lean"] = (v, r)
    if exercise == "ready":
        out["stance_ratio"] = stance_ratio(clean)
    return out
