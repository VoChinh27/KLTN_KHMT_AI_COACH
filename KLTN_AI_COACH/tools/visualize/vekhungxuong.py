#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minh họa 4 giai đoạn XUẤT PHÁT THẤP — khung xương + hình nộm 2D.
ĐỒNG BỘ với sprint_start.yaml V2.4-FINAL.

CÁCH HOẠT ĐỘNG
--------------
* Script ĐỌC TRỰC TIẾP file YAML (fixed_reference_ranges, feature_specs,
  feature_weights, phase_weights). Nếu không tìm thấy YAML hoặc thiếu PyYAML
  sẽ dùng bản sao nhúng sẵn của V2.4 (FALLBACK_SPEC bên dưới).
* Tư thế dựng bằng ĐỘNG HỌC THUẬN (khai báo góc + độ dài đoạn) rồi ĐO LẠI góc
  từ chính hình vẽ -> so với khoảng chuẩn YAML -> ✓/✗.
* Dưới mỗi pha có BẢNG ĐẶC TRƯNG: giá trị đo, khoảng chuẩn, level (P/S/I),
  weight, kết quả. Đặc trưng không thể hiện được trên 1 khung hình 2D
  (wrist_width_ratio, cadence_trend, ...) được ghi rõ là "không minh họa".

THAY ĐỔI V2.4 SO VỚI V2.3
-------------------------
1. drive_phase weight = 0.00  -> KHÔNG tính vào overall (chỉ hiển thị tham khảo).
   Overall = vao_cho 0.25 + static 0.30 + sprint_start 0.25, chuẩn hóa /0.80.
2. front_leg_full_extension, rear_leg_leaves_first, arm_leg_opposition:
   PRIMARY -> SECONDARY (weight 0.5).
3. static: knee_front [60,180], trunk_thigh [10,110].
4. sprint_start: at_knee_front_angle [70,180].
5. Tolerance: knee_front 30, trunk_thigh 25.

Chạy:  python draw_sprint_start_phases.py [đường_dẫn_yaml]   (hiện + lưu PNG)
       python draw_sprint_start_phases.py --no-show          (chỉ lưu PNG)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

plt.rcParams["font.family"] = "DejaVu Sans"  # có đủ dấu tiếng Việt

# ======================================================================
# 0. SPEC (đọc từ YAML, có fallback V2.4)
# ======================================================================
DEFAULT_YAML = "config/sport_configs/sprint_start.yaml"

# feature: (min, max, level, weight)
FALLBACK_SPEC = {
    "phase_weights": {"vao_cho": 0.25, "static": 0.30, "sprint_start": 0.25, "drive_phase": 0.0},
    "features": {
        "vao_cho": {
            "knee_front_angle": (30.0, 85.0, "PRIMARY", 1.0),
            "knee_back_angle": (30.0, 95.0, "PRIMARY", 1.0),
            "wrist_width_ratio": (0.30, 2.50, "PRIMARY", 1.0),
            "elbow_left": (120.0, 180.0, "SECONDARY", 0.5),
            "elbow_right": (120.0, 180.0, "SECONDARY", 0.5),
            "trunk_thigh_angle": (30.0, 110.0, "PRIMARY", 1.0),
            "head_trunk_alignment": (75.0, 175.0, "SECONDARY", 0.5),
            "com_support_region": (None, None, "INFO", 0.0),
        },
        "static": {
            "knee_front_angle": (60.0, 180.0, "PRIMARY", 1.0),
            "knee_back_angle": (90.0, 155.0, "PRIMARY", 1.0),
            "hip_above_shoulder_norm": (0.3, 2.0, "PRIMARY", 1.0),
            "shoulder_forward_norm": (0.0, 25.0, "SECONDARY", 0.5),
            "elbow_left": (80.0, 180.0, "SECONDARY", 0.5),
            "elbow_right": (80.0, 180.0, "SECONDARY", 0.5),
            "trunk_thigh_angle": (10.0, 110.0, "PRIMARY", 1.0),
        },
        "sprint_start": {
            "at_trunk_angle_vertical": (35.0, 115.0, "PRIMARY", 1.0),
            "front_leg_full_extension": (1.0, 1.0, "SECONDARY", 0.5),
            "rear_leg_leaves_first": (1.0, 1.0, "SECONDARY", 0.5),
            "arm_leg_opposition": (1.0, 1.0, "SECONDARY", 0.5),
            "leg_ground_angle": (25.0, 65.0, "PRIMARY", 1.0),
            "at_knee_front_angle": (70.0, 180.0, "SECONDARY", 0.5),
            "first_step_timing": (0.0, 8.0, "SECONDARY", 0.5),
        },
        "drive_phase": {
            "step_length_trend": (-1.0, 1.0, "SECONDARY", 0.5),
            "cadence_trend": (-1.0, 1.0, "SECONDARY", 0.5),
            "contact_position_convergence": (-1.0, 1.0, "SECONDARY", 0.5),
            "trunk_angle_vertical_trend": (-15.0, 10.0, "SECONDARY", 0.5),
        },
    },
}


def load_spec(path):
    """Trả về (spec, nguồn). spec cùng cấu trúc FALLBACK_SPEC."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ranges = {ph: {r["feature_key"]: (float(r["min"]), float(r["max"]))
                       for r in lst}
                  for ph, lst in cfg["fixed_reference_ranges"].items()}
        feats = {}
        for ph, fs in cfg["feature_specs"].items():
            feats[ph] = {}
            for feat, meta in fs.items():
                lo, hi = ranges.get(ph, {}).get(feat, (None, None))
                w = cfg.get("feature_weights", {}).get(ph, {}).get(feat)
                if w is None:
                    w = cfg["level_weights"].get(meta["level"], 0.0)
                feats[ph][feat] = (lo, hi, meta["level"], float(w))
        spec = {"phase_weights": {k: float(v) for k, v in cfg["phase_weights"].items()},
                "features": feats}
        return spec, f"YAML: {path}"
    except Exception as exc:  # thiếu file / thiếu PyYAML / sai cấu trúc
        return FALLBACK_SPEC, f"bản nhúng V2.4 (không đọc được YAML: {type(exc).__name__})"


_args = [a for a in sys.argv[1:] if not a.startswith("--")]
SPEC, SPEC_SRC = load_spec(_args[0] if _args else DEFAULT_YAML)
print(f"[spec] dùng {SPEC_SRC}")

PHASE_TITLE = {"vao_cho": "Vào chỗ", "static": "Sẵn sàng",
               "sprint_start": "Bộc phát", "drive_phase": "Chạy lao"}
LEVEL_CH = {"PRIMARY": "P", "SECONDARY": "S", "INFO": "I"}
ROWS = {ph: {} for ph in SPEC["phase_weights"]}      # giá trị đo được từng pha


def rng(phase, feat):
    lo, hi, _, _ = SPEC["features"][phase][feat]
    return (lo, hi)


def unit_of(feat):
    if feat.endswith("_trend") or feat.endswith("_norm") or feat.endswith("_ratio"):
        return ""
    if any(k in feat for k in ("angle", "elbow", "alignment")):
        return "°"
    return ""


def in_range(phase, feat, val):
    lo, hi = rng(phase, feat)
    return lo - 1e-9 <= val <= hi + 1e-9


def rec(phase, feat, val):
    ROWS[phase][feat] = float(val)


# ======================================================================
# 1. HẰNG SỐ HÌNH HỌC / MÀU
# ======================================================================
TORSO, THIGH, SHANK = 1.00, 0.92, 0.92
UARM, FARM, FOOT = 0.66, 0.60, 0.42
HEAD_R = 0.19
ARM_LEN = UARM + FARM
GROUND_CLEAR = 0.05

SKIN, SKIN_FAR = "#F0C29B", "#D2A27C"
SHIRT, SHIRT_FAR = "#3B82F6", "#2B63BD"
SHORTS, SHORTS_FAR = "#334155", "#232E3C"
SHOE, SHOE_FAR = "#EF4444", "#B93030"
HAIR = "#3A2A20"
OUTLINE = "#1F2937"
BONE, JOINT = "#0B4FB3", "#FF9F1C"
COL_KNEE_F, COL_KNEE_B = "#16A34A", "#EA7C0C"
COL_TRUNK, COL_HIP, COL_EXT = "#7C3AED", "#C026D3", "#0E7490"


def ang(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


def deg_of(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def interior(v, a, b):
    u, w = a - v, b - v
    c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def leg_fk(hip, thigh_dir, knee_angle, foot_dir):
    knee = hip + THIGH * ang(thigh_dir)
    ankle = knee + SHANK * ang(thigh_dir - (180.0 - knee_angle))
    toe = ankle + FOOT * ang(foot_dir)
    return dict(hip=hip, knee=knee, ankle=ankle, toe=toe)


def arm_fk(shoulder, upper_dir, elbow_angle):
    elbow = shoulder + UARM * ang(upper_dir)
    wrist = elbow + FARM * ang(upper_dir + (180.0 - elbow_angle))
    return dict(shoulder=shoulder, elbow=elbow, wrist=wrist)


def arm_to_ground(shoulder, ground_y=GROUND_CLEAR, lean_behind=True):
    dy = shoulder[1] - ground_y
    dx = np.sqrt(max(ARM_LEN ** 2 - dy ** 2, 0.0))
    wrist = np.array([shoulder[0] - (dx if lean_behind else -dx), ground_y])
    d = wrist - shoulder
    n = np.linalg.norm(d)
    if n > ARM_LEN:
        wrist = shoulder + d / n * ARM_LEN
        d = wrist - shoulder
        n = ARM_LEN
    elbow = shoulder + d / n * UARM
    return dict(shoulder=shoulder, elbow=elbow, wrist=wrist)


def shift_pose(J, delta):
    for key, val in J.items():
        if isinstance(val, np.ndarray):
            J[key] = val + delta
        elif isinstance(val, dict):
            shift_pose(val, delta)


# ======================================================================
# 2. VẼ HÌNH NỘM
# ======================================================================
def _capsule(ax, p1, p2, w1, w2, color, z):
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-9:
        ax.add_patch(Circle(p1, w1 / 2, fc=color, ec="none", zorder=z))
        return
    n = np.array([-d[1], d[0]]) / L
    poly = [p1 + n * w1 / 2, p2 + n * w2 / 2, p2 - n * w2 / 2, p1 - n * w1 / 2]
    ax.add_patch(Polygon(poly, closed=True, fc=color, ec="none", zorder=z))
    ax.add_patch(Circle(p1, w1 / 2, fc=color, ec="none", zorder=z))
    ax.add_patch(Circle(p2, w2 / 2, fc=color, ec="none", zorder=z))


def draw_chain(ax, segs, z, outline=0.035):
    for p1, p2, w1, w2, _ in segs:
        _capsule(ax, p1, p2, w1 + outline, w2 + outline, OUTLINE, z)
    for p1, p2, w1, w2, c in segs:
        _capsule(ax, p1, p2, w1, w2, c, z + 0.1)


def _lerp(a, b, t):
    return a + (b - a) * t


def draw_leg(ax, leg, far, z):
    sk, sh, so = (SKIN_FAR, SHORTS_FAR, SHOE_FAR) if far else (SKIN, SHORTS, SHOE)
    hip, knee, ankle, toe = leg["hip"], leg["knee"], leg["ankle"], leg["toe"]
    cut = _lerp(hip, knee, 0.58)
    u = (toe - ankle) / np.linalg.norm(toe - ankle)
    segs = [
        (hip, cut, 0.36, 0.30, sh),
        (cut, knee, 0.25, 0.20, sk),
        (knee, ankle, 0.20, 0.12, sk),
        (ankle - u * 0.06, toe, 0.17, 0.11, so),
    ]
    draw_chain(ax, segs, z)


def draw_arm(ax, arm, far, z):
    sk, sr = (SKIN_FAR, SHIRT_FAR) if far else (SKIN, SHIRT)
    s, e, w = arm["shoulder"], arm["elbow"], arm["wrist"]
    cut = _lerp(s, e, 0.50)
    fd = (w - e) / np.linalg.norm(w - e)
    segs = [
        (s, cut, 0.22, 0.19, sr),
        (cut, e, 0.17, 0.14, sk),
        (e, w, 0.14, 0.10, sk),
        (w, w + fd * 0.11, 0.13, 0.10, sk),
    ]
    draw_chain(ax, segs, z)


def draw_head(ax, shoulder, head_dir, face_dir, z):
    neck_len = 0.16
    center = shoulder + (neck_len + HEAD_R) * ang(head_dir)
    _capsule(ax, shoulder, shoulder + (neck_len + 0.02) * ang(head_dir), 0.15, 0.13, OUTLINE, z)
    _capsule(ax, shoulder, shoulder + (neck_len + 0.02) * ang(head_dir), 0.12, 0.10, SKIN, z + 0.05)
    ax.add_patch(Circle(center, HEAD_R + 0.02, fc=OUTLINE, ec="none", zorder=z + 0.1))
    ax.add_patch(Circle(center, HEAD_R, fc=SKIN, ec="none", zorder=z + 0.2))
    ax.add_patch(Wedge(center, HEAD_R + 0.005, face_dir + 55, face_dir + 250,
                       fc=HAIR, ec="none", zorder=z + 0.3))
    nose = center + (HEAD_R + 0.045) * ang(face_dir)
    base = center + HEAD_R * 0.96 * ang(face_dir)
    nrm = np.array([-np.sin(np.radians(face_dir)), np.cos(np.radians(face_dir))])
    ax.add_patch(Polygon([nose, base + nrm * 0.035, base - nrm * 0.035],
                         fc=SKIN, ec=OUTLINE, lw=1.0, zorder=z + 0.35))
    ax.add_patch(Circle(center + 0.09 * ang(face_dir) + 0.06 * nrm, 0.017, fc=OUTLINE, zorder=z + 0.4))
    return center


def draw_torso(ax, hip, shoulder, z):
    draw_chain(ax, [(hip, shoulder, 0.34, 0.40, SHIRT)], z)


def draw_skeleton(ax, J, head_center):
    lines = [[J["hip"], J["shoulder"]], [J["shoulder"], head_center]]
    for leg in (J["leg_far"], J["leg_near"]):
        lines.append([leg["hip"], leg["knee"], leg["ankle"], leg["toe"]])
    for arm in (J["arm_far"], J["arm_near"]):
        lines.append([arm["shoulder"], arm["elbow"], arm["wrist"]])
    for pts in lines:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="white", lw=6.5, solid_capstyle="round", zorder=30)
        ax.plot(xs, ys, color=BONE, lw=3.4, solid_capstyle="round", zorder=31)
    joints = [J["hip"], J["shoulder"], head_center]
    for leg in (J["leg_far"], J["leg_near"]):
        joints += [leg["knee"], leg["ankle"], leg["toe"]]
    for arm in (J["arm_far"], J["arm_near"]):
        joints += [arm["elbow"], arm["wrist"]]
    jx, jy = zip(*joints)
    ax.scatter(jx, jy, s=46, c=JOINT, edgecolors="#1F2937", linewidths=1.0, zorder=32)


def draw_mannequin(ax, J, head_dir, face_dir, skeleton=True):
    draw_arm(ax, J["arm_far"], True, 3.0)
    draw_leg(ax, J["leg_far"], True, 4.0)
    draw_torso(ax, J["hip"], J["shoulder"], 5.0)
    draw_leg(ax, J["leg_near"], False, 6.0)
    hc = draw_head(ax, J["shoulder"], head_dir, face_dir, 7.0)
    draw_arm(ax, J["arm_near"], False, 8.0)
    if skeleton:
        draw_skeleton(ax, J, hc)
    return hc


# ======================================================================
# 3. MÔI TRƯỜNG
# ======================================================================
def draw_ground(ax, x0, x1):
    ax.add_patch(Rectangle((x0, -0.55), x1 - x0, 0.55, fc="#C8553D", ec="none", zorder=2))
    ax.add_patch(Rectangle((x0, -0.07), x1 - x0, 0.07, fc="#E07A5F", ec="none", zorder=2.1))
    for y in (-0.30, -0.50):
        ax.plot([x0, x1], [y, y], color="white", lw=2.2, zorder=2.2)
    ax.plot([x0, x1], [0, 0], color="#7F2F1E", lw=2.2, zorder=2.3)


def draw_block(ax, foot):
    u = (foot["toe"] - foot["ankle"])
    u = u / np.linalg.norm(u)
    ns = np.array([u[1], -u[0]])
    a = foot["ankle"] - u * 0.04 + ns * 0.07
    b = foot["toe"] + u * 0.05 + ns * 0.07
    pad = [a, b, b + ns * 0.07, a + ns * 0.07]
    ax.add_patch(Polygon(pad, closed=True, fc="#F59E0B", ec=OUTLINE, lw=1.5, zorder=1.6))
    top_a, top_b = a + ns * 0.07, b + ns * 0.07
    brace = [top_a, top_b, np.array([top_b[0] + 0.05, 0.0]), np.array([top_a[0] - 0.55, 0.0])]
    ax.add_patch(Polygon(brace, closed=True, fc="#9CA3AF", ec=OUTLINE, lw=1.2, zorder=1.5))
    ax.add_patch(Rectangle((top_a[0] - 0.65, 0.0), (b[0] - top_a[0]) + 0.85, 0.07,
                           fc="#4B5563", ec=OUTLINE, lw=1.2, zorder=2.4))


# ======================================================================
# 4. NHÃN GÓC (đọc khoảng chuẩn từ SPEC)
# ======================================================================
def label(ax, text_xy, text, color, anchor=None, ha="center"):
    ax.annotate(
        text, xy=anchor if anchor is not None else text_xy, xytext=text_xy,
        ha=ha, va="center", fontsize=9.5, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.5, shrinkA=0, shrinkB=0),
        zorder=40,
    )


def draw_angle(ax, v, a, b, r, color, text_xy, name, phase=None, feat=None, ha="center"):
    """Cung góc TRONG tại đỉnh v. Nếu có (phase, feat): so với khoảng chuẩn YAML
    và ghi giá trị vào bảng. Nếu không: chỉ hiển thị giá trị."""
    a1, a2 = deg_of(a - v), deg_of(b - v)
    d = (a2 - a1 + 180.0) % 360.0 - 180.0
    t1, t2 = (a1, a1 + d) if d > 0 else (a2, a2 - d)
    ax.add_patch(Wedge(v, r, t1, t2, fc=color, ec=color, alpha=0.28, lw=0, zorder=25))
    ax.add_patch(Wedge(v, r, t1, t2, fc="none", ec=color, lw=2.4, zorder=26))
    value = interior(v, a, b)
    if phase is None:
        txt = f"{name} = {value:.0f}°"
    else:
        lo, hi = rng(phase, feat)
        ok = in_range(phase, feat, value)
        rec(phase, feat, value)
        txt = (f"{name} = {value:.0f}°\n"
               f"chuẩn [{lo:g} ; {hi:g}]°  {'✓' if ok else '✗'}")
    mid = ang((t1 + t2) / 2.0)
    label(ax, text_xy, txt, color, anchor=v + mid * r, ha=ha)
    return value


# ======================================================================
# 5. CÁC TƯ THẾ
# ======================================================================
def pose_on_marks():
    """PHA 1 — VÀO CHỖ: knee_front ∈ [30,85], knee_back ∈ [30,95],
    trunk_thigh ∈ [30,110]; tay duỗi chống đất."""
    t, trunk_thigh, Kf, Kb = 50.0, 60.0, 65.0, 90.0
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    thigh_f = t - trunk_thigh
    front = leg_fk(hip, thigh_f, Kf, foot_dir=-14.0)
    ground = front["toe"][1] - GROUND_CLEAR
    s = (ground + 0.12 - hip[1]) / THIGH
    thigh_b = -180.0 - np.degrees(np.arcsin(np.clip(s, -1, 1)))
    back = leg_fk(hip, thigh_b, Kb, foot_dir=-30.0)
    J = dict(hip=hip, shoulder=shoulder, leg_near=front, leg_far=back)
    shift_pose(J, np.array([0.0, -ground]))
    J["arm_near"] = arm_to_ground(J["shoulder"])
    J["arm_far"] = arm_to_ground(J["shoulder"] + np.array([-0.08, 0.0]))
    return J, dict(head_dir=62.0, face_dir=-8.0, trunk_dir=t)


def pose_set():
    """PHA 2 — SẴN SÀNG (V2.4): knee_front ∈ [60,180], knee_back ∈ [90,155],
    hip_above ∈ [0.3,2.0], trunk_thigh ∈ [10,110]."""
    r, trunk_thigh, Kf, Kb = 0.50, 21.0, 90.0, 130.0
    t = -np.degrees(np.arcsin(r))
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    thigh_f = t - trunk_thigh
    front = leg_fk(hip, thigh_f, Kf, foot_dir=-55.0)
    best, best_err = None, 1e9
    for th in np.linspace(-150, -88, 3000):
        cand = leg_fk(hip, th, Kb, foot_dir=-40.0)
        err = abs(cand["toe"][1] - front["toe"][1])
        if err < best_err:
            best, best_err = cand, err
    ground = front["toe"][1] - GROUND_CLEAR
    J = dict(hip=hip, shoulder=shoulder, leg_near=front, leg_far=best)
    shift_pose(J, np.array([0.0, -ground]))
    J["arm_near"] = arm_to_ground(J["shoulder"])
    J["arm_far"] = arm_to_ground(J["shoulder"] + np.array([-0.08, 0.0]))
    return J, dict(head_dir=70.0, face_dir=-18.0, trunk_dir=t, r=r)


def pose_explosive():
    """PHA 3 — BỘC PHÁT: at_trunk ∈ [35,115], leg_ground ∈ [25,65],
    at_knee_front ∈ [70,180] (chân lăng), chân đạp duỗi gần thẳng."""
    lean = 45.0
    t = 90.0 - lean
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    push = leg_fk(hip, 222.5, 176.0, foot_dir=-75.0)
    swing = leg_fk(hip, -35.0, 120.0, foot_dir=-30.0)
    J = dict(hip=hip, shoulder=shoulder, leg_near=push, leg_far=swing,
             arm_near=arm_fk(shoulder, -45.0, 98.0),
             arm_far=arm_fk(shoulder, 150.0, 140.0))
    shift_pose(J, np.array([0.0, -(push["toe"][1] - GROUND_CLEAR)]))
    return J, dict(head_dir=35.0, face_dir=-10.0, trunk_dir=t, lean=lean)


# Thân dựng dần qua 3 bước (độ nghiêng so với phương thẳng đứng)
DRIVE_LEANS = [47.0, 42.5, 38.0]


def pose_drive():
    """PHA 4 — CHẠY LAO (tham khảo, weight 0): minh họa bước thứ 3."""
    lean = DRIVE_LEANS[-1]
    t = 90.0 - lean
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    stance = leg_fk(hip, -72.0, 162.0, foot_dir=0.0)
    swing = leg_fk(hip, -18.0, 78.0, foot_dir=-38.0)
    J = dict(hip=hip, shoulder=shoulder, leg_near=stance, leg_far=swing,
             arm_near=arm_fk(shoulder, 160.0, 98.0),
             arm_far=arm_fk(shoulder, -50.0, 92.0))
    shift_pose(J, np.array([0.0, -(stance["ankle"][1] - 0.10)]))
    return J, dict(head_dir=55.0, face_dir=-6.0, trunk_dir=t, lean=lean)


# ======================================================================
# 6. KHUNG HÌNH
# ======================================================================
XLIM, YLIM = (-3.0, 3.0), (-0.55, 3.45)


def setup_axes(ax, title, sub="", bg="#EAF3FB"):
    ax.set_facecolor(bg)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14.5, fontweight="bold", color="#1D4ED8", pad=22 if sub else 8)
    if sub:
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, color="#475569", style="italic")
    draw_ground(ax, *XLIM)


def center_x(J, x):
    shift_pose(J, np.array([x - J["hip"][0], 0.0]))


def record_arms(phase, J):
    """elbow_left = tay xa, elbow_right = tay gần (quy ước minh họa)."""
    for feat, arm in (("elbow_left", J["arm_far"]), ("elbow_right", J["arm_near"])):
        if feat in SPEC["features"][phase]:
            rec(phase, feat, interior(arm["elbow"], arm["shoulder"], arm["wrist"]))


def phase1(ax):
    P = "vao_cho"
    setup_axes(ax, "1. Vào chỗ (On your marks)",
               "Gối sau chạm đất · hông thấp hơn vai · tay chống đất")
    J, m = pose_on_marks()
    center_x(J, 0.35)
    draw_block(ax, J["leg_near"]); draw_block(ax, J["leg_far"])
    hc = draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    f, b = J["leg_near"], J["leg_far"]
    ax.add_patch(Circle(b["knee"] + np.array([0, -0.12]), 0.05,
                        fc="#6B7280", ec=OUTLINE, zorder=9))
    draw_angle(ax, J["hip"], J["shoulder"], f["knee"], 0.62, COL_TRUNK,
               (J["hip"][0] - 1.15, J["hip"][1] + 1.35), "trunk_thigh", P, "trunk_thigh_angle")
    draw_angle(ax, f["knee"], f["hip"], f["ankle"], 0.42, COL_KNEE_F,
               (f["knee"][0] + 1.15, f["knee"][1] + 0.85), "knee_front", P, "knee_front_angle")
    draw_angle(ax, b["knee"], b["hip"], b["ankle"], 0.42, COL_KNEE_B,
               (b["knee"][0] - 0.75, b["knee"][1] + 0.95), "knee_back", P, "knee_back_angle")
    record_arms(P, J)
    rec(P, "head_trunk_alignment", interior(J["shoulder"], J["hip"], hc))


def phase2(ax):
    P = "static"
    setup_axes(ax, "2. Sẵn sàng (Set)")
    J, m = pose_set()
    center_x(J, 0.35)
    draw_block(ax, J["leg_near"]); draw_block(ax, J["leg_far"])
    draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    f, b = J["leg_near"], J["leg_far"]
    hip, sh = J["hip"], J["shoulder"]
    xr = hip[0] - 1.45
    for y in (hip[1], sh[1]):
        ax.plot([xr - 0.15, hip[0] if y == hip[1] else sh[0]], [y, y],
                color=COL_HIP, ls=":", lw=1.6, zorder=24)
    ax.annotate("", xy=(xr, hip[1]), xytext=(xr, sh[1]),
                arrowprops=dict(arrowstyle="<->", color=COL_HIP, lw=2.2), zorder=26)
    r = (hip[1] - sh[1]) / TORSO
    rec(P, "hip_above_shoulder_norm", r)
    lo, hi = rng(P, "hip_above_shoulder_norm")
    ok = in_range(P, "hip_above_shoulder_norm", r)
    label(ax, (xr - 0.2, (hip[1] + sh[1]) / 2 + 1.0),
          f"hip_above = {r:.2f} × thân\nchuẩn [{lo:g} ; {hi:g}]  {'✓' if ok else '✗'}",
          COL_HIP, anchor=(xr, (hip[1] + sh[1]) / 2))
    draw_angle(ax, hip, sh, f["knee"], 0.62, COL_TRUNK,
               (hip[0] + 0.9, hip[1] + 1.25), "trunk_thigh", P, "trunk_thigh_angle")
    draw_angle(ax, f["knee"], f["hip"], f["ankle"], 0.42, COL_KNEE_F,
               (f["knee"][0] + 1.35, f["knee"][1] + 0.75), "knee_front", P, "knee_front_angle")
    draw_angle(ax, b["knee"], b["hip"], b["ankle"], 0.42, COL_KNEE_B,
               (b["knee"][0] - 1.1, b["knee"][1] + 0.7), "knee_back", P, "knee_back_angle")
    record_arms(P, J)


def phase3(ax):
    P = "sprint_start"
    setup_axes(ax, "3. Bộc phát (Explosive start)",
               "Thân lao về trước · chân đạp duỗi hết ra sau · chân lăng gập đưa lên")
    J, m = pose_explosive()
    center_x(J, 0.9)
    draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    hip, sh = J["hip"], J["shoulder"]
    push, swing = J["leg_near"], J["leg_far"]

    # at_trunk_angle_vertical
    ax.plot([hip[0], hip[0]], [hip[1], hip[1] + 1.25],
            color=COL_TRUNK, ls="--", lw=1.8, zorder=24)
    draw_angle(ax, hip, hip + np.array([0, 1.0]), sh, 0.75, COL_TRUNK,
               (hip[0] - 1.6, hip[1] + 1.45), "at_trunk_angle_vertical",
               P, "at_trunk_angle_vertical")

    # at_knee_front_angle (chân lăng) — SECONDARY
    draw_angle(ax, swing["knee"], swing["hip"], swing["ankle"], 0.42, COL_KNEE_F,
               (swing["knee"][0] - 1.0, swing["knee"][1] + 1.0),
               "at_knee_front", P, "at_knee_front_angle")

    # front_leg_full_extension (boolean, SECONDARY): gối chân đạp ≥ 170°
    ext = interior(push["knee"], push["hip"], push["ankle"])
    passed = ext >= 170.0
    rec(P, "front_leg_full_extension", 1.0 if passed else 0.0)
    ax.scatter([push["knee"][0]], [push["knee"][1]], s=260, facecolors="none",
               edgecolors=COL_EXT, linewidths=2.6, zorder=33)
    label(ax, (push["knee"][0] - 1.35, push["knee"][1] + 0.95),
          f"gối chân đạp = {ext:.0f}°\nfront_leg_full_extension  {'✓' if passed else '✗'}",
          COL_EXT, anchor=push["knee"])

    # leg_ground_angle
    base = push["toe"]
    ax.plot([base[0] - 0.2, base[0] + 1.7], [GROUND_CLEAR, GROUND_CLEAR],
            color="#6B7280", ls="--", lw=1.5, zorder=24)
    draw_angle(ax, push["ankle"], push["ankle"] + np.array([1.0, 0.0]), push["hip"],
               0.7, "#B45309",
               (push["ankle"][0] + 1.55, push["ankle"][1] + 0.55),
               "leg_ground", P, "leg_ground_angle")

    ax.annotate("", xy=(2.85, 2.85), xytext=(2.2, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.4), zorder=40)


def phase4(ax):
    P = "drive_phase"
    setup_axes(ax, "4. Chạy lao (Drive phase) — chỉ tham khảo",
               "weight pha = 0.00 → KHÔNG tính vào overall · bước dài dần, thân dựng dần",
               bg="#F1F5F9")
    J, m = pose_drive()
    center_x(J, 0.15)
    hip = J["hip"]
    for lean, a in ((DRIVE_LEANS[0], 0.35), (DRIVE_LEANS[1], 0.5)):
        end = hip + TORSO * ang(90 - lean)
        ax.plot([hip[0], end[0]], [hip[1], end[1]],
                color=COL_TRUNK, ls="--", lw=2.2, alpha=a, zorder=24)
    draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    sh = J["shoulder"]
    ax.plot([hip[0], hip[0]], [hip[1], hip[1] + 1.25],
            color=COL_TRUNK, ls=":", lw=1.6, zorder=24)
    # góc thân hiện tại (không có khoảng chuẩn riêng trong YAML -> chỉ hiển thị)
    draw_angle(ax, hip, hip + np.array([0, 1.0]), sh, 0.8, COL_TRUNK,
               (hip[0] - 1.75, hip[1] + 1.15), "trunk (bước 3)")
    ax.annotate("", xy=hip + 1.12 * ang(90 - DRIVE_LEANS[2]),
                xytext=hip + 1.12 * ang(90 - DRIVE_LEANS[0]),
                arrowprops=dict(arrowstyle="-|>", color=COL_TRUNK, lw=2.0,
                                connectionstyle="arc3,rad=0.25"), zorder=27)

    # trunk_angle_vertical_trend: hồi quy tuyến tính góc thân theo bước
    trunk_slope = float(np.polyfit(np.arange(len(DRIVE_LEANS)), DRIVE_LEANS, 1)[0])
    rec(P, "trunk_angle_vertical_trend", trunk_slope)
    lo, hi = rng(P, "trunk_angle_vertical_trend")
    ok = in_range(P, "trunk_angle_vertical_trend", trunk_slope)
    label(ax, (hip[0] - 1.75, hip[1] + 0.3),
          f"trunk_trend = {trunk_slope:+.1f}°/bước\nchuẩn [{lo:g} ; {hi:g}]  {'✓' if ok else '✗'}",
          COL_TRUNK, anchor=hip + 0.6 * ang(90 - 44.0))

    # dấu chân + độ dài bước
    foot_x = J["leg_near"]["ankle"][0]
    xs = [foot_x - 2.35, foot_x - 1.25, foot_x]
    for x in xs:
        ax.add_patch(Polygon([(x - 0.18, 0.0), (x + 0.32, 0.0),
                              (x + 0.32, -0.12), (x - 0.18, -0.12)],
                             fc="#6B1F0D", ec="white", lw=1.2, zorder=2.6))
    ghost = foot_x + 1.45
    ax.add_patch(Polygon([(ghost - 0.15, 0.0), (ghost + 0.28, 0.0),
                          (ghost + 0.28, -0.07), (ghost - 0.15, -0.07)],
                         fc="none", ec="#7C2D12", ls="--", lw=1.4, zorder=2.6))
    y_arrow = -0.27
    pts = xs + [ghost]
    steps = [xb - xa for xa, xb in zip(pts[:-1], pts[1:])]
    for xa, xb in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=(xb + 0.05, y_arrow), xytext=(xa + 0.05, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=2.0), zorder=3)
    for (xa, xb), s in zip(zip(pts[:-1], pts[1:]), steps):
        ax.text((xa + xb) / 2 + 0.05, y_arrow - 0.12, f"{s:.2f}",
                ha="center", va="top", fontsize=8.5,
                color="white", fontweight="bold", zorder=4)

    step_slope = float(np.polyfit(np.arange(len(steps)), steps, 1)[0])
    rec(P, "step_length_trend", step_slope)
    lo, hi = rng(P, "step_length_trend")
    ok = in_range(P, "step_length_trend", step_slope)
    label(ax, (foot_x - 1.6, 0.75),
          f"step_length_trend = {step_slope:+.2f}\nchuẩn [{lo:g} ; {hi:g}]  {'✓' if ok else '✗'}",
          "#9A3412", anchor=(foot_x - 1.25, 0.0), ha="center")

    ax.annotate("", xy=(2.85, 2.85), xytext=(2.2, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.4), zorder=40)


# ======================================================================
# 7. BẢNG ĐẶC TRƯNG DƯỚI MỖI PHA
# ======================================================================
def fmt_val(phase, feat, v):
    lo, hi, _, _ = SPEC["features"][phase][feat]
    if lo is not None and lo == hi:                 # boolean
        return "pass" if v >= 0.5 else "fail"
    return (f"{v:.2f}" if (hi is None or hi - lo <= 5 or abs(v) < 10) else f"{v:.0f}") + unit_of(feat)


def phase_weight_text(phase):
    pw = SPEC["phase_weights"]
    total = sum(pw.values())
    w = pw[phase]
    if w <= 0:
        return f"trọng số pha {w:.2f}  (0% — KHÔNG tính vào overall, chỉ tham khảo)"
    return f"trọng số pha {w:.2f}  ({w / total * 100:.1f}% overall, chuẩn hóa /{total:.2f})"


def draw_table(ax, phase, idx):
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="white",
                           ec="#CBD5E1", lw=1.2, zorder=0))
    kw = dict(fontsize=8.6, transform=ax.transAxes, va="top")
    ax.text(0.015, 0.975, f"{idx}. {PHASE_TITLE[phase]} — {phase_weight_text(phase)}",
            fontsize=9.6, fontweight="bold", color="#0F172A",
            transform=ax.transAxes, va="top")
    X = (0.015, 0.40, 0.50, 0.66, 0.73, 0.80)      # cột: tên, đo, chuẩn, lv, w, kết quả
    for x, h, ha in zip(X, ("đặc trưng", "đo", "chuẩn", "lv", "w", "kết quả"),
                        ("left", "right", "left", "left", "left", "left")):
        ax.text(x, 0.86, h, color="#475569", ha=ha, fontweight="bold", **kw)
    step = 0.093
    y = 0.86 - step
    for feat, (lo, hi, level, w) in SPEC["features"][phase].items():
        v = ROWS[phase].get(feat)
        rg = "—" if lo is None else f"[{lo:g} ; {hi:g}]"
        if level == "INFO":
            val_s, res, col = "—", "info", "#64748B"
        elif v is None:
            val_s, res, col = "—", "không minh họa", "#94A3B8"
        else:
            ok = in_range(phase, feat, v)
            val_s, res, col = fmt_val(phase, feat, v), ("✓ OK" if ok else "✗ LỆCH"), \
                ("#15803D" if ok else "#B91C1C")
        for x, t, ha in zip(X, (feat, val_s, rg, LEVEL_CH[level], f"{w:.1f}", res),
                            ("left", "right", "left", "left", "left", "left")):
            ax.text(x, y, t, color=col, ha=ha, **kw)
        y -= step


def print_report():
    print(f"\n{'Pha':<14}{'Đặc trưng':<30}{'Đo':>9}  {'Chuẩn':<14}Kết quả")
    print("-" * 78)
    for phase, feats in SPEC["features"].items():
        for feat, (lo, hi, level, w) in feats.items():
            v = ROWS[phase].get(feat)
            if v is None:
                res, vs = ("info" if level == "INFO" else "không minh họa"), "—"
            else:
                res = "OK" if in_range(phase, feat, v) else "LỆCH CHUẨN"
                vs = f"{v:.2f}"
            rg = "—" if lo is None else f"[{lo:g}, {hi:g}]"
            print(f"{phase:<14}{feat:<30}{vs:>9}  {rg:<14}{res}")
        print("-" * 78)


# ======================================================================
# 8. CHẠY
# ======================================================================
def main():
    fig = plt.figure(figsize=(18, 17.5), facecolor="#F8FAFC")
    gs = GridSpec(4, 2, figure=fig, height_ratios=[3.2, 1.3, 3.2, 1.3],
                  hspace=0.22, wspace=0.06, left=0.03, right=0.97, top=0.895, bottom=0.04)
    fig.suptitle("MÔ PHỎNG 4 GIAI ĐOẠN XUẤT PHÁT THẤP (THEO SPEC sprint_start V2.4-FINAL)",
                 fontsize=19, fontweight="bold", color="#0F172A", y=0.985)
    pw = SPEC["phase_weights"]
    fig.text(0.5, 0.955,
             f"Overall = vao_cho {pw['vao_cho']:.2f} + static {pw['static']:.2f} + "
             f"sprint_start {pw['sprint_start']:.2f} (chuẩn hóa /{sum(pw.values()):.2f}) · "
             f"drive_phase {pw['drive_phase']:.2f} = chỉ tham khảo",
             ha="center", fontsize=11, color="#334155")

    fns = (phase1, phase2, phase3, phase4)
    phases = ("vao_cho", "static", "sprint_start", "drive_phase")
    for i, (fn, ph) in enumerate(zip(fns, phases)):
        row, col = 2 * (i // 2), i % 2
        ax = fig.add_subplot(gs[row, col])
        fn(ax)
        draw_table(fig.add_subplot(gs[row + 1, col]), ph, i + 1)

    fig.text(0.5, 0.012,
             "Khung xương xanh + khớp cam = keypoint pipeline đo · cung màu = góc được chấm · "
             "giá trị đo từ chính hình vẽ · lv: P=PRIMARY, S=SECONDARY, I=INFO · "
             f"nguồn spec: {SPEC_SRC}",
             ha="center", fontsize=9.5, color="#475569")
    out = "4_phases_mannequin_Spec_v2.4.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Đã lưu ảnh: {out}")
    print_report()
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()