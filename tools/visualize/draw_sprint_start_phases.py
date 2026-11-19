#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minh họa 4 giai đoạn XUẤT PHÁT THẤP (Spec v2.1) — khung xương + hình nộm 2D.

KHÁC BẢN CŨ
-----------
1. Dựng tư thế bằng ĐỘNG HỌC THUẬN (forward kinematics): chỉ khai báo GÓC và
   ĐỘ DÀI ĐOẠN, vị trí khớp được TÍNH ra -> chi không bị dài/ngắn tùy tiện,
   góc vẽ ra KHỚP với số trong spec (script tự đo lại và in bảng kiểm tra).
2. Hình nộm có hình khối thật: thân thon, đùi/cẳng chân/tay thon, quần-áo-da,
   giày, tay, đầu có tóc + mũi (biết người đang quay mặt về hướng nào).
3. Phủ KHUNG XƯƠNG xanh + khớp cam lên hình nộm (giống ảnh tham khảo).
4. Có bàn đạp xuất phát, đường chạy (track) + vạch làn, tay chống đất đúng chỗ.
5. Cung góc luôn vẽ đúng GÓC TRONG giữa 2 đoạn, nhãn có đường dẫn + giá trị
   ĐO ĐƯỢC + khoảng chuẩn + dấu ✓/✗.
6. Pha 3 sửa lại cho đúng thực tế: chân đạp duỗi RA SAU (leg_ground 40-45°),
   chân lăng gập gối đưa ra trước (bản cũ vẽ chân duỗi ra trước — sai).

Chạy:  python draw_sprint_start_phases.py          (hiện + lưu PNG)
       python draw_sprint_start_phases.py --no-show (chỉ lưu PNG)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

plt.rcParams["font.family"] = "DejaVu Sans"  # có đủ dấu tiếng Việt

# ======================================================================
# 0. HẰNG SỐ
# ======================================================================
# Độ dài đoạn (đơn vị: 1.0 = chiều dài thân vai-hông, giống body_scale)
TORSO, THIGH, SHANK = 1.00, 0.92, 0.92
UARM, FARM, FOOT = 0.66, 0.60, 0.42
HEAD_R = 0.19
ARM_LEN = UARM + FARM
GROUND_CLEAR = 0.05  # độ cao mũi giày / cổ tay khi "chạm đất"

# Màu
SKIN, SKIN_FAR = "#F0C29B", "#D2A27C"
SHIRT, SHIRT_FAR = "#3B82F6", "#2B63BD"
SHORTS, SHORTS_FAR = "#334155", "#232E3C"
SHOE, SHOE_FAR = "#EF4444", "#B93030"
HAIR = "#3A2A20"
OUTLINE = "#1F2937"
BONE, JOINT = "#0B4FB3", "#FF9F1C"
COL_KNEE_F, COL_KNEE_B = "#16A34A", "#EA7C0C"
COL_TRUNK, COL_HIP, COL_EXT = "#7C3AED", "#C026D3", "#0E7490"

# ======================================================================
# 1. HÌNH HỌC CƠ BẢN
# ======================================================================
def ang(deg):
    """Vector đơn vị theo góc (độ) tính từ +x, ngược chiều kim đồng hồ (y hướng lên)."""
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


def deg_of(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def interior(v, a, b):
    """Góc trong (độ) tại đỉnh v giữa 2 tia v->a và v->b."""
    u, w = a - v, b - v
    c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def leg_fk(hip, thigh_dir, knee_angle, foot_dir):
    """Chân (người quay mặt sang PHẢI): gối gập thì cẳng chân xoay thuận
    chiều kim đồng hồ so với đùi -> shank_dir = thigh_dir - (180 - K)."""
    knee = hip + THIGH * ang(thigh_dir)
    ankle = knee + SHANK * ang(thigh_dir - (180.0 - knee_angle))
    toe = ankle + FOOT * ang(foot_dir)
    return dict(hip=hip, knee=knee, ankle=ankle, toe=toe)


def arm_fk(shoulder, upper_dir, elbow_angle):
    """Tay: khuỷu gập thì cẳng tay xoay NGƯỢC chiều kim đồng hồ so với cánh tay."""
    elbow = shoulder + UARM * ang(upper_dir)
    wrist = elbow + FARM * ang(upper_dir + (180.0 - elbow_angle))
    return dict(shoulder=shoulder, elbow=elbow, wrist=wrist)


def arm_to_ground(shoulder, ground_y=GROUND_CLEAR, lean_behind=True):
    """Tay duỗi thẳng chống xuống đất (vai hơi vượt ra trước bàn tay)."""
    dy = shoulder[1] - ground_y
    dx = np.sqrt(max(ARM_LEN ** 2 - dy ** 2, 0.0))
    wrist = np.array([shoulder[0] - (dx if lean_behind else -dx), ground_y])
    d = wrist - shoulder
    n = np.linalg.norm(d)
    if n > ARM_LEN:  # không với tới đất -> duỗi thẳng theo hướng đó (tay lơ lửng)
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
    """segs = [(p1, p2, w1, w2, color), ...]. Vẽ viền trước, tô màu sau để
    các khớp nối liền mạch, không bị 'đường cắt' giữa các đoạn."""
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
        (hip, cut, 0.36, 0.30, sh),          # ống quần
        (cut, knee, 0.25, 0.20, sk),         # đùi (da)
        (knee, ankle, 0.20, 0.12, sk),       # cẳng chân
        (ankle - u * 0.06, toe, 0.17, 0.11, so),  # giày
    ]
    draw_chain(ax, segs, z)


def draw_arm(ax, arm, far, z):
    sk, sr = (SKIN_FAR, SHIRT_FAR) if far else (SKIN, SHIRT)
    s, e, w = arm["shoulder"], arm["elbow"], arm["wrist"]
    cut = _lerp(s, e, 0.50)
    fd = (w - e) / np.linalg.norm(w - e)
    segs = [
        (s, cut, 0.22, 0.19, sr),             # tay áo
        (cut, e, 0.17, 0.14, sk),
        (e, w, 0.14, 0.10, sk),
        (w, w + fd * 0.11, 0.13, 0.10, sk),   # bàn tay
    ]
    draw_chain(ax, segs, z)


def draw_head(ax, shoulder, head_dir, face_dir, z):
    neck_len = 0.16
    center = shoulder + (neck_len + HEAD_R) * ang(head_dir)
    _capsule(ax, shoulder, shoulder + (neck_len + 0.02) * ang(head_dir), 0.15, 0.13, OUTLINE, z)
    _capsule(ax, shoulder, shoulder + (neck_len + 0.02) * ang(head_dir), 0.12, 0.10, SKIN, z + 0.05)
    ax.add_patch(Circle(center, HEAD_R + 0.02, fc=OUTLINE, ec="none", zorder=z + 0.1))
    ax.add_patch(Circle(center, HEAD_R, fc=SKIN, ec="none", zorder=z + 0.2))
    # tóc: nửa sau-trên của đầu
    ax.add_patch(Wedge(center, HEAD_R + 0.005, face_dir + 55, face_dir + 250,
                       fc=HAIR, ec="none", zorder=z + 0.3))
    # mũi: tam giác nhỏ chỉ hướng nhìn
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
    """Khung xương xanh + khớp cam (lớp trên cùng) — halo trắng để thấy rõ
    trên mọi màu nền của hình nộm."""
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
    """Thứ tự lớp: tay xa -> chân xa -> thân -> chân gần -> đầu -> tay gần."""
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
# 3. MÔI TRƯỜNG: ĐƯỜNG CHẠY, BÀN ĐẠP
# ======================================================================
def draw_ground(ax, x0, x1):
    ax.add_patch(Rectangle((x0, -0.55), x1 - x0, 0.55, fc="#C8553D", ec="none", zorder=2))
    ax.add_patch(Rectangle((x0, -0.07), x1 - x0, 0.07, fc="#E07A5F", ec="none", zorder=2.1))
    for y in (-0.30, -0.50):
        ax.plot([x0, x1], [y, y], color="white", lw=2.2, zorder=2.2)
    ax.plot([x0, x1], [0, 0], color="#7F2F1E", lw=2.2, zorder=2.3)


def draw_block(ax, foot):
    """Bàn đạp: mặt đạp nằm sát đế giày (vuông góc với phương đạp)."""
    u = (foot["toe"] - foot["ankle"])
    u = u / np.linalg.norm(u)
    ns = np.array([u[1], -u[0]])           # pháp tuyến phía đế giày
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
# 4. NHÃN GÓC
# ======================================================================
def label(ax, text_xy, text, color, anchor=None, ha="center"):
    ax.annotate(
        text, xy=anchor if anchor is not None else text_xy, xytext=text_xy,
        ha=ha, va="center", fontsize=9.5, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.5, shrinkA=0, shrinkB=0),
        zorder=40,
    )


def draw_angle(ax, v, a, b, r, color, text_xy, name, spec, unit="°", fmt="{:.0f}",
               measured=None, ha="center"):
    """Cung góc TRONG tại đỉnh v giữa v->a và v->b + nhãn có giá trị ĐO ĐƯỢC.
    spec = (lo, hi). Trả về giá trị đo được."""
    a1, a2 = deg_of(a - v), deg_of(b - v)
    d = (a2 - a1 + 180.0) % 360.0 - 180.0
    t1, t2 = (a1, a1 + d) if d > 0 else (a2, a2 - d)
    ax.add_patch(Wedge(v, r, t1, t2, fc=color, ec=color, alpha=0.28, lw=0, zorder=25))
    ax.add_patch(Wedge(v, r, t1, t2, fc="none", ec=color, lw=2.4, zorder=26))
    value = interior(v, a, b) if measured is None else measured
    ok = spec[0] - 1e-9 <= value <= spec[1] + 1e-9
    txt = (f"{name} = {fmt.format(value)}{unit}\n"
           f"chuẩn [{fmt.format(spec[0])}–{fmt.format(spec[1])}]{unit}  {'✓' if ok else '✗'}")
    mid = ang((t1 + t2) / 2.0)
    label(ax, text_xy, txt, color, anchor=v + mid * r, ha=ha)
    REPORT.append((name, value, spec, ok))
    return value


REPORT = []


# ======================================================================
# 5. CÁC TƯ THẾ
# ======================================================================
def pose_on_marks():
    """PHA 1 — VÀO CHỖ: gối sau chạm đất, hông thấp hơn vai, tay chống đất."""
    t, trunk_thigh, Kf, Kb = 37.5, 21.0, 105.0, 127.0
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    thigh_f = t - trunk_thigh
    front = leg_fk(hip, thigh_f, Kf, foot_dir=-14.0)
    ground = front["toe"][1] - GROUND_CLEAR
    # gối sau CHẠM đất: giải góc đùi sau sao cho y(gối sau) = đất + 0.12
    s = (ground + 0.12 - hip[1]) / THIGH
    thigh_b = -180.0 - np.degrees(np.arcsin(np.clip(s, -1, 1)))
    back = leg_fk(hip, thigh_b, Kb, foot_dir=-50.0)
    J = dict(hip=hip, shoulder=shoulder, leg_near=front, leg_far=back)
    shift_pose(J, np.array([0.0, -ground]))
    J["arm_near"] = arm_to_ground(J["shoulder"])
    J["arm_far"] = arm_to_ground(J["shoulder"] + np.array([-0.08, 0.0]))
    return J, dict(head_dir=62.0, face_dir=-8.0, trunk_dir=t)


def pose_set():
    """PHA 2 — SẴN SÀNG: hông nâng cao hơn vai, gối trước ~90°, gối sau ~130°."""
    r, trunk_thigh, Kf, Kb = 0.50, 21.0, 90.0, 130.0
    t = -np.degrees(np.arcsin(r))                   # hông cao hơn vai r × thân
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    thigh_f = t - trunk_thigh
    front = leg_fk(hip, thigh_f, Kf, foot_dir=-55.0)
    # chân sau: tìm góc đùi để mũi giày sau ngang mũi giày trước (cùng đạp 1 mặt đất)
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
    """PHA 3 — BỘC PHÁT: chân đạp DUỖI RA SAU (≥170°, hợp mặt đất 40-45°),
    chân lăng gập gối ~70° đưa ra trước, 2 tay đánh ngược chiều."""
    lean = 28.0
    t = 90.0 - lean
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    push = leg_fk(hip, 222.5, 176.0, foot_dir=-75.0)     # chân đạp (gần camera)
    swing = leg_fk(hip, -31.0, 71.0, foot_dir=-50.0)     # chân lăng (xa)
    J = dict(hip=hip, shoulder=shoulder, leg_near=push, leg_far=swing,
             arm_near=arm_fk(shoulder, -45.0, 98.0),     # tay trước (khuỷu ~90°)
             arm_far=arm_fk(shoulder, 150.0, 140.0))     # tay sau
    shift_pose(J, np.array([0.0, -(push["toe"][1] - GROUND_CLEAR)]))
    return J, dict(head_dir=42.0, face_dir=-10.0, trunk_dir=t, lean=lean)


def pose_drive():
    """PHA 4 — CHẠY LAO: thân dựng dần, bước dài dần (minh họa bước thứ 3)."""
    lean = 38.0
    t = 90.0 - lean
    hip = np.zeros(2)
    shoulder = hip + TORSO * ang(t)
    stance = leg_fk(hip, -72.0, 162.0, foot_dir=0.0)     # chân chống (gần)
    swing = leg_fk(hip, -18.0, 78.0, foot_dir=-38.0)     # chân lăng (xa)
    J = dict(hip=hip, shoulder=shoulder, leg_near=stance, leg_far=swing,
             arm_near=arm_fk(shoulder, 160.0, 98.0),
             arm_far=arm_fk(shoulder, -50.0, 92.0))
    shift_pose(J, np.array([0.0, -(stance["ankle"][1] - 0.10)]))
    return J, dict(head_dir=55.0, face_dir=-6.0, trunk_dir=t, lean=lean)


# ======================================================================
# 6. KHUNG HÌNH
# ======================================================================
XLIM, YLIM = (-3.0, 3.0), (-0.55, 3.45)


def setup_axes(ax, title, sub=""):
    ax.set_facecolor("#EAF3FB")
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


def phase1(ax):
    setup_axes(ax, "1. Vào chỗ (On your marks)", "Gối sau chạm đất · hông thấp hơn vai · tay chống đất")
    J, m = pose_on_marks()
    center_x(J, 0.35)
    draw_block(ax, J["leg_near"]); draw_block(ax, J["leg_far"])
    hc = draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    f, b = J["leg_near"], J["leg_far"]
    # gối sau chạm đất: đánh dấu
    ax.add_patch(Circle(b["knee"] + np.array([0, -0.12]), 0.05, fc="#6B7280", ec=OUTLINE, zorder=9))
    draw_angle(ax, J["hip"], J["shoulder"], f["knee"], 0.62, COL_TRUNK,
               (J["hip"][0] - 1.15, J["hip"][1] + 1.35), "trunk_thigh", (19, 23))
    draw_angle(ax, f["knee"], f["hip"], f["ankle"], 0.42, COL_KNEE_F,
               (f["knee"][0] + 1.15, f["knee"][1] + 0.85), "knee_front", (90, 110))
    draw_angle(ax, b["knee"], b["hip"], b["ankle"], 0.42, COL_KNEE_B,
               (b["knee"][0] - 0.75, b["knee"][1] + 0.95), "knee_back", (115, 138))


def phase2(ax):
    setup_axes(ax, "2. Sẵn sàng (Set)")
    J, m = pose_set()
    center_x(J, 0.35)
    draw_block(ax, J["leg_near"]); draw_block(ax, J["leg_far"])
    hc = draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    f, b = J["leg_near"], J["leg_far"]
    hip, sh = J["hip"], J["shoulder"]
    # hip_above: chênh cao hông - vai (theo tỉ lệ thân)
    xr = hip[0] - 1.45
    for y in (hip[1], sh[1]):
        ax.plot([xr - 0.15, hip[0] if y == hip[1] else sh[0]], [y, y], color=COL_HIP, ls=":", lw=1.6, zorder=24)
    ax.annotate("", xy=(xr, hip[1]), xytext=(xr, sh[1]),
                arrowprops=dict(arrowstyle="<->", color=COL_HIP, lw=2.2), zorder=26)
    r = (hip[1] - sh[1]) / TORSO
    ok = 0.35 <= r <= 0.70
    REPORT.append(("hip_above", r, (0.35, 0.70), ok))
    label(ax, (xr - 0.2, (hip[1] + sh[1]) / 2 + 1.0),
          f"hip_above = {r:.2f} × thân\nchuẩn [0.35–0.70]  {'✓' if ok else '✗'}",
          COL_HIP, anchor=(xr, (hip[1] + sh[1]) / 2))
    draw_angle(ax, hip, sh, f["knee"], 0.62, COL_TRUNK,
               (hip[0] + 0.9, hip[1] + 1.25), "trunk_thigh", (19, 23))
    draw_angle(ax, f["knee"], f["hip"], f["ankle"], 0.42, COL_KNEE_F,
               (f["knee"][0] + 1.35, f["knee"][1] + 0.75), "knee_front", (85, 95))
    draw_angle(ax, b["knee"], b["hip"], b["ankle"], 0.42, COL_KNEE_B,
               (b["knee"][0] - 1.1, b["knee"][1] + 0.7), "knee_back", (120, 140))


def phase3(ax):
    setup_axes(ax, "3. Bộc phát (Explosive start)", "Thân lao về trước · chân đạp duỗi hết ra sau · chân lăng gập đưa lên")
    J, m = pose_explosive()
    center_x(J, 0.9)
    hc = draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    hip, sh = J["hip"], J["shoulder"]
    push = J["leg_near"]
    # trục thẳng đứng + góc thân
    ax.plot([hip[0], hip[0]], [hip[1], hip[1] + 1.25], color=COL_TRUNK, ls="--", lw=1.8, zorder=24)
    draw_angle(ax, hip, hip + np.array([0, 1.0]), sh, 0.75, COL_TRUNK,
               (hip[0] - 1.6, hip[1] + 1.45), "trunk", (25, 30))
    # duỗi chân đạp (góc gối) — gần thẳng nên dùng chú thích thay cho cung
    ext = interior(push["knee"], push["hip"], push["ankle"])
    ok = ext >= 170
    REPORT.append(("leg_extension", ext, (170, 180), ok))
    ax.scatter([push["knee"][0]], [push["knee"][1]], s=260, facecolors="none",
               edgecolors=COL_EXT, linewidths=2.6, zorder=33)
    label(ax, (push["knee"][0] - 1.35, push["knee"][1] + 0.95),
          f"leg_extension = {ext:.0f}°\nchuẩn ≥ 170°  {'✓' if ok else '✗'}",
          COL_EXT, anchor=push["knee"])
    # góc chân đạp so với mặt đất
    base = push["toe"]
    ax.plot([base[0] - 0.2, base[0] + 1.7], [GROUND_CLEAR, GROUND_CLEAR], color="#6B7280", ls="--", lw=1.5, zorder=24)
    leg_vec_pt = push["hip"]
    a_ground = interior(push["ankle"], push["ankle"] + np.array([1.0, 0.0]), leg_vec_pt)
    draw_angle(ax, push["ankle"], push["ankle"] + np.array([1.0, 0.0]), leg_vec_pt, 0.7, "#B45309",
               (push["ankle"][0] + 1.55, push["ankle"][1] + 0.55), "leg_ground", (40, 45))
    # mũi tên hướng chạy
    ax.annotate("", xy=(2.85, 2.85), xytext=(2.2, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.4), zorder=40)


def phase4(ax):
    setup_axes(ax, "4. Chạy lao (Drive phase)", "Mỗi bước dài hơn bước trước · thân nâng dần (minh họa, phóng đại tỉ lệ)")
    J, m = pose_drive()
    center_x(J, 0.15)
    hip = J["hip"]
    # bóng mờ thân ở các bước trước: thân dựng dần
    for lean, a in ((47.0, 0.35), (42.5, 0.5)):
        end = hip + TORSO * ang(90 - lean)
        ax.plot([hip[0], end[0]], [hip[1], end[1]], color=COL_TRUNK, ls="--", lw=2.2, alpha=a, zorder=24)
    hc = draw_mannequin(ax, J, m["head_dir"], m["face_dir"])
    sh = J["shoulder"]
    ax.plot([hip[0], hip[0]], [hip[1], hip[1] + 1.25], color=COL_TRUNK, ls=":", lw=1.6, zorder=24)
    draw_angle(ax, hip, hip + np.array([0, 1.0]), sh, 0.8, COL_TRUNK,
               (hip[0] - 1.75, hip[1] + 1.15), "trunk (bước này)", (30, 45))
    ax.annotate("", xy=hip + 1.12 * ang(90 - 38.5 + 0.0), xytext=hip + 1.12 * ang(90 - 47.0),
                arrowprops=dict(arrowstyle="-|>", color=COL_TRUNK, lw=2.0, connectionstyle="arc3,rad=0.25"), zorder=27)
    label(ax, (hip[0] - 1.75, hip[1] + 0.3),
          "trunk_trend < −1°/bước\n(thân dựng dần)", COL_TRUNK,
          anchor=hip + 0.6 * ang(90 - 44.0))
    # dấu chân: bước dài dần (phóng đại để dễ nhìn)
    foot_x = J["leg_near"]["ankle"][0]
    xs = [foot_x - 2.35, foot_x - 1.25, foot_x]
    for x in xs:
        ax.add_patch(Polygon([(x - 0.18, 0.0), (x + 0.32, 0.0), (x + 0.32, -0.12), (x - 0.18, -0.12)],
                             fc="#6B1F0D", ec="white", lw=1.2, zorder=2.6))
    ghost = foot_x + 1.45
    ax.add_patch(Polygon([(ghost - 0.15, 0.0), (ghost + 0.28, 0.0), (ghost + 0.28, -0.07), (ghost - 0.15, -0.07)],
                         fc="none", ec="#7C2D12", ls="--", lw=1.4, zorder=2.6))
    y_arrow = -0.27
    pts = xs + [ghost]
    for xa, xb in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=(xb + 0.05, y_arrow), xytext=(xa + 0.05, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=2.0), zorder=3)
    for xa, xb in zip(pts[:-1], pts[1:]):
        ax.text((xa + xb) / 2 + 0.05, y_arrow - 0.12, f"{(xb - xa):.2f}", ha="center", va="top",
                fontsize=8.5, color="white", fontweight="bold", zorder=4)
    label(ax, (foot_x - 1.6, 0.75), "step_length_trend > +3% / bước\n(độ dài bước tăng dần)",
          "#9A3412", anchor=(foot_x - 1.25, 0.0), ha="center")
    ax.annotate("", xy=(2.85, 2.85), xytext=(2.2, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.4), zorder=40)


# ======================================================================
# 7. CHẠY
# ======================================================================
def main():
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.2), facecolor="#F8FAFC")
    fig.suptitle("MÔ PHỎNG 4 GIAI ĐOẠN XUẤT PHÁT THẤP (THEO CHUẨN SPEC v2.1)",
                 fontsize=19, fontweight="bold", color="#0F172A", y=0.985)
    for fn, ax in zip((phase1, phase2, phase3, phase4), axes.flat):
        REPORT.append(("---", 0, (0, 0), True))
        fn(ax)
    fig.text(0.5, 0.008,
             "Khung xương xanh + khớp cam = các điểm keypoint mà pipeline đo · cung màu = góc được chấm · "
             "giá trị trong nhãn là góc ĐO từ chính hình vẽ, ✓ = nằm trong khoảng chuẩn",
             ha="center", fontsize=10, color="#475569")
    plt.tight_layout(rect=(0, 0.02, 1, 0.965), h_pad=2.5, w_pad=1.0)
    out = "4_phases_mannequin_Spec_v2.1.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Đã lưu ảnh: {out}\n")

    print(f"{'Đặc trưng':<16}{'Đo được':>10}   {'Chuẩn':<14}Kết quả")
    for name, val, spec, ok in REPORT:
        if name == "---":
            print("-" * 52)
            continue
        print(f"{name:<16}{val:>10.2f}   [{spec[0]:g}, {spec[1]:g}]".ljust(44) + ("OK" if ok else "LỆCH CHUẨN"))
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()