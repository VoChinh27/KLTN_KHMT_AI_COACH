"""
visualize_dataset.py
====================
Đánh giá CHẤT LƯỢNG + ĐỘ PHÙ HỢP của các video HLV đã trích xuất (đầu ra của
extract_pose_dataset.py) TRƯỚC KHI đưa vào xây khoảng chuẩn / AI Coach.

Quy trình ra quyết định cho mỗi video (theo model chính):
    1. GATE (loại cứng)  : detect thấp, valid thấp, mất khung dài, video quá ngắn,
                           fps nghi ngờ... -> REJECT, không tính điểm.
    2. ĐIỂM KỸ THUẬT     : tiêu chí RIÊNG theo loại động tác (running / sprint_start),
                           suy ra từ tên video. Tiêu chí = GIẢ ĐỊNH ban đầu, chỉnh trong CONFIG.
    3. OUTLIER           : so với CHÍNH nhóm video của cùng loại động tác (modified z-score,
                           median/MAD) -> video lệch bất thường bị đẩy sang REVIEW.
    4. NHẤT QUÁN MODEL   : YOLO và MediaPipe cho kết quả gần nhau => số liệu đáng tin hơn.
    => ACCEPT / REVIEW / REJECT + lý do cụ thể từng video.

Đầu ra (OUT_DIR):
    01..09_*.png            biểu đồ đánh giá
    video_decision.csv      bảng quyết định + lý do từng video
    accepted_videos.txt     danh sách video ACCEPT (dùng làm đầu vào bước xây khoảng chuẩn)
    review_videos.txt       danh sách video cần xem lại bằng mắt
    evaluation_report.txt   báo cáo văn bản

Chạy:
    python visualize_dataset.py
    python visualize_dataset.py --agg <_aggregated> --out <_figures> --primary-model yolo
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # chỉ lưu file, không cần giao diện (tránh lỗi Tk trên Windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# =========================================================
#                        CONFIG
# =========================================================
AGG_DIR = r"C:\Users\FPTSHOP\Desktop\KLTN_AI_COACH\test_posemodal_rtmpose\_aggregated"
OUT_DIR = r"C:\Users\FPTSHOP\Desktop\KLTN_AI_COACH\test_posemodal_rtmpose\_figures"

# Suy ra loại động tác từ tên video (khớp đầu tiên thắng). Sửa regex cho đúng quy ước đặt tên của bạn.
ACTIVITY_RULES: List[Tuple[str, str]] = [
    ("sprint_start", r"sprint|start|xuat[_-]?phat"),
    ("running", r"run"),
]
COACH_REGEX = r"(HLV\d+)"

# ---- GATE: không đạt = REJECT (dữ liệu kỹ thuật không đủ tin cậy, chưa bàn tới kỹ thuật) ----
# (cột, kiểu, ngưỡng, mã, thông điệp)
GATES = [
    ("pct_detected", "min", 0.85, "LOW_DETECT", "tỉ lệ frame detect được thấp"),
    ("pct_valid", "min", 0.70, "LOW_VALID", "tỉ lệ frame hợp lệ thấp"),
    ("longest_undetected_gap_s", "max", 1.0, "LONG_GAP", "có đoạn mất khung quá dài (s)"),
    ("duration_s", "min", 2.0, "TOO_SHORT", "video quá ngắn (s)"),
]
FLAG_GATES = {  # cờ qc_flags từ extract_pose_dataset.py -> REJECT
    "VFR_SUSPECT": ("VFR", "fps thay đổi (VFR) làm sai vận tốc/cadence"),
    "FPS_ASSUMED": ("FPS_ASSUMED", "không đọc được fps, đang giả định 30"),
}

# ---- ĐIỂM KỸ THUẬT: tiêu chí theo loại động tác (GIẢ ĐỊNH, cần HLV/giáo trình xác nhận) ----
@dataclass
class Criterion:
    name: str
    col: str
    kind: str                 # "min" | "max" | "range"
    value: object             # số hoặc (lo, hi)
    weight: float = 1.0
    optional: bool = False    # True: thiếu dữ liệu thì bỏ qua thay vì tính 0 điểm


SOFT: Dict[str, List[Criterion]] = {
    "running": [
        Criterion("knee_rom_L", "angle_knee_left_rom", "min", 55.0, 1.0),
        Criterion("knee_rom_R", "angle_knee_right_rom", "min", 55.0, 1.0),
        # Đối xứng theo ROM từng chân (KHÔNG dùng chênh lệch góc tức thời vì khi chạy 2 chân lệch pha)
        Criterion("rom_symmetry_knee", "rom_symmetry_knee", "max", 0.15, 1.5),
        Criterion("cadence_spm", "cadence_spm", "range", (150.0, 195.0), 1.0),
        Criterion("xmodel_knee_rom", "xmodel_knee_rom_diff", "max", 0.15, 1.0, optional=True),
    ],
    "sprint_start": [
        Criterion("static_duration_s", "static_duration_s", "min", 1.0, 1.5),
        Criterion("launch_prominence", "launch_prominence_ratio", "min", 1.3, 1.5),
        Criterion("xmodel_static_knee_front", "xmodel_static_knee_front_diff", "max", 10.0, 1.0, optional=True),
    ],
}
MIN_STEP_PEAKS = 6          # ít đỉnh hơn mức này thì cadence không đáng tin
ACCEPT_SCORE = 75.0
REJECT_SCORE = 50.0

# ---- OUTLIER ----
OUTLIER_METRICS: Dict[str, List[str]] = {
    "running": ["cadence_spm", "knee_rom_mean", "rom_symmetry_knee", "angle_hip_left_rom",
                "angle_hip_right_rom", "trunk_lean_signed_deg_mean",
                "angle_knee_left_mean", "angle_knee_right_mean"],
    "sprint_start": ["static_angle_knee_front_mean", "static_angle_knee_back_mean",
                     "static_ratio_hip_above_shoulder_mean", "static_ratio_wrist_width_mean",
                     "static_angle_elbow_left_mean", "static_angle_elbow_right_mean",
                     "static_angle_trunk_mean", "launch_angle_trunk"],
}
OUTLIER_Z = 3.5            # ngưỡng modified z-score (Iglewicz-Hoaglin)
MIN_GROUP_FOR_OUTLIER = 5
MIN_REF_VIDEOS = 8         # số video ACCEPT tối thiểu / loại động tác để khoảng chuẩn thống kê ổn định (giả định)

# ---- Đối chiếu giáo trình (đồng bộ với fixed_reference_ranges trong sprint_start.yaml) ----
TEXTBOOK = {
    "static_angle_knee_front_mean": (85.0, 95.0, "Gối chân trước (°)"),
    "static_angle_knee_back_mean": (120.0, 140.0, "Gối chân sau (°)"),
    "static_ratio_hip_above_shoulder_mean": (0.22, 0.44, "Hông cao hơn vai (tỉ lệ)"),
    "static_angle_elbow_left_mean": (165.0, 180.0, "Khuỷu trái (°)"),
    "static_angle_elbow_right_mean": (165.0, 180.0, "Khuỷu phải (°)"),
    "static_ratio_wrist_width_mean": (0.45, 0.75, "Độ rộng 2 tay (tỉ lệ)"),
    "static_ratio_hand_ground_clearance_mean": (-0.05, 0.15, "Tay cách đất (tỉ lệ)"),
    "launch_angle_trunk": (35.0, 55.0, "Góc thân lúc bật (° so với phương đứng)"),
}
STATIC_VEL_MAX = 0.6

MODEL_COLORS = {"rtmpose": "#1f77b4", "yolo": "#1f77b4", "mediapipe": "#ff7f0e"}
STATUS_COLORS = {"ACCEPT": "#2ca02c", "REVIEW": "#ffbf00", "REJECT": "#d62728"}
ACTIVITY_MARKERS = {"running": "o", "sprint_start": "s", "unknown": "^"}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

_OUT = Path(OUT_DIR)


# =========================================================
#                      TIỆN ÍCH
# =========================================================
def save_fig(fig, name: str) -> None:
    path = _OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")


def fmt(v, nd: int = 2) -> str:
    return "NaN" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def infer_activity(video_id: str) -> str:
    s = video_id.lower()
    for act, pat in ACTIVITY_RULES:
        if re.search(pat, s):
            return act
    return "unknown"


def shade_runs(ax, t: np.ndarray, mask: np.ndarray, color: str, alpha: float, label: Optional[str] = None) -> None:
    """Tô nền các đoạn liên tục mask==True."""
    if len(t) < 2 or not mask.any():
        return
    dt = float(np.median(np.diff(t)))
    i, n, first = 0, len(mask), True
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            ax.axvspan(t[i] - dt / 2, t[j - 1] + dt / 2, color=color, alpha=alpha,
                       label=label if first else None, lw=0)
            first = False
            i = j
        else:
            i += 1


def box_points(ax, d: pd.DataFrame, metric: str, group_col: str, groups: Sequence[str],
               hue_col: str, hues: Sequence[str], colors: Dict[str, str]) -> None:
    """Boxplot + điểm rải thủ công (ổn định với n nhỏ, không phụ thuộc phiên bản seaborn)."""
    rng = np.random.default_rng(0)
    width = 0.8 / max(len(hues), 1)
    for gi, g in enumerate(groups):
        for hi, h in enumerate(hues):
            vals = d[(d[group_col] == g) & (d[hue_col] == h)][metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            pos = gi + (hi - (len(hues) - 1) / 2) * width
            ax.boxplot([vals], positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=colors.get(h, "gray"), alpha=0.35),
                       medianprops=dict(color="black"))
            ax.scatter(pos + rng.uniform(-width * 0.25, width * 0.25, len(vals)), vals, s=16,
                       color=colors.get(h, "gray"), edgecolor="k", linewidth=0.3, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


# =========================================================
#                      NẠP DỮ LIỆU
# =========================================================
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if {"angle_knee_left_rom", "angle_knee_right_rom"} <= set(df.columns):
        l, r = df["angle_knee_left_rom"], df["angle_knee_right_rom"]
        df["knee_rom_mean"] = (l + r) / 2.0
        df["rom_symmetry_knee"] = (l - r).abs() / df["knee_rom_mean"].where(df["knee_rom_mean"] > 1e-6)
    return df


def add_cross_model(df: pd.DataFrame) -> pd.DataFrame:
    """Độ lệch giữa các model CHỈ khi có đủ 2 model. Nếu chỉ có rtmpose → bỏ qua."""
    models = set(df["model"].unique())
    if len(models) < 2:
        return df
    # Chỉ so sánh YOLO vs MediaPipe nếu cả 2 có mặt (giữ nguyên logic cũ)
    if not {"yolo", "mediapipe"} <= models:
        return df
    specs = [("knee_rom_mean", "xmodel_knee_rom_diff", "rel"),
             ("static_angle_knee_front_mean", "xmodel_static_knee_front_diff", "abs")]
    for src, dst, kind in specs:
        if src not in df.columns:
            continue
        p = df.pivot_table(index="video_id", columns="model", values=src, aggfunc="first")
        if not {"yolo", "mediapipe"} <= set(p.columns):
            continue
        diff = (p["yolo"] - p["mediapipe"]).abs()
        if kind == "rel":
            diff = diff / ((p["yolo"] + p["mediapipe"]) / 2.0).where(lambda s: s > 1e-6)
        df[dst] = df["video_id"].map(diff)
    return df


def load_data(agg_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    agg = Path(agg_dir)
    sp = agg / "summary_video_level.csv"
    if not sp.exists():
        raise FileNotFoundError(f"Thiếu {sp}. Hãy chạy xong extract_pose_dataset.py trước.")
    df = pd.read_csv(sp)
    missing = [c for c in ("video_id", "model", "pct_detected", "pct_valid") if c not in df.columns]
    if missing:
        raise ValueError(f"summary_video_level.csv thiếu cột bắt buộc: {missing}")

    n0 = len(df)
    df = df.drop_duplicates(["video_id", "model"], keep="last").reset_index(drop=True)
    if len(df) < n0:
        print(f"[LOAD] Bỏ {n0 - len(df)} dòng trùng (video_id, model) - giữ dòng cuối")
    if "qc_flags" not in df.columns:
        df["qc_flags"] = ""
    df["qc_flags"] = df["qc_flags"].fillna("").astype(str)
    df["coach_id"] = df["video_id"].str.extract(COACH_REGEX, flags=re.IGNORECASE, expand=False).fillna("UNKNOWN")
    df["activity"] = df["video_id"].map(infer_activity)
    df = add_cross_model(add_derived(df))

    print(f"[LOAD] summary: {len(df)} dòng | {df['video_id'].nunique()} video | "
          f"{df['coach_id'].nunique()} HLV | model: {sorted(df['model'].unique())}")
    print(f"[LOAD] loại động tác: {df.drop_duplicates('video_id')['activity'].value_counts().to_dict()}")

    dfm = None
    mp = agg / "metrics_frame_level.csv"
    if mp.exists():
        want = ["video_id", "model", "frame", "timestamp", "detected", "is_valid", "com_vel_norm",
                "angle_knee_left_sm", "angle_knee_right_sm", "angle_knee_front_sm", "angle_knee_back_sm"]
        have = pd.read_csv(mp, nrows=0).columns
        dfm = pd.read_csv(mp, usecols=[c for c in want if c in have], low_memory=False)
        for c in ("detected", "is_valid"):
            if c in dfm.columns:
                dfm[c] = dfm[c].astype(str).str.lower().isin(["true", "1", "1.0"])
        print(f"[LOAD] metrics (rút gọn cột): {len(dfm):,} dòng")
    else:
        print("[LOAD] Không có metrics_frame_level.csv -> bỏ qua các biểu đồ theo thời gian")
    return df, dfm


def pick_primary_model(df: pd.DataFrame, requested: str) -> str:
    models = sorted(df["model"].unique())
    if "rtmpose" in models:
        print(f"[MODEL] Dùng model chính: rtmpose")
        return "rtmpose"
    med = df.groupby("model")["pct_valid"].median()
    print("[MODEL] median pct_valid: " + ", ".join(f"{m}={v:.2f}" for m, v in med.items()))
    if requested != "auto":
        if requested in models:
            return requested
        print(f"[MODEL] '{requested}' không có trong dữ liệu -> chọn tự động")
    chosen = med.idxmax()
    print(f"[MODEL] Model chính (tự chọn theo pct_valid cao hơn): {chosen}")
    return chosen

# =========================================================
#              GATE + ĐIỂM + OUTLIER + QUYẾT ĐỊNH
# =========================================================
def eval_gates(row: pd.Series) -> List[Tuple[str, str]]:
    fails = []
    for col, kind, thr, code, msg in GATES:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v) or (kind == "min" and v < thr) or (kind == "max" and v > thr):
            fails.append((code, f"{msg}: {col}={fmt(v)} (yêu cầu {'≥' if kind == 'min' else '≤'} {thr})"))
    flags = set(row["qc_flags"].split(";"))
    for flag, (code, msg) in FLAG_GATES.items():
        if flag in flags:
            fails.append((code, msg))
    return fails


def soft_score(v, kind: str, val) -> float:
    """0..1, giảm tuyến tính khi lệch khỏi ngưỡng."""
    if v is None or pd.isna(v):
        return 0.0
    if kind == "min":
        return 1.0 if v >= val else max(0.0, v / val)
    if kind == "max":
        return 1.0 if v <= val else max(0.0, 1.0 - (v - val) / val)
    lo, hi = val
    if lo <= v <= hi:
        return 1.0
    return max(0.0, 1.0 - min(abs(v - lo), abs(v - hi)) / (0.5 * (hi - lo)))


def eval_soft(row: pd.Series, activity: str) -> Tuple[float, List[str]]:
    num = den = 0.0
    reasons: List[str] = []
    for c in SOFT.get(activity, []):
        if c.col not in row.index:
            continue
        v = row[c.col]
        if c.optional and pd.isna(v):
            continue
        if c.name == "cadence_spm" and not (row.get("n_step_peaks", np.nan) >= MIN_STEP_PEAKS):
            v = np.nan
            reasons.append(f"cadence không tin cậy (n_step_peaks={fmt(row.get('n_step_peaks', np.nan), 0)} < {MIN_STEP_PEAKS})")
        s = soft_score(v, c.kind, c.value)
        num += c.weight * s
        den += c.weight
        if s < 0.5 and not (c.name == "cadence_spm" and pd.isna(v)):
            reasons.append(f"{c.name}={fmt(v)} (yêu cầu {c.kind} {c.value})")
    return (100.0 * num / den if den > 0 else float("nan")), reasons


def compute_outliers(vt: pd.DataFrame, prim: pd.DataFrame) -> pd.DataFrame:
    """Modified z-score theo từng loại động tác, chỉ tính trên video đã qua GATE."""
    cols = sorted({c for v in OUTLIER_METRICS.values() for c in v})
    z = pd.DataFrame(np.nan, index=vt["video_id"], columns=cols)
    base = prim.set_index("video_id")
    for act, g in vt[vt["gate_ok"]].groupby("activity"):
        if len(g) < MIN_GROUP_FOR_OUTLIER:
            continue
        for c in OUTLIER_METRICS.get(act, []):
            if c not in base.columns:
                continue
            x = base.loc[g["video_id"], c]
            med = x.median()
            mad = (x - med).abs().median()
            if not np.isfinite(mad) or mad < 1e-9:
                continue
            z.loc[g["video_id"], c] = 0.6745 * (x - med) / mad
    return z.dropna(axis=1, how="all")


def build_decisions(df: pd.DataFrame, primary: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    multi = set()
    if "n_persons_max" in df.columns and "yolo" in df["model"].values:
        y = df[df["model"] == "yolo"]
        multi = set(y.loc[y["n_persons_max"] > 1, "video_id"])

    prim = df[df["model"] == primary].copy()
    models = sorted(df["model"].unique())
    keep = ["pct_detected", "pct_valid", "duration_s", "fps", "n_persons_max", "cadence_spm", "n_step_peaks",
            "knee_rom_mean", "rom_symmetry_knee", "static_duration_s", "static_start_s", "static_end_s",
            "launch_time_s", "launch_prominence_ratio", "xmodel_knee_rom_diff", "xmodel_static_knee_front_diff"]

    rows = []
    for vid, g in df.groupby("video_id"):
        base = {"video_id": vid, "coach_id": g["coach_id"].iloc[0], "activity": g["activity"].iloc[0],
                "primary_model": primary}
        for m in models:  # gate từng model -> biết model còn lại có dùng được không
            gm = g[g["model"] == m]
            base[f"gate_ok_{m}"] = bool(len(gm) and not eval_gates(gm.iloc[0]))
        gp = g[g["model"] == primary]
        if gp.empty:
            rows.append({**base, "gate_ok": False, "gate_codes": "NO_PRIMARY_DATA", "gate_msgs": "model chính không có dữ liệu cho video này",
                         "score": np.nan, "soft_reasons": "", "warn_codes": ""})
            continue
        r = gp.iloc[0]
        fails = eval_gates(r)
        score, soft_reasons = (float("nan"), []) if fails else eval_soft(r, base["activity"])
        warns = []
        if vid in multi:
            warns.append("MULTI_PERSON")
        if base["activity"] == "unknown":
            warns.append("UNKNOWN_ACTIVITY")
        rows.append({**base, **{k: r[k] for k in keep if k in r.index},
                     "gate_ok": not fails, "gate_codes": ";".join(c for c, _ in fails),
                     "gate_msgs": " | ".join(m for _, m in fails), "score": score,
                     "soft_reasons": " | ".join(soft_reasons), "warn_codes": ";".join(warns)})
    vt = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)

    z = compute_outliers(vt, prim)
    out_list = []
    for vid in vt["video_id"]:
        zr = z.loc[vid].dropna() if vid in z.index else pd.Series(dtype=float)
        out_list.append(";".join(f"{c}({v:+.1f})" for c, v in zr.items() if abs(v) > OUTLIER_Z))
    vt["outliers"] = out_list

    def decide(r) -> str:
        if not r["gate_ok"]:
            return "REJECT"
        if pd.isna(r["score"]):
            return "REVIEW"
        if r["score"] < REJECT_SCORE:
            return "REJECT"
        if r["score"] >= ACCEPT_SCORE and not r["warn_codes"] and not r["outliers"]:
            return "ACCEPT"
        return "REVIEW"

    vt["status"] = vt.apply(decide, axis=1)
    return vt, z


# =========================================================
#                      BIỂU ĐỒ
# =========================================================
def plot_qc_overview(df: pd.DataFrame, vt: pd.DataFrame) -> None:
    piv = df.pivot_table(index="video_id", columns="model", values=["pct_detected", "pct_valid"], aggfunc="first")
    piv.columns = [f"{m}\n{k.replace('pct_', '')}" for k, m in piv.columns]
    piv = piv.sort_index()
    n = len(piv)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.28 * n + 2)), gridspec_kw={"width_ratios": [2.2, 1]})
    sns.heatmap(piv, annot=n <= 45, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=axes[0],
                cbar_kws={"label": "Tỉ lệ"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 7})
    axes[0].set_title("Tỉ lệ frame detect / valid theo video và model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", labelsize=7)

    codes = pd.Series([c for s in vt["gate_codes"] for c in s.split(";") if c]
                      + [c for s in vt["warn_codes"] for c in s.split(";") if c]
                      + [("OUTLIER" if s else "") for s in vt["outliers"]]).replace("", np.nan).dropna()
    ax = axes[1]
    if len(codes):
        vc = codes.value_counts()
        ax.barh(vc.index, vc.values, color="#d62728")
        ax.invert_yaxis()
        ax.set_xlabel("Số video")
    else:
        ax.text(0.5, 0.5, "Không có cờ lỗi/cảnh báo", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Mã lỗi (GATE) / cảnh báo")
    sc = vt["status"].value_counts()
    fig.suptitle(f"Tổng quan chất lượng dữ liệu  |  ACCEPT={sc.get('ACCEPT', 0)}  "
                 f"REVIEW={sc.get('REVIEW', 0)}  REJECT={sc.get('REJECT', 0)}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "01_qc_overview.png")



def plot_model_agreement(df: pd.DataFrame) -> None:
    models = set(df["model"].unique())
    if len(models) < 2:
        print("[SKIP] 02: chỉ có 1 model → bỏ qua so sánh cross-model")
        return
    if not {"yolo", "mediapipe"} <= models:
        print(f"[SKIP] 02: cần YOLO + MediaPipe. Hiện có: {sorted(models)}")
        return


def plot_distributions(df: pd.DataFrame) -> None:
    cands = ["pct_valid", "cadence_spm", "knee_rom_mean", "rom_symmetry_knee", "angle_hip_left_rom",
             "trunk_lean_signed_deg_mean", "hip_height_rel_ankle_mean", "com_vel_norm_mean",
             "static_duration_s", "launch_prominence_ratio"]
    metrics = [m for m in cands if m in df.columns and df[m].notna().sum() >= 3]
    if not metrics:
        return
    acts = [a for a in ("running", "sprint_start", "unknown") if a in set(df["activity"])]
    models = sorted(df["model"].unique())
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        box_points(ax, df, m, "activity", acts, "model", models, MODEL_COLORS)
        ax.set_title(m)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.legend(handles=[Patch(facecolor=MODEL_COLORS.get(m, "gray"), alpha=0.5, label=m) for m in models],
               loc="upper right")
    fig.suptitle("Phân bố metric theo loại động tác và model (mỗi chấm = 1 video)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_distributions.png")


def plot_outlier_heatmap(z: pd.DataFrame, vt: pd.DataFrame) -> None:
    if z.empty:
        print(f"[SKIP] 04: chưa đủ video (≥{MIN_GROUP_FOR_OUTLIER}/loại động tác, đã qua GATE) để phát hiện outlier")
        return
    order = vt.sort_values(["activity", "video_id"])
    order = order[order["video_id"].isin(z.dropna(how="all").index)]
    zz = z.loc[order["video_id"]]
    labels = [f"[{a[:3]}] {v}" for a, v in zip(order["activity"], order["video_id"])]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * zz.shape[1] + 4), max(5, 0.3 * len(zz) + 2)))
    sns.heatmap(zz.clip(-6, 6), cmap="coolwarm", center=0, vmin=-6, vmax=6, ax=ax, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "modified z-score"}, yticklabels=labels)
    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            v = zz.iat[i, j]
            if pd.notna(v) and abs(v) > OUTLIER_Z:
                ax.text(j + 0.5, i + 0.5, "✱", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Video lệch bất thường so với nhóm cùng loại (✱ = |z| > {OUTLIER_Z})")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "04_outliers.png")


def plot_coach_consistency(df: pd.DataFrame, vt: pd.DataFrame, primary: str) -> None:
    prim = df[df["model"] == primary].merge(vt[["video_id", "status"]], on="video_id")
    for act, metrics in (("running", ["knee_rom_mean", "rom_symmetry_knee", "cadence_spm", "trunk_lean_signed_deg_mean"]),
                         ("sprint_start", ["static_angle_knee_front_mean", "static_angle_knee_back_mean",
                                           "static_ratio_hip_above_shoulder_mean", "launch_angle_trunk"])):
        d = prim[prim["activity"] == act]
        metrics = [m for m in metrics if m in d.columns and d[m].notna().sum() >= 2]
        coaches = sorted(d["coach_id"].unique())
        if len(d) < 3 or not metrics:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.4), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for gi, c in enumerate(coaches):
                sub = d[d["coach_id"] == c]
                vals = sub[m].dropna()
                if len(vals) >= 2:
                    ax.boxplot([vals.to_numpy()], positions=[gi], widths=0.5, showfliers=False,
                               boxprops=dict(color="gray"), medianprops=dict(color="black"))
                ax.scatter(np.full(len(sub), gi) + np.random.default_rng(gi).uniform(-0.12, 0.12, len(sub)),
                           sub[m], c=[STATUS_COLORS[s] for s in sub["status"]], s=36, edgecolor="k",
                           linewidth=0.4, zorder=3)
            ax.set_xticks(range(len(coaches)))
            ax.set_xticklabels(coaches, rotation=35, ha="right")
            ax.set_title(m)
        fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in STATUS_COLORS.items()], loc="upper right")
        fig.suptitle(f"Nhất quán GIỮA và TRONG HLV - {act} (model {primary}). Hộp hẹp = HLV ổn định; "
                     "các hộp lệch nhau = khác phong cách", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, f"05_coach_consistency_{act}.png")


def plot_ranking(vt: pd.DataFrame) -> None:
    order = {"ACCEPT": 0, "REVIEW": 1, "REJECT": 2}
    d = vt.assign(_o=vt["status"].map(order), _s=vt["score"].fillna(-1)).sort_values(["_o", "_s"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(13, max(6, 0.32 * len(d) + 1.5)))
    y = np.arange(len(d))
    ax.barh(y, d["score"].fillna(0), color=[STATUS_COLORS[s] for s in d["status"]], edgecolor="black", linewidth=0.4)
    for yi, (_, r) in zip(y, d.iterrows()):
        txt = fmt(r["score"], 1) if pd.notna(r["score"]) else "-"
        note = r["gate_codes"] or r["warn_codes"] or ("OUTLIER" if r["outliers"] else "")
        ax.text(r["score"] + 1 if pd.notna(r["score"]) else 1, yi, f"{txt}  {note}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{a[:3]}] {v}" for a, v in zip(d["activity"], d["video_id"])], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(ACCEPT_SCORE, color="green", ls="--", alpha=0.6)
    ax.axvline(REJECT_SCORE, color="orange", ls="--", alpha=0.6)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Điểm kỹ thuật (0-100). Video bị GATE không có điểm (thanh = 0)")
    ax.set_title("Quyết định theo video: ACCEPT / REVIEW / REJECT")
    ax.legend(handles=[Patch(facecolor=c, label=s) for s, c in STATUS_COLORS.items()], loc="lower right")
    fig.tight_layout()
    save_fig(fig, "06_decision_ranking.png")


def plot_time_series(vt: pd.DataFrame, dfm: Optional[pd.DataFrame], primary: str) -> None:
    if dfm is None or "timestamp" not in dfm.columns:
        print("[SKIP] 07: không có dữ liệu theo frame")
        return
    cand = vt.dropna(subset=["score"]).sort_values("score", ascending=False)
    if cand.empty:
        cand = vt
    picks = [cand.iloc[0]]
    if len(cand) >= 3:
        picks.append(cand.iloc[len(cand) // 2])
    if len(cand) >= 2:
        picks.append(cand.iloc[-1])
    rej = vt[vt["status"] == "REJECT"]
    if len(rej) and rej.iloc[0]["video_id"] not in {p["video_id"] for p in picks}:
        picks.append(rej.iloc[0])

    models = sorted(dfm["model"].unique())
    fig, axes = plt.subplots(len(picks), 2, figsize=(16, 3.5 * len(picks)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1.4]})
    for (a1, a2), r in zip(axes, picks):
        vid, act = r["video_id"], r["activity"]
        kcols = (("angle_knee_front_sm", "front"), ("angle_knee_back_sm", "back")) if act == "sprint_start" \
            else (("angle_knee_left_sm", "L"), ("angle_knee_right_sm", "R"))
        sub_p = dfm[(dfm["video_id"] == vid) & (dfm["model"] == primary)].sort_values("timestamp")
        for m in models:
            s = dfm[(dfm["video_id"] == vid) & (dfm["model"] == m)].sort_values("timestamp")
            for (col, lab), color in zip(kcols, ("#1f77b4", "#d62728")):
                if col in s.columns and len(s):
                    a1.plot(s["timestamp"], s[col], color=color, lw=1.2 if m == primary else 0.8,
                            ls="-" if m == primary else "--", alpha=1 if m == primary else 0.7,
                            label=f"knee_{lab} ({m})")
        if len(sub_p):
            t = sub_p["timestamp"].to_numpy()
            shade_runs(a1, t, ~sub_p["detected"].to_numpy(bool), "gray", 0.35, "không detect")
            shade_runs(a1, t, (~sub_p["is_valid"].to_numpy(bool)) & sub_p["detected"].to_numpy(bool), "orange", 0.25, "invalid")
            if "com_vel_norm" in sub_p.columns:
                a2.plot(t, sub_p["com_vel_norm"], color="k", lw=1)
        a2.axhline(STATIC_VEL_MAX, color="green", ls="--", alpha=0.6, label=f"ngưỡng tĩnh {STATIC_VEL_MAX}")
        if pd.notna(r.get("static_start_s", np.nan)):
            a2.axvspan(r["static_start_s"], r["static_end_s"], color="green", alpha=0.15, label="đoạn tĩnh")
        if pd.notna(r.get("launch_time_s", np.nan)):
            a2.axvline(r["launch_time_s"], color="red", label="gia tốc cực đại")
        a1.set_title(f"[{r['status']}] {vid}  score={fmt(r['score'], 1)}", fontsize=10,
                     color=STATUS_COLORS[r["status"]])
        a1.set_ylabel("Góc gối (°)")
        a1.set_xlabel("Thời gian (s)")
        a1.legend(fontsize=6, ncol=3, loc="upper right")
        a2.set_title("Vận tốc COM / body_scale (1/s)", fontsize=10)
        a2.set_xlabel("Thời gian (s)")
        a2.legend(fontsize=6)
    fig.suptitle("Chuỗi thời gian mẫu (tốt nhất / giữa / kém nhất / 1 video REJECT) - xem mắt bằng chứng cho điểm số",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_time_series_examples.png")


def plot_detection_timeline(vt: pd.DataFrame, dfm: Optional[pd.DataFrame], primary: str, bins: int = 60) -> None:
    if dfm is None or "is_valid" not in dfm.columns:
        print("[SKIP] 08: không có dữ liệu theo frame")
        return
    rows, names = [], []
    d = dfm[dfm["model"] == primary]
    for vid in vt.sort_values("pct_valid")["video_id"]:
        s = d[d["video_id"] == vid].sort_values("frame")
        if s.empty:
            continue
        chunks = np.array_split(s["is_valid"].to_numpy(float), min(bins, len(s)))
        vals = np.full(bins, np.nan)
        vals[:len(chunks)] = [c.mean() for c in chunks]
        rows.append(vals)
        names.append(vid)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(13, max(5, 0.28 * len(rows) + 2)))
    sns.heatmap(np.vstack(rows), cmap="RdYlGn", vmin=0, vmax=1, ax=ax, yticklabels=names,
                cbar_kws={"label": "Tỉ lệ frame hợp lệ"})
    ax.set_xlabel("Tiến trình video (0% → 100%)")
    ax.set_xticks([])
    ax.set_title(f"Dòng thời gian hợp lệ theo video (model {primary}). Vệt đỏ = mất khung/invalid. "
                 "Xếp từ video kém nhất lên trên")
    plt.yticks(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "08_validity_timeline.png")


def plot_textbook_check(df: pd.DataFrame, vt: pd.DataFrame, primary: str) -> None:
    d = df[(df["model"] == primary) & (df["activity"] == "sprint_start")].merge(vt[["video_id", "status"]], on="video_id")
    metrics = [m for m in TEXTBOOK if m in d.columns and d[m].notna().sum() >= 1]
    if d.empty or not metrics:
        print("[SKIP] 09: không có video sprint_start / cột static_*, launch_*")
        return
    ncols = 4
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        lo, hi, title = TEXTBOOK[m]
        ax.axhspan(lo, hi, color="green", alpha=0.15, label="khoảng giáo trình")
        vals = d[m]
        inside = ((vals >= lo) & (vals <= hi)).sum()
        ax.scatter(np.arange(len(d)), vals, c=[STATUS_COLORS[s] for s in d["status"]], s=42, edgecolor="k", linewidth=0.4)
        ax.set_title(f"{title}\ntrong khoảng: {inside}/{int(vals.notna().sum())}", fontsize=9)
        ax.set_xticks([])
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Video xuất phát thấp vs khoảng chuẩn giáo trình (dải xanh). Nếu đa số video HLV nằm NGOÀI dải: "
                 "kiểm tra lại quy đổi (vd body_scale~45cm) hoặc chất lượng video", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "09_textbook_check.png")


# =========================================================
#                   BÁO CÁO + XUẤT FILE
# =========================================================
def write_outputs(df: pd.DataFrame, vt: pd.DataFrame, primary: str) -> None:
    L: List[str] = []
    add = L.append
    add("=" * 72)
    add("BÁO CÁO ĐÁNH GIÁ VIDEO HLV (RTMPose-m ONNX + YOLO/ByteTrack)")
    add("=" * 72)
    add(f"Số video: {len(vt)} | HLV: {vt['coach_id'].nunique()} | model chính: {primary}")
    add("Loại động tác: " + ", ".join(f"{k}={v}" for k, v in vt["activity"].value_counts().items()))
    add("Ngưỡng GATE: " + "; ".join(f"{c} {'≥' if k == 'min' else '≤'} {t}" for c, k, t, _, _ in GATES))
    add(f"Quyết định: ACCEPT nếu điểm ≥ {ACCEPT_SCORE:.0f} và không cảnh báo/outlier; REJECT nếu GATE hỏng hoặc điểm < {REJECT_SCORE:.0f}")
    add("LƯU Ý: ngưỡng điểm kỹ thuật là GIẢ ĐỊNH ban đầu - hãy đối chiếu với HLV/giáo trình và xem biểu đồ 07 bằng mắt.")
    add("")

    med = df.groupby("model")[["pct_detected", "pct_valid"]].median()
    add("Chất lượng theo model (median): " + " | ".join(
        f"{m}: detect={r.pct_detected:.2f}, valid={r.pct_valid:.2f}" for m, r in med.iterrows()))
    add("")

    for status in ("ACCEPT", "REVIEW", "REJECT"):
        d = vt[vt["status"] == status].sort_values("score", ascending=False)
        add(f"--- {status}: {len(d)} video ---")
        for _, r in d.iterrows():
            reasons = " | ".join(x for x in (r["gate_msgs"], r["soft_reasons"],
                                             ("cảnh báo: " + r["warn_codes"]) if r["warn_codes"] else "",
                                             ("outlier: " + r["outliers"]) if r["outliers"] else "") if x)
            alt = [c.replace("gate_ok_", "") for c in vt.columns if c.startswith("gate_ok_")
                   and c != f"gate_ok_{primary}" and r[c]]
            if status == "REJECT" and not r["gate_ok"] and alt:
                reasons += f" | (model {', '.join(alt)} đạt GATE - thử dùng model đó)"
            add(f"  • {r['video_id']:42s} [{r['activity'][:6]}] score={fmt(r['score'], 1):>5}  {reasons}")
        add("")

    add("--- ĐỘ SẴN SÀNG XÂY KHOẢNG CHUẨN (video ACCEPT) ---")
    acc = vt[vt["status"] == "ACCEPT"]
    for act in ("running", "sprint_start"):
        a = acc[acc["activity"] == act]
        total = (vt["activity"] == act).sum()
        if total == 0:
            continue
        per_coach = a["coach_id"].value_counts().to_dict()
        warn = ""
        if len(a) < MIN_REF_VIDEOS:
            warn = (f"  ⚠ ít hơn {MIN_REF_VIDEOS} -> khoảng chuẩn thống kê (mỗi video chỉ 1 điểm) kém ổn định; "
                    "cân nhắc khoảng chuẩn cố định hoặc build_mixed()")
        if len(per_coach) == 1 and len(a):
            warn += "  ⚠ chỉ 1 HLV -> khoảng chuẩn phản ánh phong cách của 1 người"
        add(f"  {act}: {len(a)}/{total} video | theo HLV: {per_coach}{warn}")
    add("")

    text = "\n".join(L)
    print("\n" + text)
    (_OUT / "evaluation_report.txt").write_text(text, encoding="utf-8")
    vt.to_csv(_OUT / "video_decision.csv", index=False, float_format="%.4f")
    (_OUT / "accepted_videos.txt").write_text("\n".join(vt.loc[vt["status"] == "ACCEPT", "video_id"]) + "\n", encoding="utf-8")
    (_OUT / "review_videos.txt").write_text("\n".join(vt.loc[vt["status"] == "REVIEW", "video_id"]) + "\n", encoding="utf-8")
    print(f"\n[SAVE] {_OUT / 'video_decision.csv'}\n[SAVE] {_OUT / 'accepted_videos.txt'}\n"
          f"[SAVE] {_OUT / 'review_videos.txt'}\n[SAVE] {_OUT / 'evaluation_report.txt'}")


# =========================================================
#                          MAIN
# =========================================================
def run_step(label: str, fn, *args) -> None:
    print(f"\n[{label}]")
    try:
        fn(*args)
    except Exception as e:  # một biểu đồ lỗi không làm hỏng cả phiên đánh giá
        print(f"[LỖI] {label}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def main() -> int:
    global _OUT
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # tránh crash ký tự đặc biệt trên PowerShell
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

    ap = argparse.ArgumentParser(description="Đánh giá video HLV từ dữ liệu đã trích xuất")
    ap.add_argument("--agg", default=AGG_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--primary-model", default="auto", help="auto | yolo | mediapipe")
    a = ap.parse_args()
    _OUT = Path(a.out)
    _OUT.mkdir(parents=True, exist_ok=True)

    df, dfm = load_data(a.agg)
    primary = pick_primary_model(df, a.primary_model)
    vt, z = build_decisions(df, primary)

    run_step("1/9 QC tổng quan", plot_qc_overview, df, vt)
    run_step("2/9 Đồng thuận model", plot_model_agreement, df)
    run_step("3/9 Phân bố", plot_distributions, df)
    run_step("4/9 Outlier", plot_outlier_heatmap, z, vt)
    run_step("5/9 Nhất quán HLV", plot_coach_consistency, df, vt, primary)
    run_step("6/9 Xếp hạng quyết định", plot_ranking, vt)
    run_step("7/9 Chuỗi thời gian mẫu", plot_time_series, vt, dfm, primary)
    run_step("8/9 Timeline hợp lệ", plot_detection_timeline, vt, dfm, primary)
    run_step("9/9 Đối chiếu giáo trình", plot_textbook_check, df, vt, primary)
    write_outputs(df, vt, primary)
    print(f"\nHOÀN TẤT. Mọi kết quả ở: {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())