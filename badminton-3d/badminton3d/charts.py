"""Biểu đồ cho từng video (dashboard, đồ thị góc theo thời gian so với chuẩn, chất lượng dữ liệu)."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from badminton3d.analyze import Context
from badminton3d.segment import READY_STATES

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C_GOOD, C_MID, C_BAD, C_GREY = "#2e9e5b", "#e8a33d", "#d64545", "#9aa0a6"
STATE_COL = {"INVALID": "#d9d9d9", "MOVING": "#f4a6a6", "SETTLING": "#ffe08a", "READY": "#9bd99b"}
PHASE_COL = {"chuan_bi": "#ffe08a", "rut_vot": "#f4c28a", "danh_cau": "#9bd99b", "ket_thuc": "#9bc7e8"}


def short(label: str) -> str:
    return label.split(' (')[0]


def score_color(s: Optional[float]) -> str:
    if s is None:
        return C_GREY
    return C_GOOD if s >= 85 else (C_MID if s >= 60 else C_BAD)


def _timeline_panel(ax, ctx: Context, cfg: dict) -> None:
    p = ctx.pose
    t = np.arange(p.n) / p.fps
    if ctx.exercise == "ready":
        sg = ctx.ready_seg
        inv = {v: k for k, v in READY_STATES.items()}
        for k in range(p.n):
            ax.axvspan(t[k] - 0.5 / p.fps, t[k] + 0.5 / p.fps, color=STATE_COL[inv[sg.state[k]]], lw=0)
        ax.plot(t, sg.speed, color="k", lw=0.9)
        ax.axhline(cfg["ready"]["v_move_mps"], color="#b00", ls=":", lw=0.8)
        ax.set_ylabel("Tốc độ thân (m/s)")
        ax.set_ylim(0, max(1.5, float(np.nanpercentile(sg.speed, 98)) * 1.1))
        ax.legend(handles=[Patch(color=STATE_COL["READY"], label="Đã về tư thế sẵn sàng → ĐƯỢC CHẤM"), Patch(color=STATE_COL["SETTLING"], label="Đang ổn định (chưa chấm)"),
                           Patch(color=STATE_COL["MOVING"], label="Di chuyển đỡ cầu (không chấm)"), Patch(color=STATE_COL["INVALID"], label="Thiếu dữ liệu")],
                  loc="upper right", fontsize=7, ncol=2, frameon=False)
    else:
        sg = ctx.serve_seg
        ax.plot(t, sg.activity, color="tab:red", lw=0.9, label="Hoạt động của tay (m/s)")
        ax.plot(t, sg.body, color="tab:blue", lw=0.9, label="Tốc độ thân (m/s)")
        for r in sg.reps:
            for k, (a, b) in r.phases.items():
                ax.axvspan(a / p.fps, (b + 1) / p.fps, color=PHASE_COL[k], alpha=0.6, lw=0)
            ax.axvline(r.peak / p.fps, color="k", lw=0.8)
        labels = cfg["features"]["serve"]["phase_labels"]
        ax.legend(handles=[Patch(color=PHASE_COL[k], label=v) for k, v in labels.items()] + [plt.Line2D([], [], color="tab:red", label="Tay"), plt.Line2D([], [], color="tab:blue", label="Thân")],
                  loc="upper right", fontsize=7, ncol=3, frameon=False)
        ax.set_ylabel("m/s")
        ax.set_ylim(0, max(1.2, float(np.nanpercentile(sg.activity, 99)) * 1.15))
    ax.set_xlabel("Thời gian (s)")
    ax.set_xlim(0, t[-1])


def dashboard(ctx: Context, res: dict, cfg: dict, path: str) -> None:
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[0.9, 1.3, 1.3], hspace=0.42, wspace=0.55)
    sc = res.get("score")
    overall = sc["overall"] if sc else None
    # (a) điểm tổng
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.5, 0.72, "—" if overall is None else f"{overall:.0f}", ha="center", va="center", fontsize=54, color=score_color(overall), fontweight="bold")
    ax.text(0.5, 0.40, "/ 100", ha="center", fontsize=12, color="#555")
    ax.text(0.5, 0.22, sc["grade"] if sc else res.get("status", ""), ha="center", fontsize=15, color=score_color(overall), fontweight="bold")
    ax.text(0.5, 0.04, f"{res['video']}  ·  {'Sẵn sàng' if res['exercise'] == 'ready' else 'Phát cầu'}", ha="center", fontsize=8, color="#444")
    # (b) điểm từng góc
    ax = fig.add_subplot(gs[0, 1:])
    if sc and sc["units"]:
        if res["exercise"] == "ready":
            names, vals = [], []
            agg: Dict[str, list] = {}
            for u in sc["units"]:
                for f in u["features"]:
                    if f["score"] is not None:
                        agg.setdefault(f["label"], []).append(f["score"])
            for k, v in agg.items():
                names.append(short(k))
                vals.append(float(np.mean(v)))
            order = np.argsort(vals)
            ax.barh([names[i] for i in order], [vals[i] for i in order], color=[score_color(vals[i]) for i in order])
            for y, i in enumerate(order):
                ax.text(vals[i] + 1, y, f"{vals[i]:.0f}", va="center", fontsize=8)
            ax.set_xlim(0, 112)
            ax.set_xlabel("Điểm từng góc (TB các đoạn được chấm)")
        else:
            phs = list(cfg["features"]["serve"]["phase_labels"])
            feats = [k for k in cfg["features"]["serve"] if k not in ("phase_weights", "phase_labels")]
            M = np.full((len(feats), len(phs)), np.nan)
            for i, f in enumerate(feats):
                for j, ph in enumerate(phs):
                    vals = [x["score"] for u in sc["units"] for x in (u["phases"].get(ph, {}).get("features", [])) if x["feature"] == f and x["score"] is not None]
                    if vals:
                        M[i, j] = np.mean(vals)
            im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(len(phs)), [cfg["features"]["serve"]["phase_labels"][p] for p in phs])
            ax.set_yticks(range(len(feats)), [short(cfg["features"]["serve"][f]["label"]) for f in feats])
            for i in range(len(feats)):
                for j in range(len(phs)):
                    ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Điểm")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, res.get("message", "Không có dữ liệu để chấm"), ha="center", va="center", wrap=True, fontsize=9, color="#a00")
    ax.set_title("Điểm từng góc so với chuẩn HLV", fontsize=10, loc="left")
    # (c) timeline
    ax = fig.add_subplot(gs[1, :2])
    _timeline_panel(ax, ctx, cfg)
    ax.set_title("Đoạn nào được chấm?", fontsize=10, loc="left")
    # (d) lỗi cần sửa
    ax = fig.add_subplot(gs[1, 2])
    errs = (sc or {}).get("errors", [])[:6]
    if errs:
        lab = [f"{short(e['label'])}\n{e['where']}" for e in errs][::-1]
        ax.barh(lab, [e["lost_points"] for e in errs][::-1], color=C_BAD)
        for y, e in enumerate(errs[::-1]):
            ax.text(e["lost_points"] + 0.1, y, f"+{e['potential_gain']:.1f}", va="center", fontsize=8, color="#444")
        ax.set_xlabel("Điểm đang bị mất (số '+' = điểm lấy lại được nếu sửa)")
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Không có lỗi vượt dung sai" if sc and sc["units"] else "—", ha="center", va="center", color=C_GOOD, fontsize=11)
    ax.set_title("Cần sửa gì để tăng điểm", fontsize=10, loc="left")
    fig.suptitle(f"Kết quả chấm tư thế — {res['video']}", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def angles_timeline(ctx: Context, reference: Optional[dict], res: dict, cfg: dict, path: str) -> None:
    feats = list(ctx.feats)
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 1.7 * len(feats) + 0.8), sharex=True)
    t = np.arange(ctx.pose.n) / ctx.pose.fps
    spec = cfg["features"][ctx.exercise]
    for ax, name in zip(np.atleast_1d(axes), feats):
        f = ctx.feats[name]
        v = np.where(f["rel"] > 0, f["value"], np.nan)
        ax.plot(t, f["value"], color="#c9ced3", lw=0.8)
        ax.plot(t, v, color="#1f5fbf", lw=1.2)
        sub = f["substituted"] & ~np.isnan(f["value"])
        if sub.any():
            ax.scatter(t[sub], f["value"][sub], s=5, color="#e8a33d", zorder=3)
        if reference:
            if ctx.exercise == "ready" and name in reference["features"]:
                r = reference["features"][name]
                ax.axhspan(r["center"] - r["tol"], r["center"] + r["tol"], color="#2e9e5b", alpha=0.18, lw=0)
                ax.axhline(r["center"], color="#2e9e5b", lw=0.8)
            elif ctx.exercise == "serve":
                for rep in (ctx.serve_seg.reps if ctx.serve_seg else []):
                    for ph, (a, b) in rep.phases.items():
                        r = reference.get("phases", {}).get(ph, {}).get(name)
                        if r:
                            ax.fill_between(t[a:b + 1], r["center"] - r["tol"], r["center"] + r["tol"], color="#2e9e5b", alpha=0.25, lw=0)
        if ctx.exercise == "ready":
            for a, b in ctx.ready_seg.episodes:
                ax.axvspan(t[a], t[b], color="#9bd99b", alpha=0.22, lw=0)
        else:
            for rep in ctx.serve_seg.reps:
                ax.axvspan(t[rep.prep_start], t[rep.end], color="#9bd99b", alpha=0.18, lw=0)
        ax.set_ylabel(f"{spec[name]['label']}\n({spec[name]['unit']})", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Thời gian (s)")
    np.atleast_1d(axes)[0].legend(handles=[plt.Line2D([], [], color="#1f5fbf", label="Góc đo được"), plt.Line2D([], [], color="#c9ced3", label="Không đủ tin cậy"),
                                           plt.Line2D([], [], marker="o", color="#e8a33d", lw=0, markersize=4, label="Dùng bên/tay thay thế"),
                                           Patch(color="#2e9e5b", alpha=0.3, label="Chuẩn HLV ± dung sai"), Patch(color="#9bd99b", alpha=0.4, label="Đoạn được chấm")],
                                  loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=7, frameon=False)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def quality_chart(res: dict, cfg: dict, path: str) -> None:
    q = res["quality"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    jv = q["joint_valid_pct"]
    axs[0].barh(list(jv)[::-1], [jv[j] for j in list(jv)[::-1]], color=[C_GOOD if jv[j] >= 90 else (C_MID if jv[j] >= 60 else C_BAD) for j in list(jv)[::-1]])
    axs[0].set_xlim(0, 105)
    axs[0].set_title("Khớp nào thấy rõ (visibility ≥ ngưỡng)", fontsize=9, loc="left")
    axs[0].tick_params(axis="y", labelsize=7)
    spec = cfg["features"][res["exercise"]]
    fv, fs = q["feature_valid_pct"], q["feature_substituted_pct"]
    names = list(fv)[::-1]
    axs[1].barh([spec[n]["label"] for n in names], [fv[n] for n in names], color="#1f5fbf", label="Valid theo từng góc")
    axs[1].barh([spec[n]["label"] for n in names], [fv[n] * fs[n] / 100 for n in names], color="#e8a33d", label="Trong đó dùng bên/tay thay thế")
    axs[1].set_xlim(0, 105)
    axs[1].legend(fontsize=7, loc="lower right", frameon=False)
    axs[1].set_title(f"Valid theo từng góc (cũ: 10 khớp cùng lúc = {q['legacy_valid_pct']:.0f}%)", fontsize=9, loc="left")
    axs[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
