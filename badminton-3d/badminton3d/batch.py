"""Chạy cả thư mục Input, dựng chuẩn HLV, đo độ phủ (Valid theo từng góc) và kiểm chứng chéo (bỏ-một-video) + biểu đồ tổng hợp."""
from __future__ import annotations

import base64
import html
import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from badminton3d import __version__
from badminton3d.analyze import Context, analyze_context, build_context, exercise_of, load_annotation, role_of
from badminton3d.charts import C_BAD, C_GOOD, C_GREY, C_MID, score_color
from badminton3d.coverage import coverage_for_video, summarize
from badminton3d.pose_io import PoseData
from badminton3d.reference import build_reference, save_reference
from badminton3d.runner import get_pose, process_video, try_reference
from badminton3d.scoring import grade

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def list_videos(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Không thấy thư mục: {folder}")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXT))


def _ctx_for(video: str, cfg: dict, out_root: str, racket_hand: Optional[str]) -> Optional[Context]:
    stem = os.path.splitext(os.path.basename(video))[0]
    ex = exercise_of(stem)
    if ex is None:
        return None
    ann = load_annotation(video)
    df = get_pose(video, cfg, os.path.join(out_root, stem))
    return build_context(df, cfg, ex, ann.get("racket_hand") or racket_hand or cfg["camera_protocol"]["racket_hand"], ann.get("serve_windows_s"))


def build_reference_cmd(exercise: str, videos: List[str], cfg: dict, out_root: str, reference_dir: str, racket_hand: Optional[str] = None) -> dict:
    """Dựng chuẩn từ các video HLV (nên chọn 1–3 video tập ĐÚNG)."""
    units, arm_side, yaw = {}, {}, {}
    for v in videos:
        ctx = _ctx_for(v, cfg, out_root, racket_hand)
        stem = os.path.splitext(os.path.basename(v))[0]
        if ctx is None or ctx.exercise != exercise:
            raise ValueError(f"{stem}: không phải video bài '{exercise}'")
        if not ctx.units:
            print(f"  [bỏ qua] {stem}: không có đoạn nào để chấm (ready: di chuyển liên tục; serve: không thấy cú phát)")
            continue
        units[stem] = ctx.units
        if ctx.arm:
            arm_side[stem] = ctx.arm["side"]
        from badminton3d.signals import view_yaw_deg
        yaw[stem] = round(view_yaw_deg(ctx.clean), 1)
        print(f"  [dùng]   {stem}: {len(ctx.units)} đơn vị")
    if not units:
        raise RuntimeError("Không video nào có đơn vị để dựng chuẩn.")
    ref = build_reference(exercise, units, cfg, {"arm_side": arm_side, "view_yaw_deg": yaw, "racket_hand": racket_hand or cfg["camera_protocol"]["racket_hand"]})
    save_reference(ref, os.path.join(reference_dir, f"{exercise}.json"))
    return ref


# ------------------------------------------------------------------ độ phủ
def run_coverage(videos: List[str], cfg: dict, out_root: str, batch_dir: str, racket_hand: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for v in videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        ex = exercise_of(stem)
        if ex is None:
            continue
        pose = PoseData.from_df(get_pose(v, cfg, os.path.join(out_root, stem)))
        ann = load_annotation(v)
        rows += coverage_for_video(pose, ex, cfg, ann.get("racket_hand") or racket_hand or cfg["camera_protocol"]["racket_hand"])
    det = pd.DataFrame(rows)
    os.makedirs(batch_dir, exist_ok=True)
    det.to_csv(os.path.join(batch_dir, "coverage_detail.csv"), index=False)
    summarize(det).to_csv(os.path.join(batch_dir, "coverage_summary.csv"), index=False)
    chart_coverage(det, batch_dir)
    return det


def chart_coverage(det: pd.DataFrame, batch_dir: str) -> None:
    s = summarize(det)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [6, 5]})
    for ax, ex in zip(axes, ("ready", "serve")):
        d = s[s.exercise == ex]
        x = np.arange(len(d))
        w = 0.2
        for k, (col, lab, c) in enumerate([("B0", "Cũ: 10 khớp cùng lúc", "#b0b7bf"), ("S1", "Valid theo từng góc", "#7fb3e6"),
                                           ("S2", "+ nội suy khoảng mất ngắn", "#3d84cc"), ("S3", "+ bên/tay thay thế", "#1b5aa8")]):
            ax.bar(x + (k - 1.5) * w, d[col], w, label=lab, color=c)
        ax.set_xticks(x, [v.replace("coach_", "").replace("trainee_", "T:") for v in d.video], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("% frame dùng được")
        m = d[["B0", "S1", "S2", "S3"]].mean()
        ax.set_title(("Bài sẵn sàng (ready)" if ex == "ready" else "Bài phát cầu (serve)") + "   |   TB: " + " → ".join(f"{m[c]:.0f}%" for c in ("B0", "S1", "S2", "S3")), loc="left", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Độ phủ dữ liệu: từ %Valid cũ đến Valid theo từng góc + xử lý mất khớp", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(batch_dir, "coverage_stages.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    for stage, title, fname in (("S1", "Valid THÔ theo từng góc (chưa xử lý mất khớp)", "coverage_heatmap_raw.png"), ("S3", "Valid sau xử lý mất khớp (góc dùng để chấm)", "coverage_heatmap_final.png")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
        for ax, ex in zip(axes, ("ready", "serve")):
            d = det[(det.exercise == ex) & (det.stage == stage)]
            if d.empty:
                ax.axis("off")
                continue
            M = d.pivot(index="name", columns="video", values="pct")
            im = ax.imshow(M.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(M.shape[1]), [c.replace("coach_", "").replace("trainee_", "T:") for c in M.columns], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(M.shape[0]), list(M.index), fontsize=8)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
            ax.set_title("ready" if ex == "ready" else "serve", loc="left", fontsize=10)
        fig.suptitle(title, fontweight="bold", x=0.01, ha="left")
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="% frame")
        fig.savefig(os.path.join(batch_dir, fname), dpi=130, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ kiểm chứng chéo + biểu đồ tổng hợp
def run_evaluation(videos: List[str], cfg: dict, out_root: str, reference_dir: str, batch_dir: str, racket_hand: Optional[str] = None) -> pd.DataFrame:
    """Điểm KHÔNG LẠC QUAN: video nằm trong tập dựng chuẩn thì chấm bằng chuẩn dựng từ CÁC VIDEO CÒN LẠI (bỏ-một-video)."""
    ctxs: Dict[str, Context] = {}
    for v in videos:
        c = _ctx_for(v, cfg, out_root, racket_hand)
        if c is not None:
            ctxs[os.path.splitext(os.path.basename(v))[0]] = c
    rows, loo_detail = [], []
    for ex in ("ready", "serve"):
        ref_full = try_reference(reference_dir, ex)
        if ref_full is None:
            continue
        members = [s for s in ref_full["built_from"] if s in ctxs]
        for stem, c in ctxs.items():
            if c.exercise != ex:
                continue
            in_ref = stem in members
            if in_ref:
                others = {s: ctxs[s].units for s in members if s != stem and ctxs[s].units}
                if not others:
                    rows.append({"video": stem, "role": role_of(stem), "exercise": ex, "in_reference": True, "mode": "bỏ-một-video", "ref_videos": 0, "score": None, "grade": "Cần ≥ 2 video HLV để kiểm chứng chéo", "units": len(c.units)})
                    continue
                ref = build_reference(ex, others, cfg)
            else:
                ref, others = ref_full, {s: None for s in ref_full["built_from"]}
            r = analyze_context(c, cfg, ref, stem)
            sc = r.get("score")
            rows.append({"video": stem, "role": role_of(stem), "exercise": ex, "in_reference": in_ref, "mode": "bỏ-một-video" if in_ref else "độc lập",
                         "ref_videos": len(others), "score": None if not sc else sc["overall"], "grade": r["status"] if not sc or sc["overall"] is None else sc["grade"],
                         "units": len(c.units), "status": r["status"]})
    df = pd.DataFrame(rows)
    os.makedirs(batch_dir, exist_ok=True)
    df.to_csv(os.path.join(batch_dir, "evaluation.csv"), index=False)
    chart_scores(df, batch_dir)
    chart_reference_bands(ctxs, cfg, reference_dir, batch_dir)
    chart_segments(ctxs, batch_dir)
    return df


def chart_scores(df: pd.DataFrame, batch_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, ex in zip(axes, ("ready", "serve")):
        d = df[df.exercise == ex].reset_index(drop=True)
        if d.empty:
            ax.axis("off")
            continue
        y = np.arange(len(d))
        vals = d.score.fillna(0).values
        bars = ax.barh(y, vals, color=[score_color(s) if s == s else "#d0d3d6" for s in d.score])
        for b, (_, r) in zip(bars, d.iterrows()):
            if r["in_reference"]:
                b.set_hatch("//")
                b.set_edgecolor("white")
            ax.text(min(vals[b.get_y() < 0 or 0:][0] if False else (r.score if r.score == r.score else 0), 100) + 1.5, b.get_y() + b.get_height() / 2,
                    ("không có đoạn để chấm" if r.score != r.score else f"{r.score:.0f}" + (" (bỏ-một-video)" if r["in_reference"] else " (độc lập)")), va="center", fontsize=7)
        ax.set_yticks(y, [v.replace("coach_", "HLV·").replace("trainee_", "Tập·") for v in d.video], fontsize=8)
        ax.set_xlim(0, 135)
        ax.axvline(85, color=C_GOOD, ls=":", lw=0.8)
        ax.axvline(60, color=C_MID, ls=":", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Điểm (0–100)")
        ax.set_title("Bài sẵn sàng (ready)" if ex == "ready" else "Bài phát cầu (serve)", loc="left", fontsize=10)
    axes[1].legend(handles=[Patch(facecolor="#bbb", hatch="//", edgecolor="white", label="HLV trong tập dựng chuẩn — chấm bằng chuẩn của các video CÒN LẠI"),
                            Patch(facecolor="#bbb", label="Video ngoài tập dựng chuẩn — chấm độc lập"),
                            Patch(facecolor="white", edgecolor="#999", label="Người tập (trainee_*): CHƯA CÓ — sẽ tự thêm vào biểu đồ khi có video")],
                   loc="upper center", bbox_to_anchor=(-0.25, -0.18), ncol=1, fontsize=8, frameon=False)
    fig.suptitle("Điểm các video (kiểm chứng không lạc quan)", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(batch_dir, "scores_overview.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_reference_bands(ctxs: Dict[str, Context], cfg: dict, reference_dir: str, batch_dir: str) -> None:
    from badminton3d.features import weighted_median
    for ex in ("ready", "serve"):
        ref = try_reference(reference_dir, ex)
        if ref is None:
            continue
        if ex == "ready":
            feats = ref["features"]
        else:
            ph = "danh_cau" if "danh_cau" in ref["phases"] else next(iter(ref["phases"]))
            feats = ref["phases"][ph]
        names = list(feats)
        fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names) + 1, 4.2))
        for ax, n in zip(np.atleast_1d(axes), names):
            r = feats[n]
            ax.axhspan(r["center"] - r["tol"], r["center"] + r["tol"], color="#2e9e5b", alpha=0.22)
            ax.axhline(r["center"], color="#2e9e5b", lw=1)
            for k, (stem, v) in enumerate(r["per_video"].items()):
                ax.scatter([0.0], [v], s=60, color="#1f5fbf", zorder=3)
                ax.annotate(stem.replace("coach_", ""), (0.0, v), xytext=(6, 0), textcoords="offset points", fontsize=6, va="center")
            k2 = 0
            for stem, c in ctxs.items():
                if c.exercise != ex or stem in ref["built_from"] or not c.units:
                    continue
                vals = []
                for u in c.units:
                    st = u["features"][n] if ex == "ready" else u["phases"].get(ph, {}).get(n)
                    if st and st["value"] == st["value"] and st["rel"] >= cfg["quality"]["min_reliability_to_score"]:
                        vals.append(st["value"])
                if vals:
                    ax.scatter([1.0 + 0.0 * k2], [float(np.median(vals))], s=45, marker="s", color=C_MID if role_of(stem) == "coach" else C_BAD, zorder=3)
                    ax.annotate(stem.replace("coach_", "").replace("trainee_", "T:"), (1.0, float(np.median(vals))), xytext=(5, 0), textcoords="offset points", fontsize=6, va="center")
                    k2 += 1
            ax.set_xlim(-0.5, 2.0)
            ax.set_xticks([0, 1], ["HLV (chuẩn)", "Video khác"], fontsize=7)
            ax.set_title(f"{r['label']}\n({r['unit']})", fontsize=8)
        fig.legend(handles=[plt.Line2D([], [], marker="o", color="#1f5fbf", lw=0, label="HLV dùng dựng chuẩn"), plt.Line2D([], [], marker="s", color=C_MID, lw=0, label="HLV khác (ngoài tập dựng chuẩn)"),
                            plt.Line2D([], [], marker="s", color=C_BAD, lw=0, label="Người tập (khi có)")], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"Chuẩn HLV ± dung sai — {'ready' if ex == 'ready' else 'serve, pha chạm cầu'}  (vùng xanh = khoảng chấp nhận)", fontweight="bold", x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(batch_dir, f"reference_bands_{ex}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)


def chart_segments(ctxs: Dict[str, Context], batch_dir: str) -> None:
    rows = []
    for stem, c in ctxs.items():
        total = c.pose.n / c.pose.fps
        scored = (sum(b - a + 1 for a, b in c.ready_seg.episodes) if c.exercise == "ready" else sum(r.end - r.prep_start + 1 for r in c.serve_seg.reps)) / c.pose.fps
        rows.append((stem, c.exercise, scored, total - scored))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.2))
    names = [r[0].replace("coach_", "HLV·").replace("trainee_", "Tập·") for r in rows]
    ax.barh(names, [r[2] for r in rows], color=C_GOOD, label="Đoạn được chấm")
    ax.barh(names, [r[3] for r in rows], left=[r[2] for r in rows], color="#d9d9d9", label="Không chấm (di chuyển đỡ cầu / nhặt cầu / chưa vào tư thế)")
    ax.invert_yaxis()
    ax.set_xlabel("Giây")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Mỗi video có bao nhiêu giây được chấm", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(batch_dir, "scored_time.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ chạy cả thư mục
def run_batch(input_dir: str, cfg: dict, out_root: str, reference_dir: str, racket_hand: Optional[str] = None, render: bool = True, force_pose: bool = False) -> List[dict]:
    videos = list_videos(input_dir)
    if not videos:
        raise RuntimeError(f"Thư mục Input không có video nào: {input_dir}")
    results = []
    for k, v in enumerate(videos, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        print(f"[{k}/{len(videos)}] {stem} ...", flush=True)
        try:
            res, _ = process_video(v, cfg, out_root, reference_dir, racket_hand, render, force_pose)
        except Exception as e:                                   # một video lỗi không làm dừng cả lô
            res = {"video": stem, "status": "ERROR", "message": f"{type(e).__name__}: {e}"}
            print(f"    LỖI: {res['message']}")
        sc = res.get("score")
        print(f"    -> {res['status']}" + (f" | điểm {sc['overall']:.1f} ({sc['grade']})" if sc and sc.get("overall") is not None else ""))
        results.append(res)
    batch_dir = os.path.join(out_root, "_tong_hop")
    run_coverage(videos, cfg, out_root, batch_dir, racket_hand)
    ev = run_evaluation(videos, cfg, out_root, reference_dir, batch_dir, racket_hand)
    write_summary_csv(results, ev, batch_dir)
    write_index(results, ev, cfg, out_root, batch_dir)
    return results


def write_summary_csv(results: List[dict], ev: pd.DataFrame, batch_dir: str) -> None:
    rows = []
    for r in results:
        sc = r.get("score")
        rows.append({"video": r.get("video"), "role": r.get("role"), "exercise": r.get("exercise"), "status": r.get("status"),
                     "score_with_full_reference": None if not sc else sc.get("overall"), "grade": None if not sc else sc.get("grade"),
                     "legacy_valid_pct": r.get("quality", {}).get("legacy_valid_pct"),
                     "mean_feature_valid_pct": (np.mean(list(r["quality"]["feature_valid_pct"].values())) if r.get("quality") else None)})
    out = pd.DataFrame(rows)
    if not ev.empty:
        out = out.merge(ev[["video", "mode", "score"]].rename(columns={"score": "score_unbiased", "mode": "unbiased_mode"}), on="video", how="left")
    out.to_csv(os.path.join(batch_dir, "results.csv"), index=False, float_format="%.2f")


def _b64(p: str) -> str:
    with open(p, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def write_index(results: List[dict], ev: pd.DataFrame, cfg: dict, out_root: str, batch_dir: str) -> None:
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#222}.wrap{max-width:1150px;margin:0 auto;padding:24px}"
           "h2{margin-top:30px;border-bottom:2px solid #dde3e8;padding-bottom:6px}table{border-collapse:collapse;width:100%;background:#fff;font-size:14px}"
           "th,td{padding:7px 9px;border-bottom:1px solid #e6eaee;text-align:left}th{background:#f0f3f6}img{max-width:100%;background:#fff;border-radius:6px;margin:8px 0}"
           ".muted{color:#778;font-size:13px}a{color:#1f5fbf}")
    p = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Tổng hợp</title><style>{css}</style></head><body><div class='wrap'><h1>Tổng hợp kết quả</h1>"
         f"<div class='muted'>badminton3d {__version__} · {len(results)} video</div><h2>Danh sách video</h2><table><tr><th>Video</th><th>Bài</th><th>Trạng thái</th><th>Điểm (chuẩn đầy đủ)</th>"
         "<th>Điểm không lạc quan</th><th>Xếp loại</th><th>%Valid cũ → theo góc</th><th>Báo cáo</th></tr>"]
    ev_idx = ev.set_index("video") if not ev.empty else None
    for r in results:
        sc = r.get("score")
        un = ev_idx.loc[r["video"], "score"] if ev_idx is not None and r.get("video") in ev_idx.index else None
        q = r.get("quality")
        p.append(f"<tr><td>{html.escape(str(r.get('video')))}</td><td>{r.get('exercise', '')}</td><td>{r.get('status')}</td>"
                 f"<td>{'—' if not sc or sc.get('overall') is None else f'{sc['overall']:.0f}'}</td><td>{'—' if un is None or un != un else f'{un:.0f}'}</td><td>{html.escape(sc['grade']) if sc else ''}</td>"
                 f"<td>{'—' if not q else f'{q['legacy_valid_pct']:.0f}% → {np.mean(list(q['feature_valid_pct'].values())):.0f}%'}</td>"
                 f"<td><a href='../{html.escape(str(r.get('video')))}/report.html'>mở</a></td></tr>")
    p.append("</table><p class='muted'>“Điểm không lạc quan”: video nằm trong tập dựng chuẩn được chấm bằng chuẩn dựng từ các video CÒN LẠI (bỏ-một-video); "
             "video ngoài tập dựng chuẩn được chấm độc lập. Người tập (trainee_*) chưa có — sẽ tự xuất hiện khi thêm video vào Input.</p>")
    for fname, title in (("coverage_stages.png", "Độ phủ dữ liệu trước / sau khi xử lý mất khớp"), ("coverage_heatmap_raw.png", "Valid thô theo từng góc"),
                         ("coverage_heatmap_final.png", "Valid sau xử lý mất khớp"), ("scores_overview.png", "Điểm các video"),
                         ("reference_bands_ready.png", "Chuẩn HLV — ready"), ("reference_bands_serve.png", "Chuẩn HLV — serve"), ("scored_time.png", "Thời gian được chấm")):
        f = os.path.join(batch_dir, fname)
        if os.path.isfile(f):
            p.append(f"<h2>{title}</h2><img src='{_b64(f)}'>")
    p.append("</div></body></html>")
    with open(os.path.join(batch_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("".join(p))
