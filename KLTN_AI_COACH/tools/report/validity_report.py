#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/validity_report.py
So điểm hệ thống với điểm HLV (ICC, MAE, Spearman, Bland-Altman, theo pha, theo lỗi).

Chạy: python tools/validity_report.py --annotators HLV_A HLV_B --stems-file data/label_pool/clips_test.txt
Đầu ra: console + output/validity/validity_report.json (+ PNG nếu có matplotlib)
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage1_data.data_labeling import LabelStore          # noqa: E402
from stage5_eval.score_validity_eval import icc           # noqa: E402

PHASES = ("vao_cho", "static", "sprint_start")


def f(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def boot_ci(mat, kind, n=2000, seed=0):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        v = icc(mat[rng.integers(0, len(mat), len(mat))], kind)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) >= n // 2 \
        else (float("nan"), float("nan"))


def band(x):                                   # Koo & Li (2016), áp dụng cho CẬN DƯỚI của CI
    if not np.isfinite(x): return "không kết luận"
    return "kém" if x < .5 else "trung bình" if x < .75 else "tốt" if x < .9 else "xuất sắc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--labels", default=str(ROOT / "data/labels.jsonl"))
    ap.add_argument("--system", default=str(ROOT / "output/validity/system_scores.json"))
    ap.add_argument("--stems-file", help="chỉ dùng các clip này (tập TEST)")
    ap.add_argument("--out-dir", default=str(ROOT / "output/validity"))
    a = ap.parse_args()
    sysd = json.loads(Path(a.system).read_text(encoding="utf-8"))
    clips, meta = sysd["clips"], sysd.get("meta", {})
    subset = {s.strip() for s in Path(a.stems_file).read_text(encoding="utf-8").splitlines()
              if s.strip()} if a.stems_file else None

    anns = defaultdict(dict)
    for an in LabelStore(a.labels).iter_all():
        if an.sport == "sprint_start" and an.phase == "combined" and an.annotator_id in a.annotators:
            anns[an.video_id][an.annotator_id] = an

    vids, excl = [], Counter()
    for vid, d in sorted(anns.items()):
        if subset is not None and vid not in subset: continue
        if set(d) != set(a.annotators): excl["thiếu HLV chấm"] += 1; continue
        s = clips.get(vid)
        if not s or "error" in s: excl["hệ thống lỗi/chưa chấm"] += 1; continue
        if s["overall"] is None: excl["hệ thống không ra điểm tổng (<3 pha chấm được)"] += 1; continue
        vids.append(vid)
    n, k = len(vids), len(a.annotators)
    if n < 3:
        print(f"✗ Chỉ có {n} clip đủ điều kiện {dict(excl)}"); return 2

    coach = np.array([[anns[v][h].overall_score for h in a.annotators] for v in vids])
    syst = np.array([clips[v]["overall"] for v in vids])
    cm, mat_s = coach.mean(1), np.column_stack([coach, syst])
    R = {"n_clips": n, "n_annotators": k, "excluded": dict(excl), "meta": meta,
         "coach_score_range": [float(cm.min()), float(cm.max())]}

    print(f"\n=== {n} clip | {k} HLV | khoảng điểm HLV {cm.min():.0f}–{cm.max():.0f} ===")
    if n < 20: print("⚠ n < 20: khoảng tin cậy sẽ rất rộng, chỉ nên coi là thăm dò.")
    if cm.max() - cm.min() < 30: print("⚠ Khoảng điểm HLV < 30: ICC bị hạ thấp giả tạo (ít phương sai giữa clip).")
    if excl: print("Loại:", dict(excl), "→ tỉ lệ hệ thống không ra điểm là kết quả cần báo cáo.")

    # ---- ICC
    for kind in ("ICC2_1", "ICC3_1"):
        ic_c, ic_s = icc(coach, kind), icc(mat_s, kind)
        ci_c, ci_s = boot_ci(coach, kind), boot_ci(mat_s, kind)
        R[kind] = {"inter_coach": ic_c, "inter_coach_ci95": ci_c, "with_system": ic_s,
                   "with_system_ci95": ci_s, "ratio": ic_s / ic_c if ic_c and ic_c > 0 else None}
        print(f"{kind}: giữa HLV = {f(ic_c)} [{f(ci_c[0])};{f(ci_c[1])}] ({band(ci_c[0])}) | "
              f"HLV+hệ thống = {f(ic_s)} [{f(ci_s[0])};{f(ci_s[1])}] ({band(ci_s[0])}) | "
              f"tỉ lệ = {f(R[kind]['ratio'])}")
    r21 = R["ICC2_1"]["ratio"]
    R["meets_90pct_of_ceiling_ICC2_1"] = bool(r21 is not None and r21 >= 0.9)
    print("Tiêu chí dự án (ICC hệ thống+HLV ≥ 90% ICC giữa HLV, loại ICC2_1):",
          "ĐẠT" if R["meets_90pct_of_ceiling_ICC2_1"] else "CHƯA ĐẠT")
    print("  (ICC2_1 phạt cả sai lệch hệ thống; ICC3_1/Spearman chỉ xét tính nhất quán thứ hạng.)")

    # ---- sai số, tương quan, Bland-Altman
    diff = syst - cm
    rho, p = spearmanr(syst, cm)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    R["agreement"] = {"mae": float(np.abs(diff).mean()), "rmse": float(np.sqrt((diff ** 2).mean())),
                      "spearman": float(rho), "spearman_p": float(p), "bias": bias,
                      "loa95": [bias - 1.96 * sd, bias + 1.96 * sd]}
    print(f"MAE {f(R['agreement']['mae'])} | RMSE {f(R['agreement']['rmse'])} | Spearman {f(rho)} (p={f(p, 3)}) | "
          f"bias (hệ thống−HLV) {bias:+.1f} | LoA95 [{bias-1.96*sd:+.1f}; {bias+1.96*sd:+.1f}]")

    # ---- theo pha
    R["per_phase"] = {}
    for ph in PHASES:
        xs, ys = [], []
        for v in vids:
            c = [anns[v][h].phase_scores[ph] for h in a.annotators if ph in anns[v][h].phase_scores]
            sp = clips[v]["phases"].get(ph)
            if len(c) == k and sp is not None:
                xs.append(np.mean(c)); ys.append(sp)
        if len(xs) >= 5:
            r_, p_ = spearmanr(ys, xs)
            R["per_phase"][ph] = {"n": len(xs), "spearman": float(r_), "mae": float(np.mean(np.abs(np.array(ys) - xs))),
                                  "bias": float(np.mean(np.array(ys) - xs))}
            print(f"  pha {ph:<13} n={len(xs):>3} Spearman {f(r_)} MAE {f(R['per_phase'][ph]['mae'])} "
                  f"bias {R['per_phase'][ph]['bias']:+.1f}")
        else:
            print(f"  pha {ph:<13} không đủ dữ liệu (n={len(xs)})")

    # ---- lỗi: truth = tag được > 1/2 số HLV đánh dấu
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    for v in vids:
        cnt = Counter(t for h in a.annotators for t in set(anns[v][h].error_tags))
        truth = {t for t, c in cnt.items() if c > k / 2}
        pred = {x["tag"] for x in clips[v]["fail_tags"]}
        for t in truth | pred:
            i = 0 if (t in truth and t in pred) else 1 if t in pred else 2
            per[t][i] += 1
            tp, fp, fn = tp + (i == 0), fp + (i == 1), fn + (i == 2)
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    R["tags"] = {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc,
                 "per_tag": {t: dict(zip(("tp", "fp", "fn"), c)) for t, c in per.items()}}
    print(f"Lỗi kỹ thuật: precision {f(pr)} | recall {f(rc)} (TP {tp}, FP {fp}, FN {fn})")
    for t, (a_, b_, c_) in sorted(per.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:8]:
        print(f"   {t:<46} TP {a_} FP {b_} FN {c_}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "validity_report.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(cm, syst); ax[0].plot([0, 100], [0, 100], "k--", lw=1)
        ax[0].set(xlabel="Điểm HLV (TB)", ylabel="Điểm hệ thống", title=f"Spearman={rho:.2f}", xlim=(0, 100), ylim=(0, 100))
        ax[1].scatter((cm + syst) / 2, diff); ax[1].axhline(bias, color="r")
        for y in (bias - 1.96 * sd, bias + 1.96 * sd): ax[1].axhline(y, color="gray", ls="--")
        ax[1].set(xlabel="Trung bình (hệ thống, HLV)", ylabel="Hệ thống − HLV", title="Bland–Altman")
        fig.tight_layout(); fig.savefig(out / "validity.png", dpi=150)
    except ImportError:
        pass
    print(f"\nĐã ghi {out / 'validity_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())