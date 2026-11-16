#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/summarize_app_demo.py
Gộp các <clip>.json do demo.app_demo ghi ra thành bảng theo từng tiêu chí.

Dùng:
    python tools/summarize_app_demo.py output/runs/<tag>/app_demo

Ghi ra cùng thư mục:
    feature_summary.csv  : 1 dòng / (pha, tiêu chí): tỉ lệ đạt, rớt, NaN, trung vị giá trị đo...
    clip_feature_matrix.csv : 1 dòng / clip, mỗi cột = điểm 1 tiêu chí (pha.tiêu_chí)
    clip_overview.csv    : điểm tổng, điểm từng pha, cờ chất lượng, lý do bỏ qua pha
Phân loại tiêu chí (ngưỡng đặt sẵn, chỉnh ở đầu file):
    CEILING   : >= 90% clip đạt >= 95 điểm     -> không phân biệt được
    MASS_FAIL : >= 50% clip dưới 60 điểm        -> rớt hàng loạt
    MISSING   : > 25% clip không có giá trị/NaN -> đo không ổn định
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CEIL_SCORE, CEIL_FRAC = 95.0, 0.90
FAIL_SCORE, FAIL_FRAC = 60.0, 0.50
MISS_FRAC = 0.25


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main(folder: str) -> None:
    d = Path(folder)
    files = sorted(p for p in d.glob("*.json"))
    if not files:
        sys.exit(f"Không thấy file .json trong {d}")

    feat_rows, over_rows = [], []
    for p in files:
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[bỏ qua] {p.name}: {e}")
            continue
        clip = p.stem
        row = {"clip": clip, "overall": js.get("overall_score"),
               "confidence": js.get("score_confidence"),
               "quality_flags": ";".join(k for k, v in (js.get("quality_flags") or {}).items() if v)}
        for phase, sp in (js.get("sub_phases") or {}).items():
            row[f"score_{phase}"] = sp.get("score") if sp.get("is_scored") else None
            if not sp.get("is_scored"):
                row[f"skip_{phase}"] = sp.get("skip_reason")
            for fs in sp.get("feature_scores") or []:
                feat_rows.append({
                    "clip": clip, "phase": phase, "feature": fs.get("feature_key"),
                    "value": fs.get("student_value"), "ref_center": fs.get("reference_center"),
                    "score": fs.get("score"), "level": fs.get("level"),
                    "type": fs.get("scoring_type"), "in_range": fs.get("is_in_range"),
                })
        over_rows.append(row)

    f = pd.DataFrame(feat_rows)
    pd.DataFrame(over_rows).to_csv(d / "clip_overview.csv", index=False, encoding="utf-8-sig")
    if f.empty:
        sys.exit("Không có feature_scores trong JSON (pha có thể đều không được chấm).")

    f["score"] = pd.to_numeric(f["score"], errors="coerce")
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    n_clips = f["clip"].nunique()

    out = []
    for (phase, feat), g in f.groupby(["phase", "feature"]):
        s = g["score"]
        n_obs = len(g)
        valid = s.dropna()
        n_nan = int(s.isna().sum()) + (n_clips - n_obs)
        frac = lambda m: float(m.sum()) / max(len(valid), 1)  # noqa: E731
        ceil = frac(valid >= CEIL_SCORE)
        fail = frac(valid < FAIL_SCORE)
        miss = n_nan / n_clips
        tag = []
        if len(valid) and ceil >= CEIL_FRAC:
            tag.append("CEILING")
        if len(valid) and fail >= FAIL_FRAC:
            tag.append("MASS_FAIL")
        if miss > MISS_FRAC:
            tag.append("MISSING")
        v = g["value"].dropna()
        out.append({
            "phase": phase, "feature": feat, "level": g["level"].iloc[0], "type": g["type"].iloc[0],
            "n_clips": n_clips, "n_valid": len(valid), "n_nan": n_nan,
            "mean_score": round(valid.mean(), 1) if len(valid) else None,
            "std_score": round(valid.std(), 1) if len(valid) > 1 else None,
            "pct_ge95": round(100 * ceil, 0), "pct_lt60": round(100 * fail, 0),
            "pct_in_range": round(100 * g["in_range"].astype(float).mean(), 0) if g["in_range"].notna().any() else None,
            "value_median": round(v.median(), 2) if len(v) else None,
            "value_min": round(v.min(), 2) if len(v) else None,
            "value_max": round(v.max(), 2) if len(v) else None,
            "ref_center": round(pd.to_numeric(g["ref_center"], errors="coerce").median(), 2),
            "flag": ",".join(tag),
        })
    summ = pd.DataFrame(out).sort_values(["phase", "feature"])
    summ.to_csv(d / "feature_summary.csv", index=False, encoding="utf-8-sig")

    f["col"] = f["phase"] + "." + f["feature"]
    f.pivot_table(index="clip", columns="col", values="score", aggfunc="first").round(1) \
        .to_csv(d / "clip_feature_matrix.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summ.drop(columns=["n_clips", "type"]).to_string(index=False))
    print(f"\nĐã ghi: feature_summary.csv, clip_feature_matrix.csv, clip_overview.csv trong {d}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Dùng: python tools/summarize_app_demo.py output/runs/<tag>/app_demo")
    main(sys.argv[1])