#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_system_scores.py
Chấm các clip bằng ĐÚNG đường của app_demo (Pipeline.process_sprint_start_video), lưu
output/validity/system_scores.json (điểm tổng, điểm từng pha, các lỗi hệ thống báo).

Tiên quyết: YAML đã đóng băng; KHÔNG còn data/reference_models/sprint_start_static.json hỏng.
Chạy: python tools/run_system_scores.py --stems-file data/label_pool/clips.txt
"""
import argparse, hashlib, json, math, os, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from core.pipeline import Pipeline  # noqa: E402

TAG_PHASES = ("vao_cho", "static", "sprint_start")
WARN = 60.0
OUT = ROOT / "output/validity/system_scores.json"


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def find_video(stem):
    hits = list((ROOT / "data/raw_videos").rglob(f"{stem}.mp4"))
    return hits[0] if hits else None


def summarize(res):
    out = {"overall": float(res.overall_score) if fin(res.overall_score) else None,
           "confidence": res.score_confidence, "phase_source": res.phase_source,
           "n_phases_scored": res.n_phases_scored,
           "quality_flags": [k for k, v in res.quality_flags.items() if v],
           "phases": {}, "fail_tags": []}
    for ph, sp in res.sub_phase_results.items():
        if not sp.has_score():
            out["phases"][ph] = None
            continue
        sr = sp.result.score_report
        out["phases"][ph] = float(sr.overall_score) if fin(sr.overall_score) else None
        if ph not in TAG_PHASES:
            continue
        for fs in sr.feature_scores:
            if fs.is_info_only or not fin(fs.score_0_100) or fs.score_0_100 >= WARN:
                continue
            if fs.error_class == "data_quality" or not fin(fs.student_value):
                continue                      # không đo được -> không phải lỗi kỹ thuật xác định
            d = "low" if (fs.scoring_type == "boolean" or fs.student_value < fs.reference_center) else "high"
            out["fail_tags"].append({"tag": f"{ph}.{fs.feature_key}:{d}", "error_class": fs.error_class})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-file", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stems = [s.strip() for s in Path(a.stems_file).read_text(encoding="utf-8").splitlines()
             if s.strip() and not s.startswith("#")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"meta": {}, "clips": {}}
    ref_files = sorted(p.name for p in (ROOT / "data/reference_models").glob("*.json"))
    data["meta"] = {
        "yaml_md5": hashlib.md5((ROOT / "config/sport_configs/sprint_start.yaml").read_bytes()).hexdigest(),
        "reference_model_files": ref_files,
    }
    if ref_files:
        print(f"⚠ data/reference_models có file {ref_files}: pipeline sẽ DÙNG chúng thay cho YAML. "
              "Xóa/đổi tên nếu muốn chấm bằng YAML.")

    pipe = Pipeline.from_config("config/system_config.yaml", sport="sprint_start")
    for i, stem in enumerate(stems, 1):
        if stem in data["clips"] and not a.force:
            print(f"[{i}/{len(stems)}] bỏ qua {stem}"); continue
        video = find_video(stem)
        if video is None:
            data["clips"][stem] = {"error": "không thấy video"}; print(f"✗ {stem}: thiếu video"); continue
        try:
            data["clips"][stem] = summarize(pipe.process_sprint_start_video(str(video)))
            s = data["clips"][stem]
            print(f"[{i}/{len(stems)}] {stem}: overall={s['overall']} phases={s['phases']}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(limit=1)
            data["clips"][stem] = {"error": f"{type(exc).__name__}: {exc}"}
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")   # lưu từng clip
    print(f"Đã ghi {OUT}")


if __name__ == "__main__":
    main()