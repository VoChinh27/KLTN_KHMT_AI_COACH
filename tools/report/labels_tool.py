#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/labels_tool.py
  sheet  : tạo pool video ẩn danh (mã C001...) + phiếu chấm CSV cho từng HLV
  import : nhập phiếu đã chấm vào data/labels.jsonl (kiểm tra chặt, mặc định all-or-nothing)

Ví dụ (từ thư mục gốc dự án):
  python tools/labels_tool.py sheet  --clips-file data/label_pool/clips.txt \
         --calib coach_HLV03_running_v005 coach_HLV04_running_v006 coach_HLV09_running_v009 \
         --annotators HLV_A HLV_B
  python tools/labels_tool.py import --sheet data/label_sheets/HLV_A.csv --annotator HLV_A --dry-run
  python tools/labels_tool.py import --sheet data/label_sheets/HLV_A.csv --annotator HLV_A
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage1_data.data_labeling import Annotation, LabelStore, LabelValidationError  # noqa: E402

YAML_PATH = ROOT / "config/sport_configs/sprint_start.yaml"
POOL, SHEETS = ROOT / "data/label_pool", ROOT / "data/label_sheets"
MAP_PATH = POOL / "_mapping.csv"          # GIỮ BÍ MẬT với người chấm
LABEL_PHASES = ("vao_cho", "static", "sprint_start")
STEM_RE = re.compile(r"^(coach|trainee)_([A-Za-z0-9]+)_")
COLS = ["code", "overall_score", *LABEL_PHASES, "error_tags", "notes"]


def vocab() -> dict:
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    out = {}
    for ph in LABEL_PHASES:
        for key, advice in (cfg.get("feedback_advice", {}).get(ph) or {}).items():
            out[f"{ph}.{key}"] = advice
    return out


def read_map() -> dict:
    if not MAP_PATH.exists():
        return {}
    with open(MAP_PATH, encoding="utf-8-sig", newline="") as f:
        return {r["code"]: r["stem"] for r in csv.DictReader(f)}


def write_map(mp: dict) -> None:
    POOL.mkdir(parents=True, exist_ok=True)
    with open(MAP_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "stem"])
        w.writerows(sorted(mp.items()))


def find_video(stem: str) -> Path:
    hits = list((ROOT / "data/raw_videos").rglob(f"{stem}.mp4"))
    if not hits:
        raise FileNotFoundError(f"không thấy video '{stem}.mp4' trong data/raw_videos")
    return hits[0]


# --------------------------------------------------------------------- sheet
def cmd_sheet(a) -> int:
    stems = [s.strip() for s in Path(a.clips_file).read_text(encoding="utf-8").splitlines()
             if s.strip() and not s.startswith("#")]
    mp, known = read_map(), {}
    known = {v: k for k, v in mp.items()}
    rng = random.Random(a.seed)

    calib = [s for s in dict.fromkeys(a.calib or []) if s not in known]
    for i, s in enumerate(calib, 1):
        code = f"CAL{len([c for c in mp if c.startswith('CAL')]) + 1:02d}"
        mp[code] = s; known[s] = code
    new = [s for s in dict.fromkeys(stems) if s not in known]
    rng.shuffle(new)                                    # mã không lộ thứ tự/HLV
    n0 = len([c for c in mp if c.startswith("C") and not c.startswith("CAL")])
    for i, s in enumerate(new, 1):
        code = f"C{n0 + i:03d}"
        mp[code] = s; known[s] = code
    write_map(mp)

    POOL.mkdir(parents=True, exist_ok=True); SHEETS.mkdir(parents=True, exist_ok=True)
    for code, stem in mp.items():
        dst = POOL / f"{code}.mp4"
        if not dst.exists():
            shutil.copy2(find_video(stem), dst)

    voc = vocab()
    with open(SHEETS / "error_vocab.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "mo_ta"]); w.writerows(voc.items())

    cal_codes = sorted(c for c in mp if c.startswith("CAL"))
    main_codes = sorted(c for c in mp if not c.startswith("CAL"))
    for ann in a.annotators:
        order = main_codes[:]
        random.Random(f"{a.seed}-{ann}").shuffle(order)  # thứ tự riêng từng HLV
        path = SHEETS / f"{ann}.csv"
        if path.exists() and not a.overwrite:
            print(f"[giữ nguyên] {path} đã tồn tại (dùng --overwrite để tạo lại, CẨN THẬN mất điểm đã chấm)")
            continue
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(COLS)
            for c in cal_codes + order:
                w.writerow([c, "", "", "", "", "", ""])
        print(f"[ok] {path}  ({len(cal_codes)} clip hiệu chỉnh + {len(order)} clip chính)")
    print(f"Pool video: {POOL} | từ vựng lỗi: {SHEETS / 'error_vocab.csv'}")
    print(f"BẢO MẬT: không đưa {MAP_PATH.name} cho người chấm.")
    return 0


# -------------------------------------------------------------------- import
def num(s: str, name: str):
    s = (s or "").strip().replace(",", ".")       # Excel locale VN dùng dấu phẩy thập phân
    if not s:
        return None
    v = float(s)
    if not 0 <= v <= 100:
        raise ValueError(f"{name}={v} ngoài [0,100]")
    return v


def cmd_import(a) -> int:
    mp, voc = read_map(), vocab()
    if not mp:
        print(f"✗ Thiếu {MAP_PATH}"); return 2
    raw = Path(a.sheet).read_text(encoding="utf-8-sig")
    header = raw.splitlines()[0]
    delim = ";" if header.count(";") > header.count(",") else ","   # Excel VN hay xuất dấu ';'
    rows = list(csv.DictReader(raw.splitlines(), delimiter=delim))
    missing_cols = [c for c in COLS if c not in (rows[0].keys() if rows else [])]
    if missing_cols:
        print(f"✗ Thiếu cột: {missing_cols}"); return 2

    anns, errors, warns, skipped = [], [], [], 0
    for ln, row in enumerate(rows, 2):
        code = (row["code"] or "").strip()
        if code.startswith("CAL"):
            continue                                   # clip hiệu chỉnh: không nhập
        if code not in mp:
            errors.append(f"dòng {ln}: mã '{code}' không có trong mapping"); continue
        if not (row["overall_score"] or "").strip():
            skipped += 1; continue                     # chưa chấm
        stem = mp[code]
        m = STEM_RE.match(stem)
        if not m:
            errors.append(f"dòng {ln}: tên '{stem}' sai quy ước"); continue
        try:
            overall = num(row["overall_score"], "overall_score")
            phase_scores = {}
            for ph in LABEL_PHASES:
                v = num(row[ph], ph)
                if v is not None:
                    phase_scores[ph] = v
        except ValueError as e:
            errors.append(f"dòng {ln} ({code}): {e}"); continue
        tags = [t.strip() for t in (row["error_tags"] or "").split("|") if t.strip()]
        bad = [t for t in tags if t not in voc]
        if bad:
            errors.append(f"dòng {ln} ({code}): tag ngoài từ vựng {bad}"); continue
        if phase_scores and abs(overall - sum(phase_scores.values()) / len(phase_scores)) > 25:
            warns.append(f"dòng {ln} ({code}): điểm tổng {overall:g} lệch >25 so với TB các pha")
        anns.append(Annotation(
            video_id=stem, subject_id=m.group(2), subject_role=m.group(1), sport="sprint_start",
            annotator_id=a.annotator, phase="combined", overall_score=overall,
            phase_scores=phase_scores, error_tags=tags, notes=(row["notes"] or "").strip()))

    print(f"Đọc {len(rows)} dòng | hợp lệ {len(anns)} | chưa chấm {skipped} | lỗi {len(errors)} | cảnh báo {len(warns)}")
    for w in warns:
        print("  ⚠", w)
    if errors:
        for e in errors:
            print("  ✗", e)
        if not a.skip_invalid:
            print("→ DỪNG, chưa nhập gì (sửa phiếu hoặc dùng --skip-invalid)."); return 1
    if a.dry_run:
        print("(dry-run: không ghi gì)"); return 0

    store, added, dup = LabelStore(str(ROOT / "data/labels.jsonl")), 0, 0
    for an in anns:
        try:
            store.add(an); added += 1
        except LabelValidationError as e:
            dup += 1; print("  ✗", e)
    print(f"Đã nhập {added} nhãn, trùng/từ chối {dup}. File: data/labels.jsonl")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--clips-file", required=True, help="mỗi dòng 1 stem video (clip chính)")
    s.add_argument("--calib", nargs="*", help="stem các clip hiệu chỉnh (mã CAL..)")
    s.add_argument("--annotators", nargs="+", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    i = sub.add_parser("import")
    i.add_argument("--sheet", required=True); i.add_argument("--annotator", required=True)
    i.add_argument("--dry-run", action="store_true"); i.add_argument("--skip-invalid", action="store_true")
    a = ap.parse_args()
    return cmd_sheet(a) if a.cmd == "sheet" else cmd_import(a)


if __name__ == "__main__":
    sys.exit(main())