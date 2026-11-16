import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#!/usr/bin/env python3
"""Ẩn danh video gửi HLV: đổi tên theo mã (CAL01.., C001..), xoá metadata, xáo ngẫu nhiên.

Chuẩn bị file selection.txt: mỗi dòng 1 tên file gốc (đúng như trong thư mục nguồn).
  - n-cal dòng ĐẦU (mặc định 3) = video hiệu chỉnh -> CAL01.. (giữ đúng thứ tự)
  - các dòng SAU = video chính -> C001.. (bị XÁO NGẪU NHIÊN khi gán mã)
  Dùng --n-cal 0 nếu gửi 12 video, không có video hiệu chỉnh.

Chạy:
  python anonymize_videos.py --src video_raw --select selection.txt \
      --out goi_HLV/Danh_sach_Video --mapping private/_mapping.csv --seed 2026

Yêu cầu: ffmpeg + ffprobe. KHÔNG gửi _mapping.csv cho HLV.
"""
import argparse
import csv
import json
import random
import subprocess
from pathlib import Path


def probe(path: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    num, den = d["streams"][0]["r_frame_rate"].split("/")
    return {
        "fps": round(float(num) / float(den), 2),
        "duration_s": round(float(d["format"]["duration"]), 2),
        "resolution": f'{d["streams"][0]["width"]}x{d["streams"][0]["height"]}',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping", required=True, help="đặt NGOÀI thư mục gửi HLV")
    ap.add_argument("--n-cal", type=int, default=3, help="số video hiệu chỉnh ở đầu danh sách (0 nếu không dùng)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reencode", action="store_true",
                    help="re-encode thay vì -c copy (dùng nếu metadata vẫn còn sót)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.mapping).parent.mkdir(parents=True, exist_ok=True)

    names = [l.strip() for l in open(args.select, encoding="utf-8-sig") if l.strip()]
    if len(names) <= args.n_cal:
        raise SystemExit(f"Danh sách chỉ có {len(names)} video, không đủ cho n-cal={args.n_cal}")
    for n in names:
        if not (src / n).exists():
            raise SystemExit(f"Không thấy file: {src / n}")
    if Path(args.mapping).resolve().parent == out.resolve():
        raise SystemExit("mapping phải nằm NGOÀI thư mục gửi HLV")

    cal, main_vids = names[:args.n_cal], names[args.n_cal:]
    random.Random(args.seed).shuffle(main_vids)
    plan = [(f"CAL{i+1:02d}", n) for i, n in enumerate(cal)] + \
           [(f"C{i+1:03d}", n) for i, n in enumerate(main_vids)]

    rows = []
    for code, orig in plan:
        dst = out / f"{code}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src / orig),
               "-map_metadata", "-1", "-map_chapters", "-1"]
        cmd += (["-c:v", "libx264", "-crf", "18", "-c:a", "aac"] if args.reencode else ["-c", "copy"])
        cmd += ["-fflags", "+bitexact", "-flags:v", "+bitexact", str(dst)]
        subprocess.run(cmd, check=True)
        info = probe(dst)
        rows.append([code, orig, info["fps"], info["duration_s"], info["resolution"]])
        print(f"{orig}  ->  {dst.name}  {info}")

    with open(args.mapping, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "original_filename", "fps", "duration_s", "resolution"])
        w.writerows(rows)
    print(f"\nĐã ghi mapping (GIỮ RIÊNG, KHÔNG GỬI): {args.mapping}")
    print("Kiểm tra: ffprobe -show_format <file> không còn creation_time/title/encoder gốc.")


if __name__ == "__main__":
    main()

