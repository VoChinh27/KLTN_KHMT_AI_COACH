# tools/scan_cache_vs_fresh.py
"""
Quét toàn bộ video trong data/raw_videos/COACH, so sánh pose cache vs trích lại.
Mục đích: phát hiện cache bị lệch hệ toạ độ / scale / sinh từ video khác.

Cách dùng:
    python tools/scan_cache_vs_fresh.py
    python tools/scan_cache_vs_fresh.py --pattern "coach_HLV*"
    python tools/scan_cache_vs_fresh.py --config config/system_config.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# --- Tự thêm project root vào sys.path để import được core / stage5_eval ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import Pipeline
from stage5_eval.loso_cv_eval import read_pose_cache

# COCO 17 keypoints: hip trái = 11, hip phải = 12
LEFT_HIP, RIGHT_HIP = 11, 12


def hip_xy(p):
    """Trả về toạ độ (x, y) trung bình 2 hông từ một PoseFrame. Có thể chứa NaN."""
    xy = np.asarray(p.xy, dtype=float)
    return (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2.0


def scan_one(pipe, mp4: Path, verbose: bool = True):
    """Trả về dict kết quả cho 1 video, hoặc raise nếu lỗi."""
    stem = mp4.stem
    cache_path = Path("data/pose_cache") / f"{stem}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Không thấy cache: {cache_path}")

    cache, _ = read_pose_cache(cache_path)
    fps, w, h = pipe._read_video_meta(str(mp4))
    fresh = pipe._extract_tracked(str(mp4), w, h)

    fc = {p.frame_idx: p for p in cache}

    dists = []
    n_common = 0
    for p in fresh:
        q = fc.get(p.frame_idx)
        if q is None:
            continue
        n_common += 1
        a = hip_xy(q)
        b = hip_xy(p)
        # Bỏ qua frame mà cache hoặc fresh có hip NaN
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        dists.append(float(np.linalg.norm(a - b)))

    dists = np.asarray(dists, dtype=float)
    n_valid = int(dists.size)
    n_nan = n_common - n_valid
    frac_nan = (n_nan / n_common) if n_common else float("nan")

    if n_valid:
        med = float(np.median(dists))
        p90 = float(np.percentile(dists, 90))
        # Tỉ lệ frame lệch > nửa chiều cao khung → dấu hiệu lệch hệ toạ độ
        frac_far = float(np.mean(dists > 0.5 * h))
    else:
        med = p90 = frac_far = float("nan")

    res = {
        "stem": stem,
        "n_cache": len(cache),
        "n_fresh": len(fresh),
        "n_common": n_common,
        "n_valid": n_valid,
        "frac_nan": frac_nan,
        "median": med,
        "p90": p90,
        "frac_far": frac_far,
        "h": h,
    }
    if verbose:
        verdict = "OK"
        if not n_valid:
            verdict = "KHÔNG SO SÁNH ĐƯỢC"
        elif frac_far >= 0.5:
            verdict = "⚠ CACHE HỎNG (lệch >50% frame)"
        elif med > 0.1 * h:
            verdict = "⚠ NGHI NGỜ (median lớn)"
        print(
            f"{stem:35s} cache={len(cache):4d} fresh={len(fresh):4d} "
            f"common={n_common:4d} valid={n_valid:4d} "
            f"median={med:8.1f}px p90={p90:8.1f}px "
            f"frac_far={frac_far:.2f} nan={frac_nan:.2f}  -> {verdict}",
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser(description="So sánh pose cache vs trích lại cho mọi video COACH.")
    ap.add_argument("--config", default="config/system_config.yaml",
                    help="Đường dẫn system_config.yaml (mặc định: config/system_config.yaml)")
    ap.add_argument("--videos-dir", default="data/raw_videos/COACH",
                    help="Thư mục chứa video (mặc định: data/raw_videos/COACH)")
    ap.add_argument("--pattern", default="*.mp4",
                    help="Glob pattern lọc file (mặc định: *.mp4)")
    args = ap.parse_args()

    videos = sorted(Path(args.videos_dir).glob(args.pattern))
    if not videos:
        print(f"Không tìm thấy video nào khớp {args.videos_dir}/{args.pattern}")
        sys.exit(1)

    print(f"Tìm thấy {len(videos)} video. Đang khởi tạo Pipeline…", flush=True)
    pipe = Pipeline.from_config(args.config, "sprint_start")

    rows, errors = [], []
    for idx, mp4 in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] {mp4.name}", flush=True)
        try:
            rows.append(scan_one(pipe, mp4, verbose=True))
        except Exception as e:
            errors.append((mp4.stem, f"{type(e).__name__}: {e}"))
            print(f"{mp4.stem:35s} LỖI: {type(e).__name__}: {e}", flush=True)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("TỔNG KẾT")
    print("=" * 100)
    if not rows:
        print("Không có video nào quét thành công.")
    else:
        bad = [r for r in rows if r["n_valid"] and r["frac_far"] >= 0.5]
        warn = [r for r in rows if r["n_valid"] and 0.1 * r["h"] < r["median"] and r["frac_far"] < 0.5]
        ok = [r for r in rows if r["n_valid"] and r["median"] <= 0.1 * r["h"]]
        print(f"  OK        : {len(ok):3d}")
        print(f"  Nghi ngờ  : {len(warn):3d}")
        print(f"  HỎNG      : {len(bad):3d}  -> {[r['stem'] for r in bad]}")
        print(f"  Lỗi       : {len(errors):3d}")
        if bad:
            print("\n  Danh sách cache cần xoá + trích lại:")
            for r in bad:
                print(f"    - data/pose_cache/{r['stem']}.json  (median={r['median']:.1f}px, frac_far={r['frac_far']:.2f})")
    if errors:
        print("\n  Video lỗi:")
        for s, msg in errors:
            print(f"    - {s}: {msg}")


if __name__ == "__main__":
    main()