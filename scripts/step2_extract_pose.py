"""
scripts/step2_extract_pose.py
=============================
Trích pose cache (đã tracking + smoothing) cho TẤT CẢ video trong
data/raw_videos/COACH/, lưu vào data/pose_cache/<video_id>.json.

Backend pose đọc từ config/system_config.yaml (mặc định: RTMPose-m ONNX +
YOLO/ByteTrack).

QUAN TRỌNG — Cache phải sinh ra bằng CHÍNH pipeline lúc chấm điểm:
    Cache dùng `Pipeline._extract_tracked()` (bên trong có `_keep_largest_person`
    — lọc người có bbox lớn nhất mỗi frame). Nếu tự gọi `extract_video` +
    `track_and_filter` riêng (như phiên bản cũ) thì sẽ THIẾU bước lọc người
    lớn nhất → những clip có người phụ trong khung (vd người ngồi ghế) sẽ bị
    bám nhầm sang người phụ, cache hoàn toàn sai.

Chạy:
    python -m scripts.step2_extract_pose
    hoặc: python scripts/step2_extract_pose.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json

import yaml
import cv2

from stage2_pose.pose_extractor import PoseExtractor  # static method, không cần instance
from stage2_pose.signal_smoother import SignalSmoother
from core.pipeline import Pipeline

FPS_META_PATH = "data/pose_cache/video_meta.json"
SYSTEM_CONFIG_PATH = "config/system_config.yaml"


# ---------------------------------------------------------------------------
# Đọc thông tin backend từ system_config.yaml (chỉ để in log, không build model)
# ---------------------------------------------------------------------------
def load_backend_from_config(config_path: str = SYSTEM_CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    backend = cfg.get("pose_backend", "rtmpose")
    kwargs = cfg.get("pose_backend_kwargs", {}) or {}
    return backend, kwargs


def main() -> None:
    # 1) In thông tin backend từ config
    backend, kwargs = load_backend_from_config()
    print(f"[step2] Backend: {backend}")
    if backend == "rtmpose":
        print(f"[step2] RTMPose ONNX: {kwargs.get('rtmpose_onnx')}")
        print(f"[step2] YOLO weights:  {kwargs.get('yolo_weights')}")
        print(f"[step2] Device:        {kwargs.get('rtmpose_device')} / {kwargs.get('yolo_device')}")

    # 2) Build Pipeline 1 lần duy nhất — Pipeline tự tạo pose extractor từ config.
    #    Từ đây về sau chỉ dùng `pipe._extract_tracked()` cho TẤT CẢ video
    #    → đảm bảo cache khớp 100% với lúc chấm điểm.
    pipe = Pipeline.from_config(SYSTEM_CONFIG_PATH, "sprint_start")
    smoother = SignalSmoother()

    # 3) Đọc file meta fps cũ (nếu có)
    if os.path.exists(FPS_META_PATH):
        with open(FPS_META_PATH, "r", encoding="utf-8") as f:
            video_meta = json.load(f)
    else:
        video_meta = {}

    video_paths = sorted(glob.glob("data/raw_videos/COACH/*.mp4"))
    print(f"Tìm thấy {len(video_paths)} video, bắt đầu xử lý...\n")

    # 4) Đảm bảo thư mục cache tồn tại
    os.makedirs("data/pose_cache", exist_ok=True)

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for path in video_paths:
        video_id = os.path.splitext(os.path.basename(path))[0]
        out_path = f"data/pose_cache/{video_id}.json"

        if os.path.exists(out_path):
            print(f"[BỎ QUA] {video_id} đã trích xuất rồi")
            n_skip += 1
            continue

        # Đọc đúng width/height/fps của CHÍNH video này
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        print(f"[ĐANG XỬ LÝ] {video_id}  ({width}x{height}, fps={fps:.1f}) ...")
        try:
            # Đây là điểm mấu chốt: dùng CHÍNH logic của Pipeline
            # (đã có _keep_largest_person + track_and_filter bên trong)
            tracked = pipe._extract_tracked(path, width, height)
        except Exception as exc:
            print(f"[LỖI] {video_id}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        if not tracked:
            print(f"[LỖI] {video_id}: không phát hiện được người nào — kiểm tra lại video gốc.")
            n_fail += 1
            continue

        smoothed = smoother.process(tracked)
        PoseExtractor.save_sequence(smoothed, out_path)  # static method

        video_meta[video_id] = fps
        with open(FPS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(video_meta, f, ensure_ascii=False, indent=2)

        print(f"[XONG] {video_id} -> {out_path} ({len(smoothed)} frame, fps={fps:.1f})\n")
        n_ok += 1

    # 5) Đóng pose extractor bên trong Pipeline (nếu có)
    try:
        if hasattr(pipe, "_get_pose_extractor"):
            pipe._get_pose_extractor().close()
    except Exception:
        pass

    print(
        f"Hoàn tất Bước 2.  OK={n_ok}  BỎ QUA={n_skip}  LỖI={n_fail}"
    )


if __name__ == "__main__":
    main()