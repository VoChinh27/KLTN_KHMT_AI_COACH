# Giai đoạn 1 — Video ingestion và metadata

Ngày: 2026-10-08 (Asia/Saigon)

## Phạm vi và implementation

Triển khai metadata và streaming frame cho video đơn camera; không chạy pose,
preprocessing, feature, repetition, phase hoặc assessment.

- Backend tối thiểu: `opencv-python==5.0.0.93`, import lazy.
- `VideoMetadata`: video identity/path, FPS, frame count, resolution, decoder và
  timestamp policy.
- `VideoFrame`: index từ 0, timestamp giây, decoded image.
- `OpenCVVideoReader.frames()`: iterator decode lazy và giải phóng capture.
- CLI: `python -m gym_coach.video` và `gym-video-info`, JSON success/error.
- Structured reasons cho unreadable, too short, invalid FPS/count/resolution và
  decoded-count mismatch.

## Chính sách timestamp

Timestamp tính bằng `frame_index / fps` và ghi explicit
`fps_derived_cfr_assumption`. Đây là proposed engineering policy S3; chưa có
tuyên bố nhận diện hoặc hỗ trợ VFR.

## Bằng chứng kiểm thử

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

Kết quả thực tế: `28 passed in 1.52s`.

Test gồm foundation, schema invalid values, synthetic MJPG video 4 frames,
streaming index/timestamp, minimum frames, missing file, decoder trả metadata
sai, CLI JSON và integration trên:

`Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi`

CLI thực tế trả 25 FPS, 267 frames, 320x240, decoder `opencv:FFMPEG`. Các số
này chỉ mô tả đúng file integration, không phải thống kê toàn dataset.

## Definition of Done

| Tiêu chí | Bằng chứng | Kết quả |
|---|---|---|
| Metadata và timestamp được test | synthetic known fixture + real prefix | PASS |
| Lỗi đầu vào có structured reason | negative decoder/missing/short tests | PASS |
| CLI không hard-code path | positional path, success/error JSON | PASS |
| Streaming, không preload mặc định | iterator API và prefix integration | PASS |
| Không pose/form assessment | video package chỉ ingestion | PASS |

## Hạn chế và bước tiếp theo

- Chưa hỗ trợ timestamp VFR thực.
- Container frame count có thể sai; full iteration kiểm tra mismatch.
- Chưa audit codec portability/toàn dataset.
- Legacy venv đang có hai OpenCV distributions; manifest mới chỉ chọn một.

Tiếp theo là GĐ2: MediaPipe pose extraction, joint mapping version và provenance
theo joint/frame; chưa nội suy, normalize hoặc đánh giá kỹ thuật.
