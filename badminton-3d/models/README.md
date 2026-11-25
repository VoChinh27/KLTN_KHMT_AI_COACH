# models/

- `pose_landmarker_full.task` — MediaPipe Pose Landmarker (đã kèm).

## Tùy chọn: bộ dò người YOLO (top-down)

Mặc định **không cần**: hệ thống tự chọn/bám người bằng MediaPipe nhiều người + quét ô vuông (xem `docs/RESULTS.md` mục 11).
Bật YOLO khi lớp học đông / người đi lại giữa khung / người chiếm < 15% chiều cao khung:

1. Xuất mô hình (máy có mạng): `pip install ultralytics && yolo export model=yolov8n.pt format=onnx` rồi chép `yolov8n.onnx` vào thư mục này.
2. Trong `config/badminton3d.yaml`: `pose.detector: yolo_onnx` (đường dẫn `pose.yolo_model: models/yolov8n.onnx`).
3. Chạy lại với `--force-pose` để trích pose bằng bộ dò mới.

Cần cài thêm `pip install onnxruntime`. Lưu ý: phần giải mã đầu ra YOLO có kiểm thử tự động bằng dữ liệu giả; chưa được chạy với trọng số thật trong gói này.
