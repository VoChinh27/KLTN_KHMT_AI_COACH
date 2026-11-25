# Output/ — kết quả

Mỗi video `X.mp4` trong `Input/` tạo ra thư mục `Output/X/`:

| File | Nội dung |
|---|---|
| `report.html` | **Báo cáo chính**: điểm, xếp loại, lỗi cần sửa + cách sửa + điểm lấy lại được, chi tiết từng góc, chất lượng dữ liệu, biểu đồ |
| `report.json` | cùng nội dung, dạng máy đọc |
| `dashboard.png` | tổng quan: điểm, điểm từng góc, đoạn được chấm, lỗi cần sửa |
| `angles_timeline.png` | từng góc theo thời gian so với chuẩn HLV ± dung sai |
| `quality.png` | khớp/góc nào thấy rõ; chỗ nào phải dùng bên/tay thay thế |
| `overlay.mp4` | video đã làm rõ ảnh, khung xương (xanh = đạt, đỏ = đoạn sai), bảng điểm tiếng Việt, icon tick/X |
| `angles.csv` | giá trị từng góc + độ tin cậy theo từng frame |
| `pose.csv` | pose MediaPipe đã trích (lưu để lần sau chạy nhanh) |

`Output/_tong_hop/` — tổng hợp cả lô: `index.html`, `results.csv`, `evaluation.csv`, `coverage_*.csv/png` (độ phủ trước/sau xử lý mất khớp),
`scores_overview.png`, `reference_bands_*.png`, `scored_time.png`.

Thư mục này hiện chứa kết quả chạy mẫu trên 9 video được chọn (4 HLV + 5 người tập); `_luu_tru/` giữ pose đã trích của các video bị lọc. Xoá đi và chạy lại `python -m badminton3d run` bất cứ lúc nào.
