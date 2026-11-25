# badminton3d — chấm điểm tư thế cầu lông bằng MediaPipe 3D

Chấm điểm **tư thế sẵn sàng** (`ready`) và **phát cầu trái tay** (`serve`) bằng cách so **từng góc 3D** của người tập với **chuẩn dựng từ 1–3 video HLV tập đúng**.
Đặt video vào `Input/`, chạy một lệnh, nhận kết quả trong `Output/`: **điểm bao nhiêu, đang mắc lỗi gì, cần sửa gì để tăng điểm**, kèm biểu đồ và video minh hoạ.

## Điểm khác so với bản trước

| Yêu cầu | Cách giải quyết |
|---|---|
| %Valid thấp do mất khớp | **Valid theo từng góc** (chỉ cần các khớp của góc đó) + nội suy khoảng mất ngắn + bên/tay thay thế → độ phủ ready 86% → 99%, serve 61% → 99% (đọc kỹ `docs/RESULTS.md` mục 1) |
| `ready`: HLV di chuyển đỡ cầu thì không chấm | Chỉ chấm khi tốc độ thân đã xuống thấp liên tục ≥ 0,3 giây (đã **trở về** tư thế sẵn sàng) |
| `serve`: nhặt cầu thì không chấm | Một lần chuẩn = **từ bước chuẩn bị đến phát xong**; phần nhặt cầu / đi lại trước đó bị loại |
| Mất góc khuỷu tay vì bị che | Dùng khuỷu tay của **tay camera thấy rõ nhất** (và cảnh báo); khuyến nghị quay từ phía tay vợt |
| Chuẩn từ 1–3 video HLV | `build-reference`: giá trị chuẩn + dung sai từng góc (từng pha với serve) |
| Input / Output, điểm, lỗi, cách sửa | `Input/` → `Output/<video>/report.html` |
| Số liệu, biểu đồ chứng minh | `Output/_tong_hop/` (độ phủ, điểm bỏ-một-video, chuẩn HLV, thời gian được chấm); **tự thêm người tập khi có video** |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # hoặc: pip install -e .
```
Python ≥ 3.9. Model MediaPipe (`models/pose_landmarker_full.task`) đã kèm theo.

## Chạy

```bash
python -m badminton3d run                                # chạy TẤT CẢ video trong Input/ -> Output/
python -m badminton3d analyze Input/trainee_serve_01.mp4 # chạy MỘT video
```
Mở `Output/_tong_hop/index.html` (tổng hợp) hoặc `Output/<video>/report.html` (từng video). Gói đã kèm kết quả chạy mẫu trên 9 video được chọn (4 HLV + 5 người tập; xem `docs/SELECTION.md`).

**Với video mới:** đặt `trainee_ready_01.mp4` / `trainee_serve_01.mp4` vào `Input/` (đặt tên theo `<role>_<action>_<id>.mp4`, xem `Input/README.md`) rồi `run`. Video người tập sẽ tự xuất hiện trong các biểu đồ tổng hợp, tách nhóm với HLV.

**Dựng lại chuẩn khi có thêm video HLV đúng:**

```bash
python -m badminton3d build-reference ready Input/coach_ready_01.mp4 Input/coach_ready_04.mp4
python -m badminton3d build-reference serve Input/coach_serve_02.mp4 Input/coach_serve_07.mp4
```

| Lệnh | Việc làm |
|---|---|
| `run` | toàn bộ `Input/` → `Output/` (báo cáo từng video + tổng hợp) |
| `analyze <video>` | một video |
| `build-reference <ready\|serve> <video...>` | dựng chuẩn từ video HLV |
| `coverage` | đo độ phủ: %Valid cũ → Valid theo từng góc → sau xử lý mất khớp |
| `evaluate` | điểm không lạc quan (bỏ-một-video) + biểu đồ tổng hợp |
| `record <ready\|serve>` | **quay video trực tiếp** từ webcam / điện thoại, dừng quay là chấm điểm ngay (Output y như `analyze`) |
| `check <video>` | kiểm tra chất lượng video: FPS, rung máy, kích thước người, nhiễu pose |
| `config` | in cấu hình đang dùng (`config/badminton3d.yaml`) |

## Kết quả mỗi video (`Output/<video>/`)

`report.html` (báo cáo chính) · `report.json` · `dashboard.png` · `angles_timeline.png` · `quality.png` · `overlay.mp4` · `angles.csv` · `pose.csv`. Chi tiết: `Output/README.md`.

Báo cáo gồm: điểm /100 và xếp loại · **bảng "Cần sửa gì để tăng điểm"** (góc, hiện tại so với chuẩn, hướng dẫn, điểm lấy lại được) · chi tiết từng góc · chất lượng dữ liệu và xử lý mất khớp · **cảnh báo độ tin cậy** (chuẩn ít video, lệch góc quay, tay không chắc là tay vợt, ...).

## Giao diện đồ họa đa môn (Streamlit)

Giao diện là **đa môn thể thao**: ô **«Chọn Môn thể thao»** ở đầu thanh bên (mặc định Cầu lông) tự nạp bài tập, thư mục video (`Input/<môn>/`), chuẩn HLV (`Reference/<môn>/`) và cấu hình (`config/<môn>.yaml`) từ `config/sports.yaml`; không có tên bài tập nào được ghi cứng trong `app_ui.py`. Nút **«➕ Thêm Môn thể thao / Bài tập mới»** tạo nhanh thư mục và file YAML mẫu cho môn mới. Chấm điểm môn mới cần gắn engine của môn đó: xem `docs/MULTISPORT.md` và mẫu `docs/engine_template.py`.

```bash
pip install streamlit            # hoặc: pip install -r requirements.txt
streamlit run app_ui.py
```
Bốn tab, tiếng Việt: **Chấm điểm Video** (tải lên hoặc chọn trong `Input/`, chọn bài và tay vợt, xem điểm, video overlay, bảng «Cần sửa gì để tăng điểm», biểu đồ, báo cáo HTML) · **Quay & Chấm trực tiếp** (webcam / camera điện thoại, bấm SPACE để quay và dừng) · **Chuẩn HLV & Cấu hình** (xem chuẩn ± dung sai, dựng lại chuẩn từ 1–3 video HLV, chỉnh ngưỡng vung xả đà / nhấc gót / giới hạn điểm mà không mất chú thích trong file YAML) · **Thống kê & Báo cáo tổng hợp** (chạy toàn bộ và nhúng `index.html`). Giao diện chỉ gọi lõi (`badminton3d/ui_backend.py` nối với `runner`, `cli`), không sửa lõi. Video overlay được chuyển sang H.264 bằng ffmpeg (có sẵn qua `imageio-ffmpeg` nếu cài) để trình duyệt phát được; không có ffmpeg thì hiện khung hình chính và nút tải. Tab quay trực tiếp cần chạy Streamlit **trên máy có camera**.

## Quay trực tiếp

```bash
python -m badminton3d record serve --racket-hand right      # webcam 0: bấm nút xanh (hoặc SPACE) để quay, bấm lại để dừng
python -m badminton3d record ready --source 1               # webcam thứ hai
python -m badminton3d record serve --source http://192.168.1.5:8080/video   # điện thoại làm webcam (IP Webcam, DroidCam...)
```
Dừng quay là hệ thống lưu `Input/trainee_serve_liveNN.mp4` rồi chấm và mở `Output/<video>/report.html` (cùng định dạng với video có sẵn). Chưa thử với webcam thật trong gói này — xem `docs/RESULTS.md` mục 12.

## Kiểm tra lỗi riêng của phát cầu và chất lượng video

Báo cáo serve có thêm **vung xả đà quá dài** (bắt được) và **nhấc gót** (số đo có, ngưỡng để bạn chỉnh trong `config/badminton3d.yaml` mục `faults`), mỗi lỗi có mức nhẹ/nặng và điểm trừ; mỗi báo cáo có mục "Chất lượng video (FPS, rung, nhiễu)". Nhiều người trong khung: khai báo `person_roi` (xem `Input/README.md`) hoặc bật YOLO (`models/README.md`).

## Quy định quay video

Camera cố định (chân đế), thấy toàn thân, **người chiếm ≥ 50% chiều cao khung**, một người mục tiêu; **cùng góc quay với video dựng chuẩn** (hệ thống cảnh báo khi lệch quá 30°); **bài serve quay từ phía tay vợt**. Lý do và số liệu: `docs/RESULTS.md` mục 2.

## Cấu trúc

```
Input/            video cần chấm (+ file đánh dấu tay <video>.json, tuỳ chọn)
Reference/        chuẩn HLV: ready.json, serve.json
Output/           kết quả (mẫu kèm theo)
config/           badminton3d.yaml — mọi ngưỡng
badminton3d/      mã nguồn: pose_io · cleaning · angles · signals · segment · features · reference · scoring · coverage
                  · charts · render · report · runner · batch · cli
tests/            119 kiểm thử (dữ liệu giả lập, không cần video)
docs/             MULTISPORT.md (kiến trúc đa môn) · engine_template.py · METHOD.md (phương pháp) · RESULTS.md (số liệu, hiệu chỉnh, hạn chế) · SELECTION.md + selection_table.csv (lọc video)
models/           pose_landmarker_full.task
```

## Kiểm thử

```bash
python -m unittest discover -s tests -t . -v
```

## Tình trạng và hạn chế (đọc trước khi dùng điểm số)

- **Người tập sai đã có (18 video, chọn 5)** nhưng **chưa có nhãn lỗi có chủ đích** từng video, nên chưa đo được tỉ lệ bắt lỗi theo nhãn. Ready: 7/7 người tập sai bị bắt (29–41 điểm, HLV 93–94). Serve: lỗi nặng bị bắt, lỗi nhẹ (thân đổ ra trước) vẫn được "Tốt" — xem `docs/RESULTS.md` mục 3.
- **Chuẩn dựng từ 2 video mỗi bài; **chưa có `coach_ready` cùng bối cảnh quay với người tập****; dung sai tối thiểu đã hiệu chỉnh bằng dữ liệu (`docs/RESULTS.md` mục 5) nhưng chỉ trên 2–3 video HLV.
- **Góc 3D thay đổi theo góc quay**; video lệch góc so với chuẩn có điểm kém tin cậy (đã cảnh báo).
- **Tay vợt bị che** khi quay từ phía đối diện: `coach_serve_05` (đã chuyển vào `_luu_tru`) không tự phát hiện được cú phát. Với video mới nên khai báo `racket_hand` trong `Input/<video>.json`.
- **Đứng yên chưa chắc là sẵn sàng** (`coach_ready_03`). Bộ **nhận diện tư thế đang tập** và **so khớp chuỗi thời gian DTW** là bước tiếp theo (`docs/RESULTS.md` mục 7–8).
