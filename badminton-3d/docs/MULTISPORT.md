# Kiến trúc đa môn thể thao (Multi-Sport)

Giao diện `app_ui.py` **không chứa tên bài tập, thư mục hay file cấu hình của môn nào**. Mọi thứ đọc từ một file đăng ký:
`config/sports.yaml`. Chọn môn ở đầu thanh bên → giao diện tự nạp bài tập, thư mục video, chuẩn HLV, cấu hình ngưỡng của môn đó.

## 1. Bố cục dữ liệu

| Thành phần | Môn mới (tạo từ giao diện) | Cầu lông (giữ bố cục cũ) |
|---|---|---|
| Cấu hình ngưỡng | `config/<môn>.yaml` | `config/badminton3d.yaml` |
| Video đầu vào | `Input/<môn>/` | `Input/` |
| Chuẩn HLV | `Reference/<môn>/<bài>.json` | `Reference/<bài>.json` |
| Kết quả | `Output/<môn>/<video>/`, `Output/<môn>/_tong_hop/` | `Output/<video>/`, `Output/_tong_hop/` |

Cầu lông giữ bố cục cũ để dòng lệnh `python -m badminton3d ...` và gói dữ liệu hiện có không bị ảnh hưởng. Muốn chuyển sang `Input/badminton/` v.v. chỉ cần
chuyển thư mục và sửa 4 đường dẫn trong `config/sports.yaml` (và truyền `--input/--output/--reference/--config` khi dùng dòng lệnh).

## 2. File đăng ký `config/sports.yaml`

```yaml
default: badminton                 # môn chọn sẵn khi mở giao diện
sports:
  badminton:
    name: Cầu lông (Badminton)
    icon: "🏸"
    engine: badminton3d.engine_badminton:BadmintonEngine   # null = chưa có bộ chấm điểm
    config: config/badminton3d.yaml
    input: Input
    reference: Reference
    output: Output
    hand_label: Tay cầm vợt        # nhãn ô chọn tay thuận (nếu bài có hand: true)
    tips: [...]                    # mẹo quay video hiện ở thanh bên
    actions:
      ready: {label: "Tư thế sẵn sàng (ready)", short: "Sẵn sàng", hand: false}
      serve: {label: "Phát cầu trái tay (serve)", short: "Phát cầu", hand: true}
```

- **Bài tập (`actions`)**: khóa (`ready`, `serve`, `forehand`...) vừa là mã bài vừa là phần nhận diện trong tên file `<vai trò>_<bài>_<số>.mp4`
  (khóa dài nhất khớp được ưu tiên, ví dụ `serve_kick` thắng `serve`).
- **`hand: true`**: giao diện mới hiện ô chọn tay thuận cho bài đó và lưu vào `<Input>/<video>.json`.

## 3. Thêm môn / bài tập mới ngay trên giao diện

Nút **«➕ Thêm Môn thể thao / Bài tập mới»** ở thanh bên:

- **Môn thể thao mới:** nhập mã (chữ thường không dấu, ví dụ `tennis`, `chay_bo`), tên hiển thị, biểu tượng và danh sách bài tập, mỗi dòng
  `ma_bai | Nhãn hiển thị`. Hệ thống tạo `Input/<mã>/`, `Reference/<mã>/`, `Output/<mã>/`, `config/<mã>.yaml` mẫu (có chú thích) và đăng ký vào
  `config/sports.yaml`. Lỗi giữa chừng thì hoàn tác, không để lại thư mục dở dang.
- **Bài tập mới cho môn đang chọn:** thêm một bài vào registry và một khối mẫu vào file cấu hình của môn.

Tương đương dòng lệnh Python: `badminton3d.sports.create_sport(...)`, `badminton3d.sports.add_action(...)`.

> **Lưu ý quan trọng.** Tạo môn trên giao diện chuẩn bị **dữ liệu và cấu hình**. Việc **chấm điểm** cần một *engine* của môn đó: bộ chấm điểm hiện có
> (so góc 3D, tách pha, chuẩn HLV) được thiết kế riêng cho cầu lông. Môn chưa có engine vẫn dùng được: tải/quản lý video, xem và sửa cấu hình,
> xem file chuẩn; các nút chấm điểm / quay / dựng chuẩn báo rõ «chưa có bộ chấm điểm (engine)».

## 4. Gắn engine cho một môn

Engine là một lớp Python (khởi tạo không tham số) khai báo bằng `engine: "package.module:TenLop"`. Hợp đồng (xem `docs/engine_template.py`):

| Thành phần | Ý nghĩa |
|---|---|
| `name`, `supported_actions` | tên engine; các bài engine chấm được (bài khác hiện «chưa có bộ chấm») |
| `analyze(sport, video, action, hand, force) -> dict` | chấm một video, ghi kết quả vào `sport.output/<video>/`, trả `report` |
| `build_reference(sport, action, videos) -> (mã, log)` | dựng chuẩn từ 1–3 video HLV, ghi `sport.reference_file(action)` |
| `reference_rows(sport, action) -> (thông tin, dòng)` | (tuỳ chọn) bảng chuẩn ± dung sai để hiển thị |
| `record(sport, action, source, role, hand, max_seconds) -> (mã, log)` | (tuỳ chọn) quay trực tiếp; log chứa dòng `[SCORED] <video> -> ...` |
| `run_all(sport) -> (mã, log)` | (tuỳ chọn) chạy toàn bộ video và tạo `_tong_hop/` |
| `quick_settings` | (tuỳ chọn) mô tả bằng dữ liệu các ngưỡng hay chỉnh để giao diện tự vẽ form |

**Báo cáo** `report.json` dùng cùng cấu trúc cho mọi môn để giao diện hiển thị được ngay:

```json
{"status": "SCORED", "exercise": "forehand", "duration_s": 4.0, "warnings": ["..."],
 "score": {"overall": 77.0, "grade": "Khá",
           "errors": [{"where": "Chạm bóng", "label": "Góc khuỷu", "unit": "°", "value": 160, "center": 130, "tol": 15,
                       "severity": "nặng", "advice": "Gập khuỷu thêm", "potential_gain": 8.0}]}}
```
Tệp tuỳ chọn trong thư mục kết quả: `overlay.mp4`, `dashboard.png`, `angles_timeline.png`, `quality.png`, `report.html`.

## 5. Trình sửa cấu hình chung

Tab «Chuẩn HLV & Cấu hình» có hai lớp: **form nhanh** do `engine.quick_settings` mô tả (ví dụ cầu lông: vung xả đà, nhấc gót, giới hạn điểm) và mục
**«Tất cả tham số (nâng cao)»** tự liệt kê mọi tham số đơn (bật/tắt, số, chữ) của `config/<môn>.yaml`, nhóm theo mục cấp 1, hiển thị chú thích làm gợi ý.
Khi lưu chỉ đổi đúng các dòng thay đổi, **giữ nguyên chú thích**, có bản sao lưu `.yaml.bak` và tự khôi phục nếu file thành không hợp lệ.
Tham số nằm trong khối một dòng `{...}` chỉ đọc được, sửa trực tiếp trong file YAML.

## 6. Các file liên quan

| File | Vai trò |
|---|---|
| `config/sports.yaml` | đăng ký môn và bài tập |
| `badminton3d/sports.py` | đọc/ghi registry, `create_sport`, `add_action`, nạp engine |
| `badminton3d/engine_badminton.py` | engine cầu lông (bọc `runner`, `record`, `build-reference`, `run`) |
| `badminton3d/ui_backend.py` | hàm nối giao diện với engine, theo từng môn |
| `app_ui.py` | giao diện Streamlit, không chứa tên môn/bài |
| `docs/engine_template.py` | mẫu engine cho môn mới |
| `tests/test_sports.py` | kiểm thử registry, tạo môn, engine giả |
