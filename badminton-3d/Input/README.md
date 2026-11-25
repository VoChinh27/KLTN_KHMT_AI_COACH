# Input/ — đặt video cần chấm vào đây

**Đặt tên theo `<role>_<action>_<id>.mp4`:**

| Phần | Giá trị | Ý nghĩa |
|---|---|---|
| `role` | `coach` / `trainee` | HLV / người tập (biểu đồ tự tách nhóm theo tiền tố này) |
| `action` | `ready` / `serve` | tư thế sẵn sàng / phát cầu trái tay — hệ thống dựa vào chữ này để biết chấm bài nào |

Ví dụ: `coach_serve_07.mp4`, `trainee_ready_01.mp4`. Định dạng H.264 (.mp4/.mov/.avi/.mkv).

## Quy định quay (ảnh hưởng trực tiếp đến độ chính xác)

- Camera cố định, thấy **toàn thân**, một người trong khung.
- **Cùng góc quay với video HLV dựng chuẩn** (xem `Reference/README.md`): MediaPipe 3D thay đổi theo góc quay, hệ thống sẽ cảnh báo khi lệch quá 30°.
- **Bài serve: quay từ phía TAY VỢT** (tay vợt là tay gần camera). Tay bị thân người che thì MediaPipe không thấy.

## Đánh dấu tay (tuỳ chọn): `<tên video>.json`

Dùng khi cú phát không tự phát hiện được hoặc cần khai báo tay vợt:

```json
{
  "serve_windows_s": [[2.6, 4.5]],
  "racket_hand": "right"
}
```
`serve_windows_s`: mỗi cặp `[bắt đầu, kết thúc]` (giây) là một lần phát từ bước chuẩn bị đến khi phát xong. `racket_hand`: `left` | `right` | `auto`.
Ví dụ có sẵn: `coach_serve_05.json`.


## Khai báo tay vợt (khuyến nghị cho bài serve)

Tạo `Input/<tên video>.json` cùng tên với video:

```json
{"racket_hand": "right"}
```

`left` / `right` = tay cầm vợt của người trong video (theo **người đó**, không theo ảnh). Nếu bỏ trống hệ thống tự chọn (`auto`) và có thể chọn nhầm khi một tay bị che. Có thể thêm `"person_roi": [0, 0, 0.5, 1]` (x0, y0, x1, y1 theo tỉ lệ 0–1 của khung) khi trong khung có nhiều người và hệ thống chọn nhầm người; chạy lại với `--force-pose`. Có thể thêm `"serve_windows_s": [[3.1, 4.5]]` để đánh dấu khoảng thời gian cú phát (giây).

## Thư mục `_luu_tru/`

Video đã lọc bỏ (không chạy khi `run`) — lý do từng video ở `docs/SELECTION.md`. Muốn dùng lại: chuyển video (và file `.json` nếu có) ra `Input/`.
