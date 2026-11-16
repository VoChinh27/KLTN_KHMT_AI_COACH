# Định nghĩa nhãn pha chạy xuất phát thấp

## Quy ước chung

Nhãn áp dụng cho một lượt chạy xuất phát thấp trong một clip. Chỉ gán điều có thể quan sát trên video; không suy đoán ý định vận động viên. Frame bắt đầu/kết thúc đều là chỉ số frame **bao gồm hai đầu**. Pha không xuất hiện phải đánh dấu vắng mặt trong schema, không tạo một đoạn giả. Nếu khớp bị che, mờ hoặc sự kiện nằm giữa hai frame, đánh dấu theo frame đầu tiên có bằng chứng nhìn thấy và ghi chú ca khó.

Bốn pha theo thứ tự: `vao_cho` → `static` → `sprint_start` → `drive_phase`. Năm mốc là đầu `vao_cho`, đầu `static`, đầu `sprint_start`, đầu `drive_phase`, và frame cuối `drive_phase`.

## Định nghĩa từng pha và mốc

| Pha/mốc | Quy tắc gán có thể quan sát | Trường hợp khó cần ghi chú |
|---|---|---|
| `vao_cho` | Bắt đầu tại frame đầu tiên vận động viên bước/di chuyển vào vị trí xuất phát hoặc bắt đầu hạ người để vào tư thế. Kết thúc ở frame ngay trước khi tư thế ổn định của `static` bắt đầu. | Nếu clip bắt đầu khi vận động viên đã đứng sẵn, gán `vao_cho` vắng mặt; không suy ngược từ tư thế. |
| `static` | Bắt đầu ở frame đầu tiên tư thế xuất phát đã ổn định: tay/chân đặt, thân cúi và không còn điều chỉnh lớn. Kết thúc ngay trước sự kiện khởi phát chạy. | Các rung/điều chỉnh nhỏ không kết thúc pha. Nếu không thấy tay chạm đất do góc quay, dùng toàn bộ tư thế và ghi chú mức che khuất. |
| `sprint_start` | Bắt đầu ở frame đầu tiên có chuyển động chủ động rời tư thế tĩnh hướng vào bước chạy: tay rời điểm tì hoặc chân bắt đầu đạp/rời vị trí, tùy dấu hiệu nào xuất hiện trước và nhìn thấy rõ. | Không lấy frame nghiêng thân đơn thuần nếu chưa có dấu hiệu khởi phát chi. Nếu video 30 fps khiến sự kiện chỉ nằm giữa hai frame, chọn frame đầu tiên có chuyển động nhìn thấy. |
| `drive_phase` | Bắt đầu tại frame đầu tiên của bước chạy sau pha đạp/bật ban đầu, khi vận động viên đã chuyển sang chu kỳ chạy tiến về trước. Kết thúc ở frame cuối cùng còn quan sát được vận động viên chạy trong clip. | Không dùng mốc thời gian cố định. Nếu người ra khỏi khung hình, kết thúc ở frame cuối còn đủ bằng chứng; ghi chú việc mất track. |

## Quy trình gán nhãn

1. Xem toàn clip một lượt, sau đó tua chậm quanh từng mốc.
2. Gán theo thứ tự thời gian và kiểm tra các pha không chồng lấn, không đảo thứ tự.
3. Dùng thao tác đánh dấu vắng mặt nếu pha không xảy ra/không quan sát được; phân biệt rõ với pha bị che một phần.
4. Ghi `annotator_id`, lý do vắng mặt và ghi chú khi ranh giới không chắc.
5. Với kiểm tra độ lặp lại, gán lại ngẫu nhiên ít nhất 6–8 clip sau tối thiểu một tuần, không xem nhãn cũ.

## Quy tắc phân xử

Khi hai người gán lệch nhau, cùng xem đoạn ±1 giây quanh mốc và đối chiếu định nghĩa trên. Không chỉnh định nghĩa để hợp với dự đoán của model. Nếu không thể xác định từ video, đánh dấu cần duyệt/không quan sát được thay vì ép ra một frame.
