# Tiêu chí nghiệm thu phân đoạn pha

Các tiêu chí này phải được chốt trước khi xem kết quả trên tập kiểm định. Tách dữ liệu theo người, không chia ngẫu nhiên theo frame/clip của cùng một người. Tập kiểm định cuối không được dùng chọn đặc trưng, ngưỡng hoặc siêu tham số.

## Định nghĩa dung sai

Với mỗi loại biên, đặt `tol = max(3 frame, P90 sai lệch giữa hai lần gán độc lập)`. Nếu `P90 > 5 frame`, dừng huấn luyện để thống nhất lại định nghĩa nhãn. Báo kết quả cả theo frame và mili giây; không gộp các loại biên nếu phân bố độ khó khác nhau.

## Cổng chất lượng dữ liệu

- 100% nhãn dùng được phải qua kiểm tra schema, phạm vi frame, thứ tự pha và pha vắng mặt.
- Tất cả clip đánh giá phải ghép đúng pose cache theo stem; báo riêng clip thiếu cache, tỷ lệ frame NaN và nhóm HLV/học viên.
- Không huấn luyện trên nhãn `auto` chưa được người duyệt. Chỉ `manual` và `auto_reviewed` được đưa vào train theo chính sách đã ghi.
- Ghi rõ số người và số clip trong từng fold. Không báo kết quả nếu một nhóm có người xuất hiện đồng thời ở train và test.

## Cổng chấp nhận model

Đánh giá leave-one-person-out và bootstrap theo clip. So với heuristic baseline trên cùng fold. Model chỉ được tích hợp nếu đồng thời:

1. Trung vị sai số tuyệt đối của từng biên không vượt `tol` tương ứng; báo P50/P90/P95 và tỷ lệ trong dung sai.
2. Tỷ lệ tất cả biên của một clip nằm trong dung sai đạt ít nhất 80% tổng thể và không dưới 70% riêng HLV hoặc học viên.
3. So với heuristic, cải thiện tỷ lệ biên trong dung sai ít nhất 5 điểm phần trăm; khoảng tin cậy bootstrap 95% của chênh lệch không cho thấy suy giảm. Nếu mẫu quá nhỏ để kết luận, chỉ được chạy shadow mode.
4. Tỷ lệ đảo thứ tự bằng 0 sau decode; nhận diện pha vắng mặt được báo riêng và không thấp hơn baseline.
5. ECE ≤ 0.10 và recall phát hiện lỗi biên vượt dung sai ≥ 80%. Nếu chưa đạt, confidence không được dùng để tự bỏ qua review.
6. Không nhóm nào (HLV/học viên, góc quay nếu có) suy giảm quá 5 điểm phần trăm so với baseline.

## Cổng vận hành

- Giai đoạn đầu luôn yêu cầu review; không chấm điểm clip có `needs_review=true`.
- Mỗi dự đoán lưu model version, nguồn pha, confidence từng biên và cờ chất lượng.
- Chỉ cho phép bỏ review sau khi model vượt toàn bộ cổng, có phê duyệt và bật rõ `allow_auto_unreviewed`; lấy mẫu kiểm tra ngẫu nhiên tối thiểu 10%.
- Báo cáo phải chứa cấu hình, commit/version code, danh sách clip/fold, seed và đường dẫn artifact để tái lập.
