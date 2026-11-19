# Protocol nhãn temporal squat độc lập

Version `squat-count-video-annotation-v1`, schema `squat-count-labels/0.1.0`.
Engineering research protocol S3; chưa được kiểm chứng độ đồng thuận. Chỉ
BodyWeightSquats do dataset/người dùng khai báo; không nhận diện bài tập, không
định nghĩa form đúng/sai hoặc độ sâu chuẩn. Protocol bytes được hash trong package.

## Đơn vị nhãn và chu kỳ

Đơn vị là toàn bộ clip gốc, một subject quan sát liên tục và source SHA-256.
Một lần hoàn chỉnh là một excursion quan sát rõ: từ trạng thái trên cục bộ,
hạ cơ thể/gập chân, đạt vùng thấp của excursion, rồi nâng trở lại trạng thái trên
cục bộ. Không yêu cầu góc/depth/lockout chuẩn. Hai đầu của excursion và sự liên tục
của cùng người phải thấy được. Không lấy local extrema của model làm định nghĩa nhãn.
Chuyển động quá nhỏ để phân biệt với chỉnh tư thế/đảo chiều nhỏ phải uncertain;
không tự đặt ngưỡng số hoặc đoán một lần lặp.

- Start: sample cuối ở vùng trên trước khi bắt đầu hạ liên tục.
- End: sample đầu trở lại vùng trên sau nâng, trước hold hoặc lần hạ tiếp theo.
- Top giữ lâu: end của rep trước và start rep sau có thể khác frame. Nếu đảo chiều
  ngay tại top thì có thể chung một frame; không cộng hai lần event đó.
- Bottom `event`: một sample thấp nhất nhìn xác định được. `interval`: cả khoảng
  đáy không phân biệt được một sample duy nhất/đang hold. `unavailable`: không xác
  định được, null kèm reason. Không lấy midpoint interval làm physical reversal;
  đây không phải annotation arrival/departure vật lý hay pause duration.
- Count và boundaries độc lập: có thể đếm rõ nhưng chưa định vị chính xác biên;
  count vẫn available, boundaries partial/unavailable. Exhaustive boundaries cần
  một interval cho mỗi rep đã đếm. Count=0 chỉ khi đã xem toàn clip và xác nhận
  không có excursion hoàn chỉnh; không dùng0 cho thiếu dữ liệu.
- Partial ở đầu/cuối: excursion bị clip cắt, ghi visible interval và loại
  start/end/both; không cộng vào complete count. Một clip bắt đầu/đóng ngay ở top
  vẫn cần thấy đủ bằng chứng theo trên, nếu không chắc thì partial/uncertain.
- Đứng yên/giữ đáy không thêm rep. Các lần đảo chiều nhỏ/nhún không rõ chu kỳ phải
  có uncertain interval. Không yêu cầu point-resolution producer xác nhận mới gán
  nhãn video; rater phải dựa vào video độc lập.

## Dữ liệu không xác định

Camera chuyển động không tự là chuyển động người. Nếu không phân biệt được, ghi
camera_motion_ambiguous. Ra khỏi khung, occlusion, mất hình, cắt cảnh, đổi người,
nhiều người không chọn được subject đều có uncertain interval và reason. Không
nối chu kỳ qua các vùng đó. Partial do biên clip khác thiếu đoạn ở giữa clip.
Mỗi uncertain interval ghi `affects_count`: true làm count toàn clip unavailable;
những rep quan sát rõ ở phần còn lại vẫn được lưu nhưng không giả tổng count.
Không có subject rõ thì count unavailable và subject_description=null.

Frame index0..N−1; timestamp=i/FPS, tolerance kiểm tra số học1e−9s, không phải
tolerance accuracy. Interval nhãn là inclusive samples; elapsed=(end−start)/FPS,
không thêm1frame. Duration container=N/FPS, sample cuối=(N−1)/FPS. Chỉ CFR
assumption hiện hành; chưa gán nhãn VFR bằng PTS. Kiểm tra full decode trước giao
package; lỗi decode không được biến thành clip0rep. Không rotate/crop/overlay.

## Quy trình hai lượt độc lập và phân xử

1. Giao riêng `annotation_package/videos`, source manifest, protocol và template
   pass_a cho người A; pass_b cho người B. Không giao `private/`, predictions,
   pose, quality, split/performance hoặc kết quả của người kia. Gói chỉ chứa video
   gốc và source/timeline identity; giữ action label, không có count gợi ý.
2. Xem toàn video, tua/frame-step khi cần. Ghi người/phương pháp
   `manual_video_review`, thời điểm timezone, subject, nguồn, trạng thái count,
   cycles/partials/uncertain. Tự khai independent/blinded; schema không chứng minh
   rater nói đúng hoặc qualified. Không tự gọi rater là chuyên gia.
3. Giữ hai file gốc. Kiểm tra khác biệt count, status, intervals và missingness
   bằng công cụ `disagreement`; không tự resolve hai count bằng trung bình.
4. Rater thứ ba phân xử, có tên/time, mô tả field/decision/reason và hashes của
   hai lượt cùng final label. Cả trường hợp đồng thuận cũng log `review_all`.
   Pending adjudication không được evaluator dùng như nhãn final. Nếu không
   thể quyết định, final giữ unavailable; không ép consensus.
5. Lưu bundle cuối ở file mới. Protocol/hash/source phải khớp package, không sửa
   template/receipt cũ. Hai lượt cùng annotator bị từ chối. Final adjudicator
   khác hai rater. Synthetic method chỉ dùng cho synthetic tests, không real clips.

Template trống: annotator/time/method/subject/count=null, lists rỗng, status=pending,
reason=annotation_pending. Không phải đã xem0rep. Schema có version/hash và export
JSON schema. Bộ60 ảnh nhãn joint pixel là nhiệm vụ khác, không thay nhãn temporal.

## Evaluation khóa trước nhãn

Count signed error=prediction−reference; MAE và exact-count-match chỉ trên clips
có count final và prediction estimated, kèm denominator toàn cohort, label coverage,
prediction coverage, comparable coverage, abstained/failed/not_run/candidates_only.
Predicted0 chỉ là ước lượng0; reference0 là quan sát theo protocol; candidate_count
không tự chuyển thành estimated count. Unavailable luôn null và reason.

Rep-event matching chỉ khi final boundaries exhaustive và prediction estimated có
events_exhaustive. Monotone one-to-one: tối đa số match, rồi nhỏ nhất tổng absolute
start/end error, tie theo thứ tự input. Cả start và end lệch không quá tolerance
cấu hình (default0,20s, S3 association, không chuẩn chuyên môn). Không tune trên test.
Precision=TP/(TP+FP), recall=TP/(TP+FN); mẫu số0 trả null, không giả1. Timing signed/
MAE giây và frame chỉ trên matched pairs, báo misses/extras riêng. Bottom timing
chỉ khi cả hai có point event, không co interval thành point; interval nằm ngoài
timing denominator và được đếm riêng.

Reference partial/uncertain intervals là mask inclusive: predictions giao mask bị
loại khỏi event matching và **được báo riêng**. Count toàn clip không đổi; một
prediction đếm partial vẫn có thể gây count error. Reference complete cycle không
được giao mask. Incomplete boundary annotation không đủ để phạt false positives,
nên không tính event metrics trên clip đó. Candidate-only predictions được báo
số ứng viên, không được dùng để tuyên bố accuracy. Empty labels => NOT_EVALUATED,
MAE/exact/precision/recall/timing=null. Synthetic evaluation là test công cụ.
