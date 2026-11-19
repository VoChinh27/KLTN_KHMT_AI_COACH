# Nhánh nhãn AI thử nghiệm

Version `squat-ai-visual-review-v1`, schema `squat-ai-labels/0.1.0`.
Chủ project cho phép nhánh này trong request SHA-256
`d3886e12240e42de05d355c836f2a4a37cff33fc2b31c3e8c39f1a22c38a7075`.
Quyết định này không phê duyệt count admission, training hoặc documented human viewpoint.

Giữ semantics count/top→lower→bottom→top, partial, uncertain của protocol khóa
`squat-count-video-annotation-v1` (SHA trong từng record). Không yêu cầu độ sâu,
góc, lockout hay kỹ thuật chuẩn. Đây là AI reference yếu, không independent ground
truth, không human validation/adjudication. Nhánh human và templates giữ nguyên.

Quan sát trực tiếp ảnh nguồn bằng `view_image`; decode/trích ảnh chưa là đã xem.
Contact sheets có index/time và mọi frame; uniform resize/JPEG, không crop/overlay.
Record liệt kê đúng sheet đã hiển thị, hash và frame indices. Nếu chưa xem đủ
timeline, count toàn clip phải null. Ngay cả đủ frame, ảnh nhỏ/che khuất/clip edge
có thể khiến count unavailable. Count=0 chỉ khi thực sự xác định không có chu kỳ
hoàn chỉnh trên toàn clip. `observed_complete_count` là số thấy rõ, không tổng suy diễn.

Count và boundaries độc lập. Không tạo event chính xác giả: `cycle_hints` lưu
khoảng có thể chứa start/bottom/end, không là event, pause hoặc exhaustive targets.
Hints không chuyển sang Cycle của evaluator hiện hành. Khi chưa định vị đủ, hints
rỗng, boundaries unavailable và có reason. Partial interval là phạm vi hỗ trợ
quan sát phần chu kỳ bị cắt, không được cộng vào complete count. Uncertain ảnh hưởng
count làm whole_clip_count=null. Các khoảng có frame/time CFR và không vượt nguồn.

Lượt A rồi B mở lại ảnh nguồn, không đọc file nhãn A để tạo B. Cùng assistant vẫn
có context/memory: không blind, không hai rater độc lập. Giữ hai JSON và hash riêng;
so sánh status/count/partials/hints và ghi bất đồng. Reconciliation chỉ AI, không
trung bình count hoặc chọn theo prediction. Unresolved giữ null. Đồng thuận AI
không phải độ chính xác. Không chạy evaluator human với nhãn AI.

Pilot đúng 5 clip đã khóa. Sau kiểm tra schema/linkage/count semantics và tests,
mở rộng remaining112 theo source path tăng dần, checkpoint write-once từng pass.
Không loại clip khó, không tune thuật toán theo nhãn validation/development-test.

Viewpoint là task riêng trên đúng33 IDs đã khóa. Dùng rubric định tính trong
VIEWPOINT_REVIEW.md: side nhìn bên; front chính diện; oblique xiên; unknown không
rõ/back/mixed. Không suy độ/calibration. Quan sát toàn timeline, ghi subject,
framing/occlusion/camera changes; không lấy majority làm side toàn clip. AI review
luôn `visually_assessed_unverified`, `documented=false`; raw camera UNKNOWN không đổi.

Provenance: AI actor/method, thời điểm ghi record thực tế (không timestamp người),
source path/bytes/timeline, protocol hashes, evidence tool/sheets, pass hash.
Kiểm tra hash không chứng minh đã xem hoặc nhãn đúng. NOT_VALIDATED,
official_eligibility=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS.
Không inference, real counting adapter hoặc training trong quy trình này.
