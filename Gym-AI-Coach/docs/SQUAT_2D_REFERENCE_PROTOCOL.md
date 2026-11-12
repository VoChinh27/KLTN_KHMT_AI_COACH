# Protocol tham chiếu squat 2D độc lập

2026-10-21. Protocol `squat-blinded-pixel-reference-v1`, schema annotation0.1.0,
toolkit `squat-2d-reference-v1`. S3 engineering protocol, NOT_VALIDATED.
Không là rubric kỹ thuật, không mở G5.7/G6/assessment. File bytes SHA-256 được
khóa trong sampling/package; sửa protocol sau này phải tạo run/version mới.

## Câu hỏi, nguồn và mẫu khóa trước

So sánh vị trí landmark projected2D và góc gối projected2D trên development
subset đã chọn, không đo góc vật lý3D, generalization hay chất lượng động tác.
Nguồn40 clips/80 pairs chỉ từ batch run5ed2bf... và replay runefc12e...
đã kiểm tra bytes, config, timeline và numerical identity. Không inference mới.

Sampling v1: sort đường dẫn; trước hết duyệt actions allow,degrade,skip,
nếu action chưa được clip đã chọn bao phủ thì lấy clip đường dẫn nhỏ nhất
trong nhóm gXX chưa chọn có action đó. Sau đó thêm clip đường dẫn nhỏ nhất
của các nhóm chưa chọn đến tối đa12 clips. Mỗi nhóm tối đa1 clip; nếu không
đủ nhóm thì giữ ít hơn12, không backfill cùng nhóm. Actions là metadata gate
toàn clip, không chọn theo giá trị/prediction/error hay hình ảnh đẹp.
gXX là nhóm tên dataset, không là danh tính người đã xác nhận.
Mỗi clip lấy unique `floor((N-1)*k/4)`, k=0..4, thứ tự tăng; tối đa5 frames.
Timestamp = index/FPS, giả định CFR hiện hành; không gọi là VFR PTS đo được.
Khóa sampling.json/hash TRƯỚC decode/export/evaluation. Không thay mẫu không
gán được; giữ null/reason. Không ngầm loại frame biên hoặc đoạn khó.

## Gán nhãn mù và tọa độ

Chỉ chuyển thư mục annotation_package cho annotator: PNG native resolution,
manifest nguồn/frame/image hash, protocol và template; không chuyển sampling
private có gate, evaluation_inputs, old prediction/quality/skeleton hoặc demo.
Không crop, resize, xoay/lật, overlay. Xem ảnh ở native scale hoặc zoom đồng
nhất để gán tọa độ gốc; không lưu tọa độ của ảnh preview đã scale.

Đánh dấu tâm khớp projected2D theo hình ảnh: tâm vùng khớp hông (ước lượng
vị trí tâm khớp dưới trang phục nếu thật sự phân biệt được), tâm khớp gối và
tâm cổ chân. Đây là định nghĩa thao tác cần kiểm tra sự nhất quán giữa người
gán, không khẳng định thấy chính xác tâm xương/khớp3D. Nếu không thấy rõ tâm
hoặc không phân biệt được bên giải phẫu, chọn uncertain/occluded, không đoán.
Không dùng prediction để định vị hoặc gán anatomical side.

Sáu keys bắt buộc: left/right_hip, left/right_knee, left/right_ankle.
Left/right là bên giải phẫu của người trong ảnh, KHÔNG phải bên trái/phải
canvas. Không tự đổi bên theo chân gần camera. Side không xác định =>
uncertain cho các joint liên quan; ghi lý do `anatomical_side_uncertain`.
Nếu nhiều người/không rõ subject mà model đang theo, đánh uncertain và ghi
`subject_identity_unresolved`; không chọn một người để làm đẹp sai số.

Pixel centers: gốc tâm pixel trên-trái(0,0), x sang phải,y xuống dưới;
tọa độ float nằm trong [0,W-1]×[0,H-1], đơn vị pixel ảnh gốc.
Raw MediaPipe chuyển sang pixel bằng `(x_norm*W, y_norm*H)` đúng quy ước
dimensionless_image_width_height hiện có. Không dùng W-1 thay W cho model,
không clip landmark ngoài ảnh. Điểm raw ngoài miền tâm pixel là unavailable.
Khoảng cách Euclid pixel; không chuyển cm. Ảnh không vuông phải dùng cả W,H.

States: visible cần finite xy; occluded, out_of_frame, uncertain và pending
đều xy=null và reason không rỗng. Không gán vị trí joint che khuất giả chính xác.
Template chưa điền: pending; annotator/method/time/independence/note=null.
Hoàn tất một frame cần xử lý cả6 joints (có thể unavailable), status completed,
annotator thật, method manual_blinded_pixel, thời gian ISO8601 có timezone,
independent_of_model=true, blinded_to_predictions=true, ghi nguồn/cách gán
trong provenance_note. Fields tự khai không chứng minh authenticity/qualification.
Không tự điền tên người xác nhận. Chỉ nộp completed frames thực hiện được;
frames không có trong submission vẫn giữ denominator và reference_pending.

## Lặp nhãn, độc lập và bất đồng

Chưa có người gán nhãn trong milestone chuẩn bị này. Khi thực hiện: giữ ít
nhất hai bản gán độc lập trên cùng frozen images, người gán không nhìn model
hoặc bản của nhau; một lượt lặp mù của người gán sau khoảng nghỉ được ghi rõ.
Lưu riêng từng document/time/annotator/method. Evaluator chạy riêng từng bản,
không merge duplicate samples hoặc trung bình tọa độ tự động. Báo khoảng cách
inter/intra-rater, bất đồng visibility/side và góc tham chiếu, cùng missingness.
Một người adjudicate phải xem ảnh gốc và ghi quyết định/reason/nguồn bản gốc;
giữ bản trước adjudication, không ghi đè. Không tự đặt ngưỡng agreement/accuracy
pass bằng fixture. Chỉ xem xét accuracy đáng tin sau source/identity review,
side/subject agreement và đánh giá chất lượng nhãn; chưa có nghiệm thu định lượng.

## Evaluator và mẫu số

Intake từ chối extra schema, nonfinite/types/bounds, trùng sample, thiếu/sai
keys bên khớp, hash/video/frame/time/dimensions/image/protocol/sampling mismatch,
nhãn synthetic/model trên real package. Hash liên kết bytes, không xác nhận
người gán hoặc ground truth. CLI kiểm tra mọi file của package trước tính.

Hai nhánh riêng: raw_landmark_px dùng raw observed và in-frame từng joint;
processed_angle_deg chỉ dùng số đo available đã lưu, canonical side, degree,
metric allow toàn clip. Giữ nguyên sequence skip/degrade và policy framewise;
không nâng gate theo ảnh/ref, không recompute góc model từ raw để vượt gate.
Không tính raw diagnostic knee angle trong phiên bản này. Góc reference dùng
joint_angle(hip,knee,ankle) trên native pixel xy; vector zero => reference
degenerate_geometry/null. A=C khác B vẫn là0° hợp lệ. Smoothing không mặc
nhiên tốt hơn; masks của hai nhánh khác nhau, không so trực tiếp MAE của chúng.

MAE, median absolute error, p90 linear interpolation tại0.9*(n-1), n hợp lệ.
Overall micro: mỗi comparable joint/angle trọng số1; by clip/side và by side
giữ mẫu số đầy đủ. Macro: trung bình MAE của clip-side có n>0, báo số nhóm
đóng góp; nhóm rỗng vẫn tồn tại. Denominator landmark=6×sample frames;
angle=2×sample frames. Reference/model availability được đếm độc lập, lý do
multi-label có thể vượt denominator; không loại missing reference âm thầm.
Không labels => REFERENCE_PENDING, tất cả accuracy metrics null,n=0;
không accuracy=0, không validation pass. Labels có mặt chỉ là diagnostic,
không tự đổi NOT_VALIDATED hoặc official/temporal/form status.

## Giới hạn và acceptance công cụ

Tests synthetic với đáp án hình học viết độc lập, ảnh không vuông, bounds,
khớp trùng, missing/occluded, wrong side/hash/frame và duplicates/no labels.
Test pass chỉ chứng minh implementation theo contract, không model accuracy.
Dataset action label, agent viewpoint, coverage/visibility và model coordinates
không là reference độc lập. Source cameraUNKNOWN/side unverified giữ nguyên.
Không form/phase/rep/score/feedback; assessment=false, official=false,
G5.7 blocked; G6/toàn G5 IN_PROGRESS. Chưa holdout, subject split hay camera
calibration. Native resolution thấp/occlusion và tâm joint dưới trang phục
có thể làm nhãn không xác định; phải giữ chúng trong coverage báo cáo.
