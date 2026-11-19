# Hướng đếm squat — contract nghiên cứu đề xuất

## Quyết định mới đã phê duyệt — 2026-10-26

Chủ project đồng ý nhánh **video-level count-only từ raw saved2D pose + AI weak
counts**, không yêu cầu metric góc đạt side-view. Đã triển khai riêng trong
`gym_coach.counting.research`, version `squat-raw-pose-count-research-v1`.
Đây là thay đổi phương pháp được phê duyệt rõ; proposal fixed-left/angle-gated
phía dưới giữ làm lịch sử, **không được áp dụng ngầm vào code temporal cũ**.

Lý do:104 labels chỉ có whole-count, không exhaustive event targets cho ba heads;
raw pose giữ thông tin chuyển động/masks mà không biến framewise angle allowance
thành temporal permission. Admission mới kiểm nguồn/hash/timeline/config, complete
8-joint observed/in-frame support ≥70%, incomplete span≤10frames, length5..1500;
flags ambiguity/discontinuity/unstable chặn input. Quy tắc được freeze trước
batch kết quả, không dùng target/count để chọn input hoặc sửa gate.

Kết quả:112 checked;50 input accepted,62 rejected;47 có input+AI target, gồm
33 train/3 validation/11 development-test. Validation chỉ1gXX, không enough cho
claim robust generalization. Giữ112 denominator,8 labels null, dữ liệu gốc/split,
QualityReports, raw cameraUNKNOWN, geometry temporal_analysis_eligible=false,
v1/v2 synthetic-only, G5.7 và assessment. Research admission không là official allow.

Đã có loader, model nhỏ, baseline/evaluator/training CLI; **chưa fit/train/baseline
evaluation**. User tự chạy sau bàn giao; quyền implementation không quyền cho agent
training. [Contract, hash, tests và lệnh](SQUAT_COUNT_TRAINING_PLAN.md).
READY FOR USER-RUN EXPERIMENT; NOT_VALIDATED/official=false/assessment=false.
Viewpoint33 chỉ AI unverified; không bắt buộc human annotation lại cho thí nghiệm
weak reference này, cũng không giả thay thế independent validation.

## Lịch sử proposal fixed-left chưa triển khai

2026-10-24. User chốt ưu tiên count → independent labels → controlled temporal
learning → form sau. Web/desktop giữ nguyên, phát triển UI và đa bài tập tạm dừng.

Contract mới trong `gym_coach.counting` là dataset/annotation/evaluation contract,
không temporal admission capability. Không sửa `SquatTemporalInput` synthetic-only,
v1/v2 producer, point-resolution abstention, profile hoặc QualityReport. Chưa có
real-video count producer hay baseline real evaluation. Candidate count không
confirmed repetition count; framewise geometry permission không temporal permission.

Đầu ra tương lai phân biệt estimated count, candidate-only, abstained, failed và
not_run; reference count riêng có available/unavailable/pending. Boundary partials,
bottom event/interval/unavailable, count không có exact boundaries đều thể hiện được.
Protocol nhãn video độc lập không cần pose/góc đúng hoặc point-turn physical truth.
Nhãn temporal không phải nhãn form, viewpoint hay joint-coordinate reference.

Spec §2 có allowance đếm thử nghiệm viewpoint unknown, nhưng acceptance hiện hành
G5.7 yêu cầu documented side + sequence/bilateral knees allow, và code temporal chỉ
nhận synthetic. Đây là **khác biệt cần quyết định riêng**, chưa tự áp dụng allowance
đó. Một future count-only admission (ví dụ fixed single-side) cần proposal, tests và
phê duyệt phạm vi; không dùng milestone data này để nới gate hiện tại. Nhãn video
có thể chuẩn bị cho cả video khó dù backend hiện abstain.

Mức readiness độc lập: decoder-readable; annotation-ready; labels-adjudicated;
baseline-admissible; training-approved. Đạt bước trước không tự đạt bước sau.
G5.7 vẫn blocked, G6/toàn G5 IN_PROGRESS, NOT_VALIDATED, official=false,
assessment=false, REFERENCE_PENDING. Training đầu tiên cần MODEL TRAINING GATE
và phê duyệt mới; request hiện tại không phải approval training.

## Quyết định phương pháp cần review — 2026-10-25, CHƯA PHÊ DUYỆT

Cập nhật evidence sau kiểm tra pose/quality phần thiếu: đã có112 pose/quality
records (41 reuse +71 fresh strict),0/112 đạt sequence allow +left metric allow;
33 strict sequence-allow clips có left chỉ bị chặn viewpoint_unknown.71 mới chưa
preprocessing. Chi tiết ở [readiness report](SQUAT_COUNT_DATA_READINESS_REPORT.md).
Các số41/71 trong lập luận proposal ban đầu bên dưới là trạng thái trước batch,
không phải số clip còn thiếu hiện nay. Proposal vẫn CHƯA PHÊ DUYỆT, policy không đổi.

Đọc video thành công chỉ chứng minh decoder/source timeline. Geometry schema0.4
ghi `measurement_scope=framewise_projected_2d`, `temporal_analysis_eligible=false`;
policy framewise có thể cho số dù movement-analysis sequence bị skip. Vì vậy
không được dùng số available để vượt sequence gate hoặc đổi ID thật thành
`synthetic-*`. `SquatTemporalInput` hiện chỉ synthetic; nhánh QualityReport đòi
side + sequence allow + đúng cả hai knee metrics/joints allow. Nhãn video độc
lập giúp đánh giá đáp án nhưng cũng không tự cấp admission cho signal.

**Phương án nhỏ nhất đề nghị phê duyệt:** một contract/adapter count-only opt-in
riêng cho BodyWeightSquats, output candidate/estimated/unavailable rõ nghĩa,
không phase/pause/form. Pilot khai báo gối trái cố định trước kết quả, không
fallback sang phải. Giữ sequence allow và canonical left-knee metric allow,
exact source/hash/timeline, observed/imputed/missing và boundaries/gaps. Không
đổi model, thresholds, QualityReport, v1/v2, G5.7 hay policy góc. Giảm yêu cầu
**hai bên** thành **bên đã khai báo** chỉ trong hợp đồng nghiên cứu mới, phải
được user duyệt; không tự xem framewise capability hiện hành là capability mới.
Viewpoint chưa xác nhận vẫn UNKNOWN/visually_assessed_unverified; metric skip
vì viewpoint vẫn skip. Input thiếu/gap làm whole-count unavailable, không biến
candidates ở đoạn nhìn rõ thành tổng count. Pure kernel có thể dùng lại sau
review, nhưng không lách synthetic-only constructor để nhận real input.

Evidence lịch sử trước feasibility: phạm vi này **chưa mở được clip nào trong41 saved snapshots**:40 sequence không
allow; clip g15_c03 sequence allow nhưng left/right metrics đều skip/viewpoint_unknown.
Khi đó 71 clip chưa có snapshot xác minh không được tính là fail hoặc đủ điều kiện.
Hiện 71 đã có pose/quality-only evidence; 0/112 vẫn đạt subset proposal. Worksheet
review viewpoint và pilot temporal mới không là approval phương pháp hoặc real admission.
Không đề nghị bỏ sequence gate, chuyển skip/degrade thành allow hoặc cho phép
unknown-view metric trong quyết định này. Muốn đổi thêm admission cần một quyết
định phương pháp khác có evidence, không gộp thành approval mặc định.

Rủi ro: phép chiếu2D/occlusion và một bên không đại diện toàn chuyển động; smoothed/
imputed signal có thể tạo hoặc dịch candidate; không có subject identity/holdout
chưa xem. Giữ abstention, full denominators, independent labels và frozen split.
User cần quyết định **có chấp nhận nghiên cứu count-only tách khỏi gate official,
với fixed-left/sequence-allow/left-metric-allow như trên hay không**. Chưa có
quyết định này nên không implementation/admission/baseline real trong lượt intake.
Training do user tự chạy sau MODEL TRAINING GATE; phê duyệt proposal không phải
quyền cho agent train. Bước nhãn video có thể tiến hành ngay, độc lập với proposal.
