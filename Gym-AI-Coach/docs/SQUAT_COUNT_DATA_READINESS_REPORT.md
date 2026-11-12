# Squat count — dữ liệu, nhãn và evaluator trước training

Chốt 2026-10-25 (Asia/Saigon); preparation bắt đầu2026-10-24. **POSE/QUALITY
FEASIBILITY EXECUTED / TEMPORAL_LABELS_PENDING / REAL_BASELINE_BLOCKED / TRAINING_NOT_RUN**.
Preparation/intake ban đầu không inference; lượt feasibility mới đã được user
cho phép inference phần thiếu, kết quả ở mục dưới. Không Git repository, không
commit/diff giả. Không thêm dependency, train, UI, action recognition, form
assessment hoặc sửa threshold/profile/gate.
NOT_VALIDATED, official_eligibility=false, assessment=false, REFERENCE_PENDING;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.

Run mới: [run-4c26d8ba677f42fb9d0751a11688b25d](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/).
Dùng **dataset-final/** để bàn giao; **dataset/** là lần đầu và được giữ nguyên.
[Hướng dẫn/CLI](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/HUONG_DAN_SU_DUNG.md).

## Review viewpoint và pilot temporal — bàn giao 2026-10-25

**VIEWPOINT_REVIEW_PENDING / TEMPORAL_LABELS_PENDING / TRAINING NOT_READY.**
[Trang bàn giao cho điều phối viên](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/START_HERE.md),
[33 clip + hướng dẫn viewpoint](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/VIEWPOINT_REVIEW.md),
[worksheet pending](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/viewpoint-review.pending.json).
Đây là chuẩn bị review, chưa xác nhận viewpoint hoặc gán nhãn temporal.
Không có Git repository; không giả commit/diff. Không decode/xem video, inference,
preprocessing, real counting, UI hoặc training trong lượt này.

### Evidence và lựa chọn đã khóa

Verify 661 file trong inventory run feasibility, 130 file dataset-final, byte SHA
của 112 nguồn. Giữ 112 pose/quality records: 71 inference mới của milestone trước,
41 snapshot reuse; chỉ 41 có preprocessing. Source camera UNKNOWN, QualityReport,
split 67/21/24, protocol và hai templates không thay đổi.

`selection-plan.json` được lưu trước khi trích danh sách; SHA-256
`2b125a89ae2b3eb30bef5d9c42c384dd66f6c2c73b35fbe97a0f6a20389a154b`.
Script `scripts/prepare_squat_review_pilot.py` chỉ chọn metadata từ evidence đã có.
Điều kiện viewpoint chính xác: cohort fresh_strict_unknown/reused_strict_unknown;
completed/reused; source UNKNOWN, report unknown; sequence eligible=true/action=allow;
left metric eligible=false/action=skip, required_joints đúng left_hip/left_knee/
left_ankle và reasons đúng một phần tử viewpoint_unknown. Kết quả **33**, sort path;
không chọn lại bằng hình ảnh. Mỗi record có path/ID/hash/FPS/frame_count/resolution,
toàn interval 0..N−1 và sample time 0..(N−1)/FPS, selection reason, trạng thái pending.
Reviewer/time/protocol reference/viewpoint kết luận đều null; phạm vi cần xem không
được trình bày như interval đã review. Không dùng coverage làm viewpoint evidence.

Code hiện chỉ có enum unknown/side/front/oblique và gate theo enum; không có validator
chứng minh annotation authority/interval/protocol. Docs SCHEMA_CONVENTIONS và G5.7
yêu cầu per-video source/hash + protocol ID/version/source + người/ngày xác nhận +
interval + hướng camera/framing/occlusion, nhưng chưa có rubric viewpoint được duyệt.
Hướng dẫn draft mới tái sử dụng các yêu cầu này, không đặt ngưỡng độ hoặc calibration.
Đổi góc phải ghi interval riêng; uncertain/transitions unknown, không majority-side,
không lấy đoạn thuận lợi đại diện toàn clip. Camera sau lưng chưa có enum riêng:
ghi unknown + notes thay vì tự map. Current gate một viewpoint/sequence chưa hỗ trợ
interval admission. Không sửa raw/report hoặc chạy gate lại trong lượt này.

**Quyết định cần chủ project phê duyệt:** rubric phân loại định tính, người có thẩm
quyền xác nhận và quy trình chấp nhận evidence/hash/interval. Đề xuất nằm ngay trong
hướng dẫn review, chưa được ban hành để mở gate. Việc này độc lập với proposal
count-only fixed-left (vẫn chưa duyệt), không tự mở G5.7 hay temporal admission.

### Pilot temporal độc lập với danh sách viewpoint

Chỉ dùng frozen split: sort train sources theo group_id rồi path, lấy clip đầu mỗi
gXX và 5 group đầu. Không đọc quality/predictions để chọn; không thay clip khó.

| Video pilot | gXX | Vai trò đã khóa |
|---|---|---|
|v_BodyWeightSquats_g01_c01.avi|g01|train|
|v_BodyWeightSquats_g05_c01.avi|g05|train|
|v_BodyWeightSquats_g06_c01.avi|g06|train|
|v_BodyWeightSquats_g08_c01.avi|g08|train|
|v_BodyWeightSquats_g09_c01.avi|g09|train|

Đây là pilot usability protocol, không representative accuracy sample/holdout.
Giao riêng [hướng dẫn A](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/RATER_A.md)
và [hướng dẫn B](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/RATER_B.md):
có link tới 5 video nguyên bytes, protocol/manifest mù và đúng template riêng đã có.
Không nhân bản video hoặc tạo lại schema/package. Hai templates vẫn **112 records**;
sau khi làm pilot, 107 record chưa xem của mỗi lượt phải vẫn pending. Hiện toàn
224 pass records pending, trong đó 10 lượt pilot pending; 0 final/ground truth.
Mỗi người tự khai provenance thật, xem toàn clip, giữ count/boundaries unavailable
khi không rõ, không đoán 0. Lưu file mới dưới annotations/squat_count/<session-moi>/;
không sửa template. CLI intake hiện có được hướng dẫn, chỉ chạy khi có submission.
Không giao quality/viewpoint/split/predictions hay nhãn người kia. File system local
không enforce blinding: điều phối viên phải giao riêng; chưa có rater hoặc blinding
thực tế nào được xác nhận trong lượt này.

### Intake và kiểm thử thực tế

Search tên và nội dung metadata trong project active, không ngoài project; exclusions
env/archive/retired/generated test-temp/source-snapshot/prior intake receipts được lưu.
23 name hits; 13 temporal content hits là 4 pending templates, 4 pending bundles,
4 schema và 1 pixel template. 105 viewpoint content hits là metadata/schema/index
hoặc agent notes; reviewer tìm được chỉ Codex agent, status visually_assessed_unverified,
không có documented=true hoặc human-confirmed record. Không tìm được submission mới.
Typed source/hash/frame-time/protocol contracts của các temporal hits được kiểm tra;
historical dataset và dataset-final phân biệt bằng manifest hash. Không coi copies
là rater độc lập; không chạy lại assemble/evaluator trên pending để tạo report trùng.
Search content lần đầu thiếu explicit root đã đọc piped stdin; receipt final dùng
`.` mới là evidence search project, kết quả đầu không dùng kết luận.

**Tiếp nhận mới: 0; completed independent clips: 0/112; adjudicated: 0/112;
agreement denominator: 0, agreement=null/NOT_EVALUATED.** Schema/linkage hợp lệ
không chứng minh nhãn đúng hoặc người gán độc lập. Dừng phần phụ thuộc nhãn tại đây.

Tests mới: **24 passed / 0 fail/error/skip/warnings (0,55s)**. Suite liên quan:
**113 passed / 0 failed/error/skipped/deselected/warnings (4,32s)**, bao gồm 24 tests
mới nên không cộng hai lượt thành 137 test khác nhau. Command:

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_squat_review_pilot.py tests/test_squat_counting.py tests/test_pose_feasibility_batch.py -q -p no:cacheprovider --tb=short
```

Exact command có basetemp/JUnit, logs và receipts nằm trong run mới. Kiểm tra chọn
train/group cố định, không phụ thuộc quality, source/split/hash/timeline, cohort đủ,
canonical joints, pending/null, JSON roundtrip, không sửa input và tamper/path escape.
Không chạy full suite, browser QA, real-data inference/counting hoặc tính accuracy.
Chỉ thêm script/test chọn danh sách, hướng dẫn/evidence mới và cập nhật docs hiện có;
không đổi core/schema/profile/dependency. Checklist 71 thiếu snapshot phía dưới đã
gắn nhãn lịch sử, không xóa evidence preparation.

Tiếp theo: người review viewpoint và hai rater thực hiện các file đã giao; reviewer
thứ ba adjudicate sau intake. Training **NOT_READY** vì chưa labels/adjudication,
documented input/gate/admission real và baseline; targets/loss/pairing/resource chưa
freeze. Khi đủ MODEL TRAINING GATE chỉ bàn giao config/lệnh cho **user tự chạy**.
Giữ NOT_VALIDATED, official_eligibility=false, assessment=false, G5.7 blocked,
G6/toàn G5 IN_PROGRESS; tests không chứng minh model accuracy.

## Cập nhật pose/quality trên phần thiếu — 2026-10-25

**Đã chạy thực tế71 clip mới, tái sử dụng41 snapshot, đủ112 nguồn; training NOT_READY.**
[Run/evidence](../outputs/squat_pose_feasibility/run-d9d05ac71671461390499a419e766e14/),
[bảng đủ112 clip](../outputs/squat_pose_feasibility/run-d9d05ac71671461390499a419e766e14/RESULTS_112.md),
[CSV đầy đủ source/hash/reasons](../outputs/squat_pose_feasibility/run-d9d05ac71671461390499a419e766e14/results-112.csv),
[JSON theo từng clip/condition/joint](../outputs/squat_pose_feasibility/run-d9d05ac71671461390499a419e766e14/summary-51245fe0a31e46ebb1ecc6c532e8fb5b.json).
Các mục preparation/intake phía dưới là evidence ở thời điểm trước inference.

### Danh sách và cấu hình đã khóa trước inference

Đối chiếu source bytes đủ112 với dataset-final và decoder receipts, dùng lại
validators của pipeline/counting readiness cho saved artifacts. Xác minh lại40
batch snapshots và1 strict single-video snapshot; search metadata hiện có không
tìm thêm source chưa được phủ.28 pose-only manifest aliases đều thuộc clip đã có
snapshot, không coi là28 clip mới.0 artifact incompatibility trong các snapshot
được chọn; aliases chỉ catalogued, không dùng thay snapshot đã verify.
Không chạy lại readiness, tạo annotation package hoặc chia lại dữ liệu.

`plan.json` khóa112 source IDs/paths/hashes,41 reuse/71 infer, config/profile/
model/registry và code identities. SHA-256 plan:
`b271e605f24cff08b6b33f79731bb8fc5fdcb9afc1cba36e074ff9a94260a9f9`.
Nguồn duy nhất là class BodyWeightSquats hiện có, không loại clip khó. First-missing
theo thứ tự path là `v_BodyWeightSquats_g01_c04.avi`: smoke174frames, pose174,
sequence skip; left/right skip giữ nguyên. Chỉ sau readback/hash/schema/timeline/
canonical gate đạt mới chạy70 còn lại, smoke được verify/reuse chứ không infer lại.

Orchestrator mới `scripts/squat_pose_feasibility.py` gọi nguyên `pose.run.extract_video`
và `QualityGate`, không xây extractor mới. MediaPipe0.10.14, bundled full model
SHA-256 `e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f`,
pose config hash `3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976`,
quality-s3-v1 hash `514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704`.
Hai knee keys lấy canonical joints/viewpoint requirements từ registry hiện hành.
Raw pose và quality mới đều unknown; không `--assume-side`, rotation hoặc promotion.
Không preprocessing mới: đủ pose/quality để đánh giá các prerequisites; processed
count-input gap policy ghi UNKNOWN.41 snapshot cũ có preprocessing;71 mới chỉ
pose/quality, **không gọi112 là112 full-preprocessing snapshots**.

### Kết quả đo, tách đúng cohort

Các quyết định quality dưới đây áp dụng **toàn clip**, không phải gate từng frame.
A/D/S lần lượt allow/degrade/skip. Không gộp các cohort thành quality rate đồng nhất.

| Cohort |Clips|Frames|Pose / không pose|Sequence A/D/S|Gối trái A/D/S|Gối phải A/D/S|
|---|---:|---:|---:|---:|---:|---:|
|Fresh strict UNKNOWN|71|8.905|8.899 /6|32 /13 /26|0 /0 /71|0 /0 /71|
|Reused strict UNKNOWN|1|137|137 /0|1 /0 /0|0 /0 /1|0 /0 /1|
|Reused exploratory side hypothesis|40|5.314|5.311 /3|0 /5 /35|21 /5 /14|14 /5 /21|

Decode mới71/71 thành công; decode evidence của41 reuse vẫn khớp source hash.
Pose extraction mới71/71 thành công; tổng112 có pose/quality evidence,0 chưa xử
lý,0 lỗi xử lý clip.6 frame mới không phát hiện pose giữ missing/provenance và
nằm trong mẫu số; không có clip mới mất pose toàn bộ. Không raw-joint coverage
nào được dùng như accuracy. Tổng cohort14.356 frames, không cộng nhiều lần smoke.

Fresh strict left:71/71 có reason viewpoint_unknown;48 chỉ có reason này, trong
đó32 sequence allow. Ngoài viewpoint, left có20 low_pose_visibility,7 out_of_frame,
3 excessive_missingness; right có17/9/6 tương ứng. Các reason counts có thể chồng
lắp, không cộng thành số clip. Sequence fresh có21 low_pose_visibility,9 out_of_frame,
5 excessive_missingness và13 partial_joint_coverage.32 allow cho thấy không phải
mọi clip đều thất bại pose-quality; chưa chứng minh viewpoint side hay signal accuracy.

### A: gate hiện hành; B: proposal count-only chưa được phê duyệt

- A: **0/112 đáp ứng đầy đủ**. Không có documented side evidence trong nguồn
  đã xác minh; raw camera vẫn UNKNOWN. Không clip nào đồng thời sequence allow
  và cả hai knee metrics allow trong các reports được sử dụng. Synthetic-only
  temporal contract cũng chưa nhận real video. G5.7 giữ blocked.
- B: **0/112 đạt subset quality sequence allow + left metric allow**. Method
  approval=PENDING, fixed-left temporal adapter=NOT_IMPLEMENTED. Source/hash và
  timeline/raw provenance được verify; **processed count-input gap policy=UNKNOWN**
  vì không xây/chạy adapter hoặc real counting. Không coi việc chưa kiểm tra là PASS.
- 32 fresh +1 reused strict có sequence allow và left chỉ bị chặn viewpoint_unknown.
  Đây là danh sách ứng viên để con người kiểm tra metadata, **không** là33 input
  temporal eligible, không counterfactual report đã bỏ viewpoint gate.

Nút chặn chắc chắn: viewpoint metadata thiếu ở strict metrics và contract temporal
chưa cấp quyền real input; một phần clip còn bị chất lượng quan sát chặn. Chưa có
nhãn độc lập để kết luận threshold quá chặt, model không chính xác hoặc dataset
không phù hợp counting. Không suy góc quay từ pose/coverage và không đổi policy.

**Quyết định khuyến nghị:** tiếp tục dùng dataset làm development/annotation,
ưu tiên review viewpoint có protocol cho33 clip sequence allow nêu trên, kèm
source path/ID/hash/interval thật; giữ toàn112 trong cohort và các video khó trong
thống kê. Đây là công việc tiếp theo được đề nghị, chưa gán viewpoint/labels trong
lượt này. Hai rater temporal độc lập và adjudication vẫn cần thực hiện. Nếu chọn
count-only fixed-left khác gate bilateral, user phải phê duyệt proposal riêng;
việc phê duyệt đó cũng không tự xóa metric skip do viewpoint.

### Thời gian, lỗi, tests và tiếp tục batch

71 worker attempts thành công cho71 unique videos,8.905 frames. Tổng worker
pose+readback+quality427,25s; smoke+continuation wall time497,42s (8,29phút), bao
gồm process startup/verification. Một bộ tests ngắn chạy chồng thời gian batch;
đây là operational timing, không controlled benchmark.2 invocations inference,
1 smoke result reused;0 retry/backend failure. Không train/fine-tune/optimizer/fit.

Logs giữ đầy đủ theo attempt:71 protobuf `GetPrototype` deprecation warnings,
142 native feedback-manager warnings, cùng TFLite/absl diagnostics. Không tuyên
bố inference warning-free; các warnings này không tự đổi profile hoặc gate.

New orchestration tests ban đầu9 pass/1,04s; final related suite **152 passed,
4 integration deselected,0 failed/error/skipped/pytest warnings,7,27s** (gồm11
batch tests cuối, các suite có overlap). Kiểm tra strict viewpoint/canonical joints,
tamper source/frames/quality/plan, resume không infer lại, incomplete/failure giữ
mẫu số, đủ112 rows, strict/exploratory cohorts riêng và không promote counting.
Exact command trong `tests-related-command.json`; lệnh chính:

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_pose_feasibility_batch.py tests/test_pose.py tests/test_quality.py tests/test_video.py tests/test_squat_counting.py -m "not integration" -q -p no:cacheprovider --tb=short
```

Actual71-video inference là evidence thực riêng, không giả unit fixtures thành
real integration. Không chạy toàn project suite/browser QA/temporal/accuracy/
agreement. Tests pass chỉ chứng minh phần mềm; không chứng minh pose/count accuracy.

[RUN_COMMANDS.txt](../outputs/squat_pose_feasibility/run-d9d05ac71671461390499a419e766e14/RUN_COMMANDS.txt)
ghi smoke/continuation/summarize. Resume chỉ sau khi parent/worker cũ đã dừng;
không hỗ trợ hai batch cùng chạy đồng thời. Completed output phải qua full hash,
schema, timeline/config và recomputed QualityGate trước skip. Invalid/incomplete
attempt giữ nguyên, retry vào directory mới. Backend/process lỗi một clip được
ghi riêng và tiếp tục; source/config/code identity đổi thì dừng để review, không
tự cập nhật frozen plan. JSON summary giữ112 rows kể cả failed/not_processed.

Chỉ thêm script/test audit và outputs mới; không sửa core producer/schema/profile,
không thay dataset/split/protocol/nhãn hoặc tạo overlay. Readiness **NOT_READY**:
thiếu temporal labels/adjudication, documented viewpoint/input đúng gate,
real count admission được chấp nhận và baseline/targets/loss/pairing được freeze.
Training do **user tự chạy** sau MODEL TRAINING GATE và bàn giao config/lệnh;
agent không chạy training ở bất kỳ dạng nào. NOT_VALIDATED, official=false,
assessment=false, G5.7 blocked, G6/toàn G5 IN_PROGRESS giữ nguyên.

## 0. Tiếp nhận nhãn thực tế — kiểm tra2026-10-25

**TEMPORAL_LABELS_PENDING; training readiness=NOT_READY.** Đây là cập nhật intake
trên gói đã có, không readiness pipeline mới. Evidence mới tại
[run-8cb1e1b7fe184e329167d44685edb6e9](../outputs/squat_label_intake/run-8cb1e1b7fe184e329167d44685edb6e9/).
`docs/MASTER_PROMPT.md` không có trong workspace; dùng Prompt tổng Gym hiện hành
và quy định phương pháp của `.agent/AGENT_RULES.md`, không dùng trạng thái legacy.

Đã tìm theo tên file và nội dung JSON/JSONL/CSV/TSV/YAML/TXT trong project active,
kể cả outputs. Loại môi trường/.git/node_modules, archive/retired, generated
test-temp và source-snapshot; danh sách/lệnh/phạm vi ở search receipts.13 content
hits:4 pass templates pending,4 adjudication bundles pending,4 schema files và1
pixel-reference template.12 CSV candidates là audit/coverage/generated test
inventory, không temporal labels. Không submission mới, không draft temporal được
tìm thấy trong phạm vi này. Các bản sao dataset/dataset-final không là rater mới.

| Trạng thái nhãn trên cohort112 | Kết quả |
|---|---:|
| Clip có completed independent temporal pass đủ linkage/provenance khai báo |0/112|
| Clip adjudicated đủ điều kiện |0/112|
| Cặp đủ điều kiện tính count/boundary agreement |0|
| Agreement |null / NOT_EVALUATED|

Schema chỉ kiểm tra declarations/linkage, không chứng minh đúng/độc lập/chuyên môn.
Đã validate toàn bộ source/hash/protocol/frame-time contract trong pass/bundle
hits; historical dataset links được tách khỏi dataset-final. Verify329 artifact
của run cũ và130 files dataset-final;112 video nguồn khớp112 bản sao. Gọi
`counting.annotations.assemble` hiện có trên hai templates:112 PENDING_TWO_PASSES,
0 auto-adjudicated. Không nhân bản templates/bundle mới, không tự tạo người/time,
không tính agreement trên pending. Chưa tìm thấy lỗi cản trở giao/import nên
**không sửa code/tests/schema/profile**. `package-check.json` và classification
receipt ghi kết quả; split/protocol/outputs cũ nguyên vẹn.

### Bàn giao ngay cho người gán nhãn

Gốc package:
`D:\KLTN_K28_TKD\outputs\squat_count_readiness\run-4c26d8ba677f42fb9d0751a11688b25d\dataset-final\annotation_package\`.

- Video: [videos/](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/dataset-final/annotation_package/videos/).
- Protocol đúng bytes đã khóa: [PROTOCOL.md](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/dataset-final/annotation_package/PROTOCOL.md).
- Người A: [pass_a.blank.json](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/dataset-final/annotation_package/pass_a.blank.json).
- Người B: [pass_b.blank.json](../outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/dataset-final/annotation_package/pass_b.blank.json).

Mỗi người nhận video/manifest/protocol và **chỉ một template**, không predictions,
quality, split hoặc lượt của người kia. Tự xem toàn clip, chọn subject quan sát,
gán count/cycles/Bottom/partials/uncertain theo protocol; tự ghi tên/phương pháp
`manual_video_review`/thời điểm thật có timezone/provenance. Chỉ khai independent/
blinded=true khi thực sự đáp ứng. Không đoán count hoặc boundaries không nhìn rõ.

Lưu bản mới ngoài package, ví dụ `annotations/squat_count/<session-moi>/pass_a.completed.json`
và `pass_b.completed.json`; đường dẫn này là quy ước lưu được đề nghị, **chưa có
submission tại đó**. Giữ envelope version/hash và đủ112 records; clip chưa xem
giữ pending, không xóa khỏi danh sách. Record đã xem đổi status=completed, điền
count_status và boundaries_status riêng. Available count cần count_reason=null;
exhaustive boundaries cần boundaries_reason=null và đủ cycles. Unavailable count
giữ null + reason, không thay bằng0. Filename completed không tự làm mọi record
completed. Không sửa source IDs/hash/FPS/frame_count/protocol hoặc template gốc.

Sau khi có hai file, từ project root dùng công cụ đã có (thay `<session-moi>` bằng
thư mục thực; output review phải mới):

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.counting.annotations --manifest outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/dataset-final/dataset-manifest.json --pass-a "annotations/squat_count/<session-moi>/pass_a.completed.json" --pass-b "annotations/squat_count/<session-moi>/pass_b.completed.json" --output-dir "outputs/squat_label_intake/<review-moi>"
```

Lệnh này **chưa chạy với completed labels**, vì chưa có file. Nó tạo disagreement
để người thứ ba xem/phân xử, không tự chọn đáp án. Người phân xử phải lưu final
bundle mới + log/hash/time theo protocol. Nếu chưa phân xử thì evaluator không
dùng như final GT. Không tạo labels/evaluator accuracy thay cho thao tác người.

### Blocker và trách nhiệm

| Điều kiện | Evidence/trạng thái | Ai thực hiện |
|---|---|---|
| Hai lượt nhãn và phân xử |0/112 completed,0 adjudicated; PENDING|Hai rater độc lập và người phân xử thứ ba|
| Temporal admission real |Synthetic-only; framewise `temporal_analysis_eligible=false`|User phê duyệt phương pháp; triển khai/test chỉ sau quyết định|
| Input đủ gate (lịch sử intake trước feasibility) |Khi đó 40/41 sequence skip/degrade;1 allow có hai metrics skip;71 chưa snapshot. Hiện đủ112 pose/quality, 0/112 A/B quality subset|Giữ policy; xem cập nhật feasibility/review phía trên|
| Baseline thật |NOT_RUN; không có prediction được đo trong lượt này|Sau admission + nhãn đủ điều kiện, chạy/evaluate có provenance|
| Training |NOT_READY; targets/loss/pairing/resource config chưa freeze|Agent bàn giao gate/config/lệnh khi đủ; **user tự chạy**|

Phương án tối thiểu và rủi ro trong [proposal đã cập nhật](SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md):
count-only opt-in riêng, fixed-left, giữ sequence allow và left metric allow,
không đóng G5.7; không đề nghị hạ threshold hoặc promote skip/degrade. Cần user
phê duyệt sự thay đổi bilateral→fixed-side cho nghiên cứu; hiện vẫn0/41 saved clips
đủ rule này. Không biến metadata-readable/angle-available thành counting-ready.

**Training luôn do user tự chạy**: khi đủ điều kiện mới trình MODEL TRAINING GATE
và lệnh/cấu hình thật. Không training/fine-tuning/optimizer/smoke train ở lượt này
hoặc tự chạy thay user ở lượt sau. Real baseline, inference, readiness rerun đều0.
Không đổi split67/21/24, không gọi development-test là unseen-subject holdout.
Giữ NOT_VALIDATED/official=false/assessment=false, G5.7 blocked, G6 IN_PROGRESS.
Nhánh phụ thuộc nhãn dừng tại TEMPORAL_LABELS_PENDING; không hoàn tất ground truth.

Kiểm thử lượt intake: `.venv/Scripts/python.exe -B -m pytest tests/test_squat_counting.py
-q -p no:cacheprovider --tb=short` kèm basetemp/JUnit trong run mới: **78 passed,
0 failed/error/skipped/deselected/warnings,2,26s**. Exact command/log/XML lưu tại
`tests-command.json`, `tests.txt`, `tests.xml`. Các preparation tests dùng fixtures/
mocks; không chạy lại readiness trên dataset thật. Không chạy full suite, real
inference integration hoặc browser QA. Tests xác nhận phần mềm, không accuracy
hay agreement của model/người gán. Có kiểm tra hash/link và bảo toàn sau cập nhật;
không sửa source/tests/profile/split/protocol hay artifacts cũ.

## 1. Hướng nghiên cứu và mức dùng được

Ưu tiên người dùng: BodyWeightSquats → baseline count → nhãn độc lập → temporal
model đối chứng → form sau. Web/desktop, đa bài tập và UI history tạm dừng phát
triển, giữ sản phẩm/evidence. Đã đối chiếu ground-truth spec, master prompt,
state/tasks/plan, pose/quality/preprocessing, synthetic temporal/v2/point-resolution,
independent2D reference, code và saved artifacts. Trạng thái Post-Temporal trong
`.agent/AGENT_RULES.md` là lịch sử, không thay trạng thái hiện hành.

Code `SquatTemporalInput` hiện chỉ nhận synthetic contract test. PVP S3 là baseline
thuật toán chạy được trên synthetic, chưa có capability real count. V2 chỉ giữ
anchor candidates; point-resolution abstain khi không có noise discriminator.
Sensitivity cũ30 conditions:102 full reference cycles,78 matched/24 missed,
18/30 count match; đây là kết quả synthetic lịch sử, không phép đo trên người.
Không chạy lại hoặc tune detector trong milestone này.

Spec có allowance experimental count khi viewpoint unknown, nhưng acceptance
G5.7 và code hiện nghiêm hơn. [Contract proposal riêng](SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md)
ghi khác biệt này; chưa chấp nhận admission mới hoặc biến framewise permission
thành temporal authorization. Candidate count không là số rep đã xác nhận.

## 2. Kiểm kê thực tế toàn lớp — lịch sử preparation trước feasibility

Chỉ video trong `Data/UCF-101_GYM/BodyWeightSquats/`. Không đọc/decode video ở
dataset khác hay archive. Đọc audit CSV đã có, lọc đúng lớp; liên kết visual audit
cũ bằng hash, không xem lại để gán nhãn temporal. Full decode mới ở lần prepare
đầu; dataset-final tái dùng decode receipts sau kiểm tra hash, không báo hai lần
decode là hai tập độc lập. Không crop/rotate/chuyển mã hoặc sửa nguồn.

| Hạng mục | Kết quả đo/kiểm tra |
|---|---:|
| Paths / unique byte SHA-256 / nhóm gXX |112 /112 /25|
| Tổng bytes nguồn |40.447.532|
| Full decode thành công / thất bại |112 /0|
| Tổng frames / duration N/FPS |14.356 /574,24s|
| FPS / kích thước |112 clip đều25FPS,320×240|
| Metadata khớp audit cũ |112/112|
| Verified saved pose/quality/preprocessing |41 clip,5.451 frames|
| Chưa có reusable snapshot tại thời điểm preparation |71 clip; hiện đã có pose/quality-only ở run feasibility mới|
| Problems / unclassified scope entries |0 /0|
| Nhãn temporal final đủ điều kiện tìm được |0|

Frame/time dùng `fps_derived_cfr_assumption`, sample cuối=(N−1)/FPS. Chưa kiểm
PTS/VFR hoặc accuracy timestamps vật lý. Không duplicate bytes trong112 clip;
không suy ra không có cảnh/người trùng. gXX là data group, không athlete identity.
Các same-source links khác gXX chưa xác minh, không giả metadata riêng cho bản sao.

Tái sử dụng validators có sẵn cho40 batch snapshots, kiểm tra source bytes,
manifest/files, raw→processed linkage, timeline, model asset/config và profile.
Tìm thêm `v_BodyWeightSquats_g15_c03.avi` ngoài fixed40 trong saved desktop run;
kiểm tra manifest đầy đủ và typed single-video artifacts tương tự. Source SHA:
`38c8500955b0a3abd565a5f8c593ec68b40c2e644a3a41bef0a3270290749880`.
137frames, strict sequence allow nhưng **cả hai knee metrics skip/viewpoint_unknown**.
Saved manifest/path đầy đủ trong per-video row; không chạy lại MediaPipe.

| Gate trong41 saved snapshots |allow|degrade|skip|Mẫu số|
|---|---:|---:|---:|---:|
| Sequence |1|5|35|41 clips|
| Hai knee metrics |35|10|37|82 clip–metric pairs|

Đây là **quyết định cấp clip đã lưu**, không per-frame gate hay đo accuracy.
40 clip dùng giả thuyết side chưa xác nhận; clip bổ sung strict/unknown. Không
gộp các protocol đó thành tỷ lệ official eligibility.71 không snapshot vẫn nằm
manifest/split/template/mẫu số; không gọi chúng là skip hoặc chạy inference bù.
Không clip nào được cấp real temporal admission. Cả112 source camera_view UNKNOWN;
prior visual review là visually_assessed_unverified, không documented side.

Metadata search đọc206 tài liệu outputs không thuộc tests/snapshot/archive; audit
annotation-named JSON tìm gói pixel-reference trống có version
`squat-blinded-pixel-reference-v1`. Không có qualified temporal bundle trong phạm
vi đã kiểm tra. Đây là search có phạm vi, không chứng minh không thể có file nhãn
ở bất kỳ vị trí khác. Bộ60 ảnh pixel vẫn dành cho measurement, không thay nhãn count.

## 3. Protocol, schema và package

[Protocol freeze](SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md) định nghĩa một excursion
quan sát từ trên→dưới→trên, không yêu cầu form/depth chuẩn. Shared top, hold,
Bottom event/interval/unavailable, partial biên clip và uncertain intervals tách
riêng. Count0 cần quan sát đủ, unknown=null; count có thể rõ khi boundaries chưa rõ.
Mất hình/cut/occlusion/camera motion/subject ambiguous không nối chu kỳ để đoán.

Code mới `src/gym_coach/counting/`: models, readiness, split, annotations,
evaluation. Không sửa public temporal schemas/API/producer cũ. Version labels
`squat-count-labels/0.1.0`; dataset/adjudication/prediction/evaluator dùng các
version riêng được ghi trong [schema conventions](SCHEMA_CONVENTIONS.md).
Strict types, finite values, source/hash/CFR và adjudication links được kiểm tra;
model instances cũng revalidate trước serialization, không bool→count1.

`dataset-final/annotation_package/` chứa112 bản sao byte-identical của video gốc,
lean source manifest, PROTOCOL.md, HUONG_DAN.txt và hai pass templates. Mỗi pass112
records pending; annotator/time/method/subject/count=null, cycles rỗng. Không
model predictions/quality/viewpoint assumptions/split trong gói rater. Giao riêng
passA/passB, không cho xem kết quả nhau. Công cụ assemble tạo disagreement review,
**không tự adjudicate**; final đòi người thứ ba/log/hashes/time. Hash/schema không
chứng minh năng lực hoặc tính trung thực của người gán nhãn.

`private/` giữ exposure, label search và catalog evidence; không giao cho raters.
Predictions112 not_run ở ngoài gói mù. Không tạo nhãn bằng extrema/pose/agent review.

## 4. Split và exposure đã khóa

Freeze seed `squat-count-groups-20261024-v1` trước labels/performance. Union theo
gXX, duplicate bytes và verified same-source links nếu có; SHA-256(seed + component
IDs) xếp deterministic;60/20/20 theo component (floor cho train/validation).
Không random frame/window, không chọn lại theo performance. Các cửa sổ/bytes
trùng/source liên quan đã xác minh phải ở cùng partition.

| Partition |Groups|Clips|gXX|
|---|---:|---:|---|
|train|15|67|01,05,06,08,09,10,11,12,17,18,19,20,21,22,23|
|validation|5|21|02,03,04,07,13|
|development_test|5|24|14,15,16,24,25|

Cả112 đã có visual audit exposure;41 có saved pipeline evidence. Ledger liên kết
clip với audit và các runs liên quan, không coi việc không thấy một run là bằng
chứng chưa từng dùng. **Không có pristine holdout được thiết lập; không bảo đảm
subject isolation.** Muốn claim generalization phải thiết kế cohort độc lập với
nguồn subject/capture đáng tin ở milestone sau, không đổi tên development-test.
Fit normalization/feature selection trên train; validation chọn cấu hình, test
không tune. Freeze ở đây không cấp quyền training hoặc chứng minh đủ nhãn.

## 5. Evaluator đã chạy với pending labels

CLI nhận manifest + adjudication collection + prediction collection, bắt buộc
version/source/protocol và byte-hash linkage. Giữ mọi112 nguồn, unknown khác0,
candidate-only khác estimated. Count MAE/signed(pred−ref)/exact trên comparable
clips; coverage/abstain/failure/not_run trên toàncohort. Event matching chỉ với
exhaustive labels/outputs: monotone max-cardinality rồi min boundary error;
tolerance0,20s S3 cấu hình đã khóa trước labels, không tiêu chuẩn chuyên môn.
Partial/uncertain inclusive masks loại predicted events và báo riêng; không xóa
count error. Timing Bottom chỉ point-vs-point, không lấy midpoint hoặc pause0.

Run **evaluation-final-pending/**:112 manifest rows,112 predictions not_run với
reason `real_temporal_admission_blocked`,0 adjudicated labels,0 comparable clips,
prediction/label/comparable coverage0/112. Not-run112/112; abstain0/112 và failure0/112
chỉ là trạng thái records, **không phải baseline đã chạy và không thất bại**.
MAE/signed/exact/precision/recall/timing=null, status=NOT_EVALUATED. Assembly giữ
112 pending bundles, auto_adjudicated=0. Chưa real baseline accuracy.

## 6. Lệnh, môi trường và test evidence

Môi trường thực: Python3.11.9, Pydantic2.13.5, pytest9.1.1, OpenCV5.0.0.93,
MediaPipe0.10.14 đã cài; Windows build26200. Không import inference để tạo pose,
không thay dependency và không claim clean-install compatibility. Model asset
đối chiếu `e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f`;
quality profile `514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704`;
S3 temporal profile giữ `5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`.

Từ project root, `.venv/Scripts/python.exe -B -m` với modules:

- `gym_coach.counting.readiness --output-dir <run>/dataset`:30,05s, full decode112.
- `gym_coach.counting.readiness --output-dir <run>/dataset-final --reuse-decode-from <run>/dataset`:
 24,43s, reuse112 verified decode records, thêm compatibility g15_c03.
- `gym_coach.counting.annotations --manifest <dataset-final>/dataset-manifest.json --pass-a <dataset-final>/annotation_package/pass_a.blank.json --pass-b <dataset-final>/annotation_package/pass_b.blank.json --output-dir <run>/review-final-pending`.
- `gym_coach.counting.evaluation --manifest <dataset-final>/dataset-manifest.json --annotations <run>/review-final-pending/adjudication.pending.json --predictions <dataset-final>/predictions.not_run.json --output-dir <run>/evaluation-final-pending`.

Các `<...>` ở trên viết tắt; exact command arrays, exit codes/thời gian/stdout nằm
trong `*-command.json` và `*.txt` tại run. Mọi output-dir phải mới; không ghi đè.

| Test receipt |Passed|Failed|Error|Skipped|Deselected|Pytest warnings|pytest seconds|
|---|---:|---:|---:|---:|---:|---:|---:|
|tests-first|67|2|0|0|0|0|1,98|
|tests-contracts (sau sửa)|69|0|0|0|0|0|1,56|
|tests-toolkit|77|0|0|0|0|0|2,16|
|tests-readiness-final|78|0|0|0|0|0|2,55|
|tests-related|423|0|0|0|5|0|72,25|
|tests-delivery|90|0|0|0|0|0|6,33|

Initial failures được giữ nguyên: forged bool count có thể bị JSON coerce, đã
sửa strict native revalidation; test helper metadata=None chia trước khi đến
validator, đã sửa fixture. Không còn failure ở các lần sau. Các lần chạy chồng
lắp test cases, **không cộng totals thành số unique tests**.

Exact regression command (receipt có basetemp/JUnit path):

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_squat_counting.py tests/test_temporal_squat.py tests/test_temporal_v2.py tests/test_temporal_v2_renderer.py tests/test_point_resolution.py tests/test_temporal_sensitivity.py tests/test_squat_feedback.py tests/test_squat_reference.py tests/test_video.py tests/test_quality.py tests/test_single_video_pipeline.py -m "not integration" -q -p no:cacheprovider --tb=short --basetemp outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/tests-related-temp --junitxml=outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/tests-related.xml
```

New tests kiểm tra strict/roundtrip/provenance/pending,25/30/60FPS timeline,
full/partial/shared-top/uncertain, disagreement/adjudication/hash, count metrics
tính tay, misses/extras/masks/timing/tolerance/matching, đủ112-denominator,
duplicate/group isolation, frozen ordering, safe scope/no-overwrite, decode
failure/reuse tamper, package isolation và CLI assembly/evaluation. Unit toolkit
mock decoder/saved data rõ ràng; actual112 full decode là preparation receipt,
không giả unit fixtures thành real accuracy/inference test. No heavy import guard.

Không full project suite hoặc real-data inference integration;5 integration
tests deselected có chủ đích. Không browser/UI QA mới: NOT_RUN, UI tạm dừng.
Delivery chạy `tests/test_squat_counting.py tests/test_web_delivery.py` với cùng
`-m "not integration" -q -p no:cacheprovider --tb=short`, basetemp/JUnit mới tên
tests-delivery; exact command trong receipt.90 passed gồm78 counting +12 retained
entrypoint/documentation tests. Không server/browser/inference trong test delivery.

## 7. Artifacts, files và giới hạn

Final dataset manifest SHA-256:
`17ccbceb71b2c5d031ef80851e13aeec7513da417d7d114cf3e4e4add7c9daf1`.
Split SHA-256:
`ad5d66c73151f81417931962ae4c165fb2d298593b5ba85b7601bdfa039f1613`.
Protocol bytes SHA-256:
`a908d4a8bd754e9f80e1046db0e70c0fb82e827f99571a2e3055ad3ad50cd354`.
Pending evaluation SHA-256:
`d0adc9c8a4832dc75ce253c4cf6987553be81694a2b8bd30f4b90c59c695803a`.
`dataset-final/artifact-inventory.json` liệt kê từng file/hash/size, bao gồm source
copies; receipts bind inputs/outputs. Hash chỉ xác nhận bytes/linkage, không accuracy.

New source: `counting/__init__.py`, `models.py`, `split.py`, `readiness.py`,
`annotations.py`, `evaluation.py`; tests `tests/test_squat_counting.py`.
New docs: báo cáo này, annotation protocol, research contract proposal, training
plan. Cập nhật README/PROJECT_STATE/TASKS/IMPLEMENTATION_PLAN/SCHEMA_CONVENTIONS/
DOCUMENTATION_INDEX. Existing source/tests/profile/schema/UI/evidence không sửa;
không cleanup/archive move/delete. Các artifacts milestone này đều ở run mới.

Integrity cuối đạt:299 baseline file identities,293 nguyên vẹn và6 tài liệu thay
đổi đúng phạm vi,0 missing/unexpected changes;112 source videos không đổi,112 bản
sao đúng bytes,224 pass records pending. Verify130 files mỗi dataset/dataset-final
và66 local documentation links. `delivery-integrity.json` cùng command/log lưu
bằng chứng. Lần đầu offline verifier thiếu field `no_model_predictions` trong
expected key set nên fail; đã sửa verifier để bắt buộc field đó là true. Package
không thay đổi; giữ first verifier/receipt và final pass riêng. Đây không phải
pytest failure mới hoặc lỗi dữ liệu. Snapshot code/docs cuối và run-file inventory
được lưu riêng; test scratch folders vẫn giữ nhưng không là artifact bàn giao.

## 8. Bốn câu trả lời bàn giao — lịch sử preparation trước feasibility

1. **Dùng được gì?**112 video decoder-readable có source/timeline identity, sẵn
 để gán nhãn độc lập.41 saved pose/quality/preprocessing snapshots tương thích
 để khảo sát dữ liệu; điều đó chưa cấp quyền real temporal count.71 snapshot
 chưa được xác minh/có sẵn, vẫn giữ trong cohort.
2. **Có nhãn temporal độc lập chưa?**Chưa. Hai templates mỗi112 pending,0 final;
 pixel reference và visual viewpoint review không thay được temporal labels.
3. **Đánh giá baseline thật được chưa?**Chưa. Thiếu final labels và real temporal
 admission/adapter; legacy synthetic baseline giữ nguyên. Evaluator đã sẵn sàng
 ở mức phần mềm và trả NOT_EVALUATED đúng khi nhãn/predictions chưa có.
4. **Thiếu gì trước training đầu?**Hai lượt nhãn độc lập/phân xử + audit chất lượng
 nhãn; admissible real input/baseline theo contract được chấp nhận; đủ positive/
 negative/masked targets theo split; freeze targets/loss/pairing/selection và
 dependency/resource/reproducibility plan; trình MODEL TRAINING GATE với các lệnh
 cụ thể và **đợi user phê duyệt**. Count-only labels không đủ để train event/phase
 heads. Chưa biết số rep/labels usable hoặc ngày bắt đầu train.

Bước nhỏ tiếp theo: hai lượt video annotation mù trên gói đã khóa và adjudication
có log, sau đó import để thấy coverage/bất đồng thật. [Training proposal](SQUAT_COUNT_TRAINING_PLAN.md)
chọn một GRU nhỏ để review sau readiness, không chạy train trong milestone này.
