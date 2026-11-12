# BodyWeightSquats — thực hiện annotation AI và review viewpoint

Hoàn tất nhánh thao tác AI ngày 2026-10-26; **NOT_VALIDATED**, không independent
ground truth hoặc human validation. MODEL TRAINING GATE: **NOT_READY**.

Run mới: [squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33](../outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/).
[Bảng 112 clip và liên kết từng lượt/ảnh](../outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/OBSERVATIONS_112.md),
[AI reference và bất đồng](../outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/results-v1/ai-reference.json),
[33 AI viewpoint reviews](../outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/results-v1/ai-viewpoint.json).

## Quyền, nguồn và cách thực hiện

Quyết định mới của chủ project cho phép AI tự xem/gán/review/reconcile, không yêu cầu
người dùng bố trí người. Request SHA-256 và input hashes lưu trong `decision.json`.
Không xem quyền này là approval real counting admission, training hoặc human GT.
Đã đọc quy định/state/protocol/proposal hiện hành; `docs/MASTER_PROMPT.md` không tồn
tại, dùng `GYM_REBUILD_MASTER_PROMPT.md`; không dùng các mô tả legacy để chứng minh
code hiện hành. Workspace không có Git repository; không tạo commit/diff giả.

Giữ dataset-final đã khóa:112 clips/112 byte hashes/25 gXX, 67 train/21 validation/
24 development-test; không pristine hoặc unseen-subject holdout. Nguồn video chỉ
`Data/UCF-101_GYM/BodyWeightSquats/`; không archive, bài khác hoặc model predictions.
Đã decode/trích **14.356 frames**, 25 FPS, 320×240, tổng container duration574,24s.
Time là `frame_index/FPS`, CFR assumption, không native PTS được kiểm chứng độc lập.

Thực sự dùng `view_image` xem203 contact sheets chứa mọi frame (không chỉ chạy
script trích ảnh), rồi mở lại đủ203 sheets cho lượt B. Mỗi tile160×120, không crop/
flip/rotate/pose overlay; JPEG và ảnh nhỏ hạn chế chi tiết. Không video playback,
audio hoặc browser QA; không được diễn giải thành đã xem playback ở tốc độ thật.
Evidence giữ path/hash/frame indices và source timeline; hash không chứng minh nhãn đúng.
`prepared_not_observed=true` trong manifest trích ảnh được giữ đúng nghĩa: sự kiện
đã xem nằm ở checkpoint annotation, không tự đến từ việc decode.

Pilot đúng g01_c01, g05_c01, g06_c01, g08_c01, g09_c01; count A/B lần lượt4/1/2/1/4.
`pilot-check.json` và26 tests kiểm tra quy trình trước expansion. Lượt A phần còn
lại theo source path tăng dần; lượt B phần còn lại thực tế theo path giảm dần,
trong mỗi clip vẫn timeline tăng dần. Đây là khác biệt thao tác với câu expansion
order tổng quát trong protocol; không đổi semantics/cohort hoặc chọn clip theo
kết quả. Giữ nguyên protocol/decision ban đầu để truy vết, không sửa lịch sử.
Lượt B mở ảnh nguồn, không đọc file A để tạo nhãn, nhưng cùng AI/context đã có
ký ức lượt trước: **không blind, không hai rater độc lập**.

## Kết quả đã đo từ các record

| Đại lượng | Kết quả / mẫu số |
|---|---|
| Clips đã xem cả hai lượt |112/112|
| Checkpoints AI đã lưu và verify |224/224; chưa xử lý0|
| Whole-clip count nhất quán, available |104/112|
| Whole-clip count unresolved/null |8/112|
| Cặp có cả hai count known |104/112|
| Count trùng giữa hai lượt |104/104 comparable pairs; chỉ AI consistency|
| Tổng count trên riêng104 clip available |252; không tổng rep của toàn112|
| Clip có ít nhất một trường khác |24/112|
| Trường khác nhau |cycle hints5; uncertain8; partials13; observed lower bound1|
| Exact exhaustive event labels |0/112; boundaries unavailable toàn bộ|
| Human-validated/independent GT mới |0|

Khác biệt partial/uncertain được so bằng nội dung JSON, gồm wording và khoảng
frame; không gọi tất cả24 clip là count disagreement. Chưa tính timing MAE/F1,
IAA, calibrated confidence hoặc model accuracy. Count không cần exact boundaries;
gợi ý start/bottom/end của pilot chỉ là khoảng chưa chắc chắn, không exhaustive
events/pause targets. Các clip khác ghi boundaries unavailable với reason.

Các whole-count chưa chốt (tên đầy đủ theo mẫu `v_BodyWeightSquats_<ID>.avi`):

| ID | Observed lower bound A/B | Lý do giữ whole-count null |
|---|---:|---|
|g01_c02|3/3|Top đầu clip chưa rõ, có thể đã bắt đầu hạ|
|g10_c01|4/4|Lượt cuối chưa xác định trở lại top trước cut|
|g11_c04|0/0|Không xác định start trước lượt duy nhất; 0 nhìn rõ không phải total0|
|g17_c02|2/2|Camera di chuyển/zoom và crop phần thân dưới|
|g17_c03|1/2|Khác lower bound; camera/crop và top cuối chưa rõ|
|g18_c03|5/5|Lượt cuối còn gập khi hết clip|
|g19_c03|3/3|Top cuối sát biên chưa chắc|
|g23_c04|1/1|Đầu clip đã hơi gập; thiếu start chắc chắn|

Reconciliation bảo thủ chỉ giữ count khi cả hai known và bằng nhau; không trung
bình, chọn theo detector hoặc suy missing=0. g17_c03 giữ cả original lower bounds,
reconciled lower bound=null; không tự chọn1 hay2. Whole-count của8 clip vẫn null.
Không thêm lượt xem nguồn thứ ba để giả giải quyết; `conservative_AI_reconciliation`
không human adjudication. Originals A/B được giữ nguyên, liên kết byte hash từng file.

## Viewpoint riêng, không mở gate

Đúng33 IDs từ worklist frozen, mọi frame của từng clip có trong evidence. Kết quả
AI: **19 front,13 oblique,1 unknown/mixed,0 side**. g06_c02 có phần vận động xiên
(frames0–112), phần quay hướng cuối113–121 giữ unknown; whole viewpoint=unknown.
Các biên trên là vùng hỗ trợ review, không thời điểm đổi góc vật lý đã hiệu chuẩn.
g17_c01 có chỉnh framing nhẹ; vẫn nhận định front, ghi crop đầu và chân trong khung.
Từng record lưu subject, hướng camera, framing/occlusion, interval/frame/time và hash.

Rubric định tính được tái sử dụng cho nhánh AI theo quyền mới, không có ngưỡng độ
hoặc calibration. **33/33 visually_assessed_unverified; documented=0**. Raw camera
UNKNOWN, QualityReports và thresholds giữ nguyên. Không chạy lại gate bằng enum
AI, không dùng temporal count thay evidence viewpoint. G5.7 vẫn blocked; G6/toàn
G5 IN_PROGRESS. 112 pose/quality evidence vẫn gồm41 preprocessing snapshots và71
pose/quality-only, không phải112 preprocessing snapshots.

## Implementation, kiểm tra và bảo toàn

Thêm `counting/ai_annotations.py`: schema AI strict/frozen riêng, null/reason,
timeline, full viewed support, checkpoint hash/write-once, comparator và summary.
Schema versions: `squat-ai-labels/0.1.0`, `squat-ai-viewpoint/0.1.0`;
protocol `squat-ai-visual-review-v1`; aggregation `squat-ai-reconciliation-v1`.
Canonical payload SHA-256 và byte hashes tách biệt. JSON schemas/hash inventories
trong run; không sửa schema human hoặc API temporal/geometry hiện hành.

Scripts mới: `squat_ai_review_frames.py`, `squat_ai_annotation_intake.py`,
`summarize_squat_ai_annotations.py`. Intake chỉ đóng gói quan sát AI đã nhập rõ,
không tự tạo count/viewpoint từ ảnh hoặc metadata. Resume kiểm lại source/sheet
bytes, manifest/source/frame/time và không đổi nhãn cũ. Summary bắt đủ112/224 và
33 viewpoints; thiếu/extra/duplicate là lỗi, không bỏ khỏi mẫu số.

Kết quả kiểm tra:112 raw source hashes và224 checkpoints hợp lệ;315 evidence files
(203 sheets+112 manifests) được liên kết. Inventory cũ:130 files dataset-final,
661 feasibility,27 review-pilot — **818/818 không mismatch**. Human templates,
split, nguồn, pose/QualityReports cũ không đổi. Trước cập nhật docs, toàn bộ file
đã có trong `before.json` không thay đổi. Chỉ3 docs hiện hành được cập nhật sau đó.

Môi trường: Windows10.0.26200, Python3.11.9, pytest9.1.1, Pydantic2.13.5; không
dependency/model mới. Mọi kiểm tra dưới đây đã chạy; không clean-environment claim.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_squat_ai_annotations.py -q --junitxml=outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/ai-contract-tests.xml
.venv\Scripts\python.exe -m pytest tests/test_squat_ai_annotations.py tests/test_squat_counting.py tests/test_squat_review_pilot.py tests/test_pose_feasibility_batch.py -q --junitxml=outputs/squat_ai_annotations/run-896910a745f941ea85bbbd61ceb93d33/regression-tests.xml
```

- Final AI contract suite:46 passed,0 failed/error/skipped/warnings,1,38s pytest.
- Related regression:159 passed,0 failed/error/skipped/warnings,6,16s pytest;
  wrapper7,33s. Bao gồm46 tests trên, không cộng thành205 test riêng.
- Pilot26 và intermediate linkage34 pass có XML riêng; không ghi đè receipt.
- Chưa full suite, real counting, training, pose inference hoặc browser/playback QA.
  Feasibility regression dùng mocks, không tạo MediaPipe backend.
- Có lỗi thao tác trích ảnh ban đầu do gọi API reader không tồn tại, đã sửa thành
  `OpenCVVideoReader.frames()` trước extraction; một lần in notes gặp Windows
  encoding, chạy lại UTF-8. Không biến các lỗi lệnh này thành pytest failures hoặc
  bỏ clip. Artifact final kiểm đủ cohort, processing/linkage errors=0 ở lần verify.

File chính mới/đổi: module AI,3 scripts, `tests/test_squat_ai_annotations.py`,
protocol AI/report này; cập nhật `PROJECT_STATE.md`, `TASKS.md`,
`SQUAT_COUNT_TRAINING_PLAN.md`. Không sửa code cũ, UI, producer, profile hoặc gate.

## MODEL TRAINING GATE — NOT_READY

Nhãn AI count-only đã có và có thể được xét làm weak reference cho thí nghiệm
phù hợp; không được dùng để claim independent accuracy. Chưa có:

1. Real temporal admission/input được duyệt và baseline runnable; proposal fixed-left
   vẫn đòi sequence/metric allow, không được bỏ viewpoint blocker chỉ vì có nhãn AI.
2. Target/loss/pairing phù hợp: GRU ba event heads đề xuất cần exact exhaustive
   events, trong khi112/112 chưa có. Không suy chúng từ count, hints hoặc extrema.
3. Training implementation/config/dependency freeze và evaluation design tương ứng.
   Split không đổi; development-test không trở thành unseen-subject holdout.

Independent validation vẫn thiếu. Không coi việc cho phép nhãn AI là đã phê duyệt
mọi phương pháp hoặc đủ train mô hình hiện đề xuất. Không có lệnh train thật để
bàn giao; không tạo lệnh giả. Agent không train/fit/optimizer/smoke training;
user tự chạy khi có hồ sơ MODEL TRAINING GATE phù hợp.

Bước nhỏ tiếp theo có căn cứ: chốt một thiết kế count-only dùng AI reference với
input admission và target thực sự khả dụng, đối chiếu việc33 quality candidates
chưa có side evidence phù hợp; không triển khai adapter hoặc mở gate trong lượt này.
