# BodyWeightSquats — fresh exploratory bilateral candidate audit

2026-10-18, Asia/Saigon. **NO_BILATERAL_CANDIDATE; demo thật NOT_CREATED.**
Đã chạy đủ40 clip/40 byte hashes,5.314 frames; không clip nào có hai knee
metrics cùng allow.35 sequence skip,5 degrade,0 allow. Giữ NOT_VALIDATED,
official_eligibility=false, assessment=false; G5.7
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Workspace không Git repository; không giả commit/diff.

## Phạm vi và selection đã khóa trước inference

Nguồn video duy nhất: `Data/UCF-101_GYM/BodyWeightSquats/`.
Tái sử dụng audit ngày15/10:
`outputs/stage5_7/subset-visual-20261015-077a183104c94d28a301b4d538b69de8/visual_audit.json`
và đúng image evidence của các clip đó. Không mở dataset khác, archive hoặc
video ngoài scope; không xem lại toàn bộ visual audit, không infer viewpoint
từ tên file, nhãn action, pose coverage hay tọa độ.

Audit mới nhất có112 rows:40 likely_side,46 oblique,23 front,3 uncertain.
Chọn toàn bộ40 likely_side, kiểm tra source byte SHA-256, linkage/sample hashes
và agent_viewed tại nhiều thời điểm đã lưu.40 hashes khác nhau;0 duplicate paths
cần inference lại.72 clip khác loại bị loại trước inference, lưu lý do trong
selection plan. Không dùng con số44 likely_side của audit12/10 thay cho audit
15/10, không chọn lại category sau khi biết quality.

Đây là nhận định cũ của agent, **visually_assessed_unverified**, chưa xác nhận
bởi con người/annotator chính thức; documented=false. Official source camera_view
vẫn **UNKNOWN**, raw PoseFrame.viewpoint=unknown. Giả định side áp dụng toàn clip
chỉ để exploratory; visual samples không xác nhận continuous side-view hay hai
đầu gối quan sát độc lập. Các confidence visual không phải xác suất đã hiệu chuẩn.

Plan `squat-real-bilateral-selection-v1` đã lưu trước backend/decoder:
`outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/selection/selection-plan.json`.
Byte SHA-256: `849b1a5bf05524d0480f8ac46d65b4edaf74f70067af1cbf38f9b746727b4c5f`.

Quy tắc đã khóa:

1. Dùng toàn clip; không cắt/chọn cửa sổ sau khi xem kết quả.
2. Demo cần sequence allow **và** exact left/right knee metric allow, ít nhất1
   frame có6 raw joints observed, finite, trong image bounds.
3. Xếp theo bilateral_allowed_frame_ratio giảm dần, rồi số frame allowed giảm
   dần, sau đó relative path tăng dần khi hòa. Không thêm/tune threshold.
4. Nếu không có clip đạt hai bên, dừng sau audit, không numeric/temporal/demo.

QualityGate hiện trả eligibility ở **cấp toàn sequence**, không trả decision
độc lập cho từng frame. `bilateral_allowed` là conjunction của quyết định gate
toàn clip với raw six-joint mask. Chỉ một phần joint invalid cũng có thể tạo
`degrade` dù ratio dưới engineering maximum; không tự biến valid raw frame thành
metric allow. Báo riêng raw joint coverage để tránh giấu thông tin hoặc đánh
đồng confidence với quyền đo. Không dùng single-frame QualityGate, đổi min
length hoặc tạo quality windows để mở gate.

## Pipeline mới và identities

Mỗi clip có reader và MediaPipe extractor mới, không dùng saved pose/quality
reports làm kết quả mới. Extraction1 lần/clip, source unknown; tạo bản sao
`pose-assumed-side.jsonl` chỉ đổi viewpoint metadata, giữ nguyên mọi joint/flag,
video_id/frame_index/timestamp/config/backend. Đây không phải lần inference
thứ hai hoặc annotation. Chạy QualityGate mới trên cả raw unknown lẫn bản sao
exploratory side, lưu cả hai. Official unknown knee metrics đều skip.

| Identity | Giá trị giữ nguyên |
|---|---|
| Package / Python | ai-coach-gym0.7.0 / CPython3.11.9, Windows10.0.26200 x64 |
| Backend / MediaPipe | mediapipe.solutions.pose.Pose /0.10.14 |
| Bundled model | complexity1, full model; không download/cài framework |
| Model SHA-256 | `e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f` |
| Pose config | pose-extraction-v1, visibility/detection/tracking0.5; smoothing/segmentation disabled |
| Config SHA-256 | `3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976` |
| Mapping / coordinates | mediapipe-pose-13-v1; image_normalized_xy; x-right/y-down; dimensionless |
| Quality profile | quality-s3-v1, min frames5, missing/low ratios0.30, max consecutive invalid10 |
| Quality profile SHA-256 | `514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704` |
| Canonical registry | gym-feature-registry-v2 |
| Registry SHA-256 | `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72` |
| Decoder / timestamps | OpenCV5.0.0, FFMPEG;25FPS,320×240; index/FPS CFR assumption |

Exact requirements lấy từ registry, assert trước inference:

| Metric | Ordered required joints | Allowed viewpoint |
|---|---|---|
| squat_knee_angle_left_deg | left_hip, left_knee, left_ankle | side |
| squat_knee_angle_right_deg | right_hip, right_knee, right_ankle | side |

Movement_analysis giữ8 shoulder/hip/knee/ankle joints hiện hành. Không sửa policy,
model/profile/threshold, raw metadata, schema, producer hoặc old reports. Không
preprocessing, interpolation, smoothing, numeric adapters, point-resolution/
temporal detector hoặc gộp hai bên. Không tạo góc, phase, repetition, point-turn,
pause, score, feedback kỹ thuật; các capability này vẫn unavailable trong phạm
vi lượt này. Source selection/extraction/evaluation/ranking đều exploratory.

## Kết quả đã đo — toàn bộ clip thử

**40/40 pipeline completed**, exit0,166.04s.5.311 frames có pose,3 không có pose.
Raw bilateral observed+in-frame:1.301/5.314 (**24,48%**). Đây không phải accuracy,
human visibility ground truth hay metric allow. **Bilateral allowed:0/5.314 (0%)**.

Trong bảng, raw=N/K là số frame đủ6 joints theo backend; allowed là số frame
được sequence và hai metrics cho phép. Mỗi metric L/R độc lập, không fusion.
Reasons giữ đúng codes của report, không diễn giải thành lỗi người tập.
CSV kèm full video hashes nằm ở `candidate-results.csv` trong run mới;
source/frame details và flags nằm trong từng thư mục `audit/Sxxx/`.

| File | Frames | Raw valid | Raw % | Allowed | Sequence | Left | Right | Reasons |
|---|---:|---:|---:|---:|---|---|---|---|
| `v_BodyWeightSquats_g01_c01.avi` | 267 | 8 | 3.00 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g01_c02.avi` | 252 | 14 | 5.56 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g01_c03.avi` | 250 | 35 | 14.00 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g04_c03.avi` | 88 | 4 | 4.55 | 0 | skip | skip | degrade | S:low_pose_visibility; L:low_pose_visibility; R:partial_joint_coverage |
| `v_BodyWeightSquats_g04_c04.avi` | 63 | 0 | 0.00 | 0 | skip | skip | degrade | S:low_pose_visibility; L:low_pose_visibility; R:partial_joint_coverage |
| `v_BodyWeightSquats_g05_c01.avi` | 71 | 62 | 87.32 | 0 | degrade | degrade | allow | S:partial_joint_coverage; L:partial_joint_coverage |
| `v_BodyWeightSquats_g05_c02.avi` | 135 | 133 | 98.52 | 0 | degrade | degrade | allow | S:partial_joint_coverage; L:partial_joint_coverage |
| `v_BodyWeightSquats_g05_c03.avi` | 54 | 51 | 94.44 | 0 | degrade | degrade | allow | S:partial_joint_coverage; L:partial_joint_coverage |
| `v_BodyWeightSquats_g07_c01.avi` | 123 | 61 | 49.59 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g07_c02.avi` | 103 | 45 | 43.69 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g07_c03.avi` | 83 | 45 | 54.22 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g07_c04.avi` | 54 | 32 | 59.26 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g09_c05.avi` | 229 | 8 | 3.49 | 0 | skip | skip | degrade | S:low_pose_visibility; L:low_pose_visibility; R:partial_joint_coverage |
| `v_BodyWeightSquats_g09_c06.avi` | 260 | 3 | 1.15 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g09_c07.avi` | 233 | 6 | 2.58 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g10_c03.avi` | 152 | 4 | 2.63 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g10_c04.avi` | 148 | 4 | 2.70 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g10_c05.avi` | 158 | 9 | 5.70 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g12_c01.avi` | 237 | 0 | 0.00 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g12_c02.avi` | 143 | 0 | 0.00 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g12_c03.avi` | 120 | 0 | 0.00 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g13_c01.avi` | 112 | 67 | 59.82 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g13_c02.avi` | 62 | 24 | 38.71 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g13_c03.avi` | 115 | 46 | 40.00 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g13_c04.avi` | 75 | 21 | 28.00 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g15_c01.avi` | 142 | 35 | 24.65 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g15_c02.avi` | 122 | 30 | 24.59 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g19_c03.avi` | 144 | 44 | 30.56 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g19_c04.avi` | 108 | 28 | 25.93 | 0 | skip | skip | allow | S:low_pose_visibility; L:low_pose_visibility |
| `v_BodyWeightSquats_g20_c05.avi` | 136 | 26 | 19.12 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g20_c06.avi` | 167 | 41 | 24.55 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g23_c03.avi` | 73 | 61 | 83.56 | 0 | degrade | degrade | degrade | S:partial_joint_coverage; L:partial_joint_coverage; R:partial_joint_coverage |
| `v_BodyWeightSquats_g23_c04.avi` | 66 | 58 | 87.88 | 0 | degrade | degrade | degrade | S:partial_joint_coverage; L:partial_joint_coverage; R:partial_joint_coverage |
| `v_BodyWeightSquats_g25_c01.avi` | 103 | 44 | 42.72 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c02.avi` | 110 | 30 | 27.27 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c03.avi` | 116 | 49 | 42.24 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c04.avi` | 108 | 35 | 32.41 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c05.avi` | 106 | 54 | 50.94 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c06.avi` | 101 | 38 | 37.62 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |
| `v_BodyWeightSquats_g25_c07.avi` | 125 | 46 | 36.80 | 0 | skip | allow | skip | S:low_pose_visibility; R:low_pose_visibility |

Tất cả40 clip bị loại theo gate đã khóa; không có clip được chọn cho demo.
Tổ hợp L/R:21 allow/skip,11 skip/allow,3 skip/degrade,3 degrade/allow,
2 degrade/degrade;0 allow/allow. Không phải toàn bộ phạm vi112 squat clips
đã được inference:72 clips ngoài latest likely_side pool không chạy.

Clip có raw coverage cao nhất, **không phải selected candidate**:
`v_BodyWeightSquats_g05_c02.avi` (S019),133/135=98,52%.
Frames71–72 có left_ankle low_visibility; sequence degrade, left knee degrade
với partial_joint_coverage, right knee allow. Không cắt bỏ hai frame này,
interpolate hoặc gọi numeric adapter để tạo kết quả. Clip này chưa pass dù
gần100% raw coverage; observed/allow cũng chưa chứng minh joint accuracy.

## Test/integrity evidence và warnings

Lệnh đã chạy tại workspace root, output/receipts mới:

```powershell
& .venv/Scripts/python.exe -B scripts/audit_squat_exploratory_candidates.py prepare --audit outputs/stage5_7/subset-visual-20261015-077a183104c94d28a301b4d538b69de8/visual_audit.json --output-dir outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/selection

& .venv/Scripts/python.exe -B -m pytest tests/test_squat_exploratory_candidates.py tests/test_quality.py tests/test_development_eligibility.py -m "not integration" -q --basetemp outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/tests-first-temp --junitxml=outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/tests-first.xml

& .venv/Scripts/python.exe -u -B scripts/audit_squat_exploratory_candidates.py run --selection outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/selection/selection-plan.json --selection-sha256 849b1a5bf05524d0480f8ac46d65b4edaf74f70067af1cbf38f9b746727b4c5f --output-dir outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/audit

& .venv/Scripts/python.exe -B outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/verify_audit.py
```

Tests: **67 passed,0 failed,0 errors,0 skipped,3 deselected,0 pytest warnings,
9.21s**.21 tests mới +46 existing synthetic/unit regressions.3 saved real-data
quality integrations ngoài scope bị deselect; không chạy full suite/other-data
integrations. Test numeric compatibility chỉ dùng synthetic, không gọi real
numeric adapters. pytest9.1.1/Pydantic2.13.5; không thêm dependency.

Kiểm tra độc lập saved run đạt:280 artifact file hash/size checks;40 source
video hashes;5.314 raw/assumed/frame mask identities/timestamps;80 QualityReports
strict readback và re-evaluation bằng cùng profile/requirements. Bản side chỉ
khác đúng viewpoint; source unknown giữ nguyên. So sánh baseline112 source
videos, audit và code/config: không đổi dữ liệu/model/profile/producer/schema.
`integrity.json`, preservation receipts và artifact inventory lưu evidence.

Inference có **10.622 UserWarning lặp lại cùng message**
`SymbolDatabase.GetPrototype() is deprecated...`, thu đầy đủ bằng
warnings.simplefilter(always) và lưu theo clip. Không phải10.622 lỗi khác nhau
hoặc pytest warnings; không giấu chúng trong test counts. Native MediaPipe/
TFLite feedback-manager logs giữ trong pipeline-stderr.txt. Các pipeline vẫn
completed; không sửa dependency/backend để chặn warning hoặc nới gate.

**browser_qa=NOT_RUN**; không tạo demo nên không có browser/visual QA claim.
Không dựng server/network/security workaround. Không dùng test success làm
bằng chứng pose accuracy hoặc ground truth trên vận động viên.

## Artifacts, bảo toàn và giới hạn khoa học

Run: `outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/`.
`selection/selection-plan.json`, `audit/summary.json`, `candidate-results.csv`,
commands/stdout/stderr/environment/test XML và integrity/preservation receipts.
Mỗi `audit/Sxxx/` có source.json (decoder/model/config/warnings/evidence),
raw-pose-unknown.jsonl, pose-assumed-side.jsonl, hai QualityReports,
frame-eligibility.json, result.json, artifact-manifest.json.
Không ghi đè output/audit/report cũ, không xóa/move nguồn.

Thêm script audit và21 tests, báo cáo này; cập nhật tối thiểu README,
PROJECT_STATE/TASKS. Không thay core hoặc thêm renderer cho kết quả bị gate khóa.

Giới hạn: visual opinion chỉ trên mẫu đã xem; side hypothesis không confirmed
camera protocol.2D projection/foreshortening/occlusion và subject policy
backend_most_prominent_single_subject không bảo đảm identity/multi-person hoặc
bilateral physical accuracy. Visibility là confidence backend, không calibrated
measurement uncertainty.112 clips/40 hashes không phải40 vận động viên độc lập;
các clip cùng gXX có thể cùng cảnh/người. Selection chỉ kiểm tra development
eligibility trên pool đã định, không estimate generalization hay form quality.
Frame/FPS là CFR assumption, không xác thực VFR/container timestamps.

**Đề xuất điều chỉnh phạm vi đồ án, chưa áp dụng policy mới:** tập trung demo
pipeline provenance/quality/abstention và số liệu failure coverage trên dữ liệu
thật; temporal presentation tiếp tục synthetic-only. Nếu muốn tiến tới demo
measurement một bên/framewise, cần một milestone/phạm vi acceptance riêng và
không gọi đó là bilateral hoặc5.7 pass. Không dùng kết quả audit để tự mở nhánh
đó, không đổi thresholds. Bilateral real demo và official5.7 tiếp tục blocked;
documented input lẫn sequence/bilateral allow vẫn chưa đủ evidence.
