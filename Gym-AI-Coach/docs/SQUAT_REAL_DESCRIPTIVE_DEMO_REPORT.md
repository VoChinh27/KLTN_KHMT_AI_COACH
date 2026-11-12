# BodyWeightSquats — real descriptive fixed-side demo

2026-10-20, Asia/Saigon. **EXPLORATORY_DEMO_IMPLEMENTED**, không phải5.7 pass
hoặc validation hệ thống. NOT_VALIDATED, official_eligibility=false,
assessment=false; G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT,
G6/toàn G5 IN_PROGRESS. Workspace không có Git repository; không tạo commit/diff.

## Mở sản phẩm

Run mới:
`outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/`.

- Mở `demo/START_HERE.html` hoặc `demo/overlay-h264.mp4` bằng trình phát có sẵn.
- MP4 **H.264**,1280×720,25FPS,267 frames,10,68s; decode toàn bộ đạt.
- `demo/original.avi` là video gốc copy byte-identical,320×240, không sửa nguồn.
- Giữ nguyên thư mục và sidecars; `HUONG_DAN_MO_DEMO.txt` hướng dẫn mở offline.
  Không cần Python/dependency/mạng/server để xem. Nếu chính sách local file
  chặn, dừng; không tắt bảo vệ hoặc dựng server để đi vòng.

Video hiển thị hình gốc phóng đồng dạng2× trong panel640×480, skeleton raw pose,
gối trái cố định, số đo từ processed coordinates, frame/time và quality gates.
Observed màu xanh ngọc, low visibility màu cam, missing/ngoài ảnh không vẽ.
Màu chỉ nói trạng thái dữ liệu, không đúng/sai kỹ thuật. Overlay không mang audio.
Không có phase, repetition, point-turn, pause, form score hoặc lời sửa động tác.

**browser_qa=NOT_RUN**; sampled-frame visual QA đã thực sự chạy, trình bày riêng
bên dưới. Không tuyên bố desktop/native player/browser playback đã được kiểm tra.

## Vì sao đo được một gối trong khi audit bilateral trước bị chặn

Code hiện hành: registry v2 `framewise-metric-only-v1` có
require_sequence_allow=false; geometry_policy chỉ bỏ blocker scope
movement_analysis, giữ nguyên exact metric/joints/viewpoint/usability/provenance.
geometry_adapter dùng policy đó, không sửa QualityReport. Context factory liên
kết exact Stage4 frame/report/source; readback không tự đo lại.

Tests mới và existing geometry tests xác nhận:

- Metric allow + sequence skip vẫn tạo GeometryFeatureObservation available,
  temporal_analysis_eligible=false, assessment_status blocked.
- Legacy observe_feature vẫn unavailable vì cần sequence+metric allow.
- Metric degrade/skip trả value=null với reasons/flags kể cả processed tọa độ
  usable; không loại frame khỏi mẫu số hoặc promote eligibility.

Đây là sử dụng policy đã tồn tại cho phạm vi exploratory một bên được yêu cầu,
**không** thay acceptance bilateral trước hoặc dùng một bên để mở5.7/temporal.
Các reports cũ về NO_BILATERAL_CANDIDATE giữ nguyên và vẫn đúng.

## Selection receipt trước khi tính góc

Nguồn video duy nhất: `Data/UCF-101_GYM/BodyWeightSquats/`.
Tái sử dụng đúng run
`outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/`.
Không infer lại40 clip hoặc khởi tạo MediaPipe trong lượt này.

Loader kiểm tra root/per-clip manifests, source SHA-256, frozen selection/audit
linkage, metadata/timeline, raw-unknown và assumed-side chỉ khác viewpoint,
strict QualityReports/core-count consistency, current quality/registry hashes
và đúng canonical knee requirements. Không mở dataset khác hoặc archive.

Xếp80 clip–side pairs của40 unique clips;35 pairs có metric allow,45 không có.
Xếp eligible pairs theo tỷ lệ frame có hip/knee/ankle cùng bên **observed,
finite, trong ảnh** giảm dần; hòa theo relative path rồi left/right, không ưu
tiên độ dài hoặc đổi sang clip đẹp sau khi tính góc. Các pairs allow đều đạt
raw ratio1, nên tie rule chọn đường dẫn đầu tiên, bên trái.

| Selection | Kết quả |
|---|---|
| Clip/video ID | `v_BodyWeightSquats_g01_c01.avi` |
| Relative source path | `Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi` |
| Byte SHA-256 | `1a199bf3ad385a66505fa0864a48fb9b9f9c1edfd1306d8345729ab06742c9e0` |
| Selected side | left, cố định toàn bộ267 frames |
| Canonical feature | `squat_knee_angle_left_deg` |
| Required joints | left_hip, left_knee, left_ankle |
| Raw selected-side coverage |267/267 observed/finite/in-frame |
| Sequence | skipped / skip / eligible=false; low_pose_visibility |
| Selected metric | allow / eligible=true; no reasons |
| Other knee | right metric skip; không đo/hiển thị số đo bên này |
| Original camera_view | UNKNOWN; raw PoseFrame viewpoint=unknown |
| Assumed side evidence | visually_assessed_unverified, documented=false |

`demo/selection.json` được exclusive-write với geometry_started=false trước
preprocessing/adapter, giữ source/artifact hashes, rule, decisions và phạm vi
pool. `ranking.json` lưu đủ80 pairs, kể cả rejected. File byte hashes trong
manifest và result_hash lưu identity. Không có file selection hoặc report cũ
bị sửa. Đây là **lựa chọn mẫu trình diễn dựa trên audit outcomes**, không phải
hold-out evaluation,40 vận động viên độc lập hoặc bằng chứng generalization.

## Phần tái sử dụng và phần xử lý mới

Tái sử dụng byte-identical: raw-pose-unknown.jsonl, pose-assumed-side.jsonl,
quality-unknown.json, quality-assumed-side.json, source.json của review S001.
Source video giữ nguyên bytes. Không tạo QualityReport mới vì hai knee keys và
joints/viewpoint requirements đã đầy đủ. Kiểm tra consistency không được gọi
là fresh inference hoặc fresh official gate.

Xử lý mới: preprocess_pose bằng `configs/preprocessing_s3_v1.toml` hiện hành,
build_feature_input_context cho mỗi frame, observe_geometry_feature đúng một
canonical ID, result/coverage JSON, video và HTML/guide. Factory audit lưu full
preprocessing.json, per-frame linkage/raw/processed/input/quality hashes; adapter
nhận cùng frame/report snapshot đã qua factory như workflow cuối Stage5.

| Identity/policy | Giữ nguyên |
|---|---|
| Package | ai-coach-gym0.7.0 |
| Saved backend/model | MediaPipe0.10.14, bundled full complexity1 |
| Model SHA-256 | `e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f` |
| Pose config | pose-extraction-v1, visibility/detection/tracking0.5, smooth_landmarks=false |
| Quality | quality-s3-v1; hash `514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704` |
| Registry | gym-feature-registry-v2; hash `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72` |
| Feature policy/adapter | framewise-metric-only-v1 / geometry-features-v1 |
| Numeric contract | GeometryFeatureObservation0.4.0, giữ nguyên |
| Factory | preprocessing-context-v1 / context0.2.0, giữ nguyên |
| Preprocessing | preprocessing-s3-v1,25FPS: gap4 frames, smoothing window9/polyorder2 |
| New artifact envelope | squat-real-descriptive-v1; không public FeatureSequence |

Preprocessing vẫn loại low visibility, nội suy missing gap theo profile hiện có,
aspect-ratio correction, normalization/fallback và finite-run SG. Trong mẫu đã
chọn, **801/801 selected joint inputs observed,0 imputed**; không sửa profile
để đạt kết quả. Không lấy tọa độ processed vẽ như raw skeleton. Angle có thể
khác góc nhìn trực tiếp trên raw overlay do aspect correction/smoothing;
điều đó được chú thích trên mỗi frame. Không calibration vật lý hoặc metric 3D.

## Kết quả và contract artifact

**267 available,0 unavailable /267 frames, coverage100%.** Mẫu số là toàn clip,
không lọc lỗi/đoạn không nhìn rõ.100% là data availability theo S3 policy và side
hypothesis, **không phải accuracy**. Số đo round1 decimal khi vẽ; JSON giữ float.
Mọi observation vẫn unvalidated, framewise_projected_2d, temporal_analysis_eligible=false.

Mỗi record trong result.json có video_id/frame_index/timestamp, source video SHA,
raw pose hash/joints, factory linkage, full typed observation (value/unit/status,
input joints/provenance, QualityReport snapshot/hash, reason/flags), metric action
và visually_assessed_unverified. Shared schema/API không bị thay đổi. result_hash
và manifest hashes phát hiện thay đổi nội dung, không authenticate annotator/model.

Renderer mới chỉ validate/read saved values, format1 decimal và vẽ; không tính
lại joint angle, preprocessing, QualityGate hoặc temporal. Existing synthetic
renderer/contracts không bị đổi label để nhận real input. Không fallback bên,
không averaging/fusion. Temporal object giữ null/reason not_run_no_independent_temporal_evidence.

## Decode và visual QA thực tế

Build exit0,244.30s. Factory full-source revalidation mỗi frame tạo chi phí
O(N²) hiện hữu; không tối ưu/thay core trong milestone này. Video playback đã
burn-in số đo, không cần tính geometry khi mở file.

Decode kiểm tra **tất cả267 frames**,1280×720,25FPS; toàn clip không cắt,
output_frame=source_frame=0..266, timestamp=index/25 khớp chính xác JSON.
Nominal duration267/25=10,68s; last-frame timestamp10,64s. Input CFR assumption,
không claim hỗ trợ source VFR/container PTS. Video gốc copy hash khớp source.

Writer request avc1, output decoder xác nhận **H.264** (fourcc h264), FFMPEG.
stderr ghi một lần thử FFmpeg/OpenH264 thiếu DLL/khởi tạo encoder không thành
công, nhưng VideoWriter cuối cùng opened và output decode toàn bộ đạt. Không
tải DLL/codec/framework hoặc đổi dependency; exact writer backend không được
ghi lại, không suy đoán nó từ stderr. `codec-inspection.json`/build-stderr.txt
giữ evidence này, tách khỏi pytest warnings. Không tuyên bố browser compatibility
chỉ từ codec metadata.

Đã thực sự dùng view_image(detail=original) để xem PNG **giải mã từ MP4**:

| Output frame | Seconds | Kiểm tra |
|---:|---:|---|
|0|0.00|Đầu clip; cũng là earliest maximum4 raw low-visibility joints |
|66|2.64|Quarter sample; legend và raw pose colors |
|133|5.32|Giữa; đối chiếu thêm ảnh gốc cùng frame |
|200|8.00|Three-quarter; fixed side/quality labels |
|239|9.56|Additional saved-value minimum để QA tư thế khác và text; không gọi Bottom event |
|266|10.64|Cuối clip; frame/time/footer |

Đã xem thêm original-frame133 và239. Layout/text/độ/legend/frame-time đọc được,
không thấy clipping trên6 output frames đã xem; hình gốc full-body vẫn trong
panel, raw pose projections gần/xa chồng nhau. Observed teal và low visibility
amber phân biệt được, không sửa tọa độ backend khi nhìn thấy projection yếu.
Frame239 dùng kiểm tra hiển thị, không phát hiện/gán phase hoặc đổi selection.

Selected knee không có đoạn low/missing;4 raw joints khác low tại5 mẫu phân bố
ban đầu, có màu cam. Không giả tạo một unavailable real frame cho demo; blocked
paths được kiểm tra bằng synthetic tests. `visual-qa-review.json` và additional
review ghi đúng images/scope; receipt trước inspection giữ trạng thái lúc tạo.

Đây là **sampled-frame visual QA**, không continuous playback QA, browser UI QA,
ground-truth pose/form validation. **browser_qa=NOT_RUN**, không server/network
hoặc security workaround.

## Tests và integrity

Môi trường: Windows10.0.26200 x64, CPython3.11.9, pytest9.1.1/Pydantic2.13.5,
OpenCV5.0.0.93, Pillow12.3.0/NumPy2.4.6/SciPy1.17.1 đã có. Không dependency mới,
không claim clean-environment compatibility. Unicode font lấy từ hệ thống.

| Receipt | Pass | Fail | Error | Skipped | Deselected | Warnings | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| tests-first |240|4|14|0|0|0|14.90|
| tests-new |21|0|0|0|0|0|19.26|
| **tests-related** |**681**|**0**|**0**|**0**|**4**|**0**|**93.82**|

Initial failure là lớp nối mới truy cập sai observation.input_hash; sửa sang
availability.input_hash hiện hành. Không sửa schema/domain. Sau unit21 pass,
thêm saved-renderer linkage/no-recompute regression; final681 gồm22 mới và659
existing. Không cộng lặp runs.4 real-data integrations deselect:3 saved quality
integrations và1 extra real-preview; không chạy full suite hoặc dữ liệu khác.

Tests gồm policy-v2 vs legacy, metric degrade/skip null, immutable QualityReport,
fixed side/ranking, source/hash/path validation/no inference, JSON version/hash/
coverage/frame/time, disabled gates, saved renderer no geometry/policy call,
raw missing/low draw rules, Unicode canvas và HTML escaping/local links.

Readback/decode riêng đạt:15 manifest-listed file hashes/sizes,267 observations
khớp exact processed frame/report/factory input hash,801 observed inputs,
9 local href/src links, complete video/FPS/resolution/map và original SHA.
Counts/files trong integrity.json là evidence authoritative. Những check này
không đo MAE/accuracy trên người tập. Source/old outputs bảo toàn theo baseline.

Lệnh đã chạy, các directory/receipts đều mới:

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_real_descriptive_demo.py tests/test_geometry_features.py tests/test_feature_context.py tests/test_feature_observation.py tests/test_right_knee_feature.py tests/test_preprocessing.py tests/test_quality.py tests/test_squat_feedback.py tests/test_showcase_handoff.py -m "not integration" -q --tb=short --basetemp outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/tests-related-temp --junitxml=outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/tests-related.xml

& .venv/Scripts/python.exe -u -B -m gym_coach.reporting.real_descriptive_demo --audit-run outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf --output-dir outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/demo

& .venv/Scripts/python.exe -B outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/verify_demo.py
```

Không chạy lại vào output/JUnit/basetemp đã tồn tại; chọn đường dẫn mới.
commands/stdout/stderr/XML/environment/test-summary/integrity/QA receipts được
giữ riêng. Không giả test counts từ các milestone trước.

## Files, giới hạn và bước tiếp theo

Thêm reporting/real_descriptive_demo.py, reporting/real_descriptive_render.py,
tests/test_real_descriptive_demo.py, report này; cập nhật tối thiểu README,
PROJECT_STATE/TASKS. Không đổi public exports/version, shared schemas, producer,
profiles/thresholds/model hoặc historical reports. Baseline/preservation và
artifact inventory nằm trong run mới; không xóa/move video/archive/output.

Demo chỉ mô tả một estimated projected knee angle trên1 selected development
clip dưới side hypothesis chưa confirmed. Single-subject backend không bảo đảm
tracking identity/multi-person; visibility và availability không calibrated
uncertainty/accuracy. Không có human keypoint/reference angle, official viewpoint
annotation hoặc form ground truth để claim accuracy. Không assessment/rules.

Bước nhỏ tiếp theo: browser/native-player QA bằng cơ chế được phép, và một
milestone bổ sung mẫu cố định một bên/case unavailable theo acceptance đã chốt.
Nếu muốn claim measurement accuracy cần reference độc lập và protocol riêng;
formal5.7 vẫn cần documented side linkage và fresh sequence/bilateral allow.
Không dùng demo một bên để đóng G5/G6 hoặc nâng exploratory thành official.
