# Kế hoạch triển khai AI Coach Gym sau clean restart

## Ưu tiên nghiên cứu 2026-10-25

BodyWeightSquats → baseline đếm lần → nhãn độc lập → temporal learning đối chứng
→ đánh giá kỹ thuật sau. Web/desktop/multi-exercise/form development tạm dừng;
entry points và artifacts giữ nguyên. Không thay lịch sử hoặc acceptance G5.7/G6.

1. **Đã triển khai preparation:**112 video manifest/full decode,41 verified saved
 snapshots; schema/protocol/gói nhãn mù, group/duplicate split và exposure ledger;
 count/event evaluator cùng synthetic tests.0 completed temporal labels,
 112 not_run và accuracy null. Xem [evidence](SQUAT_COUNT_DATA_READINESS_REPORT.md).
2. Hai lượt video annotation độc lập + phân xử;60 ảnh pixel reference là nhánh
 measurement riêng. Không lấy pose/extrema/agent review làm temporal ground truth.
3. Review [count contract proposal](SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md):
 baseline hiện chỉ synthetic; framewise metrics không cấp quyền temporal. Chưa
 chấp nhận count-only real admission, không sửa policy để tạo prediction.
4. Baseline real sau admission và labels đủ điều kiện, báo full denominators,
 coverage/abstention cùng errors; không tune split/tolerance/test theo performance.
5. [Training plan](SQUAT_COUNT_TRAINING_PLAN.md) là proposal, không authorization.
 Trình MODEL TRAINING GATE gồm nhãn, split, targets/loss/pairing, resource/dependency,
 baseline và lệnh cụ thể; đợi user approval trước actual fit.

Split hiện67 train/21 validation/24 development-test,15/5/5 groups. Cả112 đã có
visual exposure; không pristine holdout hoặc bảo đảm subject identity. Giữ
NOT_VALIDATED/official=false/assessment=false/REFERENCE_PENDING, G5.7 blocked,
G6/toàn G5 IN_PROGRESS. Các mục bên dưới ghi kế hoạch/evidence ở thời điểm cũ.

Cập nhật sản phẩm2026-10-24: giao diện chính WEB local; dùng pipeline/progress hiện có. Xem [web milestone](SQUAT_WEB_DEMO_REPORT.md) và [mục lục active/lịch sử](DOCUMENTATION_INDEX.md). Không đổi gate G5.7/G6 hoặc tiêu chí khoa học.

Ngày cập nhật: 2026-10-18 (Asia/Saigon). Nguồn yêu cầu: [README](../README.md),
[đặc tả Gym](GYM_TECHNICAL_GROUND_TRUTH_SPEC.md),
[Prompt tổng](GYM_REBUILD_MASTER_PROMPT.md).

## Point-resolution milestone lịch sử (2026-10-18)

Đã chốt/freeze rule/tests rồi triển khai opt-in `squat-v2-point-resolution-v1`.
Input synthetic → unchanged conservative v2 → raw support/instability checks →
noise-identifiability abstention → envelope JSON/report/HTML mới. Strict minimum
có thể là noisy hold với cùng observations; input hiện không có discriminator.
Không nhận fixture GT/noise-free override, không thêm thresholds. Accept0 points,
bounds/pause/motion unavailable; không gọi đây là validated detector. Existing
v1/v2 schema/API/profile/renderers không đổi. Separate reader dispatch cho
`squat-point-resolution-audit-v1`/envelope0.1, không silent migration.

Bước nhỏ tiếp theo: chốt capability/evidence độc lập cho noise discrimination
trước khi xem xét accepting policy mới. Giữ abstention nếu chưa có. G5.7 blocked,
G6/G5 IN_PROGRESS/NOT_VALIDATED/official=false/assessment=false.
[Frozen policy](SQUAT_POINT_RESOLUTION_POLICY.md),
[milestone evidence](STAGE6_POINT_RESOLUTION_REPORT.md).

## Lát cắt GĐ6 trước — temporal v2 (2026-10-18)

Additive/opt-in schema0.2 tách event và motion phase, Bottom components, reasons
và half-open intervals; giữ schema/API v1 và default S3 profile nguyên vẹn.
Contract/acceptance freeze trước code producer, có receipt và expected-red run.
Đường chạy: synthetic scalar input → unchanged PVP v1 → conservative v2 adapter
→ JSON/report/offline HTML. Chưa có bounds detector nên producer giữ anchors/
candidate count nhưng abstain arrival/departure/pause/motion. Positive modes chỉ
là output-contract acceptance, không ground truth được đưa vào producer.
Shared tops có một event/two roles; gap/side-switch split runs; clip-ending hold
không có full candidate/bịa departure. Renderer v2 không hiển thị/tính góc.
Không tune 30-condition sensitivity matrix, thêm dependency hoặc chạy Data.

Bước nhỏ kế tiếp: chốt riêng evidence rule cho **một** bounds/resolution detector,
giữ abstention cho nhiễu/imputed/censored cases và freeze tests trước producer
mới; chưa triển khai detector trong slice này. G6/G5 IN_PROGRESS, G5.7 blocked,
NOT_VALIDATED/official=false/assessment=false. Chi tiết và lệnh kiểm thử:
[frozen contract](SQUAT_TEMPORAL_V2_FROZEN_CONTRACT.md),
[slice report](STAGE6_TEMPORAL_V2_SLICE_REPORT.md).

## Kiến trúc được chọn

Một distribution `ai-coach-gym`, package `gym_coach`, layout `src/`.
Chạy offline trên một máy, chưa có backend/web/database. CLI GĐ4 nối video →
pose → quality → preprocessing cho audit; chưa có orchestrator toàn pipeline.
Module chỉ phụ thuộc hợp đồng chung và kết quả đầu vào được truyền tường minh;
không đọc dữ liệu hoặc import backend ML tại thời điểm import package.

```text
video -> pose -> quality -> preprocessing -> features -> segmentation
                  |              |             |              |
                  +--------------+-------------+--------------+-> reporting
                                               +--------------+-> assessment
                                                                  (BLOCKED)
```

QualityReport phải đi cùng dữ liệu qua preprocessing, features, segmentation;
reporting tổng hợp kết quả từng stage và reason, không tự tính lại thuật toán.
Assessment chỉ nhận feature/rep đủ điều kiện và rule đã duyệt ở Giai đoạn 8.

| Module | Trách nhiệm / hợp đồng | Trạng thái hiện tại (GĐ5 angles implemented; documented validation blocked) |
|---|---|---|
| `video` | path → VideoMetadata + iterator VideoFrame; index/timestamp | COMPLETE GĐ1 |
| `pose` | frame → PoseFrame, JointObservation, model/mapping version | COMPLETE GĐ2 |
| `quality` | pose sequence → QualityReport, eligibility, structured flags | COMPLETE GĐ3 |
| `preprocessing` | pose + quality + metadata/profile → PreprocessingResult | COMPLETE GĐ4; raw/quality snapshots, transforms/runs/audit |
| `features` | GĐ4 result → linked context; registry + context → readiness/numeric observation | 10 angle calculators, v2/schema0.4; legacy knees compatible; ratios/FeatureSequence blocked/chưa triển khai |
| `geometry` + `axis_geometry` | ba điểm → góc trong; bốn joints → midpoint axis/image-up angle hoặc GymError.flag | Hai primitives stdlib, độc lập registry/backend |
| `segmentation` | synthetic temporal contract + quality/FPS → PVP candidates/diagnostics | Prototype S3 synthetic-only; real admission/accuracy BLOCKED, GĐ6 IN_PROGRESS |
| `assessment` | eligible features/reps + approved registry → AssessmentResult | Namespace, BLOCKED expert validation |
| `reporting` | stage snapshots → feedback JSON + offline HTML/Canvas | AuditHeader compatible + squat-only prototype; data feedback, không assessment |

`contracts.py`, `flags.py`, `config.py` là nền dùng chung. Không tạo một cây
class cho các thuật toán chưa có. Hợp đồng domain được triển khai tại đúng giai
đoạn cùng test; tên contract trên là thiết kế, chưa phải class đang chạy.

Lý do: giảm phụ thuộc và tránh lẫn pose failure với lỗi kỹ thuật; layout `src/`
cho phép kiểm tra import của package được cài; adapter pose/video có thể được
test độc lập; schema chung giúp giữ provenance và reason qua từng stage.

## Dependencies và cấu hình

- Python thực tế: CPython 3.11.9, Windows x64; manifest yêu cầu Python >=3.11.
- Runtime GĐ0: `pydantic==2.13.5` đã có trong `.venv`, dùng validation/JSON schema.
- Dev: `pytest==9.1.1`; build: `setuptools==84.0.0`, đều có trong môi trường.
- Không copy `pip freeze`; GĐ1–2 thêm OpenCV contrib 5.0.0.93, MediaPipe 0.10.14,
  NumPy 2.4.6; GĐ4 khai báo SciPy 1.17.1 đã có sẵn, cần cho Savitzky–Golay.
  Torch và framework legacy khác không được dùng trong code mới.
- GĐ5.2 không thêm dependency: arithmetic dùng math/typing; errors dùng contracts/
  flags chung. GĐ5.3–5.5 không thêm dependency; package 0.5.4. Legacy numeric
  0.2.0, knee numeric 0.3.0 và linked context 0.2.0 là các contract riêng; metadata/registry/null-only
  schema 0.1.0 giữ nguyên.
- `configs/default.toml` chỉ chứa MVP exercise, viewpoint, paths, trạng thái
  assessment; chưa có threshold quality, feature, segmentation hoặc kỹ thuật.
- Dependency pin phản ánh môi trường đo được, chưa phải chứng nhận tương thích
  các máy khác. Chưa có lock toàn bộ dependency bắc cầu.
- Version/schema/flags: [SCHEMA_CONVENTIONS.md](SCHEMA_CONVENTIONS.md).

## Giai đoạn và acceptance criteria

Mỗi giai đoạn phải đọc trạng thái mới nhất, thêm test cùng implementation, lưu
bằng chứng và dừng tại gate. Không chuyển sang giai đoạn sau chỉ vì code chạy.

Ngoại lệ phạm vi do yêu cầu người dùng 2026-10-15: bắt đầu **prototype synthetic
GĐ6 và renderer mô tả** dù official GĐ5.7 chưa đạt. Không mở real temporal
admission, không thay acceptance criteria, không tuyên bố GĐ5/GĐ6 COMPLETE.
`GeometryFeatureObservation` numeric0.4 vẫn framewise/temporal false; temporal
capability riêng chỉ nhận synthetic, yêu cầu global sequence và exact hai knee
allow cho fixture qua Stage4. Pure signal fixtures dành riêng unit tests.
Centered mean span0.2s, excursion5°, min half-cycle5/30s là profile S3 thử nghiệm,
không chuẩn kỹ thuật; giữ nguyên Stage3/4 profiles và registry. Stage4 SG vẫn
được ghi trong nguồn; temporal làm mịn signal riêng, không sửa raw/processed.
Giữ raw overlay image_normalized_xy, không vẽ processed normalized coordinates
lên ảnh. Demo thật chỉ preview UNKNOWN metadata/blocked; không numeric/gate
giả. [Phạm vi/bằng chứng/giới hạn](GYM_RESULT_FEEDBACK_PROTOTYPE_REPORT.md).

### Giai đoạn 0 — Nền kiến trúc

- Input: ba tài liệu hiện hành, workspace và inventory môi trường.
- Output: package namespaces, manifest, config/schema/flags, audit header,
  kế hoạch/trạng thái/tasks và smoke test.
- Test: cài package offline, import ngoài workspace, chặn backend thuật toán,
  config portable/validation/hash, JSON roundtrip flags/header.
- DoD: không phụ thuộc archive; manifest/import/smoke đạt; mọi giai đoạn sau
  có input/output/test/DoD; không có scoring/ngưỡng kỹ thuật chưa xác nhận.

### Giai đoạn 1 — Video ingestion và metadata

- Input: path từ CLI/config, video synthetic/fixture nhỏ; chọn decoder tối thiểu
  sau khi kiểm tra môi trường, không mặc định đưa cả hai OpenCV distributions vào manifest.
- Output: VideoMetadata (`video_id`, FPS dương hữu hạn, frame_count không âm,
  width/height dương), iterator VideoFrame, CLI xuất JSON.
- Test: fixture có FPS/resolution/count biết trước; index từ 0; timestamp giây
  theo FPS đối với video CFR; rỗng/quá ngắn, invalid_fps, unreadable_video;
  iterator không tải toàn bộ video. VFR phải được nhận diện/ghi chính sách tường minh.
- DoD: metadata và timestamp đúng; lỗi có flag; không hard-code path; CLI
  chạy trên fixture và một development video chọn rõ ràng, không pose.

### Giai đoạn 2 — Pose extraction và provenance

- Input: iterator VideoFrame/metadata, exercise; chọn development video rõ ràng.
- Output: JointObservation/PoseFrame 2D, x/y/visibility/presence nếu có,
  index/timestamp, model identity và joint mapping version, subject policy;
  coverage per-joint/video. Mapping MediaPipe chưa được chọn lại trong GĐ0.
- Test: mapping/schema, missing/low_visibility không thành observed; lưu/đọc
  roundtrip; integration trên video development. Kiểm tra inverted pose/occlusion
  và khả năng thực tế của model trước khi phát biểu về multi_person.
- DoD: provenance bảo toàn; backend/version được ghi; subject được chọn hoặc
  limitation/flag rõ ràng; không nội suy/normalize/form assessment.

### Giai đoạn 3 — Quality gate

- Input: pose sequence, exercise/viewpoint, versioned S3 quality profile.
- Output: QualityReport, per-joint coverage/consecutive gaps, metric eligibility,
  skip/degrade reasons; excessive_missingness, low_pose_visibility, out_of_frame,
  viewpoint_unknown, insufficient_sequence_length.
- Test: chuỗi rỗng, missing đầu/cuối, gap ngắn/dài, visibility thấp, thiếu một
  bên, unknown viewpoint; integration development video.
- DoD: mỗi skip/degrade có reason; gate độc lập với form assessment; các ngưỡng
  từ đặc tả được ghi S3 và chưa validated; chưa kết luận đúng/sai kỹ thuật.

Baseline profile: `quality-s3-v1`, min 5 frames, missing ratio >0.30,
low-visibility ratio >0.30, invalid run >10 frames. Out-of-frame ratio >0.30
gate metric như extension S3 vì tọa độ ngoài ảnh không dùng được cho metric hình
học; ngưỡng này chưa validated. Gap/ratio tính theo đúng joints metric khai báo.

### Giai đoạn 4 — Preprocessing

- Input: pose + QualityReport + FPS; profile thời gian/FPS versioned S3.
- Output: processed pose giữ raw/index/timestamp, provenance và imputation mask
  per-joint; translation_status mid_hip/fallback_origin;
  scale_status torso/fallback_unit; finite segment boundaries.
- Test: linear short-gap interpolation, missing endpoints, long gaps không nối;
  không mutate input; normalize zero torso/missing hips; smoothing từng finite
  run; 30/60 FPS theo cùng thời gian; integration development video.
- DoD: joint biết observed/imputed/missing; fallback có flags; không smoothing
  xuyên gap dài; không feature/rep logic. Quy đổi baseline frame của đặc tả sang
  thời gian phải ghi giả định và chính sách làm tròn, không ngầm coi 30 FPS là phổ quát.

Implementation GĐ4: `PreprocessingResult` lưu raw snapshots/hashes, quality
snapshot/hash và processed joints; raw/processed schema riêng. Profile TOML
`preprocessing-s3-v1` quy đổi gap 5/30 s bằng floor; smoothing span 10/30 s thành
số interval chẵn gần nhất (ties up), cộng một sample, window odd ≥3 và >polyorder.
Nội suy mặc định chỉ missing, không low visibility; endpoints observed in-frame.
Working units `x, y*height/width` giữ tỷ lệ pixel; normalization mid-hip/torso,
fallback null + flag; smoothing split cả unit-status transitions. Bằng chứng và
policy đầy đủ: [STAGE4_PREPROCESSING_REPORT.md](STAGE4_PREPROCESSING_REPORT.md).

### Giai đoạn 5 — Geometry và feature extraction

- Input: processed pose + raw pose khi metric cần + quality/viewpoint.
- Output: mô tả knee/hip/torso cho squat, elbow/shoulder/body-line cho HSPU;
  registry có formula, joints, unit, source tier, measurement tier, viewpoint,
  quality requirements, role; missing trả unavailable + reason.
- Test: góc synthetic chuẩn/đối xứng, điểm trùng, non-finite, missing joint,
  fallback scale, viewpoint không phù hợp; development integration.
- DoD: calculation và policy tách nhau; không giá trị giả hoặc ngưỡng đúng/sai;
  hip-y sau mid-hip centering không được ngầm dùng làm primary signal.

GĐ5 chia nhỏ:

- **5.1 COMPLETE**: FeatureDefinition/registry metadata, safe policies, input
  snapshots/hashes, readiness/unavailable + structured flags, null-only observation
  schema và tests; 10 angle candidates unvalidated, 2 ratio candidates blocked.
  Không calculator hoặc scientific validation; xem báo cáo GĐ5.1.
- **5.2 COMPLETE**: `gym_coach.geometry.joint_angle(A,B,C)` trả float degrees
  `[0,180]`, pure arithmetic, structured invalid-input/degenerate/numerical errors.
  94 synthetic/import tests và toàn suite 328 tests đạt; không numeric feature
  observation hoặc nối extractor. [Báo cáo GĐ5.2](STAGE5_2_JOINT_ANGLE_REPORT.md).
- **5.3 COMPLETE (phạm vi một feature)**: numeric observation schema 0.2.0 và
  adapter `squat_knee_angle_left_deg`; dùng policy canonical, preserve identities/
  provenance/quality, geometry failures → unavailable. 118 unit + 2 saved real-input
  integrations đạt; full suite 448 pass/4 warnings. Real samples unavailable,
  chưa real-video numeric measurement experiment. [Báo cáo](STAGE5_3_FEATURE_OBSERVATION_REPORT.md).
- **5.4 COMPLETE**: factory full PreprocessingResult + explicit index → linked
  FeatureInputContext 0.2.0, strict detached snapshot/hashes/exact identities;
  adapter revalidates linkage. 107 unit + 2 saved real-input tests đạt, full suite
  557 pass/4 warnings. Legacy/numeric schemas không đổi; linked context lưu riêng.
  Core GĐ4 audit chỉ kiểm tra counts/spans, không tính lại eligibility.
  [Báo cáo](STAGE5_4_CONTEXT_FACTORY_REPORT.md).
- **5.5 COMPLETE**: right-knee numeric adapter `squat-right-knee-v1`;
  KneeAngleFeatureObservation 0.3.0 hỗ trợ đúng hai knee IDs, dispatch bên trái
  vẫn schema 0.2.0 và migration tường minh. 116 unit + 2 saved real-input tests
  đạt; full suite 674 pass/4 warnings. Six real samples unavailable, không ép
  allow hoặc gọi primitive. [Báo cáo](STAGE5_5_RIGHT_KNEE_FEATURE_REPORT.md).
- **5.6 COMPLETE (eligibility/data audit)**: genuine QualityGate fixture cho exact
  hai knee IDs, read-only dataset/viewpoint/report inventory. 39 new tests đạt;
  full suite 713 pass/4 warnings. 224 file/112 unique contents, không documented side-view
  input; real numeric measurement BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT.
  Không đổi package/runtime/profile/schema hoặc thêm calculator.
  [Báo cáo](STAGE5_6_DEVELOPMENT_ELIGIBILITY_REPORT.md).
- **5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**: fresh workspace input check
  xác nhận không có squat video mới hoặc documented side-view annotation liên kết
  path/ID/SHA-256. 224 files/112 unique hashes và 7 dataset metadata giữ nguyên;
  lượt 2026-10-10 chưa inference/QualityGate hoặc tests/full suite. Không sửa runtime/policy/old
  reports. [Báo cáo blocked](STAGE5_7_DOCUMENTED_SIDE_VIEW_GATE_REPORT.md).
  Acceptance sau khi người dùng cung cấp input: verify protocol/confirmer/date/
  interval/camera/framing/occlusion, fresh ingestion→pose→quality với exact hai
  canonical knee keys/joints/side policies; sequence và cả hai metric allow,
  new-run artifacts/config/model/profile/command/output hashes và tests đạt.
  GĐ5.7 chỉ gate, chưa numeric adapters hoặc góc; không relax threshold.
- **5.7A COMPLETE (visual audit/exploratory)**: người dùng cho phép review videos
  hiện có, mỗi byte-distinct content một lần. 112 unique/224 paths, 560 mẫu đã xem:
  44 likely_side, 38 oblique, 22 front, 8 uncertain, 0 documented.
  Candidate R106 chọn trước inference bằng quan sát. Hai fresh unknown/assumed-side
  runs tách biệt; canonical two-knee requirements, profile không đổi.
  Assumed side: sequence skip, left allow, right skip(low_pose_visibility).
  Numeric calls=0; data audit only, pytest/full suite NOT_RUN, counts N/A.
  Inventory/reasons/times/hash và reports/output cũ bảo toàn. Visual assessment
  không đáp ứng GĐ5.7 official acceptance; [báo cáo](STAGE5_7A_VISUAL_VIEWPOINT_AUDIT.md).
- **Implementation cuối GĐ5 COMPLETE trong phạm vi code**: đủ 10 unblocked
  angles, API `observe_geometry_feature`, registry v2 và numeric0.4; development
  audit fresh preselected 3 clips có available/unavailable thật. Hai ratios không
  tính. Legacy GĐ5.3/5.5 APIs/gates không đổi. Documented validation vẫn BLOCKED.
- Toàn GĐ5 chỉ COMPLETE khi calculation/policy đều được kiểm thử; GĐ5.1 không
  cho phép đánh dấu measurement accuracy validated hoặc bật assessment.

GĐ5.1/legacy registry v1 policy S3: side-view only, sequence/feature metric phải allow, exact
required joints, observed/imputed usable finite xy. Missing metric eligibility
là unavailable, không fallback về movement_analysis hoặc tự recompute quality.
Distance/ratio yêu cầu origin/scale hợp lệ; body-line candidate chỉ là trục vai–
hông trong ảnh, không full-body/biomechanical alignment.

GĐ5.2 nhận tọa độ do caller cung cấp, không tự xác định units/viewpoint/quality,
không rotate/flip/normalize input. Positive vector rescaling chỉ để tránh overflow/
underflow trong arithmetic, không sửa dữ liệu; denominator finite >0 trước
cosine clamp. atan2(abs(cross)/denominator, clamped cosine) giữ các góc nhỏ;
không epsilon để biến vector gần zero thành unavailable. Pure math tests không
validation pose/form; error flags stage features, source tier S3.

GĐ5.3 giữ nguyên registry v1 và null-only FeatureObservation. API mới chỉ nhận
canonical ID/definition đã chọn, snapshot qua check_feature_availability trước
gọi geometry. Numeric observation xác minh identity/metadata/hash, finite degree
value/status/reasons/flags và imputation references bracket frame. Không gọi
primitive trong schema readback hoặc recompute QualityReport. Legacy context
thiếu FPS/raw sequence; dùng factory GĐ5.4 cho linked input, không tự đặt FPS/viewpoint.

GĐ5.4 giữ full PreprocessingResult snapshot cùng context. Revalidation strict
kiểm tra raw/processed length, CFR timeline, frame equality/raw hashes, sequence/
quality/profile hashes, resolved profile, imputation endpoints và smoothing audit.
Context thêm result/processed-frame/metadata content identities, không có run UUID.
Factory/adapter tách caller dictionaries và không tính eligibility/geometry tại
factory. Numeric schema 0.2.0 tiếp tục lưu frame/quality input; context audit JSON
lưu riêng và liên kết qua measurement_input_hash. Không tự migrate/downgrade.
QualityReport chưa có raw-sequence identity độc lập: kiểm chứng equality với report
trong result và raw core statistics, không chứng minh producer hoặc exact report
origin giữa hai runs có cùng thống kê. Không rerun preprocessing để chứng minh coords.

GĐ5.5 chỉ mở dispatch cho canonical right-knee ID, không sửa registry/policy/
primitive/factory hoặc upstream schemas. Right inputs hip–knee–ankle theo đúng
thứ tự, quality allow-only và declared side; không thêm scale/origin gate cho góc.
Numeric 0.3.0 kiểm tra audit/provenance/ordered inputs và vector suy biến;
readback không đo lại hoặc nâng validation. Legacy NumericFeatureObservation
0.2.0 giữ nguyên class/file/payload; from_legacy chỉ đổi version qua thao tác
tường minh. Các tám angle candidates còn lại chưa có calculator, hai ratio
candidates vẫn blocked. Không tự chọn bên, fuse hoặc tạo FeatureSequence.

GĐ5.6 test-only fixture đủ sequence/profile, exact canonical metric keys/joints/
side policies, actual QualityGate report không sửa eligibility. Side metadata
của fixture chỉ construction evidence. Inventory đối chiếu mọi squat file trong
Data, heuristic CSV UNKNOWN và saved Stage4 quality reports; report-side field
không là chứng cứ camera protocol. Không inference toàn dataset/numeric thật.
Audit có thể COMPLETE với kết luận thiếu documented input; phép đo thật vẫn
blocked, không dùng synthetic evidence để thay real-data acceptance.

### Giai đoạn 6 — Signal, repetition và phase segmentation

Synthetic sensitivity 2026-10-18: analytic fixture version
`squat-analytic-fixture-v1` và evaluator `squat-temporal-sensitivity-v1`, 10 cases
×25/30/60FPS. Raw signal/reference sinh trước prediction; audit input/FPS
sensitivity, không parameter sweep/tuning. Profile/hash/algorithm/phase_at
nguyên vẹn. Full-cycle matching độc lập event drift; phase agreement giữ cả
Bottom hold interval và unavailable frames. 78/102 matched,24 missed,0 extra;
Bottom pause interval chưa mô hình hóa nên GĐ6 vẫn IN_PROGRESS, real validation
blocked. [Thiết kế, evidence, limits](STAGE6_SYNTHETIC_TEMPORAL_SENSITIVITY_REPORT.md).

- Input: FeatureSequence + quality + FPS + versioned S3 signal profile.
- Output: raw/smoothed signal, polarity, side source, events top_start/bottom/
  top_end, full/partial reps, Descent/Ascent, Bottom boundary, durations giây,
  descriptive ROM và flags; 0 chu kỳ phải có lý do, không khẳng định 0 rep thật.
- Test: synthetic PVP và polarity ngược, endpoints/partial reps, flat/jitter,
  short/long gaps, side switches, 30/60 FPS; development videos sau synthetic pass.
- DoD: không rep qua unresolved gap/side switch; temporal parameters FPS-aware;
  diagnostics có versions; tách execution success, coverage, count accuracy;
  chưa validated khi thiếu reference labels.

### Giai đoạn 7 — Development validation và freeze

- Input: manifest có group/participant/viewpoint/provenance và split cô lập,
  reference-count protocol/labels, pipeline GĐ1–6.
- Output: development run artifacts có config/code hash, MAE, exact match,
  signed error, coverage/failure reasons theo exercise/viewpoint; frozen config.
- Test: kiểm toán group isolation, chống evaluation-label leakage, hash/version
  và tính metric bằng fixtures biết trước; tái chạy development đã chọn.
- DoD: chỉ tune development; freeze trước evaluation; output tách execution,
  coverage, accuracy; nếu chưa có Git dùng source tree SHA-256 manifest để định
  danh code, không giả commit; không tái sử dụng số liệu archive chưa kiểm chứng.

### Giai đoạn 8 — Ground truth kỹ thuật và assessment

- Input: từng rule có S1 hoặc S2 dẫn S1, citation, expert approval về phase/
  feature/range/error/severity/feedback; viewpoint protocol; custom correct/
  incorrect dataset; annotation guide/IAA; split theo participant.
- Output khi thiếu: protocol/schema và danh sách BLOCKED, không scoring.
  Chỉ khi đủ điều kiện: registry/rules, quality-gated assessment, calibrated score
  và feedback có bằng chứng trên custom dataset.
- Test: missing approval/citation không kích hoạt rule; poor quality khóa score;
  tách technical/measurement/data/segmentation errors; participant isolation,
  calibration validation và expert-labelled evaluation độc lập.
- DoD: mọi rule có nguồn + approval; measurement tier/viewpoint rõ ràng; không
  chấm khi quality fail; score/feedback được kiểm chứng. Không có số liệu IAA/
  accuracy giả; training đầu tiên vẫn phải qua MODEL TRAINING GATE của agent rules.

## Giả định và giới hạn

- Các câu “đã triển khai” và số liệu cũ trong đặc tả/.agent chỉ mô tả lịch sử;
  không chứng minh trạng thái nền mới. Ba tài liệu hiện hành và bằng chứng mới
  điều phối rebuild; không mở hoặc restore archive.
- Metadata exercise/viewpoint do người dùng/dataset cung cấp; chưa có classifier.
- Camera 2D chưa calibration; khoảng cách không đổi sang cm.
- Dataset chưa kiểm kê/annotation/split trong GĐ0; không suy ra nhãn form từ UCF101.
- GĐ0 smoke chỉ chứng minh hợp đồng/cấu hình/import. GĐ1–4 hiện đã chạy video,
  pose, quality và preprocessing trên hai mẫu; chưa có accuracy ground truth
  hoặc segmentation, không suy ra form correctness từ coverage/execution.

Roadmap giữ thứ tự gate. GĐ0–4 và GĐ5.1–5.6 COMPLETE theo phạm vi từng bước;
GĐ5.7A COMPLETE theo phạm vi visual audit/exploratory. Implementation angles
cuối GĐ5 COMPLETE, nhưng GĐ5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT,
toàn GĐ5 IN_PROGRESS (VALIDATION_BLOCKED). Nhiệm vụ nhỏ nhất:
người dùng xác nhận protocol/annotation cho video hiện có hoặc input mới, liên kết
path/ID/SHA-256/annotator/date/interval/camera/framing/occlusion; sau verify tạo
official gate mới cho hai knee keys. R106 exploratory còn fail sequence/right
quality; annotation đơn lẻ không đủ mở measurement, cả sequence và hai metrics cần allow.
Official measurement validation BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT; không
nâng eligibility report hiện hữu hoặc suy viewpoint từ nhãn UCF101. Exploratory
numeric integration đã chạy, không thay documented gate; GĐ6 chưa bắt đầu,
assessment vẫn BLOCKED.

## Policy và acceptance cuối GĐ5

Registry v2/schema0.2 kế thừa canonical measurement metadata v1; 10 angles
`calculation_status=implemented`, measurement status/tier vẫn unvalidated và
assessment blocked. Policy `framewise-metric-only-v1` dựa đặc tả §§2.1–2.2:
phân tích chuỗi/chu kỳ cần sequence gate; mô tả từng frame chỉ dùng exact feature
metric allow + required usable joints + provenance + viewpoint. Không bỏ metric
degrade/skip, không đổi profile/threshold, không recompute report trong adapter.
Numeric0.4 giữ nguyên global decision trong audit; temporal_analysis_eligible=false.
API/schema registry v1/knee0.2/0.3 giữ sequence+metric gate cũ, không migration ngầm.

Development subset frozen: R106 từ pre-inference visual choice5.7A; HSPU first6
SHA-sorted unique contents, xem5 distributed samples/clip rồi chốt H004 likely_side
và H002 front control. Không chọn lại sau numeric outcomes. Fresh cả unknown và
visual-hypothesis branches: 2.790 observations, 534 available/2.256 unavailable.
Formal documented input chưa có; sequence/right metric của side candidates fail.

Acceptance mapping: formulas/calculation/quality separation, all-angle synthetic
tests, serialization/identity/provenance/light imports/legacy compatibility và
development pipeline đều được kiểm thử. **Documented validation subgate5.7 chưa
đạt**; whole Stage5 chưa COMPLETE, không mở Stage6–8/scoring. Evidence và test
commands: [STAGE5_FINAL_FEATURE_EXTRACTION_REPORT.md](STAGE5_FINAL_FEATURE_EXTRACTION_REPORT.md).
