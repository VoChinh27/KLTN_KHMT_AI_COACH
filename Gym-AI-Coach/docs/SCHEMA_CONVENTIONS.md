# Schema, version và structured flags

Ngày cập nhật: 2026-10-18 (Asia/Saigon). Contract GĐ0 dùng Pydantic v2, cấm unknown fields và NaN/Inf.
Model `frozen` ngăn gán lại field; các dict lồng nhau không bất biến sâu.
Serialize bằng `model_dump_json()`/`model_dump(mode="json")`, deserialize bằng
`model_validate_json()`; có thể lấy JSON Schema qua `model_json_schema()`.

## Count research preparation — additive 2026-10-25

`gym_coach.counting` tách khỏi segmentation v1/v2; không migration/admission ngầm.
Dataset `squat-count-dataset-v1`, labels `squat-count-labels/0.1.0`, protocol
`squat-count-video-annotation-v1`, adjudication `squat-count-adjudication-v1`,
prediction `squat-count-prediction-v1`, evaluator `squat-count-evaluation-v1`.
Collection envelopes có version riêng và byte SHA-256 của dataset manifest;
pass/source/hash/protocol phải khớp chính xác. Canonical object hashes dùng JSON
UTF-8 sorted compact; file hashes dùng bytes, không hoán đổi hai loại hash.

Strict/frozen/extra-forbid/finite, native model được revalidate trước JSON để
không coercion bool thành count1; JSON intake strict, nested values kiểm tra lại.
Source path/ID/gXX/byte hash + metadata CFR; gXX không là subject identity.
Timestamp=i/FPS kiểm tra1e-9s; inclusive label samples, shared full-cycle top hợp lệ.
Partials chỉ tại biên clip, uncertain nội bộ riêng; full reference cycles không
được giao masks. Bottom event/interval/unavailable, không suy physical pause.

Pending labels không có người/time/count/cycles tự sinh. Available0 khác null;
count available có thể boundaries unavailable. `affects_count=true` làm count
whole clip unavailable. Hai passes độc lập/mù, người phân xử thứ ba, log phải bind
hashes hai passes/final và time. Schema không xác thực tư cách/rater honesty.
Real data từ chối method synthetic_fixture; bộ60 pixel images không temporal GT.

Prediction estimated/candidates_only/abstained/failed/not_run tách biệt; ba trạng
thái cuối null count/empty cycles + structured Reason code/category/stage/S3.
Candidate-only cũng null count. Không sửa FlagCode cũ. Public count contracts/
evaluator/split không import video/pose/NumPy/SciPy/MediaPipe/Torch.

Evaluator giữ mọi source row/mẫu số, missing prediction là not_run; null không là0.
Count MAE/signed/exact chỉ trên final+estimated. Event matching exhaustive-only,
monotone max-cardinality rồi min boundary error, tolerance0,20s cấu hình S3.
Partial/uncertain masks loại và báo predicted events riêng, không sửa count error.
Bottom timing chỉ point-vs-point. Empty labels accuracy null/NOT_EVALUATED.
NOT_VALIDATED/official=false/assessment=false không thay đổi từ evaluation.
[Protocol](SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md),
[readiness/evidence](SQUAT_COUNT_DATA_READINESS_REPORT.md).

## Point-resolution opt-in audit envelope (2026-10-18)

`point_resolution_schema_version=0.1.0`, contract `squat-point-resolution-audit-v1`,
policy/producer `squat-v2-point-resolution-v1`, policy hash
`930d41199421d6544f775b4485ba2b347005498b836fbda6532ee16dce4e4642`.
Không sửa `SquatTemporalResultV2` đã freeze để thêm producer literal mới: envelope
riêng giữ nguyên full conservative v2 `base_result`/`base_hash`. New reader
`read_temporal_experiment_json` dispatch envelope vs segmentation v1/v2; old
readers không tự diễn giải envelope. Public exports/renderer/profile cũ không đổi.

Mỗi full PVP có check exact B-1..B+1/raw + S..E/candidate support và input hash.
Strict minimum/necessary_conditions_met chỉ là fact/điều kiện cần trên samples,
không phải point-turn event. Thiếu independent noise evidence nên status luôn
unavailable, accepted_point_count=0, point/arrival/departure/pause null. Reasons
segmentation/S3/insufficient_data và inclusive support; original flags/provenance
giữ trong base, không sửa FlagCode. No-full-PVP là reason riêng, không true0rep.
Readback kiểm tra version/linkage/count/null/reasons; hash không authenticate hay
chứng minh signal accuracy. Không recompute segmentation/QualityGate khi đọc.
NOT_VALIDATED, official=false/assessment=false; synthetic-only, G5.7 blocked.
[Policy](SQUAT_POINT_RESOLUTION_POLICY.md), [evidence](STAGE6_POINT_RESOLUTION_REPORT.md).

## Additive experimental temporal v2 (2026-10-18)

Opt-in modules `segmentation.temporal_v2_models`, `segmentation.temporal_v2` và
`reporting.temporal_v2_renderer`; không đổi old exports. Schema0.2.0,
contract `squat-temporal-components-v2`, producer `squat-v2-conservative-adapter-v1`,
policy `squat-v2-anchor-only-v1` (hash
`6a9435df2b3f6d8ccc694820a20d3cf4d75f3ba7beeea12ae4339f0dc892e43f`).
Default S3 profile/hash không đổi. Legacy input/full result + hash được giữ;
event ID SHA-256 từ input hash/run/kind/frame, event time đúng input CFR.

Evidence support inclusive, motion/pause closure `[)` riêng. Dwell contract:
Descent[S,A), Bottom[A,D), Ascent[D,E), pause D-A. Point contract: Descent[S,B),
Ascent[B,E), arrival/departure/pause unavailable/null; không gán 0s. Shared top
một ID/hai roles; at shared top phase thuộc cycle sau. Bounds-unresolved giữ
anchor/count candidate nhưng motion unavailable. Đây là mode **duy nhất producer
hiện tạo**, không có detector bounds/dwell/point-turn. Hai positive modes có
`producer_version=contract_acceptance_only`, dùng kiểm thử output semantics.
Không coi method/evidence record/hash là scientific validation.

Component status candidate/unavailable/not_run/blocked; null/empty + structured
V2Reason bắt buộc khi abstain. Reasons có category insufficient_data,
stage segmentation/source S3/support; không thay FlagCode hay semantics cũ.
Original flags nằm đầy đủ trong linked legacy result. Slice chưa hỗ trợ giữ một
bound đơn lẻ hoặc event ngoài full PVP; partial/censored runs có diagnostics,
không bịa arrival/departure/full count. Imputed provenance giữ trong input;
declared bound/point support của output-contract tests cần observed, không
đồng nhất imputed với observed. Producer chỉ xuất legacy_proxy anchors.

`read_temporal_json` dispatch chính xác 0.1.0 hoặc0.2.0, extra forbid, finite,
linkage/reasons/order/half-open/provenance checks; không migration ngầm. Dùng
`strict_json` (allow_nan=false, warnings=error), không serializer làm NaN thành
null. Chưa nhận `GeometryFeatureObservation` hoặc real input; capability
synthetic hiện hành vẫn độc lập. NOT_VALIDATED/official=false/assessment=false.
[Frozen semantics](SQUAT_TEMPORAL_V2_FROZEN_CONTRACT.md),
[evidence](STAGE6_TEMPORAL_V2_SLICE_REPORT.md).

## Identity và compatibility

| Identity | Giá trị nền | Quy tắc |
|---|---|---|
| package version | `0.7.0` | additive synthetic temporal + feedback schemas; registry/numeric0.4/legacy knees giữ nguyên |
| output schema_version | `0.1.0` | AuditHeader từ chối version khác; migration tường minh |
| config_schema_version | `0.1.0` | PipelineConfig từ chối version khác |
| config_version | `foundation-v1` | Đổi mỗi khi sửa profile/settings cho run |
| config_hash | SHA-256 | JSON validated, sorted keys, compact separators, UTF-8 |
| joint_mapping_version | `mediapipe-pose-13-v1` | 13 canonical joints; mapping version độc lập schema |
| pose_schema_version | `0.1.0` | PoseFrame/PoseSequenceSummary từ chối version khác |
| pose config_version | `pose-extraction-v1` | PoseExtractionConfig độc lập foundation config |

SemVer: trước 1.0, breaking change tăng minor; patch chỉ sửa tương thích. Từ
1.0, breaking tăng major, additive tăng minor. Không tự nâng version hoặc đọc
contract mới như cũ. Literal version và constants phải cùng được cập nhật với
test migration khi thay đổi schema. Config hash giữ path portable chưa resolve;
artifact run sau này cần thêm resolved input identity/hash và code identity.

Config load dùng path tường minh. Relative input/output paths được resolve qua
`config.paths.resolve(config_path.parent)`; config trong `configs/` sử dụng
`../Data` và `../outputs`. Không tạo thư mục khi load, không phụ thuộc cwd sau
khi caller đã chọn config path/base, không có path máy cá nhân trong code/config.
`PipelineConfig()` có defaults Data/outputs, caller phải truyền base khi resolve.

Assessment hiện `enabled=false` và status
`NOT_AVAILABLE_PENDING_EXPERT_VALIDATION`. Cấu hình bật true hoặc thêm score
không hợp lệ. Không có feature/scoring rule hay threshold nào trong config GĐ0.

## Quy ước domain dành cho giai đoạn sau

- Exercise chỉ BodyWeightSquats/HandstandPushups; viewpoint mặc định unknown.
  Side/front/oblique là metadata khai báo, không phải output classifier.
- Frame index bắt đầu 0; flag frame interval inclusive `[start,end]`; rep index
  bắt đầu 1; durations/timestamp bằng giây. Phase boundaries sau này lưu explicit
  start/bottom/end, Bottom là event chung của Descent/Ascent.
- x tăng sang phải, y xuống dưới. Pose raw và processed phân biệt coordinate
  space/units; không suy ra cm. GĐ2 khai báo mapping và coordinate units cụ thể.
- Per-joint/per-frame provenance: observed, low_visibility, imputed, missing.
  Imputed phải giữ nguồn/gap/mask; không đổi low_visibility thành observed.
- Translation: mid_hip/fallback_origin; scale: torso/fallback_unit; side source:
  left/right/bilateral/unavailable; chưa có thuật toán hay class domain này GĐ0.
- Gap dài và side switch là ranh giới, không nối rep qua đó. Feature unavailable
  dùng null + status/reason; chưa chạy stage dùng not_run, không trả số 0 giả.
- SourceTier S1/S2/S3/OBS/TBD-EXPERT độc lập MeasurementTier M0/M1/M2/M3.
  Một feature có thể có nhiều source tiers trong registry GĐ5.
- Pose coverage là tỷ lệ dữ liệu; không đặt tên assessment confidence.
  Segmentation proxy phải ghi rõ chưa validated; assessment confidence null
  trước expert validation. Không suy ra accuracy từ execution success.

## Gym Result Feedback và temporal squat prototype (0.7.0)

Các identity mới độc lập numeric0.4, không migrate hoặc sửa schemas/gates cũ:

| Contract | Schema / version | Ràng buộc |
|---|---|---|
| `SquatTemporalInput` | 0.1.0 / `squat-temporal-input-v1` | chỉ `synthetic_contract_test`, ID synthetic-, CFR zero-based, fixed declared left/right runs, PVP |
| `SquatSegmentationProfile` | 0.1.0 / `squat-temporal-s3-v1` | S3 / NOT_VALIDATED; span0.2s, excursion5°, min half-cycle5/30s |
| `SquatSegmentationResult` | 0.1.0 / `squat-pvp-hysteresis-v1` | input/profile hashes, smoothed signal, candidates/boundaries/flags, official=false |
| `SquatFeedbackResult` | 0.1.0 / `gym-squat-feedback-v1` | BWS only, source/registry/segmentation hashes, per-frame phase/metric/evidence, assessment false |
| `FeedbackSource` | `squat-feedback-source-v1` | synthetic fixture upstream hashes hoặc permitted real preview byte hashes/UNKNOWN metadata, documented=false |

Hash mới = SHA-256 JSON validated, sorted keys, compact, UTF-8
`ensure_ascii=False`, `allow_nan=False`; dùng `content_hash()` của đúng model.
Source preprocessing/raw hashes tiếp tục dùng hàm hash GĐ4 hiện hành, không đổi
canonicalization cũ. Byte SHA của video/CSV/JPEG/JSON là identity artifact khác
content hash của model. CLI ghi registry snapshot, profile, schema, full Stage4
synthetic input, numeric observations và inventory với command/environment/code
hashes; hash không chứng minh người xác nhận viewpoint hoặc accuracy.

Temporal capability không phải `GeometryFeatureObservation`/FeatureSequence:
numeric0.4 vẫn `temporal_analysis_eligible=false`. Pure synthetic signal fixture
không được nhận QualityReport giả. Fixture từ Stage4 phải có exact source/quality
hash, side policy, global sequence allow và cả hai knee metric allow với đúng
hip/knee/ankle mỗi bên. Không tự recompute/promote report trong gate này. Real
temporal admission chưa có. Samples giữ observed/imputed references/null/reasons;
imputed không thành observed. Mọi unresolved null và side switch đều split finite
runs, cả smoothing lẫn PVP; không auto side fallback hoặc bilateral fusion.

Time theo giây, values degrees hữu hạn `[0,180]`. Centered-mean radius =
`floor(span_seconds * fps / 2 + 1e-9)` capped bằng run length, cắt window tại
biên run; không nội suy signal. Excursion là amplitude hysteresis S3, **không phải
mathematical peak prominence hoặc chuẩn độ sâu**. Equal plateaus midpoint,
first/last extremum unconfirmed; chỉ confirmed interior PVP đủ hai half-duration
mới là candidate. Min spacing giây so trực tiếp, không cố định frame mọi FPS.
Không nhận VFR source timestamps. Các status: experimental_candidates /
unavailable / blocked; candidate_count=0 khi unavailable không phải true rep count.

Feedback metric lưu canonical feature/definition hash, degree value hoặc null,
unvalidated, quality action, actual ordered ProcessedJoints và numeric observation/
input hashes. Unavailable luôn có structured flags + reasons; NOT_RUN không có
joint/observation/input giả. Evidence frame interval inclusive, giây CFR. Descent/
Ascent dùng interval candidate; Top/Bottom/Return là event một frame, chưa là
phase-duration được annotation. Shared top hiển thị Return/Top của rep trước;
candidate tiếp theo vẫn giữ start riêng trong segmentation. Frame raw overlay
giữ image_normalized_xy và source frame hash; không vẽ normalized processed
coordinates. Missing không vẽ, low_visibility màu khác; không khẳng định form.

Năm flags additive: `invalid_temporal_input`, `invalid_feedback_input`,
`temporal_gate_blocked`, `documented_input_missing`, `feedback_serialization_error`.
Invalid input/readback: category input_error, ERROR, stage segmentation/reporting,
source S3. Temporal gate failure: segmentation_error/WARNING/segmentation/S3;
documented input thiếu: insufficient_data/WARNING/reporting/S3. Severity chỉ vận
hành, không severity lỗi người tập. Dùng lại gap/side-switch/partial/unstable
flags hiện hành; không đổi ý nghĩa flags cũ. Serialization/render revalidate
strict Python-mode JSON trước khi render để không biến NaN thành null/bool thành
float. JSON nhúng escape HTML delimiters, DOM dùng textContent, không external JS.

Chi tiết/test evidence: [báo cáo prototype](GYM_RESULT_FEEDBACK_PROTOTYPE_REPORT.md).

## Structured flags và errors — hợp đồng nền

### Synthetic sensitivity audit (2026-10-18)

Module fixture `segmentation.synthetic_fixtures`, version
`squat-analytic-fixture-v1`, scope `analytic_synthetic_signal_only`. CycleSpec
ghi descent/pause/ascent seconds và excursion degrees (signal amplitudes,
không chuẩn form). Half-cosine strokes, constant bottom hold, explicit guards,
gap masks và declared side switch; CFR nearest frame ties-up với epsilon1e-9.
Reference phases sinh độc lập algorithm; Bottom reference là toàn hold interval,
midpoint proxy được lưu riêng. Boundary-partial references liệt kê tách khỏi
full-cycle denominator; invalid/missed full cycles vẫn nằm trong mẫu số chính.

Evaluator `squat-temporal-sensitivity-v1`: protocol hash SHA-256 sorted compact
UTF-8 JSON/ensure_ascii=False/allow_nan=False; definition/input/result/profile
content hashes và manifest byte hashes riêng. Persist reference trước prediction.
Matching monotone1:1 max-cardinality rồi max total IoU, IoU≥0.5 và predicted
bottom nằm trong reference cycle. Đây là evaluation association policy, không
phải ngưỡng segmentation hoặc kỹ thuật. Sai lệch signed=pred−ref (frame/giây),
timing summaries chỉ matched samples; misses/extras và execution failures không
biến thành zero-error. Không match thì MAE null, không phải0.

Phase denominator là union full-reference frame intervals inclusive, kể cả gap
và cycles không phát hiện. Availability coverage khác exact-frame label agreement;
Unavailable tính là không đúng nhãn. Bottom arrival/departure errors so marker
đơn của prediction với bounds của synthetic hold, không claim algorithm đã dự
đoán pause boundaries. JSON/Markdown audit là diagnostics, không form score.
Profile hash bắt buộc giữ `5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`;
official=false, assessment=false, NOT_VALIDATED, G5.7 blocked. Không đổi
schema/API/flag semantics hiện hành; package vẫn0.7.0, audit identity/code hashes riêng.

`StructuredFlag`: code, category, severity, stage, message, source_tier,
location và evidence. Location hỗ trợ video_id, inclusive frame_start/frame_end,
joint, feature, rep_index. Severity là mức vận hành info/warning/error, không
phải mức độ lỗi kỹ thuật chuyên gia. Evidence là key/value scalar có đơn vị
trong key/message khi cần, không nhúng chuỗi dữ liệu lớn.

Các nhóm khoa học giữ riêng: technical_error, measurement_error,
insufficient_data, segmentation_error, informational_observation.
input_error, configuration_error và operational_error bổ sung để mô tả lỗi vận hành. Sự hiện diện
của enum technical_error không cho phép tạo rule/score chưa được phê duyệt.

`FlagCode` chứa tất cả mã edge case ở §7 đặc tả, segmentation_failed ở §2,
video_too_short/unreadable_video ở GĐ1 và invalid_config/assessment_unavailable.
Chúng là vocabulary, chưa phải quality gate tự chạy. Stage tạo flag chịu trách
nhiệm category/source tier phù hợp; không dùng measurement flag để gán form sai.
Quality decisions GĐ3 sẽ lưu action/eligibility/reason tường minh.

`GymError.flag` mang cùng contract để CLI/report chuyển lỗi vận hành thành
JSON; không nuốt exception lập trình bằng catch-all. Config load wrap file/TOML/
validation failures thành invalid_config và giữ cause.

`AuditHeader` hiện chỉ có input video_id, exercise/viewpoint, run_id, package/
schema/config identities, flags và trạng thái mặc định not_run cho các stage.
Nó không phải kết quả phân tích video. GĐ1 bổ sung metadata và input identity;
các giai đoạn sau mới thêm result payload đã có test. Full report không được
điền rep_count=0/score=0 khi stage chưa chạy hoặc assessment đang blocked.

## Video ingestion contract — Giai đoạn 1

`VideoMetadata` gồm `video_id`, absolute `source_path`, FPS hữu hạn dương,
`frame_count >= 0`, width/height nguyên dương, decoder identity và timestamp
policy. `duration_seconds` là thời lượng danh nghĩa `frame_count / fps`, không
phải presentation duration đã kiểm chứng cho VFR.

`VideoFrame` sở hữu `frame_index` bắt đầu 0, `timestamp_seconds` hữu hạn không
âm và decoded image. Decoder chạy streaming; không nạp toàn bộ video mặc định.
Timestamp dùng `frame_index / fps` và ghi `fps_derived_cfr_assumption`, vì
backend OpenCV không cung cấp kiểm toán VFR đáng tin cậy trong phạm vi GĐ1.

Lỗi video dùng structured flags: `unreadable_video`, `video_too_short`,
`invalid_fps`, `invalid_frame_count`, `invalid_video_resolution`. `min_frames`
là input gate S3 do caller truyền, không phải tiêu chuẩn kỹ thuật động tác.

## Pose extraction contract — Giai đoạn 2

`JointObservation`: joint, x/y nullable, visibility/presence nullable trong [0,1],
visibility_threshold, status và missing_reason. Tất cả số phải hữu hạn; không
chấp nhận imputed ở pose extraction. Joint measured cần x/y/visibility;
observed khi visibility >= threshold, low_visibility khi thấp hơn. Schema
từ chối trạng thái trái visibility. Missing có x=y=null và reason bắt buộc;
adapter loại observation hỏng, không đưa NaN/Inf hoặc tọa độ giả vào JSON.

`PoseFrame`: pose schema version, video_id, exercise/viewpoint, index/timestamp
gốc, has_pose, backend/config, tuple 13 unique joint observations và flags.
Joint threshold phải khớp frame config. has_pose chỉ nói backend trả landmark
list không rỗng; không đảm bảo cả 13 joint usable. No-pose frame giữ nguyên
index/timestamp và cả 13 missing joint với reason pose_not_detected.

`PoseBackendIdentity`: MediaPipe API/version, model asset SHA-256, mapping
version, `image_normalized_xy`, `dimensionless_image_width_height`,
`x_right_y_down`, subject policy `backend_most_prominent_single_subject` và
limitations. Tọa độ có thể ngoài [0,1]: giữ số hữu hạn và flag out_of_frame,
không clip. Chưa có mid-hip/torso normalization hoặc physical units.
Solutions Pose 0.10.14 xóa presence khỏi result: null, không suy ra presence=0/1.

`PoseExtractionConfig` version/hash được lưu đầy đủ: visibility/detection/
tracking confidence defaults 0.5 (S3); model_complexity=1 cố định để không tải
model khác; video tracking bật qua static_image_mode=false; smooth_landmarks,
smooth_segmentation và enable_segmentation đều false. Đây không phải smoothing
hay interpolation của project; backend vẫn sử dụng ROI tracking nội bộ.
Foundation config/schema chưa đổi; pose config độc lập để giữ tương thích.

Coverage với N frame, J=13:

- Per-joint pose_coverage = observed_count/N.
- Sequence pose_coverage = sum(observed_counts)/(N*J).
- required_joint_coverage tính tương tự trên 8 joint bắt buộc của exercise.
- pose_detection_ratio = frames_with_pose/N; frames_with_pose + frames_without_pose = N.
- Low_visibility và missing đếm riêng; không tính vào observed coverage.
- N=0: coverage null, counters 0, insufficient_sequence_length flag.
- Out-of-frame nhưng visibility đủ vẫn là observation backend, có flag;
  coverage này không bảo đảm geometric eligibility (quality gate là GĐ3).

`PoseSequenceSummary` lưu backend/config/hash và counters/ratios per-joint;
schema kiểm tra tổng và consistency. Accumulator stream không giữ image/pose
sequence; từ chối mixed identity/config và index/timestamp không tăng.

Serialization: UTF-8 JSONL, mỗi dòng một PoseFrame; readback validation qua
`read_pose_frames`. Summary là một JSON object. Writer dùng exclusive creation;
không overwrite artifact cũ. Backend/input thất bại có thể để lại partial JSONL;
chỉ run có manifest completed mới có bằng chứng full iteration.

Flags bổ sung: pose_not_detected, missing_joint, missing_landmark_field,
non_finite_landmark, invalid_landmark_confidence, invalid_pose_input,
invalid_pose_sequence, pose_backend_error, pose_serialization_error.
Backend init/process/cleanup errors dùng operational_error, giữ exception cause;
no pose dùng insufficient_data; visibility thấp dùng measurement_error.
Các flag generic mô tả kết quả quan sát, không tự gán nguyên nhân inverted/
occlusion hoặc lỗi vận động viên. Không phát flag multi_person từ backend không
enumerate người; limitation được ghi rõ thay thế.

## Quality gate contract — Giai đoạn 3

`QualityProfile` version/hash giữ ngưỡng S3 theo cấu hình: minimum sequence
frames, missing ratio, low-visibility ratio và maximum consecutive invalid run.
Các baseline `5`, `0.30`, `0.30`, `10` là engineering assumptions theo đặc tả
Gym, chưa validated trên development set.

`JointQuality` giữ riêng observed/low-visibility/missing và out-of-frame counts,
observed ratio, unusable ratio, gap spans inclusive theo frame ID và maximum
invalid run. Out-of-frame có thể giao với low-visibility; `invalid_ratio` đếm
union, không cộng trùng. Observed coverage mô tả provenance pose, không khẳng
định metric có thể dùng.

`MetricEligibility` áp dụng profile lên required joints của metric. `allow`
nghĩa là không gặp gate này; `degrade` nghĩa là partial coverage chưa vượt
ngưỡng; `skip` có reason machine-readable. Viewpoint chỉ gate metric khi caller
khai báo `metric_viewpoints`; viewpoint unknown mặc định được ghi flag nhưng
không chặn segmentation tổng quát. Tọa độ ngoài `[0,1]` được giữ để audit,
được đếm là không dùng được và gate metric tương ứng khi chiếm quá
`max_missing_ratio` (extension S3, chưa validated).

Quality gate yêu cầu cùng video/exercise/viewpoint, frame index liên tiếp và
timestamp tăng dần. Report không tạo form error, score, interpolation hoặc sửa
pose. `sequence_status` phản ánh eligibility của exercise required joints;
mọi `metric_eligibility` được lưu riêng để mỗi metric có gate phù hợp.

## Preprocessing contract — Giai đoạn 4

`preprocessing_schema_version=0.1.0` độc lập với pose/quality/foundation schemas.
`PreprocessingProfile` có `profile_schema_version=0.1.0`,
`profile_version=preprocessing-s3-v1`, S3 và canonical content SHA-256 như config.
Profile nằm tại `configs/preprocessing_s3_v1.toml`; caller load path tường minh.
Thay setting phải đổi version khi publish baseline mới; hash phát hiện thay đổi
ngay cả khi caller quên đổi version. ResolvedProfile lưu FPS, gap frame limit,
smoothing window và actual span seconds. Công thức làm tròn ghi trong báo cáo GĐ4.

`PreprocessingResult` chứa metadata, profile/hash/resolved profile, QualityReport
snapshot/hash, raw PoseFrame snapshots, từng raw frame hash, raw_sequence_hash
và processed frames/smoothing runs. Sequence hash = SHA-256 của chuỗi nối các
frame hash ASCII theo thứ tự. Không hash đường dẫn resolve vào profile; metadata
lưu path input do reader cung cấp, không hard-code path trong implementation.
Revalidation/snapshot tách các dict khỏi caller; models frozen chưa bất biến sâu,
hash validation khi ghi/đọc phát hiện thay đổi raw/profile/quality.

Input là full sequence CFR từ GĐ1: index 0..N-1, timestamps khớp index/FPS
(tolerance 1e-9 s khi kiểm tra; output giữ số gốc), metadata frame_count=N, cùng
video/exercise/viewpoint/backend/config. Không tự resample hoặc sửa thứ tự.
Empty sequence hỗ trợ audit counts zero/coverage null; quality dùng sentinel
identity `unknown` theo contract GĐ3. Quality core counts/spans được đối chiếu
input, custom metric eligibility/flags được giữ đúng nguyên bản. Preprocessing
được phép tạo audit output khi quality skip, không nâng quality eligibility.

`ProcessedJoint`: joint, `source_status` (raw observed/low_visibility/missing),
`status` (observed/low_visibility/imputed/missing), usable/unavailable_reason,
pre_normalization_x/y, unsmoothed_x/y, x/y cuối cùng, interpolation source,
smoothing_status (smoothed/skipped_short_run/unavailable). Processed unavailable
có tất cả tọa độ null; raw measured coords/confidence vẫn có trong snapshot.
Observed ngoài ảnh vẫn giữ observed provenance nhưng usable=false + out_of_frame.
**Observed provenance không đồng nghĩa usable/quality eligible.**

Imputed giữ source_status gốc, endpoint indices/timestamps và endpoint statuses
observed; validation kiểm tra endpoints raw in-frame, bracketing và gap/profile.
Default không nội suy low visibility; opt-in chỉ biến short interior low points
thành imputed, không bao giờ observed hoặc dùng làm endpoint. Không chain từ
imputed endpoint; head/tail/long gap và observed ngoài ảnh không được nội suy.

Working coordinate space `image_width_units_xy`: x=x_raw, y=y_raw*height/width,
axes x_right_y_down. Pre-normalization tọa độ được lưu để audit; chưa physical
calibration. Khi đủ hai hips, origin là mid-hip; khi đủ shoulders/hips và torso
norm >min_torso_scale hữu hạn, chia torso. Các nguồn anchor có thể imputed,
provenance tra từ chính processed joints của frame. Không đánh giá accuracy anchor.

`NormalizationTransform`: translation_status mid_hip/fallback_origin;
scale_status torso/fallback_unit; origin_x/y và torso_scale nullable.
Fallback origin bỏ translation, fallback scale bỏ division; giá trị metadata
null, không ghi gốc (0,0) hoặc scale 1 giả. Tọa độ output có torso units khi
scale hợp lệ, working units khi fallback. `distance_ratio_eligible` chỉ xác nhận
scale_status=torso, còn cần usable joints + quality/viewpoint gates downstream.

`SmoothingRun` giữ joint, inclusive start/end, window, normalization statuses,
smoothed/skipped_short_run. Không cross unresolved joint gaps hoặc unit-status
transitions. Short runs giữ nguyên unsmoothed coords và flag tại frame đầu có
location toàn interval. Smoothing không đổi provenance và không clamp overshoot.

Serialization là UTF-8 JSON object self-contained, exclusive writer, strict
reader, từ chối schema/version khác, NaN/Inf, hash/identity/provenance hoặc run
audit không nhất quán. Lỗi có GymError.flag: invalid_preprocessing_input,
quality_input_mismatch (input_error), invalid_config (configuration_error),
preprocessing_error/preprocessing_serialization_error (operational_error).
Fallback normalization và smoothing_short_run là measurement_error, S3,
stage preprocessing; không phát technical_error/score/form feedback.

## Feature metadata/readiness contract — Giai đoạn 5.1

Feature schema và registry schema `0.1.0`; registry version
`gym-feature-registry-v1`; mỗi definition có version `0.1.0` và stable snake_case
ID. Canonical hash = SHA-256 JSON sorted keys, compact separators, UTF-8,
ensure_ascii=false, allow_nan=false. Registry tuple immutable, unique IDs,
per-feature quality metric key bằng ID. Helper mappings được tạo mới, không
mutate registry/report. Không đọc file/config hoặc import ML/array backend khi
import public contracts/registry.

`FeatureDefinition` yêu cầu name/exercises, measurement method/definition/formula,
ordered required_joints, joint policy, quantity/unit, source references/tiers,
measurement tier, validation, viewpoint policy, quality policy, roles, reasons và
limitations. Không defaults cho safety policies. Method arity: joint_angle_abc
3 joints (A,B,C); midpoint_axis_angle_vertical 4 joints (left/right hip, left/right
shoulder); horizontal_offset_torso_ratio và normalized_euclidean_distance 2 joints.
Tất cả joints unique, đúng mapping; chưa có numeric calculation.

Kinds/units: angle/degrees, distance/torso_lengths, ratio/dimensionless_ratio;
không cm. Distance/ratio bắt buộc requires_valid_scale và requires_mid_hip_origin.
Roles chỉ descriptive/informational/segmentation_candidate; không assessment
role/threshold. Validation measurement_status unvalidated hoặc blocked, có reason;
calculation_status not_implemented và assessment_status blocked_pending_expert_validation.
M0 phải blocked; không cho phép tự ghi validated trong schema GĐ5.1.

ViewpointPolicy phân chia đủ UNKNOWN/SIDE/FRONT/OBLIQUE, không trùng, mỗi unsupported
viewpoint có reason; UNKNOWN không được support. MVP chỉ SIDE theo conservative
S3 policy, chưa validated protocol. Trục đứng cho axis candidates là image-up
(0,-1) theo y-down; trục hip→shoulder có hướng, không đổi dấu để che inverted pose.
HSPU body_line_angle chỉ đo candidate shoulder-hip axis, không đủ joints toàn thân.

Joint policy chỉ cho observed/imputed usable finite processed xy. Imputed giữ
source_status/endpoint record trong input snapshot, không trở thành observed.
Low visibility/missing/unusable/null bị khóa; malformed schema input (NaN/Inf,
missing structural joint, identity mismatch) trả GymError invalid_feature_input.
Unresolved gaps và side continuity thuộc GĐ4/GĐ6, không được suy ra trong registry.

`FeatureInputContext` lưu detached snapshots ProcessedPoseFrame và QualityReport;
video ID phải khớp, index nằm trong full Stage 4 sequence. Exercise/viewpoint đọc
từ QualityReport. Context hash gồm frame/raw_frame_hash/transforms/provenance và
quality report đầy đủ. Không sửa raw/GĐ4/report, không nâng MetricEligibility.
Caller phải lấy cùng frame/report từ một GĐ4 result; QualityReport hiện chưa có
raw-sequence hash, context identity checks chưa chứng minh exact coordinate linkage.

`FeatureAvailability` lưu registry version/hash, definition/hash, input/hash,
status ready_for_calculation/unavailable, reasons và flags. Eligibility missing
key là unavailable; exact feature joints và sequence/metric allow bắt buộc;
degrade/skip không được nâng. Serialization validation tính lại policy checks
(không quality metrics) để từ chối false readiness, lost flags/reasons hoặc hash
không khớp. Registry hash là reference tới registry snapshot, không là code hash.

`FeatureObservation` status not_computed nếu ready, unavailable nếu blocked;
**value luôn null**, computation_status not_implemented. Unavailable reasons và
flags ở `availability`; không có score/confidence hay numeric available state.
GĐ5.3 bổ sung NumericFeatureObservation 0.2.0 riêng; không
đọc numeric result như FeatureObservation schema 0.1.0. GĐ5.2 có primitive thuần;
degenerate vectors/clamp/arithmetic checks vẫn chưa chạy ở readiness và
FeatureObservation không có giá trị đo để downstream dùng nhầm.

Flags stage features/S3: unknown_feature_id, unsupported_exercise,
invalid_viewpoint, invalid_feature_input, feature_definition_blocked,
feature_joint_unusable, feature_provenance_rejected, feature_quality_blocked,
feature_metric_eligibility_missing, feature_quality_policy_mismatch;
reuse missing_joint/non_finite_landmark/viewpoint_unknown/viewpoint_mismatch/
normalization_origin_fallback/normalization_scale_fallback. Các category chỉ
input_error/measurement_error/insufficient_data; không technical_error.
Quality blockers lưu scope/action/upstream reason codes trong evidence và nguyên
report trong input snapshot. Unknown ID/exercise hoặc invalid enum dùng GymError;
known unsupported viewpoint/exercise cho feature dùng unavailable + flags.

## Pure joint-angle API — Giai đoạn 5.2

`gym_coach.geometry.Point2D = tuple[int | float, int | float] | list[int | float]`.
`joint_angle(a: Point2D, b: Point2D, c: Point2D) -> float` nhận tuple/list có đúng
hai tọa độ mỗi điểm, hữu hạn trong float range; không bool/null/string/complex.
Integer input giữ nguyên đến subtraction để không collapse điểm khác nhau do
ép float. Output là góc trong tại B, degrees `[0,180]`, full precision; không
UI rounding hoặc numeric FeatureObservation. Không có schema JSON mới cho số đo;
schemas cũ/registry version/hash không đổi, package tương thích 0.5.1.

Định nghĩa: u=A−B, v=C−B, góc giữa hai vector. Internal positive direction
rescaling bảo vệ overflow/underflow; không sửa tọa độ caller hoặc normalization
metadata. Denominator finite >0 trước division/cosine clamp `[-1,1]`;
`degrees(atan2(abs(cross)/denominator, cosine))` tránh mất góc nhỏ gần 0/180.
Chỉ vector zero bị khóa, không epsilon/near-zero threshold. A=C khác B cho 0°.

Errors dùng `GymError.flag`, severity error, stage features, source tier S3,
evidence primitive=`joint_angle` và reason chi tiết; không measured value trong
error payload. Location để trống vì primitive không có video/frame/joint identity.

| Flag code | Category | Detailed reason / context |
|---|---|---|
| invalid_geometry_input | input_error | point_type, point_dimension, coordinate_type, coordinate_not_finite, coordinate_not_representable; point A/B/C và coordinate x/y khi có |
| degenerate_geometry | measurement_error | zero_length_vector; vector BA hoặc BC |
| geometry_numerical_error | measurement_error | invalid_denominator, non_finite_ratio, invalid_angle_result; defensive arithmetic guard |

Toàn bộ input được validate trước geometry. New codes chỉ bổ sung vocabulary;
không đổi category/reason của stages trước. Flag JSON roundtrip dùng cùng
StructuredFlag schema. Invalid input không chứa NaN/Inf trong evidence.

Primitive không nhận pose/quality/registry, không phân biệt observed/imputed/
low_visibility hoặc tự kiểm tra viewpoint/scale. Caller GĐ5.3 phải áp dụng mọi
GĐ5.1 gates trước khi truyền tọa độ; toán học chạy được không nâng eligibility
hoặc validation. Tọa độ cần cùng hệ đơn vị để diễn giải góc Euclidean; API không
ngầm sửa anisotropic raw x/y. Synthetic tolerance `1e-10` degrees, rel=0, chỉ
kiểm thử arithmetic; không là accuracy bound cho MediaPipe/form hoặc threshold.

[Báo cáo và measured evidence GĐ5.2](STAGE5_2_JOINT_ANGLE_REPORT.md).

## Numeric observation và một adapter — Giai đoạn 5.3

Legacy `FeatureObservation`, `FeatureDefinition`, `FeatureAvailability` và registry
schema vẫn 0.1.0. API mới `NumericFeatureObservation.numeric_feature_schema_version`
literal **0.2.0**, hiện chỉ `squat_knee_angle_left_deg`; không implicit migration
hoặc parse numeric payload như null-only class. Adapter version squat-left-knee-v1,
primitive joint_angle/version joint-angle-2d-v1 (constant trong geometry module).
Mở rộng feature/method sau này cần schema/adapter version và tests có chủ đích.

`observe_feature(feature: str | FeatureDefinition, context: FeatureInputContext)`
nhận ID/definition canonical MVP, không có custom-registry override. Unknown ID
giữ unknown_feature_id; known feature khác trả GymError
unsupported_feature_calculation (input_error, features, error, S3); altered definition
hoặc malformed context trả invalid_feature_input. Chỉ một calculator thực sự có.

Numeric observation lưu direct feature/exercise/viewpoint, video/index/timestamp,
method/kind/unit/tier/source tiers/measurement_validation_status/assessment_status,
ordered joint_inputs (ProcessedJoint), full FeatureAvailability và quality_report/
quality_policy hashes. Registry version/hash, definition version/hash, input hash,
raw_frame_hash, normalization và QualityReport/profile/hash giữ trong availability.
Canonical metadata/hashes được đối chiếu, output identity bằng input snapshot;
imputation endpoint indices/timestamps phải bracket current frame và nằm trong
quality sequence. Status/source_status/reference không đổi; không nhân bản provenance.

| Status / computation_status | Value | Reasons/flags |
|---|---|---|
| available / computed | finite float degrees `[0,180]` | empty; readiness phải ready và vectors nonzero |
| unavailable / not_computed | null | đúng nguyên prerequisite reasons/flags |
| unavailable / failed_measurement | null | một primitive measurement error với actual video/frame/feature và A/B/C joint names |

Value strict, không bool/string/NaN/±Inf; angle/degrees bắt buộc khớp definition.
Unavailable không có numeric value; available không thiếu value. Geometry codes
chỉ degenerate_geometry/geometry_numerical_error, category measurement_error,
features/S3/error, reason detail phù hợp và location thật; không technical_error.
Primitive input errors và exceptions lập trình không bị catch-all thành unavailable.
Readback validate hashes/policies/metadata/status, không gọi primitive hoặc tính
lại quality metrics. `content_hash()` là canonical payload SHA-256; file evidence
có byte SHA-256 riêng. Hash là audit identity, không chữ ký xác thực hay accuracy.

Registry v1 `calculation_status=not_implemented` là snapshot metadata GĐ5.1 được
giữ tương thích; trạng thái execution mới ở NumericFeatureObservation. Không nâng
measurement_validation_status=unvalidated hay assessment_status khi tính thành công.
No scale gate thêm: selected angle không require scale/origin; fallback metadata
giữ null. Distance/ratio policies cũ vẫn khóa fallback và chưa có numeric adapter.

Giới hạn context: frame không có exercise/viewpoint/FPS riêng và QualityReport
không có raw-sequence hash/timeline. API kiểm tra video/index bounds, structural
schema, imputation references và mọi duplicated output identity; không thể chứng
minh timestamp/report linkage độc lập nếu caller ghép các snapshots khác nhau
cùng video. Caller phải lấy cùng GĐ4 result; factory có reference GĐ4 là bước sau.
Nested dictionaries chưa deep-immutable; adapter tách caller, strict readback/hash
audit phát hiện stale data, không khẳng định chống forgery được re-hash đồng bộ.

[Báo cáo và bằng chứng GĐ5.3](STAGE5_3_FEATURE_OBSERVATION_REPORT.md).

## Linked context factory — Giai đoạn 5.4

`build_feature_input_context(preprocessing_result: PreprocessingResult, *,
frame_index: int) -> LinkedFeatureInputContext` nhận full result, không nhận frame/
report override. Index bắt buộc, integer >=0, không bool/coercion/wrap/default;
ngoài range và empty sequence bị từ chối. Snapshot tách nested dictionaries khỏi caller.

Class mới **kế thừa FeatureInputContext**, có `context_schema_version="0.2.0"`,
`context_factory_version="preprocessing-context-v1"`. Legacy class vẫn đúng hai
fields frame/quality, không có implicit migration. Linked JSON không parse qua
legacy class (extra fields forbidden); nâng từ legacy phải cung cấp full GĐ4 result.
Numeric observation schema 0.2.0 và metadata/registry/availability/null-only 0.1.0
giữ nguyên; numeric JSON **không nhúng linkage**. Lưu context riêng cùng observation:
`context.measurement_input_hash == observation.availability.input_hash` đối chiếu
frame/quality snapshot; `context.content_hash()` định danh full linked payload.
Adapter nhận linked subtype sẽ strict revalidate toàn linkage trước readiness/primitive.
Caller cũ vẫn dùng detached context, với những giới hạn đã ghi ở GĐ5.3.

Linked fields: `frame`, `quality`, full `preprocessing_result`,
`preprocessing_result_hash`, `raw_sequence_hash`, `raw_frame_hash`,
`processed_frame_hash`, `quality_hash`, `profile_hash`, `profile_version`,
`metadata_hash`. `raw_frame` property chọn từ raw sequence theo exact frame index,
không serialize thêm duplicate raw frame. Profile identity gồm version/hash/full
profile/resolved FPS trong source snapshot, không suy đoán hoặc override.

| Identity | Canonical hashing |
|---|---|
| result, processed frame, metadata | GĐ4 content_hash: sorted compact JSON, UTF-8, **ensure_ascii=true**, allow_nan=false |
| raw frame, quality, preprocessing profile | Giữ nguyên recorded GĐ4 hashes và kiểm tra với GĐ4 content_hash |
| raw sequence | Giữ GĐ4 SHA-256 concatenated ordered raw-frame ASCII hashes |
| full linked context / measurement_input_hash | Features content_hash: **ensure_ascii=false**, sorted compact UTF-8, allow_nan=false |

Không đồng nhất `context.quality_hash` GĐ4 với `NumericFeatureObservation.quality_report_hash`
features khi payload chứa Unicode: hai thuật toán canonical hiện hành có khác
ensure_ascii, đều được kiểm tra theo đúng convention. File byte hashes riêng.
Metadata/result hash chứa source_path như stored trong GĐ4; không relocation invariant,
không thêm path máy cá nhân hard-code trong factory.

Factory intake dùng `model_dump(mode="python", warnings="error")` rồi JSON encoding
allow_nan=false (Path → string là conversion tường minh duy nhất), sau đó
`PreprocessingResult.model_validate_json(..., strict=True)`. Python mode giữ bool
trong float field để strict validator từ chối; JSON mode có thể đổi bool→1.0,
model_dump_json mặc định có thể đổi NaN→null. Adapter linked intake dùng cùng rule.
Đọc linked context: `LinkedFeatureInputContext.model_validate_json(json_text, strict=True)`.
Serialization/readback không tự tính góc hay chạy preprocessing.

Revalidation đối chiếu full raw/processed counts, metadata/CFR timeline, exact
frame identities/raw hashes, sequence hash, report/core counts/spans/quality hash,
profile/resolved profile, imputation references và smoothing-run coverage.
Linked validator so sánh selected processed frame và QualityReport bằng snapshot
nguồn, kiểm tra mọi added audit field. Không tolerance cho equality raw/processed/
context timestamp; CFR consistency GĐ4 vẫn tolerance 1e-9 s và giữ timestamp gốc.
GĐ4 validate_quality được sửa tối thiểu sang shared joint-count/span statistics;
không gọi QualityGate/eligibility. Mọi custom decisions và flags giữ nguyên.

| Flag code | Category / stage / tier | Reason |
|---|---|---|
| invalid_feature_context_index | input_error / features / S3 | invalid_frame_index, frame_index_out_of_range |
| invalid_feature_input | input_error / features / S3 | preprocessing_result_type, preprocessing_serialization_failed, preprocessing_validation_failed, context_serialization_failed, context_linkage_validation_failed; type/nonfinite/strict schema/profile overflow |
| feature_context_mismatch | input_error / features / S3 | preprocessing_validation_failed hoặc context_linkage_validation_failed; value/audit/identity consistency contradiction |

Severity error; evidence component=feature_input_context_factory và detailed reason;
validation message/cause giữ context cụ thể. Valid requested index có inclusive
frame location; không tạo video ID hoặc measured value giả. Không catch-all
exceptions lập trình, không biến input error thành technical form error/unavailable.
Known policy failures vẫn do adapter trả unavailable/value=null theo GĐ5.1/5.3.

Giới hạn: full source snapshot/revalidation O(N) mỗi context, chưa benchmark.
Frozen model chưa bất biến sâu; adapter/readback kiểm tra post-factory mutation.
Full result equality/hash xác định **nội dung result được cung cấp**, không run UUID,
chữ ký hoặc producer authentication. QualityReport chưa lưu raw-sequence hash riêng:
core-count equality không chứng minh report thực sự được tạo từ exact coordinates
hay từ run khác có cùng thống kê. Không rerun SG/normalization để chứng minh tọa độ
đã được tính đúng; provenance/audit consistency không là accuracy/form validation.

[Báo cáo/bằng chứng GĐ5.4](STAGE5_4_CONTEXT_FACTORY_REPORT.md).

## Knee numeric schema và dispatch — Giai đoạn 5.5

Legacy `NumericFeatureObservation` **0.2.0** vẫn chỉ nhận left-knee ID, class/file
và field contract không đổi. `KneeAngleFeatureObservation` là subclass mới có
`numeric_feature_schema_version="0.3.0"`, nhận đúng hai canonical knee IDs:

| Canonical ID | Ordered inputs A/B/C | Adapter version | Dispatch output |
|---|---|---|---|
| squat_knee_angle_left_deg | left_hip / left_knee / left_ankle | squat-left-knee-v1 | NumericFeatureObservation 0.2.0 |
| squat_knee_angle_right_deg | right_hip / right_knee / right_ankle | squat-right-knee-v1 | KneeAngleFeatureObservation 0.3.0 |

Class mới dùng fields/helpers legacy, thay validator canonical ID/version;
adapter_version bắt buộc và phải khớp feature ID. Một observation chỉ chứa một
bên. Quantity angle/degrees, primitive joint-angle-2d-v1; method/source tier/
measurement tier/validation lấy nguyên từ canonical registry. Không fusion/score.

Đọc JSON qua đúng class `model_validate_json(text, strict=True)`; hai literal
versions không interchange. `KneeAngleFeatureObservation.from_legacy(left)`
nhận đúng legacy class, revalidate strict rồi chỉ đổi numeric schema version
thành 0.3.0. Đây là migration opt-in, không tính lại góc, không sửa snapshots/
flags/hash hoặc nâng validation. Dispatch left vẫn tạo 0.2.0; không auto-upgrade.
Old GĐ5.3 schema snapshot giữ nguyên; shared FlagCode đã thêm hai codes ở GĐ5.4,
GĐ5.5 không thêm/sửa flags. [New schema snapshot](artifacts/stage5_5_knee_observation.schema.json).

New validator dùng Python-mode serialization/JSON allow_nan=false + strict JSON
validation cho availability và actual joint_inputs. Kiểm tra canonical definition/
registry hashes, input/quality hashes và duplicated metadata, ordered joints,
imputation references, value float hữu hạn `[0,180]`, unit/status/flags. Vector
BA hoặc BC bằng zero không thể forged available. Available không reason/flag;
prerequisite unavailable giữ nguyên reasons/flags, value null/not_computed.

Measurement unavailable null/failed_measurement phải có đúng một code/flag
degenerate_geometry hoặc geometry_numerical_error, measurement_error/features/
S3/error; location đúng video/frame/canonical ID; evidence primitive=joint_angle,
input_joint_a/b/c theo bên đã chọn. Degenerate reason/vector phải khớp actual
zero vector; numerical failure chỉ nhận known primitive reasons. Không evidence
giá trị đo giả. Invalid input/context errors tiếp tục GymError, không unavailable.

Readback không gọi primitive hoặc QualityGate; kiểm tra consistency không xác
thực giá trị angle bằng phép tính lại. Hash là content identity, không chữ ký,
producer proof hoặc accuracy. Linked context lưu riêng như GĐ5.4, numeric JSON
chỉ giữ measurement input snapshot. Registry/profile/context/upstream schemas
và policies giữ nguyên; angle không cần scale, ratio fallback vẫn bị chặn.

[Báo cáo/bằng chứng GĐ5.5](STAGE5_5_RIGHT_KNEE_FEATURE_REPORT.md).

## Eligibility/data evidence — Giai đoạn 5.6

Không version/schema runtime/numeric/flag mới; package 0.5.4, quality 0.1.0,
quality-s3-v1, registry v1, linked context 0.2.0, left numeric 0.2.0/right0.3.0
giữ nguyên. Test helper chỉ tạo deterministic synthetic PoseFrame, gọi QualityGate
thật với hai stable knee keys và đúng canonical joint/viewpoint mappings; không
manually construct MetricEligibility hoặc chỉnh report sau evaluation.

Global movement allow không suy ra feature allow. Nếu caller khai báo sai joint
set nhưng sạch dữ liệu, QualityGate có thể trả allow cho set đó; feature readiness
vẫn từ chối feature_quality_policy_mismatch. Unknown/front/oblique metric skip;
partial raw coverage degrade vẫn bị khóa ngay cả khi processed joint imputed usable.
Không recompute/promote eligibility trong factory/adapter/readback.

Inventory JSON/CSV là **audit artifact**, không numeric schema hoặc runtime
permission cho phép đo. Ghi từng relative path/video ID/byte SHA, source metadata/
camera_view, evidence type, documented/visual-review status, recorded quality
reports/metric decisions, eligible/ineligible/blocked và reasons. Copies được
group bằng byte hash, không suy participant/viewpoint độc lập. Existing reports
giữ original decisions và hashes; absent report/key là unknown/missing, không allow.

Documented viewpoint cần per-video metadata/protocol/annotation có xác nhận rõ
góc quay và nguồn/annotator/date/frame interval; action label, pose/viewpoint enum,
heuristic camera field hoặc visual impression không thay thế nguồn này.
Visual inspection đơn thuần phải visually_assessed_unverified và không documented;
audit hiện tại không visual inspection (not_performed). Data metadata chưa review
được liệt kê NOT_REVIEWED và giữ blocked; script không auto-approve annotation mới.

Synthetic artifacts ghi evidence_type=synthetic_contract_test, real_video=false,
input ID stage5_6_synthetic_contract.avi; available measurements vẫn unvalidated.
Real inventory kết luận BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT. Hash chỉ content
identity; QualityReport chưa có independent raw/video origin proof. Không inference
toàn dataset hoặc numeric real-video, không schema/threshold/scoring change.

[Báo cáo/bằng chứng GĐ5.6](STAGE5_6_DEVELOPMENT_ELIGIBILITY_REPORT.md).

## Geometry registry v2 và numeric schema 0.4 — implementation cuối GĐ5

Các mục GĐ5.1–5.6 phía trên mô tả schema/API lịch sử còn được hỗ trợ. Không
diễn giải `calculation_status=not_implemented` của registry v1 thành trạng thái
calculator mới. `DEFAULT_REGISTRY` và `observe_feature` không đổi.

| Identity mới | Giá trị | Boundary |
|---|---|---|
| registry_schema_version | `0.2.0` | GeometryFeatureRegistry, không đọc bằng registry0.1 |
| registry_version | `gym-feature-registry-v2` | 12 canonical entries, 10 implemented angles/2 blocked ratios |
| registry_hash | `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72` | source_registry_version/hash bind v1; canonical derivation được kiểm tra |
| feature_schema_version / definition_version | `0.2.0` | GeometryFeatureDefinition/Availability; giữ measurement metadata từ v1 |
| angle quality policy_version | `framewise-metric-only-v1` | require_sequence_allow=false, exact feature metric allow vẫn bắt buộc |
| ratio quality policy_version | `sequence-and-metric-v1` | blocked/not_implemented, require_sequence_allow=true |
| numeric_feature_schema_version | `0.4.0` | GeometryFeatureObservation và observe_geometry_feature |
| adapter_version | `geometry-features-v1` | Một frame/một canonical feature; chưa public FeatureSequence |
| joint primitive_version | `joint-angle-2d-v1` | primitive joint_angle hiện hành không đổi |
| axis primitive_version | `midpoint-axis-angle-vertical-2d-v1` | hip→shoulder midpoint vector so với image_up=(0,-1) |

Registry v2 từ chối thiếu/sửa formula, ordered joints, policies hoặc source v1
identity. Validation calculation_status tách measurement_status: implemented
không nâng unvalidated thành validated. Measurement tier/source tiers giữ v1;
mọi assessment_status vẫn blocked_pending_expert_validation.

Policy framewise chỉ bỏ flag global `feature_quality_blocked` có
`evidence.scope=movement_analysis`. Mọi flag metric-specific, wrong/missing key,
wrong required joints, viewpoint, finite/usability/provenance và fallback gates
vẫn áp dụng. Canonical v2 definition bắt buộc; không cho caller tự chọn policy.
Global QualityReport/sequence_status không sửa. Numeric output luôn
measurement_scope=framewise_projected_2d, temporal_analysis_eligible=false,
assessment blocked; không dùng available để mở downstream temporal/assessment.

Joint angle: góc trong ABC giữa A−B/C−B. Axis: directed angle của
`(LS+RS-LH-RH)/2` với `(0,-1)`. Cả hai đơn vị degrees, finite `[0,180]`.
Midpoint arithmetic dùng Fraction của finite input để tránh overflow/underflow;
positive rescaling nội bộ chỉ phục vụ arithmetic. Không rotate/flip/normalize
input; không thêm smoothing ngoài Stage4. Trục inverted có thể 180°, body-line
chỉ shoulder–hip proxy. Zero vector hip_to_shoulder trả degenerate_geometry,
category measurement_error, stage features, source S3; invalid input dùng
invalid_geometry_input/input_error. Numerical reasons dùng taxonomy cũ.

Numeric0.4 lưu canonical identity/method/unit/tier/validation, ordered
ProcessedJoint inputs (gồm source/status/usable/interpolation), frame/index/time,
full detached frame+QualityReport availability snapshot và hashes. Ready không
đồng nghĩa computed. Available cần finite float value/no reasons/no zero vector;
unavailable luôn value=null và exact policy flags hoặc một measurement failure
flag. Flag failure có primitive và input_joint_0..N theo ordered joints, actual
video/frame/feature location, reason/vector, không chứa giá trị đo giả.
Readback revalidates policies/metadata/hashes/provenance, không gọi geometry
hoặc recompute QualityGate. Hash không authenticate producer hay chứng minh
numeric accuracy; không tính lại góc để giả biến readback thành validation.

API intake strict Python-mode JSON (allow_nan=false, serializer warnings=error)
từ chối model_copy/construct sai kiểu, NaN/Inf/null usable tọa độ. Linked context
vẫn schema0.2 factory hiện hành; source linkage/hashes được kiểm tra trước khi
đo. Audit persist full preprocessing + frame_links mapping + first/last full
linked contexts; observation.input_hash phải khớp measurement_input_hash.
Imputed vẫn imputed và giữ endpoints; raw-gap metric degrade không được promote
chỉ vì preprocessing đã điền tọa độ. Angles không cần scale; ratio candidates
không được đo, fallback scale/origin có reason, không thay null bằng1.

Legacy knee numeric0.2/0.3 giữ nguyên serializer/classes và sequence+metric gate.
Chọn reader theo explicit version/class; old JSON không tự migrate sang0.4.
Để dùng policy mới phải tạo observation mới qua API mới; không đổi label version
của payload cũ. Metadata readiness/null-only0.1 vẫn nguyên contract.

Development audit selection bound by byte SHA before inference; script xác minh
path/ID/hash, visual record/hash/sample links và assumption phù hợp visual
category. Sai linkage trả feature_context_mismatch/input_error/features/S3 với
component=development_angle_audit, reason=selection_linkage_invalid. Audit writes
exclusive paths. Unknown original và visual-hypothesis side/front branches fresh
và tách biệt; original viewpoint giữ unknown. Source visual audit không annotation:
visually_assessed_unverified/documented=false/official_eligibility=false. Hash
không chứng minh viewpoint hoặc quyền annotator. Official gate5.7 vẫn BLOCKED.

[Report, contracts và measured evidence cuối GĐ5](STAGE5_FINAL_FEATURE_EXTRACTION_REPORT.md).
