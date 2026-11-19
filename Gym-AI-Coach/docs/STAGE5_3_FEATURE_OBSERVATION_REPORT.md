# Giai đoạn 5.3 — Numeric observation và adapter cho một feature

Ngày đo: 2026-10-08, Asia/Saigon. **GĐ5.3 COMPLETE theo phạm vi một feature**;
toàn **GĐ5 IN_PROGRESS**. Test results dưới đây đã chạy; không dùng archive,
không coi historical report là bằng chứng execution mới.

## Feature/API/phạm vi thực tế

Duy nhất **squat_knee_angle_left_deg** được tính: ordered inputs
left_hip → left_knee → left_ankle, góc trong tại knee, degrees. Chọn vì registry
đã khai báo đủ ba joints/method/policy, không cần axis construction hoặc bilateral
fusion. Không tạo ID mới, không implement feature thứ hai hoặc feature HSPU.

```python
from gym_coach.features import FeatureInputContext, NumericFeatureObservation, observe_feature

context = FeatureInputContext(frame=result.frames[0], quality=result.quality)
observation = observe_feature("squat_knee_angle_left_deg", context)
# Cũng nhận DEFAULT_REGISTRY.get("squat_knee_angle_left_deg") thay cho ID.
encoded = observation.model_dump_json()
restored = NumericFeatureObservation.model_validate_json(encoded)
```

`observe_feature(feature: str | FeatureDefinition, context: FeatureInputContext)`
trả một NumericFeatureObservation. Không batch API, FeatureSequence, extractor
toàn video, bilateral fusion, feature khác, rep/phase/form/threshold/score/feedback
hoặc training. Measurement có số chỉ là mô tả hình học **unvalidated**, không
kết luận vận động viên đúng/sai hoặc accuracy claim.

Files mới: `features/observation.py`, `features/adapter.py`,
`tests/test_feature_observation.py`, `tests/test_feature_observation_integration.py`,
report này, verification snapshot và numeric JSON Schema snapshot.
Cập nhật public features exports, additive FlagCode, package version và primitive
version constant; geometry arithmetic functions không đổi. README/plan/state/tasks/
schema được cập nhật. Không sửa feature models/registry/policy cũ hoặc GĐ0–4 code/tests/config.

## Schema/version/audit identity

| Identity | Giá trị |
|---|---|
| Package | ai-coach-gym **0.5.2** |
| Numeric observation schema | **0.2.0**, class riêng NumericFeatureObservation |
| Legacy metadata/readiness/null-only schema | **0.1.0**, giữ nguyên |
| Adapter | **squat-left-knee-v1** |
| Method / primitive | joint_angle_abc / joint_angle |
| Primitive version | **joint-angle-2d-v1** |
| Registry | **gym-feature-registry-v1**, schema 0.1.0 |
| Registry hash | bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a |
| Selected definition version | 0.1.0 |
| Selected definition hash | 875e6351320ea402ffcb377a00b95c0429d9008b9528f83b86b236d3f30bcce6 |

Numeric schema được thu hẹp có chủ đích cho một feature/method. Shared audit
field derivation dựa trên definition; mở rộng cần schema/adapter version và tests
có chủ đích. Không silent migration hoặc parse numeric payload qua FeatureObservation
cũ. Legacy value=null vẫn được test. Registry v1 calculation_status=not_implemented
là metadata snapshot GĐ5.1 được giữ tương thích; execution state mới nằm trong
NumericFeatureObservation, không chỉnh registry/hash để tuyên bố tất cả đã tính.

Observation lưu direct feature_id/exercise/viewpoint, video_id/frame_index/
timestamp_seconds, method/quantity/unit, measurement/source tiers, measurement
validation status và assessment blocked status. Ordered joint_inputs là toàn
ProcessedJoint: name, final x/y, status/source_status, usable, interpolation
endpoints/timestamps và processing metadata, giữ nguyên quan sát nguồn.

Full FeatureAvailability gồm canonical definition, registry version/hash,
definition hash, detached context/input hash, raw_frame_hash, normalization,
QualityReport/profile/hash và prerequisite flags. Numeric result thêm full
quality_report_hash/quality_policy_hash, đối chiếu với snapshot. Metadata và
duplicate identity không được caller override; imputation refs phải bracket
frame index/timestamp và nằm trong sequence. Model strict validation kiểm tra
hashes/metadata/status/flags khi đọc, không gọi primitive hay recompute quality.

Canonical `content_hash()` = SHA-256 JSON sorted keys/compact UTF-8,
ensure_ascii=false/allow_nan=false. Output files có byte SHA-256 trong evidence.
Hash là audit identity, không chữ ký hoặc bằng chứng scientific accuracy.
[Numeric JSON Schema snapshot](artifacts/stage5_3_numeric_observation.schema.json)
mô tả structure; runtime validators còn cần để kiểm tra policy/hash relationships.
Byte SHA-256: `fe01ccfb9e29d1063a730295eb904867cbebec26491769b10aad5c8616e5733c`.

## Policy/readiness và errors

1. ID phải có trong registry; supplied definition phải đúng canonical metadata.
   API không có custom-registry override. Unknown ID → unknown_feature_id;
   11 ID registry khác → unsupported_feature_calculation; altered definition/
   malformed context → invalid_feature_input. Đây là structured GymError input
   failures, không trả fake unavailable value hoặc chạy calculator khác.
2. `check_feature_availability` cũ revalidate/snapshot context và áp dụng đúng
   provenance/usability/finite xy, exercise/viewpoint, sequence **allow**, đúng
   feature metric key/required joints **allow**. Không promote degrade/skip,
   infer absent key hoặc recompute metrics. Unsupported/unknown viewpoint và
   wrong MVP exercise giữ unavailable theo policy cũ.
3. Kiểm tra imputation references hiện có bracket frame. Chỉ sau mọi prerequisite
   đạt mới lấy đúng ordered processed x/y và gọi geometry.joint_angle **một lần**.
   Không biến observed/imputed/source missing thành cùng một provenance.
4. Primitive measurement GymError degenerate_geometry/geometry_numerical_error
   được chuyển thành unavailable, null value, flags features/S3/measurement_error,
   actual video/frame/feature location và actual A/B/C joint names. Primitive input
   errors và programming exceptions được truyền rõ, không catch-all.

| Result | Value | Evidence |
|---|---|---|
| available / computed | finite degrees `[0,180]` | readiness ready, no unavailable flags; unvalidated |
| unavailable / not_computed | null | policy reasons/flags đúng nguyên bản |
| unavailable / failed_measurement | null | một structured primitive measurement reason/flag |

Schema từ chối available thiếu value, unavailable chứa value (kể cả 0), NaN/±Inf,
wrong kind/unit, stale hashes, changed identity/validation/tier, lost flags hoặc
forged available trên zero vectors. Degenerate flags phải khớp zero vector thật;
measurement error evidence không được chứa fake measured value.

Selected angle không require scale/origin; **scale fallback không thêm gate**
và scale metadata vẫn null. Các distance/ratio policies GĐ5.1 vẫn forbid fallback,
nhưng không có numeric adapter cho chúng. Không normalize/smooth/rotate data thêm.

Actual preprocessing imputation làm raw-quality metric degrade vẫn unavailable;
test giữ nguyên trạng thái đó. Để kiểm thử successful imputed schema path, có
một **fresh synthetic caller-supplied QualityReport fixture** với contract allow
và profile identity synthetic-imputed-allow-contract-v1. Fixture này không phải
QualityGate output, không thay report gốc hoặc bất kỳ report thật nào, không
chứng minh pipeline hiện tại cho phép đo trên gap đã impute. Status/source_status/
endpoint references giữ nguyên cả trong available và unavailable observations.

## Môi trường và test evidence thực tế

CPython **3.11.9**, Windows x64 (10.0.26200), Pydantic **2.13.5**, pytest **9.1.1**,
setuptools **84.0.0**. Không dependency/model/framework mới; pyproject không đổi.
Full-suite environment vẫn MediaPipe **0.10.14**, NumPy **2.4.6**, SciPy **1.17.1**,
OpenCV contrib **5.0.0.93**, protobuf **4.25.9**; legacy opencv-python cũng 5.0.0.93.
Numeric public import/JSON Schema được chạy trong fresh isolated subprocess chặn
NumPy/SciPy/OpenCV/MediaPipe/Torch. Chưa clean-environment/transitive lock test.

PowerShell tại project root:

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q tests/test_feature_observation.py tests/test_feature_observation_integration.py --junitxml=outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/new-tests.xml
& .venv/Scripts/python.exe -m pytest -q tests/test_feature_observation.py tests/test_feature_observation_integration.py --junitxml=outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/new-tests-final.xml
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/full-suite.xml
```

| Run | Pass | Fail / errors / skipped | Warnings | Terminal time |
|---|---:|---|---:|---:|
| Initial new tests | 120 | 0 / 0 / 0 | 1 | 9.75 s |
| Final new tests (118 unit + 2 sample integrations) | **120** | **0 / 0 / 0** | **0** | **9.20 s** |
| Full suite GĐ0–5.3 | **448** | **0 / 0 / 0** | **4** | **43.94 s** |

Commands exit 0; JUnit counts/hashes được parse đối chiếu. Initial Pydantic
serializer warning đến từ intentionally malformed unit enum; intake definition
được sửa dùng serialization warnings=error, wrap thành structured invalid input,
không suppress warning. 4 full-suite warnings là UserWarning protobuf
SymbolDatabase.GetPrototype deprecated tại google/protobuf/symbol_database.py:55,
từ bốn fresh MediaPipe integrations GĐ2/GĐ4; không phải feature accuracy evidence.
Pip install offline thành công ai-coach-gym 0.5.2; pip check No broken requirements found.

Unit tests cover known 90/180/0/45° và đối chiếu primitive, exact identity/ordered
inputs, canonical ID/definition, prerequisite no-call spy, quality sequence/metric
degrade/skip/absent/wrong keys/joints, wrong exercise/viewpoint, provenance/usability,
fallback scale, imputation/source refs, degeneracy/numerical errors, input/programming
error propagation, JSON invariants/hashes/flags/enums/metadata/timestamps, old-schema
boundary, immutability và heavy-import blocking. Tolerance **absolute 1e-10 degrees,
relative=0**, chỉ synthetic arithmetic; không pose accuracy bound hoặc form rule.

[Final new JUnit](../outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/new-tests-final.xml),
[full JUnit](../outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/full-suite.xml),
[environment/commands/source/input/output hashes](artifacts/stage5_3_verification.json).
Reviewable synthetic output: [available 90°](../outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/synthetic-available.observation.json),
[degenerate unavailable/null](../outputs/stage5_3/verification-c46ebb73601c43e58941bbae1d36adeb/synthetic-degenerate.observation.json).
Đây là hai fresh synthetic single-frame examples, JSON readback đã đối chiếu;
byte/content hashes trong synthetic-example-identities.json cùng directory.
Source tree SHA-256 (src/gym_coach/configs/tests/pyproject, no docs/data/outputs):
`3daed9fb7b069bed11b89ba82cfe55f3589cb3e2660473d5b8ffaa20516d1d3b`.
No Git repository/commit/diff. Before manifest 197 files, 188 unchanged, chỉ 9
existing code/docs files sửa đúng phạm vi; all old outputs giữ bytes/hash.
Geometry function AST hash trước/sau khớp; chỉ thêm version constant/export.
Không sửa dataset/archive, upstream algorithms/tests/configs hoặc legacy feature contracts/policies.

## Hai real-input sample integrations

Chạy adapter mới trên strict saved GĐ4 inputs theo measured snapshot hiện có;
**không fresh feature extraction toàn video**. Full suite vẫn fresh inference
GĐ2/GĐ4 qua tests cũ. Dùng **cùng squat left-knee ID** cho cả hai input sets;
HSPU là negative exercise-policy case, không triển khai HSPU numeric feature.

| Input / relative video path | N input frames | Sample frame IDs | Available / unavailable | Primitive calls |
|---|---:|---|---|---:|
| Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi | 267 | 0, 133, 266 | 0 / 3 | 0 |
| Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi | 78 | 0, 39, 77 | 0 / 3 | 0 |

Video IDs chính là filenames trên. Squat timestamps `0`, `5.32`, `10.64` s;
HSPU `0`, `1.56`, `3.08` s. Cả hai report action skip/viewpoint unknown, thiếu
selected feature metric key → viewpoint_unknown, feature_quality_blocked,
feature_metric_eligibility_missing. HSPU thêm unsupported_exercise. Không sửa
viewpoint hoặc report để ép có số. Sáu observations giữ value=null, identities,
provenance/quality snapshots và JSON readback; source result/file bytes không đổi.

Final output directory `outputs/stage5_3/integration-563152cdd3aa4a3394bebd0619d16858/`:
[Squat summary](../outputs/stage5_3/integration-563152cdd3aa4a3394bebd0619d16858/BodyWeightSquats.summary.json),
[HSPU summary](../outputs/stage5_3/integration-563152cdd3aa4a3394bebd0619d16858/HandstandPushups.summary.json).
Mỗi sampled frame có observation JSON riêng; report/evidence lưu exact paths/hashes.
Mỗi test run dùng directory/artifact mới, không overwrite initial/final/old files.
Coverage/measurement accuracy cho full feature sequence **chưa đo**; 0/3 available
chỉ là sample availability trong phạm vi này, không form score hoặc full-video coverage.

## Đối chiếu DoD và giới hạn

| Tiêu chí GĐ5.3 | Bằng chứng | Kết quả |
|---|---|---|
| Đúng một canonical registry feature | ID/definition equality + 11 unsupported IDs tests | PASS |
| Numeric schema có version, legacy giữ nguyên | 0.2.0 new class + old-schema reject/roundtrip tests | PASS |
| Metadata/identity/provenance/quality audit | Snapshot/hash/exact identity/ordered inputs/refs validators và tests | PASS |
| Policy trước primitive, không promote/recompute | no-call tests cho exercise/viewpoint/quality/joints; real samples | PASS |
| Fallback theo definition, không gate thêm | available angle fallback test; existing distance/ratio tests vẫn pass | PASS |
| Measurement error unavailable/null có actual identity | zero BA/BC/numerical injection + strict flags tests | PASS |
| Serialization/status/unit/kind/finite/hash strict | available/unavailable/corruption/forgery tests | PASS |
| Inputs/upstream/artifacts giữ nguyên và import nhẹ | unit/sample immutability, hash audit, isolated blocked import | PASS |
| New + full tests đạt, docs/evidence đúng scope | 120/448 passed + current docs/report/evidence | PASS |
| Không đánh dấu toàn GĐ5 COMPLETE | IN_PROGRESS; feature/integration gaps bên dưới | PASS |

- Context hiện không có raw PoseFrame/timeline/FPS riêng; QualityReport không có
  raw-sequence hash. API kiểm tra video ID/index bounds, structure và imputation
  refs; duplicate output timestamp/identity phải đúng snapshot. Nó chưa thể
  chứng minh caller ghép đúng timestamp/report/coordinates giữa các snapshots
  cùng video. Không đoán FPS hoặc tự sửa dữ liệu để che giới hạn này.
- Nested dictionaries chưa deep-immutable. Snapshots tách caller, strict readback/
  hashes phát hiện stale metadata; synchronized forgery/re-hashing không bị xem
  như chữ ký xác thực. Chưa benchmark memory/serialization video dài.
- One-frame geometry là projected 2D; perspective/depth/occlusion/inverted pose,
  viewpoint protocol và measurement accuracy chưa validation. UCF101 chỉ nhãn
  action, không form/biomechanical ground truth. Available/unvalidated không có
  kết luận đúng/sai hoặc permission assessment.
- Chưa 9 angle candidates còn lại, ratios blocked, sequence/extractor hoặc
  numeric development integration trên input đủ prerequisites. Vì vậy acceptance
  criteria toàn GĐ5 chưa đạt; GĐ6–7 chưa bắt đầu, assessment vẫn BLOCKED.

Bước nhỏ nhất tiếp theo: factory tạo FeatureInputContext từ **validated
PreprocessingResult + frame index**, đối chiếu đúng processed/raw identity và
quality snapshot để củng cố linkage. Sau gate đó mới mở rộng adapters/feature
schema và chọn development input có viewpoint/quality đủ điều kiện; không sửa
report thật để ép numeric output. Chưa thực hiện các bước sau trong GĐ5.3.
