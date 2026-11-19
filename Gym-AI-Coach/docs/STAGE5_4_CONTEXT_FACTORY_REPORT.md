# Giai đoạn 5.4 — Factory liên kết feature context với preprocessing

Ngày: **2026-10-09, Asia/Saigon**. **GĐ5.4 COMPLETE**, toàn **GĐ5 IN_PROGRESS**.
Đã đọc đủ 11 tài liệu được yêu cầu và kiểm tra current contracts/tests GĐ0–5.3.
Workspace không có Git repository; không tạo commit/diff giả, không dùng archive.
Các số test/sample dưới đây đã chạy trong lượt này.

## API và phạm vi

```python
from gym_coach.features import (
    LinkedFeatureInputContext, build_feature_input_context, observe_feature,
)

context = build_feature_input_context(result, frame_index=12)
observation = observe_feature("squat_knee_angle_left_deg", context)
restored = LinkedFeatureInputContext.model_validate_json(
    context.model_dump_json(), strict=True,
)
assert restored == context
assert observation.availability.input_hash == restored.measurement_input_hash
```

Factory chỉ nhận **full PreprocessingResult + keyword frame_index bắt buộc**.
Không có frame/report/profile override, default frame, wrap index hoặc suy đoán
timestamp/viewpoint. Index phải int >=0, không bool/float/string; phải nằm trong
sequence. Empty result hợp lệ để audit GĐ4 nhưng không có frame để chọn.

Output `LinkedFeatureInputContext` kế thừa `FeatureInputContext`. Snapshot tách
caller nested dictionaries, giữ processed frame/QualityReport/full source result;
không mutate input hoặc biến đổi coordinates/provenance/normalization/flags.
Adapter chỉ bổ sung strict intake cho linked subtype trước readiness/primitive.
Caller context cũ và toàn bộ numeric outcome/schema GĐ5.3 tiếp tục hoạt động.

Không thêm feature ID/calculator, extractor toàn video, FeatureSequence, bilateral
fusion, segmentation, repetition/phase, form rule/threshold/score/feedback hoặc
training. Duy nhất left squat knee adapter hiện hữu được gọi trong tests.

## Linkage, validation và thay đổi audit tối thiểu

Intake dùng Python-mode `model_dump(warnings="error")`, JSON allow_nan=false và
`PreprocessingResult.model_validate_json(..., strict=True)`. Chỉ Path được chuyển
thành string tường minh. Python mode tránh JSON-mode bool→1.0; encoding strict
tránh model_dump_json mặc định biến NaN→null ở nullable fields.

Factory và linked JSON validation kiểm tra:

- Full raw/processed/metadata/quality lengths và complete contiguous CFR input;
  metadata video/FPS, raw exercise/viewpoint/backend/config consistency.
- Exact raw/processed video ID/index/timestamp; selected context frame bằng đúng
  source processed frame tại index. CFR/FPS check vẫn tolerance GĐ4 1e-9 s;
  equality giữa stored frames/context là chính xác, không sửa timestamp.
- Raw frame/sequence hashes, quality snapshot/hash và raw core counts/spans.
- Preprocessing profile/hash/resolved FPS/settings, imputation original observed
  endpoints/bracketing/gap policy và smoothing-run audit coverage/unit statuses.
- Full result content identity, processed frame/metadata hashes, profile version,
  và mọi field audit lặp lại trong linked context đều bằng nguồn đã kiểm tra.

GĐ4 `validate_quality` trước đây gọi `evaluate_quality` để tạo temporary report,
chỉ so sánh identity/counts/spans nhưng vẫn tính eligibility bên trong. Đã sửa
**chỉ helper audit và hai imports**: dùng shared `_joint_quality` thống kê từng
joint + pose counts/identity/sentinel để đối chiếu core fields. Caller sequence
đã qua `validate_sequence`. Không tạo report mới hoặc chạy eligibility. Custom
decisions, flags và hashes giữ nguyên. Tests chặn QualityGate/evaluate/_eligibility
khi factory, adapter và linked readback chạy; clean/empty/missing/low/OOF core
statistics vẫn khớp QualityGate thật đã tạo trước audit.

Preprocessing arithmetic, interpolation, normalization, SG, profile/settings và
GĐ4 schemas không đổi. Registry/models/policy/geometry source và tests/configs
GĐ0–5.3 không đổi. Adapter chỉ thêm import và linked-context revalidation branch;
không đổi legacy calculation/policy/error paths.

## Versions, hashes và compatibility

| Identity | Giá trị / quy ước |
|---|---|
| Package | ai-coach-gym **0.5.3** |
| Linked context schema | **0.2.0**, class riêng, không thay legacy context |
| Factory | **preprocessing-context-v1** |
| Numeric observation / adapter | 0.2.0 / squat-left-knee-v1, shape/version giữ nguyên |
| Registry/legacy metadata/null-only | gym-feature-registry-v1 / schema 0.1.0 |
| Registry hash | bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a |
| Preprocessing schema/profile | 0.1.0 / preprocessing-s3-v1, không đổi |
| Shared vocabulary additions | invalid_feature_context_index, feature_context_mismatch |

Linked context giữ `preprocessing_result_hash`, `raw_sequence_hash`,
`raw_frame_hash`, `processed_frame_hash`, `quality_hash`, `profile_hash`,
`profile_version`, `metadata_hash`; source chứa đầy đủ metadata/profile/quality/
raw/processed/runs. `raw_frame` property chọn original raw tại exact index.

Result/processed/metadata hashes dùng canonical **GĐ4 ensure_ascii=true**; raw/
quality/profile/sequence giữ đúng hashes GĐ4 hiện hành. Full context và
measurement_input_hash dùng features canonical **ensure_ascii=false**. Cả hai
sorted keys/compact JSON/UTF-8/SHA-256/allow_nan=false. Không đồng nhất quality
hash hai conventions khi payload chứa Unicode; file bytes có hashes riêng.

[Linked JSON Schema snapshot](artifacts/stage5_4_linked_context.schema.json),
byte SHA-256 `ab20931ed14f1e39b90d2e0ce81ff1db22d8a34f02e23c5002b0925cec719f93`.
Runtime validators kiểm tra quan hệ hash/source ngoài khả năng JSON Schema.
Generated numeric JSON Schema chỉ khác snapshot GĐ5.3 ở hai shared FlagCode enum
additions; structure/version không đổi, old schema artifact không ghi đè.

Legacy JSON chỉ có frame/quality. Linked JSON không parse như legacy class;
nâng legacy cần full PreprocessingResult. **Numeric output không nhúng full
linkage** để giữ schema cũ: lưu linked context JSON riêng cùng observation và
đối chiếu measurement_input_hash với availability.input_hash. Sample artifacts
đã lưu cả hai; không implicit downgrade/migration hoặc claim numeric JSON tự đủ
để xác minh toàn source linkage.

Hash là content identity, không chữ ký bảo mật/producer authentication/run UUID.
Runs có nội dung giống nhau có cùng hash. Metadata/result hash chứa stored path,
không relocation invariant; implementation không hard-code path máy cá nhân.

## Structured errors

Mọi factory/linked-adapter input error: GymError.flag, category input_error,
stage features, severity error, source tier S3; không measured value/form error.

| Code | Detailed reason |
|---|---|
| invalid_feature_context_index | invalid_frame_index, frame_index_out_of_range |
| invalid_feature_input | preprocessing_result_type, preprocessing_serialization_failed, preprocessing_validation_failed, context_serialization_failed, context_linkage_validation_failed |
| feature_context_mismatch | preprocessing_validation_failed/context_linkage_validation_failed khi dữ liệu schema-shaped mâu thuẫn identity/hash/audit |

Type/nonfinite/serialization/strict schema/profile-overflow failures tách khỏi
audit consistency ValueErrors. Evidence có component/reason; message/cause giữ
validation context. Valid requested index có inclusive location. Không catch-all
RuntimeError hoặc che programming exception. Quality/viewpoint/provenance failures
trong context hợp lệ vẫn do adapter trả unavailable/value=null theo policy cũ.

## Môi trường và test evidence

CPython **3.11.9**, Windows x64 (10.0.26200), Pydantic **2.13.5**, pytest **9.1.1**,
setuptools **84.0.0**. Không dependency/model/framework mới. Venv còn MediaPipe
0.10.14, NumPy 2.4.6, SciPy 1.17.1, OpenCV contrib 5.0.0.93, protobuf 4.25.9;
legacy opencv-python 5.0.0.93 vẫn có. Chưa clean-environment/transitive-lock test.

Lệnh PowerShell thực tế tại project root:

```powershell
& .venv/Scripts/python.exe -m pytest -q tests/test_feature_context.py tests/test_feature_context_integration.py --junitxml=outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/new-tests.xml
& .venv/Scripts/python.exe -m pytest -q tests/test_feature_context.py tests/test_feature_context_integration.py --junitxml=outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/new-tests-repaired.xml
& .venv/Scripts/python.exe -m pytest -q tests/test_feature_context.py tests/test_feature_context_integration.py --junitxml=outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/new-tests-final.xml
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/full-suite.xml
```

| Run | Passed | Failed / errors / skipped | Warnings | Terminal time |
|---|---:|---|---:|---:|
| Initial new tests | 89 | 1 / 0 / 0 | 0 | 22.23 s |
| Repaired strict serialization tests | 95 | 0 / 0 / 0 | 0 | 21.23 s |
| Final new tests, 107 unit + 2 integration | **109** | **0 / 0 / 0** | **0** | **22.99 s** |
| Full suite GĐ0–5.4 | **557** | **0 / 0 / 0** | **4** | **60.60 s** |

Initial failure phát hiện bool trong coordinate bị JSON-mode serializer đổi
thành float; chuyển Python-mode intake và thêm usable/unusable regression tests.
Final tests thêm no-eligibility-call, core-statistics parity, JSON corruption và
profile intake guards. Final commands exit 0. Offline install 0.5.3 và pip check
exit 0, `No broken requirements found.` 4 warnings không suppress: protobuf
SymbolDatabase.GetPrototype deprecated, hai fresh pose và hai fresh preprocessing
integrations GĐ2/GĐ4; không feature accuracy evidence.

[Final new JUnit](../outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/new-tests-final.xml),
[full JUnit](../outputs/stage5_4/verification-a70932e36113422b8e8b54de5679a5be/full-suite.xml),
[commands/environment/source/input/output hashes](artifacts/stage5_4_verification.json).
Isolated `python -I` subprocess ngoài workspace chặn NumPy/SciPy/cv2/MediaPipe/
Torch: public import, schema, factory, strict JSON roundtrip và numeric call đạt.
Synthetic angle tolerance 1e-10 degrees, rel=0, chỉ kế thừa arithmetic assertion;
không là pose accuracy hoặc threshold form.

## Real saved Stage 4 sample integrations

Dùng measured snapshot GĐ4 để chọn đúng hai saved inputs, không dùng archive
hoặc số liệu lịch sử làm execution mới. Factory/adapter mới đã chạy trên các
sample; full suite riêng vẫn chạy fresh MediaPipe qua tests upstream hiện hữu.
Không chạy extractor feature toàn video. HSPU dùng cùng squat ID như negative
exercise case; không triển khai calculator HSPU.

| Relative input / video ID (filename) | Frames | Samples / timestamps seconds | Available / unavailable | Primitive calls |
|---|---:|---|---|---:|
| Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi | 267 | 0/0.0, 133/5.32, 266/10.64 | 0 / 3 | 0 |
| Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi | 78 | 0/0.0, 39/1.56, 77/3.08 | 0 / 3 | 0 |

Sáu contexts giữ exact identities/raw hashes/quality/profile/result links, JSON
roundtrip và context/input hashes đạt. Quality action **skip**, viewpoint **unknown**,
thiếu selected metric → viewpoint_unknown/feature_quality_blocked/
feature_metric_eligibility_missing. HSPU thêm unsupported_exercise và left ankle
unusable/provenance flags. Không sửa real report để ép available, không diễn giải
measurement/input failures thành lỗi kỹ thuật người tập.

Full-result content hashes: squat
`8c503dfad52f0de42e7327fe38eec902563c98aeaff6f0579e5a97f80c137630`;
HSPU `7fc3a9bdd7411a4510f6295fecb504eeb85565f344a261849112b3a51eeb1770`.
Final output directory `outputs/stage5_4/integration-67d379854d064e9cbedbdc9a317acd44/`:
[squat summary](../outputs/stage5_4/integration-67d379854d064e9cbedbdc9a317acd44/BodyWeightSquats.summary.json),
[HSPU summary](../outputs/stage5_4/integration-67d379854d064e9cbedbdc9a317acd44/HandstandPushups.summary.json).
Mỗi sample lưu context JSON + observation JSON riêng, full file/content hashes
và source relative paths. Source result/QualityReport/file bytes không đổi.
0/6 available chỉ là sample outcome, không full-video coverage/form score.

## Preservation, files và DoD

Before manifest **265 files**, **255 unchanged**, **10 changed**, missing=0.
Mọi output cũ giữ byte/hash; source/tests/configs upstream giữ nguyên trừ audit
helper nêu trên và additive flags/version/imports/adapter intake. Reversing chính
hai imports + validate_quality body khớp byte SHA-256 trước sửa, chứng minh phần
numerical preprocessing khác không đổi. Không xóa/sửa dataset/archive/output cũ.
Source tree SHA-256 (scope src/gym_coach Python, tests Python, configs TOML,
pyproject; không docs/data/outputs/pycache):
`3026326e84357429f4ee765a21cdd0885d45f120ff74bc5db1cbe766d27d6ab5`.

Mới: `features/context.py`, `tests/test_feature_context.py`,
`tests/test_feature_context_integration.py`, report này, linked JSON Schema và
verification JSON. Cập nhật features exports/adapter, preprocessing audit helper,
flags/version và README/IMPLEMENTATION_PLAN/PROJECT_STATE/TASKS/SCHEMA_CONVENTIONS.
New run artifacts dùng UUID directories và exclusive creation.

| Acceptance criterion | Evidence | Result |
|---|---|---|
| Explicit exact index/full strict source snapshot | first/middle/last, invalid index/type/empty/unsafe tests | PASS |
| Raw/processed/result/profile/quality linkage | full-source validators, mixed/hash/timeline/endpoint/JSON corruption tests | PASS |
| Provenance/normalization/flags/report preservation | synthetic all statuses + six real snapshots/roundtrips | PASS |
| No mutation/alias with caller | nested dictionaries/flag evidence mutation isolation tests | PASS |
| No eligibility recompute/promotion | core-only audit, no-call spy, custom degrade/skip tests | PASS |
| Legacy/adapter scope retained | equality available/unavailable/degenerate/fallback outcomes; old tests all pass | PASS |
| Lightweight import/execution | isolated blocked-backend subprocess | PASS |
| Real sample unavailable integrity | two saved GĐ4 inputs, six contexts, primitive calls=0 | PASS |
| New/full tests and docs/evidence | 109/557 passed, current docs + JUnit/hash evidence | PASS |
| Whole Stage 5 kept IN_PROGRESS | remaining features/eligible integration below | PASS |

## Giới hạn và bước nhỏ nhất tiếp theo

- Full source snapshot và revalidation **O(N) mỗi context**; chưa benchmark video
  dài, streaming/shared audit store hoặc optimized proof. Không decoded images.
- Frozen models chưa deep-immutable; factory/adapter tách caller và strict intake/
  readback kiểm tra stale hashes. Re-hashing forgery đồng bộ không bị coi là xác
  thực nguồn. Legacy context tự ghép vẫn có giới hạn cũ nếu caller chủ động dùng nó.
- QualityReport chưa có raw-sequence hash độc lập. Chứng minh report bằng snapshot
  trong result và raw core statistics; không chứng minh report thật sự được tạo
  từ exact coordinates/run khi hai inputs có cùng counts/spans. Không rerun
  interpolation/normalization/SG để xác minh arithmetic output hoặc authenticate
  producer. Full result content identity không là execution/run identity.
- Numeric JSON giữ legacy input; cần lưu linked context riêng để audit source.
  Không claim numeric payload tự chứa full linkage hoặc auto migration.
- Viewpoint/projection/pose accuracy/form vẫn **unvalidated**; UCF101 chỉ nhãn
  action. Chưa numeric integration trên real input đủ quality/viewpoint, chưa
  feature khác/FeatureSequence, segmentation hoặc assessment; protobuf/dual OpenCV
  và clean-environment limitations còn nguyên.

Bước nhỏ nhất tiếp theo: thiết kế version/schema và adapter riêng cho
`squat_knee_angle_right_deg`, kiểm thử bằng factory + readiness policy hiện có,
trước chọn numeric development input đủ điều kiện. Không mở GĐ6 hoặc scoring.
