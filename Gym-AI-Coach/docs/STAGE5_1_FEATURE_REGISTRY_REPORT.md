# Giai đoạn 5.1 — Feature contracts và registry metadata

Ngày: 2026-10-08, Asia/Saigon. **GĐ5.1 COMPLETE**; toàn **GĐ5 IN_PROGRESS**.
Nguồn yêu cầu là README, đặc tả/Prompt tổng, kế hoạch/trạng thái/tasks/schema
hiện hành và báo cáo GĐ4, đã đọc đủ trước khi sửa. Không sử dụng archive hoặc
coi các câu “đã triển khai” lịch sử là bằng chứng cho rebuild hiện tại.

## Phạm vi thực tế

Module `gym_coach.features` cung cấp metadata registry, prerequisite checks và
observation contract. Calculation/policy tách riêng: policy chỉ kiểm tra dữ liệu,
không chạy angle/vector/distance formula. Registry chứa 10 angle candidates
unvalidated và 2 ratio candidates blocked. Không thêm feature extractor,
FeatureSequence, bilateral fusion, rep/phase, form errors, thresholds, scoring,
feedback hoặc training. Không tạo measurement/accuracy claims từ UCF101.

Files mới: `features/models.py`, `registry.py`, `policy.py`;
`tests/test_features.py`, `test_features_integration.py`; report này và
`docs/artifacts/stage5_1_feature_registry.json`, `stage5_1_verification.json`.
Files đổi: features `__init__.py`, common `flags.py`, `_version.py`, README,
IMPLEMENTATION_PLAN, PROJECT_STATE, TASKS, SCHEMA_CONVENTIONS. Manifest/dependencies
và source/tests/configs video/pose/quality/preprocessing không đổi.

## Schema, registry và identity

- Package `ai-coach-gym==0.5.0`; feature/registry schema `0.1.0`.
- Registry version `gym-feature-registry-v1`; definition version `0.1.0`.
- Canonical registry hash:
  `bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a`.
- Canonical JSON SHA-256: sorted keys, compact separators, UTF-8,
  ensure_ascii=false, allow_nan=false. Registry tuple/definitions frozen,
  helper dictionaries tạo mới. Không backend/file I/O tại public import.
- [Registry JSON snapshot](artifacts/stage5_1_feature_registry.json) chứa đầy đủ
  formulas/joints/policies/tiers/validation/reasons/limitations, đọc lại bằng
  FeatureRegistry.model_validate_json(). Registry hash là metadata identity,
  không code/measurement accuracy identity.
- Code manifest hash (src Python/tests Python/configs TOML/pyproject):
  `1062820a3af73242a679d5688a35a7a152328ec15a5eb835cfa851ed51f090ba`.
  Workspace không có Git repository; before/source file hashes được lưu, không
  tạo commit hoặc giả Git diff. Scope hash không gồm docs/data/outputs/archive.

`FeatureDefinition` yêu cầu ID/name/exercises, definition/formula/method,
ordered joints, input policy, kind/unit, source references/tiers, measurement
tier, validation, viewpoint, quality/scale policies, roles/reasons/limitations.
Schema từ chối thiếu metadata, joint trùng/arity sai, unit không khớp kind,
provenance unsafe, unknown enums hoặc missing policy reasons. M0 phải blocked.

`FeatureInputContext` lưu detached frame/quality snapshots, giữ frame identity,
raw_frame_hash, transforms, joint status/source_status/interpolation và report
nguyên bản. `FeatureAvailability` lưu registry/definition/input identities/hashes,
ready_for_calculation/unavailable, reasons và structured flags. JSON validation
kiểm tra hashes và chạy lại **prerequisite policies**, không tính lại quality
metrics hoặc geometry. Lost flags/reasons/false readiness bị từ chối.

`FeatureObservation` chỉ có status not_computed hoặc unavailable;
**value=null**, computation_status not_implemented. Unavailable reasons/flags
được giữ trong `availability`. Numeric/available observation không hợp lệ trong
schema GĐ5.1; phần calculation sau phải có schema version mới và tests.

## Danh mục metadata MVP

A/B/C là ordered joints của `joint_angle_abc`; angle trong tại B, metadata miền
toán học [0,180] degrees. LH/RH/LS/RS là left/right hips/shoulders.
Axis angle = angle(midpoint(shoulders)-midpoint(hips), image_up=(0,-1)) trong
y-down coordinates, có hướng; không rotate/flip để làm inverted pose thành upright.
Không công thức nào được tính trong GĐ5.1.

| Stable feature ID | Ordered inputs / formula metadata | Unit / tier | Validation / role |
|---|---|---|---|
| squat_knee_angle_left_deg | left hip, knee, ankle → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| squat_knee_angle_right_deg | right hip, knee, ankle → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| squat_hip_angle_left_deg | left shoulder, hip, knee → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| squat_hip_angle_right_deg | right shoulder, hip, knee → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| squat_torso_angle_vertical_deg | LH,RH,LS,RS → directed axis angle | degrees / M2 | unvalidated, descriptive/informational |
| hspu_elbow_angle_left_deg | left shoulder, elbow, wrist → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| hspu_elbow_angle_right_deg | right shoulder, elbow, wrist → angle(A,B,C) | degrees / M1 | unvalidated, descriptive/segmentation candidate |
| hspu_shoulder_angle_left_deg | left hip, shoulder, elbow → angle(A,B,C) | degrees / M2 | unvalidated, descriptive/segmentation candidate |
| hspu_shoulder_angle_right_deg | right hip, shoulder, elbow → angle(A,B,C) | degrees / M2 | unvalidated, descriptive/segmentation candidate |
| hspu_body_line_angle_deg | LH,RH,LS,RS → directed shoulder-hip axis angle | degrees / M2 | unvalidated, descriptive/informational |
| squat_knee_to_ankle_horizontal_left_ratio | left knee, ankle → abs(delta_x_processed) | dimensionless_ratio / M2 | blocked, descriptive/informational only |
| squat_knee_to_ankle_horizontal_right_ratio | right knee, ankle → abs(delta_x_processed) | dimensionless_ratio / M2 | blocked, descriptive/informational only |

Source references là đặc tả project §§0.2,2.2,3.2–3.3/4.2–4.3, không nguồn S1
đã được chuyên gia duyệt. Joint angles ghi S3; axis/ratio candidates S3 +
TBD-EXPERT. Không gán OBS cho hiệu quả signal từ thí nghiệm legacy. M1/M2 là
candidate measurement classification, không chứng nhận accuracy. HSPU shoulder
và axis chọn M2 bảo thủ khi phụ thuộc projection; camera policy cũng chưa validation.

HSPU body-line chỉ mô tả shoulder-hip axis, không toàn thân/head/leg alignment.
Ratios blocked vì cần measurement/viewpoint protocol và validation; không phải
rule “gối vượt mũi chân”. Với torso normalization hợp lệ, delta_x_processed đã
chứa division theo torso, tương đương delta_x_working/torso_scale; **không chia
lần thứ hai**. Generic contract hỗ trợ distance/torso_lengths nhưng registry MVP
không bổ sung một distance feature ngoài các candidates của đặc tả.

## Chính sách availability

Policy này là **S3 conservative baseline**, không ngưỡng kỹ thuật:

1. Feature áp dụng đúng exercise; blocked definitions luôn unavailable.
2. Joint inputs là Stage 4 processed x/y, chỉ observed hoặc explicit imputed,
   require usable và finite. Source_status/imputation endpoints giữ nguyên;
   low_visibility/missing không trở thành observed. Không nhận raw coordinate
   null hoặc Stage 4 out-of-frame-unusable như measurement hợp lệ.
3. MVP chỉ support side viewpoint do metadata/người dùng xác nhận. Unknown
   không xác định image plane; front chưa sagittal/symmetry protocol; oblique
   chưa validation foreshortening. Mỗi unsupported case có metadata reason.
4. Sequence movement_analysis và feature metric key phải eligible + action allow.
   Degrade/skip đều blocked; không nâng quality khi imputation thành công.
   Feature metric phải khai báo đúng set/arity required joints; không được dùng
   allow cho một joint khác. Missing metric key trả unavailable, không infer
   từ global allow và không tự recompute QualityReport.
5. Distance/ratio luôn require valid mid-hip + torso scale, không fallback origin
   hoặc scale. Null scale không thay bằng 1.0. Angles không yêu cầu scale nếu
   joints/quality/viewpoint đủ prerequisites; geometry safety vẫn chưa được tính.

`quality_requirements(exercise)`/`quality_viewpoints(exercise)` trả metadata
mappings cho caller truyền vào QualityGate **khi tạo report mới trước GĐ4**;
không sửa report cũ hoặc mutate registry. `for_exercise` mặc định loại blocked
entries; tra cứu ID bị blocked vẫn được và policy trả reason.

Flags features/S3: unknown_feature_id, unsupported_exercise, invalid_viewpoint,
invalid_feature_input; feature_definition_blocked, feature_joint_unusable,
feature_provenance_rejected, feature_quality_blocked,
feature_metric_eligibility_missing, feature_quality_policy_mismatch; reuse
missing_joint/non_finite_landmark/viewpoint_unknown/viewpoint_mismatch và
normalization_origin_fallback/normalization_scale_fallback. Categories là
input_error/measurement_error/insufficient_data; không technical_error. Quality
blocker evidence giữ scope/action/upstream reason codes và context giữ report đầy đủ.

Unknown ID/exercise hoặc invalid viewpoint enum → GymError structured input reason.
Known UNKNOWN/unsupported viewpoint hay wrong MVP exercise for feature → unavailable.
Invalid structural input, NaN/Inf, null trong usable joint, identity mismatch →
invalid_feature_input, không đo hoặc trả số giả. Missing-status/unusable joints
trong valid Stage 4 frame → unavailable với joint-scoped flags.

## Test evidence và môi trường

CPython 3.11.9 Windows x64, Pydantic 2.13.5, pytest 9.1.1, setuptools 84.0.0.
Môi trường GĐ4 giữ NumPy 2.4.6, SciPy 1.17.1, MediaPipe 0.10.14, OpenCV contrib
5.0.0.93, protobuf 4.25.9; GĐ5.1 không thêm dependency/model/framework.
Legacy venv vẫn có opencv-python 5.0.0.93; chưa clean-environment test.

Commands thực tế từ project root:

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q tests/test_features.py tests/test_features_integration.py --junitxml=outputs/stage5_1/verification-0400b5b2e23e4b8cb05c0171bf2afd9b/new-tests.xml
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_1/verification-0400b5b2e23e4b8cb05c0171bf2afd9b/full-suite.xml
```

- Editable reinstall exit 0, ai-coach-gym 0.5.0, không tải dependency.
- Pip check: `No broken requirements found.`
- New tests: **89 passed in 5.95s**, 0 warnings; 87 unit + 2 real-input policy tests.
- Full suite: **234 passed, 4 warnings in 39.55s**, exit 0; 0 failures/errors/skips.
- 4 warnings protobuf SymbolDatabase.GetPrototype deprecated từ fresh MediaPipe
  pose integrations GĐ2/GĐ4 hiện có, không phải feature measurement validation.
- Tests cover unique/stable IDs, required metadata/method/joints/units/tiers/reasons,
  exhaustive viewpoint, provenance/usability/null, unknown input enums,
  blocked definitions, sequence/metric degrade/skip, absent/wrong eligibility,
  scale/origin missing/zero cho distance/ratio, angle fallback readiness,
  input immutability, serialization/hashes/lost flags/false readiness, no fake
  numeric value, immutable registry mappings và isolated public import/schema.
- Subprocess `-I` ngoài workspace chặn NumPy/SciPy/OpenCV/MediaPipe/Torch,
  import/lookup/JSON Schema thành công; GĐ0 smoke tiếp tục đạt.

[New JUnit](../outputs/stage5_1/verification-0400b5b2e23e4b8cb05c0171bf2afd9b/new-tests.xml),
[full JUnit](../outputs/stage5_1/verification-0400b5b2e23e4b8cb05c0171bf2afd9b/full-suite.xml),
[environment/commands/hash evidence](artifacts/stage5_1_verification.json).
Before manifest lưu trong cùng output directory; output file mới không ghi đè
GĐ4 XML/artifacts. Hash audit xác nhận source/tests/configs upstream và recorded
GĐ4 artifacts còn nguyên; các test GĐ0–4 vẫn pass.

Hai policy integrations đọc strict Stage 4 artifacts được chỉ định bởi snapshot
GĐ4 hiện hành: squat v_BodyWeightSquats_g01_c01.avi (267 frames) và HSPU
v_HandStandPushups_g01_c01.avi (78 frames). Chọn first/middle/last × 5 active
definitions/bài = 15 checks/bài. Tất cả unavailable vì report skipped, viewpoint
unknown và thiếu feature metric keys; không nâng quality hay tạo numeric values.
Frame/report/file bytes giữ nguyên, JSON availability roundtrip đạt. Đây là
contract/policy integration trên real input, **không phải fresh feature extraction**
hay accuracy experiment. Full suite vẫn chạy fresh video→pose→quality→preprocessing
qua các integration tests cũ, không thay behavior GĐ4.

## Đối chiếu DoD GĐ5.1

| Tiêu chí | Bằng chứng | Kết quả |
|---|---|---|
| Registry unique/stable MVP metadata, completeness | 12 entries, strict schema/lookup/negative tests | PASS |
| Joint/provenance/usability guards | Missing/low/OOF/unsafe input, imputed source tests | PASS |
| Quality/viewpoint không đạt bị khóa, không promote | Allow-only exact metric/sequence tests, missing keys, enum/viewpoint tests | PASS |
| Distance/ratio fallback bị khóa | Explicit required policies + missing/zero origin/scale tests | PASS |
| Unavailable reason/flag, no fake value | Policy consistency, null-only observation và forged-result rejection | PASS |
| Metadata/validation/reasons/flags JSON roundtrip | Registry/availability/observation tests + real-input readback | PASS |
| Public import lightweight | Blocked backend -I subprocess và GĐ0 smoke | PASS |
| Upstream behavior/test/input không sửa | Hash audit, immutability và full 234-pass suite | PASS |
| Docs và bằng chứng đúng phạm vi | 5 docs cập nhật, report/registry/verification snapshots | PASS |
| Không đánh dấu toàn GĐ5 COMPLETE | GĐ5 IN_PROGRESS; calculator NOT_STARTED | PASS |

## Hạn chế và bước nhỏ nhất tiếp theo

- Chưa geometry/measurement implementation, degeneracy/clamp checks, numeric
  FeatureObservation, FeatureSequence hay real-video measured feature output.
  Ready chỉ kiểm tra prerequisites, không chứng minh có một góc hữu hạn hợp lệ.
- Side-view/allow-only là conservative engineering policy. Chưa camera protocol,
  empirical validation, uncertainty estimation hoặc expert form ground truth.
- QualityReport GĐ3 không lưu raw-sequence hash. Context kiểm tra video/index,
  giữ raw_frame_hash và snapshot/hash; caller phải lấy frame/quality cùng GĐ4 run.
  Chưa chứng minh exact coordinate linkage cho report được caller gắn thủ công.
- Availability giữ full frame/quality snapshot để audit, chưa tối ưu serialization
  cho toàn sequence dài; không full extractor/benchmark trong GĐ5.1.
- Frozen input models vẫn có nested dicts; snapshot tách caller, hash/policy
  validation khi deserialize phát hiện thay đổi. Không có deep immutable report.
- Không dùng nhãn UCF101/pose coverage làm form/accuracy ground truth. Các candidate
  chưa expert approved; không assessment/score/feedback được mở.
- Chưa clean environment/transitive lock; protobuf warnings/dual OpenCV legacy còn.

Tiếp theo **GĐ5.2**: triển khai pure joint_angle(A,B,C) primitive với synthetic
known-angle, coincident/zero vector, NaN/Inf và symmetry tests. Sau đó mới thiết kế
numeric schema/adapter GĐ5.3 và real development feature integration; chưa làm
bước này trong lượt GĐ5.1.
