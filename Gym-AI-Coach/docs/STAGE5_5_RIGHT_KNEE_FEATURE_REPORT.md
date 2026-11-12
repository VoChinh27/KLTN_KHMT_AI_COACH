# Giai đoạn 5.5 — Numeric adapter góc gối squat bên phải

Ngày kiểm tra: **2026-10-09, Asia/Saigon**. GĐ5.5 **COMPLETE** theo phạm vi
trong yêu cầu; toàn GĐ5 **IN_PROGRESS**, assessment vẫn blocked.

## Phạm vi và workspace

Đã đối chiếu 11 tài liệu hiện hành: README, technical ground-truth spec, master
prompt, implementation plan, project state, tasks, schema conventions và reports
GĐ5.1–5.4; kiểm tra registry/policy, factory/linked context, primitive, adapter
bên trái và tests trước sửa. Workspace không có Git repository: dùng source
manifest SHA-256, không giả commit/diff. Không mở/restore archive, không sửa/xóa
dataset hoặc output cũ. Full suite hiện hữu vẫn đọc hai video fixtures GĐ2/GĐ4.

Chỉ thêm calculator cho **`squat_knee_angle_right_deg`** đã có trong registry.
Ordered inputs A/B/C: **right_hip → right_knee → right_ankle**. Primitive
`joint_angle(A,B,C)` hiện hữu tính góc trong giữa A−B và C−B tại knee,
float degrees `[0,180]`, method `joint_angle_abc`, primitive `joint-angle-2d-v1`.
Không viết lại math, không rotate/flip/normalize/smooth tại adapter.
Tọa độ là `stage4_processed_xy`, giữ coordinate/normalization metadata GĐ4;
đây là góc chiếu trong ảnh, chưa xác nhận tương đương góc giải phẫu 3D.

Chưa feature khác, bilateral fusion/symmetry, batch extractor/FeatureSequence,
segmentation/repetition/phase, form rules/threshold/score/feedback hoặc training.
Hai knee adapters đo độc lập theo ID; không tự chọn bên hoặc ghép kết quả.

## API, schema và compatibility

Package **ai-coach-gym 0.5.4**, không thêm dependency.

```python
from gym_coach.features import (
    KneeAngleFeatureObservation, NumericFeatureObservation,
    build_feature_input_context, observe_feature,
)

context = build_feature_input_context(result, frame_index=5)
right = observe_feature("squat_knee_angle_right_deg", context)
restored = KneeAngleFeatureObservation.model_validate_json(right.model_dump_json(), strict=True)
assert right == restored

left = observe_feature("squat_knee_angle_left_deg", context)
assert type(left) is NumericFeatureObservation
new_left = KneeAngleFeatureObservation.from_legacy(left)  # thao tác tường minh
```

| Identity | Giá trị / boundary |
|---|---|
| Legacy left class / schema | NumericFeatureObservation / 0.2.0, source và field contract không sửa |
| Left dispatch / adapter | Schema 0.2.0 / squat-left-knee-v1 giữ nguyên |
| New class / schema | KneeAngleFeatureObservation / 0.3.0, đúng hai canonical knee IDs |
| Right dispatch / adapter | Schema 0.3.0 / squat-right-knee-v1 |
| New schema left migration | from_legacy revalidate strict, chỉ đổi schema version, không tính lại |
| Registry | gym-feature-registry-v1 / definition schema 0.1.0 giữ nguyên |
| Factory / linked context | preprocessing-context-v1 / context schema 0.2.0 giữ nguyên |
| New FlagCodes | Không thêm; dùng conventions/codes hiện hữu |

Class mới kế thừa fields/helper audit legacy, override validator canonical ID/
version. Adapter version bắt buộc, phải khớp bên đã chọn. JSON 0.2.0 không tự đọc
như 0.3.0 và ngược lại; left dispatch không silent migration. Schema snapshot
GĐ5.3 không sửa. Khi so sánh generated schema với snapshot GĐ5.3, chỉ loại hai
shared FlagCodes đã thêm ở GĐ5.4; không có schema delta mới cho legacy ở GĐ5.5.

Observation giữ exercise/viewpoint, video/index/timestamp, method/primitive,
unit/kind, M1/S3 tiers, validation, ordered actual ProcessedJoint inputs,
source_status/status/usability/interpolation refs, registry/definition/input/
quality/policy hashes và full availability audit. Linked context JSON lưu riêng;
`context.measurement_input_hash == observation.availability.input_hash`.
Numeric JSON tiếp tục chứa frame/quality snapshot, không nhúng full linkage.

Identity đã đo:

- Registry hash: `bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a`.
- Right definition hash: `c2a60690a7414f7827d20dcd80b7d6e69f8ab714637a75542e985699f13a4c93`.
- [New schema snapshot](artifacts/stage5_5_knee_observation.schema.json), byte SHA-256:
  `2c616bb64174f91eeb556fbff3dcda7c9346106471b78507170f33d72f0fd9e2`.
- Source/test/config manifest: **63 files**, SHA-256
  `9814eb9f5926c735a446e42c907c6f4ff89642aa090e3753c839ad29ebe1118b`.
  Hash của sorted POSIX `relative_path:byte_SHA256\n`; docs/outputs/cache/data/archive
  không thuộc source tree hash. Byte hashes và canonical payload hashes phân biệt.

Hash là content identity, không chữ ký, producer authentication hoặc accuracy.

## Policy, error và strict validation

Chỉ canonical ID/definition từ DEFAULT_REGISTRY. Unknown ID có structured
unknown_feature_id; definition bị sửa có invalid_feature_input. Các 10 entries
còn lại không calculator trả unsupported_feature_calculation trước primitive.

`check_feature_availability` chạy trước phép đo, giữ policy GĐ5.1:

- Chỉ BodyWeightSquats và viewpoint side được khai báo. Unknown/front/oblique
  unavailable; không suy viewpoint hoặc form từ action label.
- Sequence **allow**, exact metric key `squat_knee_angle_right_deg` **allow**
  với đúng required joints. Missing key/wrong joints/degrade/skip bị khóa.
  Factory/adapter không gọi lại QualityGate eligibility hoặc nâng action.
- Required joints phải usable, x/y hữu hạn và status observed/imputed.
  Low visibility/missing/out-of-frame không trở thành observed hoặc số thật.
  Imputed giữ source missing và bracket endpoints/index/timestamp.
- Angle không require torso scale/origin: fallback vẫn được lưu null, không
  thêm gate hoặc thay scale bằng 1.0. Distance/ratio policies không đổi.

Available phải finite degrees, computed, không unavailable reasons/flags.
Prerequisite unavailable giữ **nguyên** reasons/flags từ policy, value null,
not_computed; không gọi primitive. Geometry failure null/failed_measurement
với đúng một structured flag measurement_error/features/S3/error:

| Code | Reason / evidence |
|---|---|
| degenerate_geometry | zero_length_vector, BA hoặc BC khớp actual zero vector |
| geometry_numerical_error | invalid_denominator, non_finite_ratio hoặc invalid_angle_result từ primitive |

Location giữ actual video_id/frame inclusive/feature ID; evidence primitive và
input_joint_a/b/c đúng right_hip/right_knee/right_ankle. Không technical_error,
không evidence value giả. Invalid input/context mismatch vẫn structured GymError;
exceptions lập trình hoặc input errors từ primitive không bị catch-all che đi.

New strict validator revalidate availability/actual joint inputs bằng Python-mode
serialization, JSON allow_nan=false và strict=True. Từ chối NaN/Inf, bool/null
coordinates giả, wrong ID/version/adapter/unit/kind/joint order/metadata/hashes,
lost reasons/flags và forged available trên BA/BC zero. Readback không gọi
primitive hoặc recompute quality; consistency validation không đo lại angle.

## Test evidence thực tế

Môi trường: CPython **3.11.9**, Windows x64 (`Windows-10-10.0.26200-SP0`),
Pydantic **2.13.5**, pytest **9.1.1**, setuptools **84.0.0**. Backend/dependencies
hiện hữu: MediaPipe **0.10.14**, NumPy **2.4.6**, SciPy **1.17.1**, protobuf
**4.25.9**, opencv-contrib-python **5.0.0.93**. Venv legacy còn metadata
opencv-python **5.0.0.93**; chưa kiểm tra clean environment hoặc đổi provider.
Public feature import/factory/schema/execution được kiểm tra subprocess isolated
chặn NumPy/SciPy/cv2/MediaPipe/Torch, không kéo heavy backends.

Lệnh đã chạy tại project root (PowerShell, không có run dự kiến):

```powershell
& .venv/Scripts/python.exe -m pytest -q tests/test_right_knee_feature.py tests/test_right_knee_feature_integration.py --junitxml=outputs/stage5_5/verification-305d721cc3dd4c3083fe0cc73973f8e2/new-tests.xml
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_5/verification-305d721cc3dd4c3083fe0cc73973f8e2/full-suite.xml
```

| Run | Passed | Failed | Errors | Skipped | Warnings | Terminal duration / exit |
|---|---:|---:|---:|---:|---:|---|
| New tests: 116 unit + 2 saved real-input integrations | 118 | 0 | 0 | 0 | 0 | 30.90 s / 0 |
| Full suite | 674 | 0 | 0 | 0 | 4 | 94.59 s / 0 |
| Offline editable install | — | — | — | — | — | Successfully installed ai-coach-gym-0.5.4 / 0 |
| pip check | — | — | — | — | — | No broken requirements found. / 0 |

New run đạt ngay lần đầu, không có repair run. Bốn warnings là UserWarning
`SymbolDatabase.GetPrototype()` deprecated từ google/protobuf/symbol_database.py:55
ở hai pose và hai fresh preprocessing MediaPipe integrations GĐ2/GĐ4; không
suppress warnings hoặc sửa dependency để che. JUnit ghi tests/errors/failures/
skipped/time; warnings/duration terminal ghi riêng trong evidence.

Tests bao gồm angles 90/180/0/45° tại frames 0/5/30, tolerance **1e-10 degrees**
absolute, rel=0, direct primitive equality; canonical dispatch/migration; quality/
viewpoint/metric/provenance gates; fallback scale; degeneracy/numerical faults;
strict tampering; detached state and heavy import guards. Tolerance chỉ là kiểm
thử hình học synthetic, không ngưỡng form hoặc pose accuracy.

Imputed actual pipeline fixture vẫn **degrade/unavailable**. Available imputed
case dùng **fresh synthetic caller report contract** version
`synthetic-right-imputed-allow-contract-v1`, giữ nguyên original report/result;
không QualityGate evidence, không ground truth hoặc promotion report thật.
Synthetic examples available90°/degenerate-null được lưu riêng trong verification
directory để review, không được báo như số đo real video.

Full count: **556 retained tests + 118 mới = 674**. Test cũ kỳ vọng mọi ID khác
left unsupported được cập nhật để loại right khỏi parameter set, vì calculator
này nay đã có; giảm một case từ baseline557. Không đổi assertion cho 10
calculators chưa triển khai. Legacy left exact payload và serialized JSON khớp
**bảy golden GĐ5.3**: một synthetic available, sáu real unavailable.

## Saved real-input integration đã đo

Source là saved GĐ4 results, không chạy lại video để tạo phép đo GĐ5.5. Cả hai
artifacts source giữ nguyên byte hashes. Full suite riêng vẫn chạy fresh upstream
MediaPipe integrations như trước. Run GĐ5.5 full-suite mới lưu tại
`outputs/stage5_5/integration-50851f1cf132411089cc7ade74089d76/`.

| Exercise / video ID | Relative video path | Source frames | Samples index / timestamp seconds | Available / unavailable | Primitive calls |
|---|---|---:|---|---|---:|
| BodyWeightSquats / v_BodyWeightSquats_g01_c01.avi | Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi | 267 | 0/0.0, 133/5.32, 266/10.64 | 0 / 3 | 0 |
| HandstandPushups / v_HandStandPushups_g01_c01.avi | Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi | 78 | 0/0.0, 39/1.56, 77/3.08 | 0 / 3 | 0 |

Ba squat samples: right hip observed/usable; right knee và ankle low_visibility/
not usable. Reasons cả ba: viewpoint_unknown, feature_quality_blocked,
feature_metric_eligibility_missing, feature_provenance_rejected, feature_joint_unusable.

HSPU samples0/39: hip/knee observed/usable, ankle low_visibility/not usable;
cùng reasons squat và thêm unsupported_exercise. Sample77: ankle observed nhưng
out_of_frame/not usable, không feature_provenance_rejected; những reasons còn
lại giữ nguyên. HSPU là negative wrong-exercise test cho squat ID, không HSPU
feature calculator. Sequence action skip, viewpoint unknown và absent feature
metric key giữ nguyên ở cả hai source reports; **0 available/6 unavailable**.
Không diễn giải sự thiếu dữ liệu thành lỗi kỹ thuật người tập.

Saved source paths/hashes, từng context/observation JSON, sample reasons/joints/
identity/hashes nằm trong [measured evidence](artifacts/stage5_5_verification.json).
Không tuyên bố feature coverage toàn267/78 frames từ sáu sampled frames.

## File changes và preservation audit

Thêm knee_observation.py và hai test files; sửa features/adapter.py,
features/__init__.py, _version.py và parameter unsupported test đã nêu. Cập nhật
README/IMPLEMENTATION_PLAN/PROJECT_STATE/TASKS/SCHEMA_CONVENTIONS, thêm báo cáo
này cùng schema/evidence. Script verification và mọi artifacts mới ở thư mục
GĐ5.5 riêng, exclusive creation không ghi đè output cũ.

Before manifest **385 existing files**: **376 byte-unchanged**, **9 changed**
đúng phạm vi trên, **0 deleted**, gồm **300 prior output files giữ nguyên**.
Legacy observation.py, registry/models/policy/context, geometry, video/pose/
quality/preprocessing source, configs/manifest/common flags và mọi old tests
khác byte-unchanged. Dataset/archive không thuộc hash audit; không bị sửa/xóa.

## Definition of Done và giới hạn

| Yêu cầu | Bằng chứng / kết quả |
|---|---|
| Đúng canonical right ID/order/primitive | Known angles + direct primitive/metadata checks PASS |
| Version mới chủ đích, legacy left giữ nguyên | Separate class/schema0.3, left0.2 dispatch, migration boundary + 7 golden checks PASS |
| Policy trước đo, không promote/recompute/scale gate thêm | Forbidden primitive/eligibility calls, policy/fallback/imputed fixtures PASS |
| Finite available, null unavailable, structured errors/audit | Strict corruption/degenerate/error mapping/readback tests PASS |
| Input/report/registry/source artifacts preserved | Detach tests + 385-file before/after audit PASS |
| Saved real samples không ép allow | 2 integrations, 6 unavailable, primitive0 PASS |
| Lightweight public import, regressions đạt | 118 new / 674 full passed PASS |
| Current docs/report/schema/measured evidence đầy đủ | Các file hiện hữu được cập nhật, JUnit/evidence/new snapshots PASS |

Measurement **unvalidated**, assessment **blocked_pending_expert_validation**.
Registry metadata calculation_status=not_implemented là snapshot GĐ5.1 giữ
compatibility; execution status ở numeric observation, không nâng scientific
validation. Không accuracy/form labels hoặc real numeric sample đủ điều kiện;
nhãn UCF101 chỉ là action. 2D projection/occlusion/camera limitations còn nguyên.

Linked context O(N) snapshot/revalidation mỗi frame chưa benchmark; không
authenticate producer/report origin hoặc đo lại processed coordinates. Frozen
models không deep-immutable; strict readback và detached intake kiểm tra stale
content, không chống đồng bộ forgery/re-hash. Chưa clean-environment compatibility.
Tám angle candidates còn lại chưa calculator; hai ratio candidates vẫn blocked.

Bước nhỏ nhất tiếp theo: **chốt một development fixture có documented side-view
và QualityGate sequence/exact-feature metric allow cho hai knee IDs**, trước
numeric real-data measurement. Không đổi reports skip/degrade để ép available,
không suy viewpoint từ action label. Toàn GĐ5 chưa COMPLETE; chưa mở GĐ6/assessment.
