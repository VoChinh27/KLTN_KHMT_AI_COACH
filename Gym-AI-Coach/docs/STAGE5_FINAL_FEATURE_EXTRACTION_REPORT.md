# Giai đoạn 5 — Geometry và Feature Extraction: báo cáo cuối implementation

Ngày: 2026-10-12 (Asia/Saigon). Workspace `D:\KLTN_K28_TKD` không có Git repository:
đã kiểm tra `git status`, không tạo commit/diff hoặc khởi tạo Git. Đọc đầy đủ
README, hai đặc tả/Prompt tổng, plan/state/tasks/schema và reports5.1–5.7A;
đối chiếu source/tests thực tế. Không đọc/khôi phục implementation từ archive.

**Implementation 10 angles COMPLETE. Toàn GĐ5 IN_PROGRESS / VALIDATION_BLOCKED.**
Core DoD của Prompt tổng (calculation/policy tách biệt, unavailable có reason/null,
không technical thresholds) và implementation acceptance Stage5 đạt. Subgate5.7
trong kế hoạch vẫn yêu cầu documented side-view + fresh official sequence/bilateral
knee allow; chưa đạt. Không xóa hoặc tự nới acceptance để đánh dấu whole Stage5
COMPLETE. Stage6–8/scoring/training chưa mở; expert/ground-truth validation chưa có.

## 1. Phạm vi thực tế và file chính

- Thêm `src/gym_coach/axis_geometry.py`: directed midpoint axis angle thuần stdlib.
- Thêm `features/geometry_registry.py`, `geometry_policy.py`,
  `geometry_observation.py`, `geometry_adapter.py`: canonical registry v2,
  versioned framewise policy, numeric0.4 và uniform one-frame/one-feature API.
- Export public API qua `features/__init__.py`; package `_version.py` thành0.6.0.
- Thêm `scripts/audit_stage5_angles.py`: precommitted selection verification,
  fresh unknown/exploratory pipeline, audit và exclusive output creation.
- Thêm `tests/test_axis_geometry.py`, `test_geometry_features.py`,
  `test_geometry_development_audit.py`; cập nhật 5 current docs theo yêu cầu và report này.

Không ratio calculator, public FeatureSequence/temporal extractor, repetition,
phase, segmentation, technique rules, severity/score/feedback hoặc model training.
Không thay đổi old joint_angle, pose backend, QualityGate/profile/threshold,
preprocessing, factory, legacy registry/adapters hoặc previous reports/artifacts.

## 2. API, canonical metadata và version boundary

```python
from gym_coach.features import (
    GEOMETRY_REGISTRY, GeometryFeatureObservation,
    build_feature_input_context, observe_geometry_feature,
)
# result.quality cần exact canonical metric requirements/viewpoints cho exercise.
context = build_feature_input_context(result, frame_index=0)
observation = observe_geometry_feature("squat_hip_angle_left_deg", context)
restored = GeometryFeatureObservation.model_validate_json(observation.model_dump_json())
assert restored == observation
```

API nhận ID hoặc canonical GeometryFeatureDefinition. Registry v2 lookup là nguồn
method/formula/ordered joints/unit/tier/source/viewpoint/quality/validation. Definition
hoặc identity/hash sai bị từ chối; unknown ID/exercise/viewpoint có structured input
error. Malformed/nonfinite input không được làm tọa độ thật; unavailable prerequisite
và measurement failure đều null kèm flags/reasons. Registry/definition/source input
revalidated; JSON readback không đo lại góc hoặc recompute QualityReport.

| Identity | Giá trị |
|---|---|
| Package/distribution | 0.6.0 |
| Registry v1 giữ nguyên | gym-feature-registry-v1 / schema0.1.0 |
| Registry v1 hash | `bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a` |
| Registry v2 | gym-feature-registry-v2 / schema0.2.0 |
| Registry v2 hash | `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72` |
| Definition/availability v2 | 0.2.0 |
| Numeric observation mới | 0.4.0 / geometry-features-v1 |
| Angle policy | framewise-metric-only-v1 |
| Joint primitive | joint_angle / joint-angle-2d-v1, không đổi |
| Axis primitive | midpoint_axis_angle_vertical / midpoint-axis-angle-vertical-2d-v1 |
| Legacy knees | observe_feature; left0.2.0/right0.3.0, gate sequence+metric cũ |
| Linked context | 0.2.0 / preprocessing-context-v1, không đổi |

Không silent migration. Reader dùng explicit class/version; old JSON không được
đổi version để giả thành observation0.4. API mới tạo phép đo mới. Registry v1 tiếp
tục lưu historical calculation_status=not_implemented; v2 mới ghi implemented cho
10 angles. Measurement status/tier không nâng; mọi assessment vẫn blocked.
Schema JSON và full registry snapshot lưu trong run root (hash ở inventory).

| Feature ID | Ordered input joints | Method | Tier | Sources | Validation/calculation |
|---|---|---|---|---|---|
| hspu_body_line_angle_deg | left_hip, right_hip, left_shoulder, right_shoulder | midpoint_axis_angle_vertical | M2 | S3,TBD-EXPERT | unvalidated/implemented |
| hspu_elbow_angle_left_deg | left_shoulder, left_elbow, left_wrist | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| hspu_elbow_angle_right_deg | right_shoulder, right_elbow, right_wrist | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| hspu_shoulder_angle_left_deg | left_hip, left_shoulder, left_elbow | joint_angle_abc | M2 | S3 | unvalidated/implemented |
| hspu_shoulder_angle_right_deg | right_hip, right_shoulder, right_elbow | joint_angle_abc | M2 | S3 | unvalidated/implemented |
| squat_hip_angle_left_deg | left_shoulder, left_hip, left_knee | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| squat_hip_angle_right_deg | right_shoulder, right_hip, right_knee | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| squat_knee_angle_left_deg | left_hip, left_knee, left_ankle | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| squat_knee_angle_right_deg | right_hip, right_knee, right_ankle | joint_angle_abc | M1 | S3 | unvalidated/implemented |
| squat_knee_to_ankle_horizontal_left_ratio | left_knee, left_ankle | horizontal_offset_torso_ratio | M2 | S3,TBD-EXPERT | blocked/not_implemented |
| squat_knee_to_ankle_horizontal_right_ratio | right_knee, right_ankle | horizontal_offset_torso_ratio | M2 | S3,TBD-EXPERT | blocked/not_implemented |
| squat_torso_angle_vertical_deg | left_hip, right_hip, left_shoulder, right_shoulder | midpoint_axis_angle_vertical | M2 | S3,TBD-EXPERT | unvalidated/implemented |

Tất cả angles là degrees `[0,180]`, viewpoint side-only conservative S3; front,
oblique, unknown không được mở. Joint angles dùng đúng góc trong ABC giữa A−B
và C−B (B là knee/hip/elbow/shoulder theo bảng). Formula/method v1 được giữ nguyên.
Axis formula: `angle_between((LS+RS-LH-RH)/2, (0,-1))` trong hệ x-right/y-down.
HSPU body-line là **trục shoulder–hip**, không full-body/head/ankle alignment.
Directed inverted axis có thể gần180°, không flip để cho gần0°.

Axis primitive validate cả4 points trước zero check; Fraction của finite floats/ints
tránh midpoint overflow, subnormal underflow và integer cancellation. Internal
positive rescaling phục vụ arithmetic; không biến đổi dữ liệu input. Reuse joint_angle
để nhận degree finite/range. Zero hip_to_shoulder trả degenerate_geometry với
zero_length_vector; invalid point/type/NaN/Inf dùng invalid_geometry_input/input_error;
known arithmetic failure dùng geometry_numerical_error/measurement_error. Stage
features/S3, có primitive/reason/vector và actual ordered joint identities trong
observation flags; không số giả. Synthetic tolerance **abs1e-10 degrees, rel0**.

## 3. Global-vs-feature gate: rationale và giới hạn

Đặc tả §§2.1–2.2 tách sequence và feature gate: sequence missing/gap gate khóa
đếm/chu kỳ; low-visibility khóa metric bị ảnh hưởng; frame feature yêu cầu đúng
keypoints/finite/provenance/viewpoint. Vì vậy v2 thêm S3 framewise policy tường minh:

1. Exact feature metric key có eligible=true/action=allow và exact joint set/count.
2. Viewpoint trong canonical policy; joints của frame usable, finite xy và status
   observed hoặc explicit imputed với valid bracketing endpoint references.
3. Mọi metric degrade/skip vẫn unavailable; absent/wrong requirements không fallback
   từ movement_analysis. Adapter không tự tính QualityReport hoặc tune threshold.
4. Chỉ bỏ blocker **global** scope movement_analysis theo policy version mới. Không
   sửa report/sequence status, không đổi eligible=false thành true, không dùng một
   metric sạch để xác nhận toàn sequence. Output temporal_analysis_eligible=false và
   assessment_status=blocked_pending_expert_validation.
5. Legacy v1/knee APIs vẫn yêu cầu global+metric allow; snapshot/JSON compatibility
   được kiểm thử. Hai ratios blocked dùng policy sequence+metric và không có primitive.

Angle invariance cho phép dùng working units khi Stage4 scale fallback nếu chính
joints cần vẫn usable; distance/ratio không được dùng fallback/null scale, không
thay bằng1. Imputed giữ source_status/status/endpoints, không gọi là observed; genuine
raw-gap metric degrade tiếp tục chặn dù tọa độ được điền. Synthetic imputed-allow
fixture chỉ kiểm tra detached contract, **không phải genuine raw-gap eligibility**.

Đây là policy engineering mô tả framewise, chưa validated camera/physical accuracy.
Side assertion không tự chứng minh documented input. Fixed left/right IDs, không
bilateral fusion hay swap bên ngầm; chưa có temporal jump detector. Formal gate5.7
vẫn cần sequence + cả2 knee metrics allow, không được thay bằng v2 local result.

## 4. Visual audit và selection trước inference

Run root: `outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd`. Selection committed UTC:
`2026-10-12T09:40:03.400521+00:00`; inference_started=false trong frozen selection.
Selection byte SHA-256: `817a45288be619aaff513389cb64a31dcf76f602b724a1402a6bf0edf7fbb3a0`.
Hash chỉ nhận diện bytes, không chứng minh viewpoint/annotator authority.

Squat reuse full visual audit5.7A (112 hashes/560 distributed samples), giữ R106
đã chọn bằng hình ảnh **trước inference của lượt5.7A**; không đổi sang clip có pose
đẹp sau khi biết right/global failures. 111 squat hashes còn lại có rejected/not
selected rationale trong frozen selection; không giả là đã xem lại chúng ở lượt này.

HSPU inventory trong Data: 256 paths/128 byte-distinct contents. Candidate pool
quy định trước inference: first6 unique hashes sorted SHA lexicographically; mỗi
clip xem5 distributed frames floor((N−1)*[0,.25,.5,.75,1]). Đã xem 6 contact sheets,
30 real images; không suy viewpoint từ filename/class/pose. 3 front, 1 rear-facing
front/back projection (cùng category front), 1 likely_side, 1 uncertain. Chọn first
full-framing front control H002 và first clear lateral full-framing H004; không
expand/reselect pool sau quality/numeric outcomes. Pool không phải dataset-wide
HSPU viewpoint validation. Tất cả visually_assessed_unverified; confidence qualitative
không calibrated probability. Camera/body/framing/occlusion/confidence rationale và
sample hashes/times nằm trong hspu_visual_audit.json.

| Review | Video | Visual category | Confidence | Frames viewed | Selection rationale |
|---|---|---|---|---|---|
| H001 | v_HandStandPushups_g01_c01.avi | front | high | 0,19,38,57,77 | Rejected: frontal projection with crop; first full-framing frontal control is H002. |
| H002 | v_HandStandPushups_g18_c06.avi | front | high | 0,25,50,75,100 | Selected frontal control: first full-framing front/back clip in deterministic pool, independently of pose results. |
| H003 | v_HandStandPushups_g15_c03.avi | front | high | 0,26,53,79,106 | Rejected: another frontal close framing; redundant with retained H002 control. |
| H004 | v_HandStandPushups_g17_c03.avi | likely_side | high | 0,18,37,55,74 | Selected: first clearly lateral full-body clip in deterministic hash pool; all sample times consistent. |
| H005 | v_HandStandPushups_g25_c04.avi | front | high | 0,35,70,105,141 | Rejected: frontal category redundant with H002 control. |
| H006 | v_HandStandPushups_g08_c02.avi | uncertain | medium | 0,23,46,69,93 | Rejected: uncertain orientation and distant framing; not a side candidate. |

H004 near-profile camera, body faces image-left, consistent five times; far arm
overlap, slight oblique component possible; toàn body trong frame, fixed gym view.
H002 rear-facing symmetric arms/torso, frontal-plane projection, đầy đủ framing;
không claim side. Sampling không chứng minh mọi frame/camera calibration hoặc
form. Assumed viewpoint áp dụng toàn exploratory branch như giả thuyết có giới hạn;
không tạo protocol/annotator chính thức. Original branch viewpoint giữ unknown.

| Selected ID | Relative path | Byte SHA-256 | Exploratory assumption |
|---|---|---|---|
| R106 | Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g25_c01.avi | 893f7b0c89d10030705eeadfca1f9681a81190540bff1b2ab77d0ab3b614261e | side |
| H002 | Data/UCF-101/HandstandPushups/v_HandStandPushups_g18_c06.avi | 095fb71ca771c70d48b02dc281c199b3f92d2071636c535b97ebd655fd0bfe54 | front |
| H004 | Data/UCF-101/HandstandPushups/v_HandStandPushups_g17_c03.avi | 0fc473b97c35bbf04dcc7e279c2e3aba970f68982e7eba70d61d3d1e27627b2e | side |

Script checks selection SHA/path/ID/video bytes/evidence hash/link/view category
và viewed sample hashes trước decoder/backend. Có regression cho wrong identity,
missing review, path escape, mismatched samples/viewpoint/documented flag/duplicates.
Final stricter selection checks đã đọc lại đúng committed selection trong6 real
integration tests, không thay selection hoặc source data.

## 5. Fresh real pipeline và measured outcomes

Chạy mới ingestion → MediaPipe → QualityGate → Stage4 preprocessing → geometry
cho **mọi frame và cả5 angle features của exercise**, ở cả unknown và exploratory
branch. Không reuse QualityReport/pose cache. 3 distinct clips ×2 branches =6 runs,
103/101/75 frames mỗi branch, tổng558 decoded frames/2790 observations.
Mọi frame có pose (0 no-pose frames); có low visibility, không đồng nghĩa form error.
**534 available, 2256 unavailable, 0 failed_measurement; 0 documented inputs.**
Không loại failure cases khỏi báo cáo. Available ở nhánh exploratory không official
eligibility/ground truth/accuracy. Original unknown branches: all5 metrics skip,
R106515/H002505/H004375 observations unavailable (tổng1395), viewpoint_unknown và
feature_quality_blocked, kèm unusable/provenance flags ở weak joints nếu có.

Global sequence: R106 unknown+side **skipped/skip low_pose_visibility**; H004
unknown+side **skipped/skip low_pose_visibility**; H002 unknown+front **eligible/allow**.
H002 cả5 metrics vẫn skip do unknown hoặc front mismatch; clean pose coverage không
mở unsupported projection. v2 không làm global skipped thành eligible.

| Candidate exploratory | Feature | Metric action | Metric reasons | Available | Unavailable |
|---|---|---|---|---|---|
| R106 | squat_hip_angle_left_deg | allow | none | 103 | 0 |
| R106 | squat_hip_angle_right_deg | skip | low_pose_visibility | 0 | 103 |
| R106 | squat_knee_angle_left_deg | allow | none | 103 | 0 |
| R106 | squat_knee_angle_right_deg | skip | low_pose_visibility | 0 | 103 |
| R106 | squat_torso_angle_vertical_deg | allow | none | 103 | 0 |
| H002 | hspu_body_line_angle_deg | skip | viewpoint_mismatch | 0 | 101 |
| H002 | hspu_elbow_angle_left_deg | skip | viewpoint_mismatch | 0 | 101 |
| H002 | hspu_elbow_angle_right_deg | skip | viewpoint_mismatch | 0 | 101 |
| H002 | hspu_shoulder_angle_left_deg | skip | viewpoint_mismatch | 0 | 101 |
| H002 | hspu_shoulder_angle_right_deg | skip | viewpoint_mismatch | 0 | 101 |
| H004 | hspu_body_line_angle_deg | allow | none | 75 | 0 |
| H004 | hspu_elbow_angle_left_deg | allow | none | 75 | 0 |
| H004 | hspu_elbow_angle_right_deg | skip | low_pose_visibility | 0 | 75 |
| H004 | hspu_shoulder_angle_left_deg | allow | none | 75 | 0 |
| H004 | hspu_shoulder_angle_right_deg | skip | low_pose_visibility | 0 | 75 |

R106 right knee59/103 low, observed44, max invalid span29; right ankle11 low,
observed92/maxgap5. Right elbow73 low/maxgap35; right wrist31 low/maxgap26.
Left hip/knee/ankle, right hip và both shoulders103 observed. Right knee/hip metrics
skip cả103 frames vì metric gate toàn selected sequence; trong59 low frames có thêm
feature_provenance_rejected/feature_joint_unusable. Không promote44 clean frames.

H004 right elbow/wrist/knee75/75 low/maxgap75; right ankle72 low/3 observed/maxgap34.
Left elbow/wrist/shoulder, both hips/shoulders75 observed nên left2/body-axis metrics
allow. Right elbow/shoulder75 unavailable có quality/provenance/unusable reasons.
H002 mọi13 joints101 observed; toàn505 exploratory observations vẫn unavailable
viewpoint_mismatch + feature_quality_blocked. Không dùng là form claim.

Measured angle ranges dưới **unverified visual hypotheses**, chỉ descriptive
stats của processed coordinates (rounded6 decimals trong bảng, full floats JSON):

| Candidate | Feature | Min deg | Max deg | Available frame count |
|---|---|---|---|---|
| R106 | squat_hip_angle_left_deg | 65.445064 | 172.659035 | 103 |
| R106 | squat_knee_angle_left_deg | 67.803477 | 179.066945 | 103 |
| R106 | squat_torso_angle_vertical_deg | 0.115348 | 28.931840 | 103 |
| H004 | hspu_body_line_angle_deg | 154.664498 | 163.576476 | 75 |
| H004 | hspu_elbow_angle_left_deg | 112.129313 | 146.249543 | 75 |
| H004 | hspu_shoulder_angle_left_deg | 110.339673 | 123.012356 | 75 |

Không synthetic values trong bảng real này; không so với góc kỹ thuật đúng/sai.
No missing raw joint ở subset này, không imputed available; missing/imputed/zero/
NaN cases được kiểm thử bằng synthetic contracts tách biệt. Raw visibility/missing
frame indices và spans ở raw_pose/quality JSON; mọi per-observation joint snapshots,
reasons/flags ở features.jsonl, không chỉ summary.

## 6. Model/config/profile và artifacts

- MediaPipe0.10.14 `mediapipe.solutions.pose.Pose`, bundled full modelcomplexity1;
  no download/framework/model mới. Model SHA
  `e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f`.
- Mapping mediapipe-pose-13-v1, native image_normalized_xy/x-right/y-down;
  single most-prominent subject, không multi-person/identity guarantee.
- Pose config pose-extraction-v1, visibility/detection/tracking .5, smooth false;
  hash `3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976`.
- Quality quality-s3-v1: min5 frames, missing/low .30, maxgap10 unchanged;
  hash `514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704`. Exact requirements/viewpoints từ registry v2.
- Preprocessing preprocessing-s3-v1, hash `33edf08cdafea19ef0ae67b67ce02b9f53b1b1764641bc9df7af58259d387c33`;
  x,y*height/width working units, provenance/normalization/finite-run smoothing cũ.
- All3 clips FPS25, resolution320×240; timestamp fps_derived_cfr_assumption,
  index/FPS bảo toàn. Decoder đo metadata; agent visual evidence là nguồn riêng.

Mỗi branch lưu raw_pose.jsonl, metadata.json, pose_summary.json, quality.json,
preprocessing.json, features.jsonl, frame_links.jsonl, first/last full linked
contexts và summary.json. Frame links nối input_hash tới raw_sequence/raw_frame/
processed/source/profile/quality/metadata hashes; factory validates complete source
mỗi frame trước dùng detached context để tránh sao chép full sequence cho5 features.
Quality/raw/preprocessing snapshot hashes được assert unchanged sau calculation.
Manifest run có selection hash/registry/environment/command. Summary có byte hashes
mọi branch artifact trước chính summary; inventory root chứa cả summary hashes.

| Summary artifact | Byte SHA-256 |
|---|---|
| R106_original_unknown | 1218c094b6c2f7b70d836e1e90ffeca3aa95cba0f5ad23bcfe610dd5d300d1dd |
| R106_exploratory | a933ffced4f768d531f68e927b1a47022529a06ac01f3d3cad4b9d3b89856f45 |
| H002_original_unknown | f1a201d2f3f56f2913ff0a8872735192263bae18eb0206b22dfacb129c83266f |
| H002_exploratory | 7dcc09d5ab9858f7680ade1be8aca75aabb9c1552a19e456c3c6123bea1758fa |
| H004_original_unknown | 51044c3fe99e232b7d1cdffbfeb4f0fa0d42fc730783f6ae150e4871ba582ae4 |
| H004_exploratory | d195adecaf92556c41f933d56c8006a95f268886b07bf41f39147929e06136ec |

Execution script snapshot `audit_script_executed_attempt1.py` SHA
`1deb45fa2dbb4e2c683c712ae1817c8639d610dbd0f47d677f59987566cb59cd` lưu code đúng lúc audit chạy.
Sau audit đã sửa guard selection missing-review message và tăng checks sample/viewpoint,
không thay branch calculation/quality hoặc source selection. Final script source hash
và schema/test/doc/source hashes nằm trong source_hashes.json/artifact_inventory.json.
Không backfill/sửa manifests hoặc summary outputs của attempt1 thành run mới.

## 7. Commands, test evidence và môi trường

Local Python 3.11.9, Windows-10-10.0.26200-SP0. Dependencies thực tế:
`{"ai-coach-gym": "0.6.0", "pydantic": "2.13.5", "pytest": "9.1.1", "setuptools": "84.0.0", "mediapipe": "0.10.14", "numpy": "2.4.6", "scipy": "1.17.1", "opencv-contrib-python": "5.0.0.93", "protobuf": "4.25.9", "Pillow": "12.3.0"}`. Pillow đã có chỉ để tạo contact sheets,
không thêm vào runtime manifest. Không thêm dependency, `pip check` exit0:
No broken requirements found. Không tuyên bố clean-environment compatibility.

Editable install `python -m pip install --no-index --no-deps --no-build-isolation -e .`
ban đầu failed WinError5 trong sandbox TEMP lúc rename metadata. Rerun cùng lệnh với
TEMP/TMP trỏ run-local pip-temp đã exit0, installed ai-coach-gym0.6.0, không network.
Không cài framework/model khác. Đây là lỗi quyền filesystem, không dependency conflict.

Fresh audit command tại workspace root (PowerShell; output mới/exclusive):

```powershell
& .venv/Scripts/python.exe -B scripts/audit_stage5_angles.py `
  --selection outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/selection.json `
  --selection-sha256 817a45288be619aaff513389cb64a31dcf76f602b724a1402a6bf0edf7fbb3a0 `
  --output outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/development-attempt1
```

Đã chạy exit0, cả6 branches completed. Sum branch elapsed **202.06s**,
không gọi đây là wall time toàncommand. Reproduce bằng output directory khác,
không rerun lệnh vào attempt1 để ghi đè. Audit1116 Python UserWarning records vì
filter always (2/frame ×558) về deprecated Protobuf GetPrototype; native12 feedback
manager warning lines, một routing warning và một XNNPACK INFO riêng với pytest.
Không suppression hoặc trình bày audit warnings như feature failure/accuracy.

Tests final dùng explicit env `GYM_STAGE5_AUDIT_DIR=outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/development-attempt1`.
6 integration tests read back/verify fresh audit generated above, không claim pytest
đã inference subset lần nữa; full suite còn chạy existing real pose/preprocessing
integrations từ đầu. Nếu env thiếu6 readback tests skipped có reason, không pass claim.

| JUnit artifact/run | Pass | Fail | Error | Skipped | Warnings | Deselected | Seconds (JUnit) |
|---|---|---|---|---|---|---|---|
| unit-attempt1.xml | 173 | 1 | 0 | 0 | 0 | 0 | 13.759 |
| unit-attempt2.xml | 174 | 0 | 0 | 0 | 0 | 0 | 6.249 |
| unit-final.xml | 184 | 1 | 0 | 0 | 0 | 6 | 9.367 |
| unit-corrected.xml | 185 | 0 | 0 | 0 | 0 | 6 | 6.841 |
| new-tests-final.xml | 243 | 0 | 0 | 0 | 0 | 0 | 22.280 |
| related-tests.xml | 627 | 0 | 0 | 0 | 0 | 0 | 61.826 |
| full-suite.xml | 956 | 0 | 0 | 0 | 4 | 0 | 126.286 |

Hai intermediate failures được giữ artifact: first imputed test dùng string trong
unsafe model_copy thay enum, đã sửa fixture; second missing visual review raised
StopIteration không message, khiến structured flag validation fail; sửa explicit
message và regression đạt. Không che failed attempts bằng kết quả dự kiến.
Final CLI summaries: **243 passed in22.28s**, **627 passed in61.83s**,
**956 passed,4 warnings in126.30s**, all exit0,0 failed/error/skipped.
Bốn pytest warnings là SymbolDatabase.GetPrototype deprecation trong hai old pose
và hai old preprocessing integrations, không phát sinh warning mới ở243 tests.

Exact final test commands (thêm JUnit paths riêng, chưa overwrite):

```powershell
$env:GYM_STAGE5_AUDIT_DIR = 'outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/development-attempt1'
& .venv/Scripts/python.exe -B -m pytest tests/test_axis_geometry.py tests/test_geometry_features.py tests/test_geometry_development_audit.py -q --junitxml=outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/new-tests-final.xml
& .venv/Scripts/python.exe -B -m pytest tests/test_geometry.py tests/test_features.py tests/test_feature_observation.py tests/test_feature_context.py tests/test_right_knee_feature.py tests/test_development_eligibility.py tests/test_development_inventory.py tests/test_quality.py tests/test_preprocessing.py -q --junitxml=outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/related-tests.xml
& .venv/Scripts/python.exe -B -m pytest -q --junitxml=outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd/full-suite.xml
```

test_evidence.json lưu cả commands/timestamps/counts/time của initial/unit/final
attempts và env; full tests713 existing +243 new =956. Tests kiểm tra known0/45/90/180
mọi angle, ordered joints/units/ranges/primitive, coincidence, translation/scale,
axis extreme finite/subnormal/huge integers, zero/nonfinite/type/null/provenance,
imputed references, exact quality/viewpoint gates, no global promotion, hashes,
JSON/legacy APIs, immutability, selection linkage và lightweight public imports.

## 8. Acceptance, bảo toàn và bước tiếp theo

| Acceptance | Kết quả |
|---|---|
| Đủ10 canonical unblocked angle calculations | PASS code + each-feature synthetic tests |
| Calculation/policy tách, no fake null measurement, no form thresholds | PASS |
| Global/feature split versioned, exact allow, no promotion | PASS engineering contract; chưa scientific validation |
| Finite geometry/errors/provenance/hash/JSON/light import | PASS |
| Legacy schemas/APIs + Stage0–5.7A tests | PASS956 tests |
| Frozen visual selection, fresh real pipeline/all outcomes | PASS exploratory audit,0 documented |
| Current docs/report + artifact hashes | Đã tạo/cập nhật theo evidence |
| Stage5.7 documented protocol + sequence/bilateral gate | BLOCKED: không official annotation, right/global fail ở side candidates |
| Camera/physical accuracy/form ground truth/expert validation | BLOCKED, chưa được đo hoặc claim |
| Whole Stage5 | IN_PROGRESS / VALIDATION_BLOCKED |

Baseline1745 scoped files trước sửa:1738 unchanged,7 modified đúng phạm vi
(README + plan/state/tasks/schema và source _version/features __init__),0 missing. Danh sách
chính xác/hashes trong preservation_audit.json; **README + plan/state/tasks/schema
là5 current documents**, hai source changes là packageversion/publicexports.
Mọi old tests/code khác, Data MVP video bytes, old outputs và historical reports
trong baseline giữ hashes. Archive/venv/cache ngoài snapshotscope, không sửa/xóa
archive/dataset; installation chỉ cập nhật distribution trong existing venv.
Không tạo Git artifacts. Source/output inventories dùng hashes, không giả commit.

Giới hạn: synthetic tests chứng minh arithmetic/contract, không accuracy; chỉ3 clips,
6 HSPU visual reviews, viewpoint sampling unverified, no camera calibration/3D/GT,
far-side occlusion/inversion risk/single-subject limits; ratio và temporal analysis
blocked. Factory full-source validation O(N) mỗi frame, audit hiện tổng O(N²),
chưa tối ưu streaming feature sequence. Không chọn lọc outcome hoặc mở assessment.

Bước nhỏ nhất tiếp theo là đóng **documented input linkage** bằng protocol/source
ID/version, confirmer/date, exact video path/ID/SHA, frame/timestamp interval và
camera/framing/occlusion; rồi chạy fresh official gate theo5.7 không đổi thresholds.
R106 right/global quality vẫn fail nên documentation đơn lẻ không đủ. Đây là
remaining data-validation requirement, không yêu cầu người dùng xác nhận để chặn
phần code trong lượt này. Không tự tạo annotation/ground truth để mở Stage6–8.
