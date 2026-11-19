# Giai đoạn 5.6 — QualityGate eligibility và development side-view inventory

Ngày kiểm tra: **2026-10-10, Asia/Saigon**. **GĐ5.6 COMPLETE (eligibility/data
audit)**; toàn **GĐ5 IN_PROGRESS**. Numeric real-video measurement vẫn
**BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**. Audit hoàn tất không đồng nghĩa
đã có dữ liệu thật đủ điều kiện hoặc đã validation feature accuracy/form.

## Phạm vi và implementation

Đã đọc/đối chiếu 10 docs yêu cầu: README, technical spec, master prompt, plan,
state, tasks, schema conventions và reports GĐ5.3/5.4/5.5; inspect current
QualityGate/profile, registry/policies, linked context/factory, adapters/tests.
Workspace không Git repository; dùng hashes, không giả commit/diff. Trạng thái
legacy `.agent`/dataset reports không được coi như execution evidence mới.

Thêm bốn files:

- `tests/stage5_6_fixture.py`: deterministic test-only PoseFrame/metadata fixture,
  actual QualityGate report và existing preprocessing, không sửa eligibility.
- `tests/test_development_eligibility.py`: 37 synthetic unit/contract tests.
- `tests/test_development_inventory.py`: 2 real inventory/source integrity tests.
- `scripts/audit_stage5_6_development.py`: read-only inventory CLI, JSON/CSV vào
  directory mới, không overwrite hoặc decode/infer/measure video.

Cập nhật README/IMPLEMENTATION_PLAN/PROJECT_STATE/TASKS/SCHEMA_CONVENTIONS;
thêm báo cáo này và verification JSON. **Không sửa src/gym_coach, config,
pyproject, dependency hoặc package version**: vẫn **0.5.4**. Registry/policy/
quality/preprocessing/context/primitive/left-right adapters và numeric schemas
giữ nguyên. Không feature ID/calculator/batch extractor/FeatureSequence, fusion,
rep/phase/segmentation, form rule/threshold/score/feedback/training mới.

## Genuine synthetic QualityGate evidence

Profile **quality-s3-v1**, S3, hash
`514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704`:
minimum frames5, max missing0.30, max low-visibility0.30, max invalid run10.
Không relax/sửa thresholds. Preprocessing dùng existing TOML
`configs/preprocessing_s3_v1.toml` / preprocessing-s3-v1.

Fixture tạo **31 frames**, index0..30, timestamp=index/FPS tại **30 và 60 FPS**,
video ID `stage5_6_synthetic_contract.avi`, metadata320×240, side được khai báo
trong construction. 13 joints observed/visibility0.9/in-frame. Hips (.4,.7) và
(.6,.7), knees (.4,.4) và (.6,.4), ankles (.2,.4) và (.8,.4); shoulders (.4,.3),
(.6,.3), các joints còn lại từ deterministic shared test builder.
Đây là các điểm toán học cố định, không mô phỏng một lần squat hoặc người tập.
Backend identity của PoseFrame ghi test; không MediaPipe inference trên fixture.

`QualityGate(QualityProfile()).evaluate(raw, metric_requirements=...,
metric_viewpoints=...)` dùng hai canonical definitions để lấy mappings:

| Scope/key | Required joints theo thứ tự | Eligible / action / reasons đã đo |
|---|---|---|
| movement_analysis / sequence | left/right shoulder, hip, knee, ankle | true / allow / empty |
| squat_knee_angle_left_deg | left_hip, left_knee, left_ankle | true / allow / empty |
| squat_knee_angle_right_deg | right_hip, right_knee, right_ankle | true / allow / empty |

QualityReport **do QualityGate thật tạo**, không tự dựng MetricEligibility,
không chỉnh report sau evaluation. Mappings gồm đúng hai stable keys, explicit
viewpoint policy side. Existing preprocessing → factory → readiness → adapters
ở selected indices0/15/30 đạt; primitive gọi đúng hai lần (một lần mỗi bên).
Hai synthetic outputs **90.0°**, tolerance absolute **1e-10 degrees**, rel=0,
direct primitive equality, measurement_validation_status **unvalidated** và
assessment blocked. Left schema0.2.0/right0.3.0 giữ nguyên. Không pose accuracy,
form correctness, real movement hay measurement validation claim.

Full-suite synthetic artifacts:
[summary](../outputs/stage5_6/synthetic-b2aed6886d274599afc9c6c049dd4afd/summary.json).
Directory có raw JSONL, quality/processed/context JSON và từng observation;
summary ghi evidence_type=synthetic_contract_test, real_video=false và byte hashes.
Initial new-test artifacts tại synthetic-a91d55636e5a4cbd9e8f9582996ae6e5 giữ nguyên.

Negative tests dùng reports **từ genuine gate**, không promotion:

- Global sequence allow nhưng key thiếu/sai/chỉ key bên kia → unavailable.
- Wrong joint set sạch có thể QualityGate allow cho set caller khai báo; readiness
  vẫn feature_quality_policy_mismatch vì không khớp canonical required joints.
- Unknown/front/oblique: generic movement có thể allow, exact knee metrics skip.
- Short sequence, low visibility/missing/out-of-frame: skip/degrade được giữ;
  một shoulder hỏng có thể chặn sequence dù knee metrics riêng vẫn allow.
- Interior missing gap được GĐ4 impute usable vẫn không nâng raw quality degrade;
  endpoint missing/low/OOF joints không usable bị khóa. Spy primitive calls=0.
- Spy chặn QualityGate evaluate sau report construction: preprocessing audit/
  factory/adapters/readback không tính lại eligibility. Inputs/config/registry/
  quality/raw/processed snapshots giữ nguyên; deterministic rerun equality đạt.

## Inventory real development inputs

Scope: mọi video squat trong **Data/** (class directory hoặc squat filename),
mọi non-video dataset metadata hiện có và existing Stage4 squat report artifacts.
Không xem archive, không chọn bài khác làm candidates, không chạy pose inference
mới toàn dataset hoặc tạo numeric real-video measurement. **Không visual review**:
not_performed; không suy viewpoint từ action label, filename/group, subject/pose
result, coverage hoặc heuristic image quality.

Đã đọc metadata/protocol fields và đối chiếu nguồn hiện hữu:

| Nguồn Data/UCF-101_GYM | Evidence tìm được / giới hạn |
|---|---|
| DATASET_STATUS.md | Action classes và historical heuristic stats; không camera annotation |
| audit/UCF101_GYM_AUDIT_CONFIG.md | Ghi camera_view explicitly UNKNOWN, automated blur/lighting/readability, no pose model |
| audit/UCF101_GYM_VIDEO_AUDIT.csv | 112 squat rows, tất cả camera_view UNKNOWN; không viewpoint protocol |
| audit/UCF101_GYM_PROBLEM_VIDEOS.csv | 0 squat rows; heuristic problem subset, không annotation |
| audit/UCF101_GYM_CLASS_SUMMARY.csv | Một aggregate squat row; không per-video viewpoint |
| audit/UCF101_GYM_AUDIT_SUMMARY.md | Historical aggregates, không per-video camera confirmation |
| audit/summary_data.json | Aggregate heuristic totals, không camera annotation |

Seven metadata file bytes được hash và giữ nguyên. Historical GOOD/USABLE values
chỉ lưu như heuristic metadata; không QualityGate evidence, không mới xác minh
readability/pose suitability. Source CSV path trong inventory explicit; original
dataset copies không được coi như có annotation riêng từ cùng filename.

| Measured inventory | Kết quả |
|---|---:|
| Data/UCF-101/BodyWeightSquats files | 112 |
| Data/UCF-101_GYM/BodyWeightSquats files | 112 |
| Candidate paths | 224 |
| Unique byte SHA-256 / duplicate pairs | 112 / 112 |
| Filename group IDs (không verified participant IDs) | 25 |
| Documented side-view candidates | 0 |
| Eligible candidate paths | 0 |
| Ineligible paths (recorded quality fails) | 1 |
| Blocked paths (thiếu documentation/report trực tiếp) | 223 |

Từng candidate có relative path/video ID/hash, metadata source/camera_view,
evidence_type=heuristic_metadata_unverified, viewpoint unknown, documented=false,
visual_review=not_performed, existing report summaries và decision/reasons:

- [JSON inventory đầy đủ](../outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/real-inventory/inventory.json).
- [CSV từng candidate](../outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/real-inventory/candidates.csv).

CLI sau full suite đối chiếu **14 existing Stage4 squat report artifacts**, tất
cả ghi cùng direct source path
`Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi`.
QualityReport strict schema/profile/hash và metadata video identity đã kiểm tra;
report source files/hashes lưu trong inventory. **Đọc decisions đã lưu**, không
reevaluate/promote chúng. Sequence skip/viewpoint unknown, hai knee keys absent.
Source path này **ineligible** bởi quality và vẫn thiếu documented side-view.
Original copy cùng content hash không được tự gán report linkage/eligibility.
Đây không phải 14 independent videos hoặc 14 validated runs/ground truth.

Saved GĐ4 source được kiểm tra thêm qua strict full PreprocessingResult tại
frames0/133/266: cả hai adapters đều unavailable/null, primitive calls=0,
report và bytes giữ nguyên. GĐ5.6 không mới infer video cho experiment numeric;
full suite vẫn chạy bốn fresh upstream pose/preprocessing integrations như trước.
Không báo sáu outcomes unavailable như full-video feature coverage hoặc form error.

Kết luận inventory: **BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**. Điều này chỉ
phản ánh nguồn được kiểm tra, không nói các video chắc chắn không quay bên cạnh.
Nếu chỉ xem bằng mắt sau này, status phải visually_assessed_unverified và không
documented. Script liệt kê metadata bổ sung chưa review và giữ blocked; không
generic annotation parser/auto-approval cho input mới.

## Input cần cung cấp cho bước tiếp theo

Một video squat kèm record gắn **đúng video path/ID/content SHA-256** và:

1. Exercise BodyWeightSquats, mục đích development, participant/group context
   và split provenance; không suy participant độc lập từ filename hoặc copies.
2. Annotation **viewpoint=side** cho interval frames/timestamps áp dụng, protocol
   ID/version hoặc tài liệu nguồn, annotator/người xác nhận và ngày xác nhận,
   mô tả rõ camera nhìn ngang và xác nhận orientation. Không chỉ một enum side
   trong pose report hoặc nhận xét visual chưa được protocol xác nhận.
3. FPS/resolution/timestamp policy; camera cố định và framing đủ vai/hông/gối/
   cổ chân, ghi occlusion/in-frame limitations. Đây là input protocol mô tả,
   không quy tắc form hoặc bảo đảm QualityGate sẽ allow.
4. Sau documented input mới tạo pose/QualityGate report mới từ nguồn đó với
   canonical hai knee requirements/viewpoints. Sequence và cả hai metrics phải
   allow; nếu degrade/skip/absent thì giữ blocked/unavailable, không sửa report.

Không cần hoặc suy form-good/bad labels cho phép đo descriptive này. Protocol
input và quality allow cũng chưa xác nhận measurement accuracy.

## Tests, môi trường và preservation evidence

Môi trường local đo được: CPython **3.11.9**, Windows x64 10.0.26200, Pydantic
**2.13.5**, pytest **9.1.1**, setuptools **84.0.0**; package **0.5.4** giữ nguyên.
Existing backends/dependencies: MediaPipe0.10.14, NumPy2.4.6, SciPy1.17.1,
OpenCV contrib5.0.0.93, legacy opencv-python5.0.0.93, protobuf4.25.9.
Không install/model/framework/dependency mới; pip check exit0. Không tuyên bố
clean-environment compatibility hoặc giải quyết dual OpenCV bằng audit này.

PowerShell tại project root, lệnh đã chạy:

```powershell
& .venv/Scripts/python.exe -m pytest -q tests/test_development_eligibility.py tests/test_development_inventory.py --junitxml=outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/new-tests.xml
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/full-suite.xml
& .venv/Scripts/python.exe scripts/audit_stage5_6_development.py --output-dir outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/real-inventory
& .venv/Scripts/python.exe -m pip check
```

| Run | Passed | Failed | Errors | Skipped | Warnings | Terminal time / exit |
|---|---:|---:|---:|---:|---:|---|
| New: 37 synthetic units + 2 real audit/integrity tests | 39 | 0 | 0 | 0 | 0 | 17.40 s / 0 |
| Full GĐ0–5.6: 674 retained + 39 new | 713 | 0 | 0 | 0 | 4 | 100.53 s / 0 |
| Read-only CLI inventory | — | — | — | — | — | exit0, blocked conclusion như bảng |
| pip check | — | — | — | — | — | exit0, No broken requirements found. |

Cả hai test runs đạt ngay, không repair/suppression. Bốn UserWarnings protobuf
SymbolDatabase.GetPrototype deprecated tại google/protobuf/symbol_database.py:55
đến từ hai pose và hai fresh preprocessing MediaPipe integrations GĐ2/GĐ4.
JUnit counts/time và warning/terminal time được ghi riêng trong
[verification evidence](artifacts/stage5_6_verification.json).

Before manifest **737 files**: **732 byte-unchanged**, đúng **5 current docs sửa**,
**0 missing/deleted** trong audit; **415 old output files** và **231 Data files**
(224 squat videos +7 metadata) byte-preserved. Toàn src/config/pyproject/.agent
và mọi existing test GĐ0–5.5 không sửa. Archive không đọc/sửa/xóa. Cache files
ngoài source/data audit có thể được pytest/interpreter tạo như các runs trước.
New artifacts dùng riêng UUID directories và exclusive files, không overwrite.

Source/test/script/config manifest **67 files**, SHA-256
`8979813fa22b0401b2e6e33df3c2f26a712970d3e9e856bc1ed29df0f7a7eb25`.
Hash của sorted POSIX relative_path:byte_SHA256 + LF; scope lần này có thêm
scripts/*.py, không docs/outputs/cache/data/archive. File identity không là chữ ký
hay accuracy. Before manifest/JUnit/file hashes có exact paths trong evidence.

## Definition of Done và giới hạn

| Yêu cầu | Bằng chứng / trạng thái |
|---|---|
| Genuine gate exact sequence + two knee allow | Real QualityGate, canonical mappings, 30/60 FPS fixture PASS |
| No global/key/joint/viewpoint/skip/degrade/usability bypass | Negative policy/primitive/evaluate spies PASS |
| Synthetic numeric remains unvalidated, source immutable | Fixtures, roundtrip/raw/report/config/registry checks PASS |
| Current squat inventory + viewpoint/quality evidence/reasons | 224 candidate rows, 7 metadata reviews, 14 saved reports PASS |
| Real report preservation, no fake documented/allow | Saved full-result checks + Data/output hash audit PASS |
| New + retained full suite, docs/measured evidence | 39/713 passed, current docs/JUnit/report/inventory PASS |
| Missing documented input leaves real measurement blocked | Explicit blocked state and user input checklist PASS |

Synthetic allow chứng minh contract path khả thi, không chứng minh real pose
đủ coverage hoặc accuracy. 2D projection, viewpoint/occlusion, profile thresholds
engineering chưa validation; UCF101 chỉ action labels. Hash/report consistency
không authenticate producer hay exact raw/video origin. Linked context O(N),
numeric schemas/remaining eight angle calculators và blocked ratio candidates
còn giới hạn GĐ5.5. Không tune/freeze dataset hoặc chọn evaluation video.

Bước nhỏ nhất tiếp theo: cung cấp/ghi một documented side-view squat input theo
checklist, rồi mới chạy pose → fresh exact-key QualityGate cho development.
GĐ5.6 audit COMPLETE; real numeric measurement BLOCKED, toàn GĐ5 IN_PROGRESS,
GĐ6–7 chưa bắt đầu và assessment vẫn blocked pending expert validation.
