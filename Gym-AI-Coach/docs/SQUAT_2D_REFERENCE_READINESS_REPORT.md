# Squat2D reference toolkit — readiness và evidence

2026-10-21, Asia/Saigon. **TOOLKIT IMPLEMENTED / REFERENCE_PENDING**.
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Không Git repository; không giả commit/diff. Không dependency mới.

## Sản phẩm và cách dùng

Run mới:
`outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/`.
[Hướng dẫn chạy](../outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/HUONG_DAN_SU_DUNG.md),
[protocol khóa trước](SQUAT_2D_REFERENCE_PROTOCOL.md),
[blank template](../outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/package/annotation_package/annotations.blank.json),
[manifest ảnh](../outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/package/annotation_package/manifest.json),
[kết quả pending thực chạy](../outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/evaluation-final-pending/evaluation.json).

`package/annotation_package/` chứa60 PNG native320×240, protocol, manifest,
template và guide. Chỉ giao thư mục này cho người gán nhãn mù. Không giao
`evaluation_inputs/` hoặc sampling private chứa gate metadata. Template trắng
không phải nhãn thực: toàn bộ coordinates/person/method/time còn null/pending.
Giữ nguyên originals/template/manifest; lưu bản đã điền ở file mới.

```powershell
# Chọn output chưa tồn tại. Thay file annotation bằng bản nhãn độc lập thật.
& .venv/Scripts/python.exe -B -m gym_coach.reporting.squat_reference --package outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/package --annotations ANNOTATIONS_COMPLETED.json --output-dir outputs/squat_2d_reference/EVALUATION_RUN_MOI
```

Builder CLI `gym_coach.reporting.squat_reference_package --output-dir NEW_PATH`
chỉ đọc fixed batch/replay và video squat hiện có; từ chối output tồn tại.
Evaluator CLI kiểm tra package hashes trước annotations, rồi ghi evaluation.json
vào thư mục mới. Không inference/preprocess lại/tính lại góc model hoặc gate.

## Kiểm kê tham chiếu thực tế

`reference-evidence-inventory.json` và `reference-search-supplement.json` ghi
nguồn, hash, phân loại và giới hạn tìm kiếm. Dataset:112 video paths trong đúng
BodyWeightSquats,0 non-video sidecar tại class folder;7 metadata files chung,
chỉ dùng rows squat.112/112 rows có camera_view/person_visibility/occlusion/
framing UNKNOWN. Config audit nói rõ đây là automatic heuristics, không joint
annotation. Không dùng GOOD, nhãn action hoặc UNKNOWN làm nhãn accuracy/form.

Đối chiếu source.json của đủ40 clips, saved raw/quality/preprocessing và các
báo cáo hiện hành: viewpoint của agent là visually_assessed_unverified,
documented=false; raw pose là MediaPipe prediction, processed là model-derived.
Không source nào trong scope này cung cấp independent six-joint pixel labels
với protocol/person/frame/video hash linkage. Tìm tên metadata/reference/
annotation/protocol ngoài Data/archive/venv/cache chỉ thấy code/tests/specs,
presentation-reference manifest của prototype và synthetic temporal protocols.
Presentation reference là tham khảo UI, không Gym label; không mở lại media đó.
Synthetic analytic references là đáp án test phần mềm, không nhãn video thật.

**0 qualified reference documents; REFERENCE_PENDING.** Không tự tạo tọa độ,
người xác nhận hoặc nhãn form. Không có human/repeated/independent annotation,
disagreement measurement hay adjudication đã thực hiện. Đây là kết quả tìm
trong các nguồn ghi inventory, không khẳng định ngoài scope không thể có nhãn.

## Identity và sampling trước extraction

Nguồn batch `outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f`;
replay `outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20`.
Kiểm tra1.033+567 existing run-file hashes/sizes theo inventories;452 batch
manifest files, replay manifest và current registry/profile identity. Full strict
Stage4/raw unknown-versus-assumed/source/quality/config validation cho40 clips.
Selected120 observations được typed readback và exact match baseline/replay,
observation hash/side/frame/time/source-video SHA đúng. Không dùng claim cũ
thay cho kiểm tra bytes trong lượt này. Selected12 source video byte hashes
kiểm tra trước và sau decode; final integrity kiểm tra thêm nguồn fixed40.

Sampling.json lưu trước decode, extraction_started=false/evaluation_started=false,
reselection_allowed=false. SHA-256:
`59714b3f9dd6dcd7467376dafb78953db014ca040ead00c4053036d24996beb2`.
Protocol bytes/hash, nguồn/config inventories, kích thước/FPS/framecount/group/
video hashes và exact indices có trong manifest. Một clip mỗi group, tối đa12;
phủ missing actions trước rồi path; floor((N−1)*k/4),k0..4. Không chọn bằng
ảnh đẹp, số đo hoặc error. gXX không phải subject identity đã xác nhận.

| Clip | Group | Frames gốc | Frame indices đã khóa | L / R gate toàn clip |
|---|---|---:|---|---|
| v_BodyWeightSquats_g01_c01.avi | g01 |267|0,66,133,199,266|allow / skip|
| v_BodyWeightSquats_g04_c03.avi | g04 |88|0,21,43,65,87|skip / degrade|
| v_BodyWeightSquats_g05_c01.avi | g05 |71|0,17,35,52,70|degrade / allow|
| v_BodyWeightSquats_g07_c01.avi | g07 |123|0,30,61,91,122|allow / skip|
| v_BodyWeightSquats_g09_c05.avi | g09 |229|0,57,114,171,228|skip / degrade|
| v_BodyWeightSquats_g10_c03.avi | g10 |152|0,37,75,113,151|skip / allow|
| v_BodyWeightSquats_g12_c01.avi | g12 |237|0,59,118,177,236|allow / skip|
| v_BodyWeightSquats_g13_c01.avi | g13 |112|0,27,55,83,111|skip / allow|
| v_BodyWeightSquats_g15_c01.avi | g15 |142|0,35,70,105,141|allow / skip|
| v_BodyWeightSquats_g19_c03.avi | g19 |144|0,35,71,107,143|skip / allow|
| v_BodyWeightSquats_g20_c05.avi | g20 |136|0,33,67,101,135|allow / skip|
| v_BodyWeightSquats_g23_c03.avi | g23 |73|0,18,36,54,72|degrade / degrade|

12/40 clips,12 groups,24 clip-side pairs:9 allow/5 degrade/10 skip.60 sampled
frames,360 landmark slots,120 angle slots; không dùng full40/5.314 frames làm
mẫu số của subset. Decode toàn1.774 frames của selected12, xuất60 đúng native
pixels; PNG readback exact pixel equality, không crop/stretch/overlay/rotation.
25FPS320×240; timestamps=index/25 theo CFR assumption, không VFR PTS.

## Contract/evaluator và kết quả thực chạy

Additive reporting-only schema0.1.0, protocol squat-blinded-pixel-reference-v1,
toolkit squat-2d-reference-v1. Không sửa numerical0.4/session/source contracts.
FrameIdentity ràng buộc video/path/SHA/frame/time/dimensions/image SHA. Exact6
anatomical keys, visible finite xy trong miền pixel centers; unavailable/null
và reason. Completed yêu cầu annotator/method/time có timezone, blinded/
independent provenance note; pending giữ null. Synthetic label bị từ chối trên
real package. Source declarations không thay chữ ký/đánh giá chất lượng nhãn.

Pixel convention: ref(0..W−1,0..H−1), raw chuyển x_norm×W,y_norm×H, không
clamp/out-of-frame thành hợp lệ. Raw observed/in-frame từng joint dùng Euclid
pixel error; processed_angle chỉ dùng saved available canonical-side degrees
và metric allow. Không tính raw diagnostic angle. Reference angle trên native
pixel coordinates, đúng aspect ratio; vector zero => unavailable/reason.
Không tự mở lại metric degrade/skip, không nâng sequence hoặc official gates.

MAE/median/p90/n theo clip-side/by-side và overall micro; macro trung bình
nonempty clip-side MAEs, báo số nhóm. p90 linear tại0.9(n−1). Model/reference
availability độc lập; lý do multi-label, không cộng chúng thành số mẫu độc lập.
Không có accuracy acceptance threshold. Không coi smoothing tốt hơn mặc định.

Lệnh thực chạy dùng blank template, có receipts riêng cho initial/final:

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.reporting.squat_reference --package outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/package --annotations outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/package/annotation_package/annotations.blank.json --output-dir outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/evaluation-final-pending
```

**REFERENCE_PENDING,0 completed/60 pending frames.**

| Nhánh | Denominator | Model available / unavailable | Reference available / unavailable | Comparable n | MAE / median / p90 |
|---|---:|---|---|---:|---|
| Raw landmark pixel |360 joint slots|292 /68|0 /360|0|null /null /null|
| Processed angle degree |120 frame-side slots|45 /75|0 /120|0|null /null /null|

Raw68 unavailable đều low_visibility;292 observed/in-frame không chứng minh đúng
landmark. Processed inputs292 observed/68 low_visibility,0 imputed trong sampled
slots.75 unavailable đều feature_quality_blocked;46 đồng thời provenance_rejected
và joint_unusable. Không đồng nhất raw coverage với numeric eligibility/accuracy.
Clip-side macro MAE null,0 contributing groups ở cả hai nhánh. Không báo error=0
trên video thật; số0 trong synthetic geometry tests là đáp án test riêng.

## Test evidence, môi trường và QA

Final related **884 passed,0 failed,0 errors,0 skipped,3 deselected,
0 pytest warnings,92.74s**, exit0.78 tests mới +806 retained related tests.
stdout/XML/command ở tests-final.txt/xml/-command.json; tested-code.json giữ
hash source/tests/protocol đã kiểm thử. Full project suite và3 real inference
integrations **NOT_RUN**; không clean-environment hoặc athlete validation claim.

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_squat_reference.py tests/test_feature_context_session.py tests/test_context_performance.py tests/test_real_batch_coverage.py tests/test_real_descriptive_demo.py tests/test_feature_context.py tests/test_geometry_features.py tests/test_feature_observation.py tests/test_right_knee_feature.py tests/test_preprocessing.py tests/test_quality.py tests/test_geometry.py -m "not integration" -q --tb=short --basetemp outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/tests-final-temp --junitxml=outputs/squat_2d_reference/run-be0cc77de33c483da73a22b890a90342/tests-final.xml
```

Tests: độc lập45°/90° trên400×200,3–4–5 pixel error, p90/micro/macro có đáp án
định trước; reference/model masks; bounds, degeneracy/A=C, nonfinite/types,
occluded/missing/uncertain, exact keys/hash/frame/time/dimensions, duplicate,
provenance, version/unknown kind, unsafe nested mutation, JSON roundtrip,
deterministic sampling/side/order, disk tampering/no overwrite, backend-free import.
Wrong-side metadata bị từ chối; tọa độ mang key hợp lệ nhưng bị con người gán
nhầm giải phẫu không thể được phát hiện tự động bằng schema — cần label review.

Lịch sử không ghi đè: first68 pass/1 fail do returned pair order phụ thuộc order
input; sửa sort sides trước freeze/extraction. Sau đó76 pass; related882 pass;
thêm strict Python-reference revalidation và unknown-kind tests, final884 pass.
Không dùng expected geometry do evaluator tự sinh để chứng minh chính nó.

Python3.11.9, Pydantic2.13.5, pytest9.1.1, OpenCV-contrib5.0.0.93,
NumPy2.4.6/SciPy1.17.1/MediaPipe0.10.14 installed nhưng không inference.
Evaluator/public import không kéo cv2/numpy/scipy/mediapipe/torch; decoder chỉ
import trong package preparation. Chuẩn bị CLI13.45s (internal12.08s) gồm
verification/export; không coi đây là benchmark model. environment/command receipts.

Thực sự xem3 native PNG bằng view_image: S001f0, S016f43, S099f72; không overlay
nhân tạo. S016 còn letterboxing/lower-leg crop của nguồn, giữ nguyên, không đoán
nhãn hoặc thay frame. sampled-image-qa.json có hash/nhận xét. Không full60 visual
audit, không xác nhận anatomical side/viewpoint chính thức. Browser/playback QA
NOT_RUN, không server/network hoặc mở video ngoài squat scope.

## Bảo toàn, giới hạn và bước tiếp theo

Code mới: reporting/squat_reference.py, reporting/squat_reference_package.py;
tests/test_squat_reference.py. Docs mới: protocol và report này; chỉ cập nhật
PROJECT_STATE/TASKS hiện hành. Không sửa model/threshold/profile/preprocess/
formula/shared schema/producer/renderer; không download/upload/train/inference.
baseline-before.json, preservation.json và final-integrity.json kiểm tra nguồn/
old artifacts/disabled gates; artifact-inventory.json giữ hashes/size toàn run.
Không xóa/move hoặc ghi đè dataset/archive/outputs/reports lịch sử.

Sẵn sàng: frozen image package, blank schema/protocol, hash-linked evaluator,
coverage/missingness và synthetic regression tools. Chưa có bằng chứng sai số
trên real video. Bước nhỏ tiếp theo là thực hiện hai lượt annotation độc lập/mù
trên chính60 frames, lưu nguồn/person/time và uncertainty, kiểm tra side/subject,
đo disagreement rồi adjudicate có log. Chưa thực hiện các công việc đó trong
milestone này; chưa có accuracy threshold đã duyệt, holdout hoặc calibration.
Không tự mở G5.7 (documented side + sequence/bilateral allow vẫn thiếu), temporal
G6 hay assessment. Không rep/phase/score/coaching. Bộ công cụ pass tests không
đồng nghĩa reference hoặc model đã được validated.
