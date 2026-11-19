# Gym Result Feedback — squat descriptive prototype

Ngày: 2026-10-15, Asia/Saigon. Package **0.7.0**. Prototype implemented;
GĐ6 **IN_PROGRESS**, GĐ5 **IN_PROGRESS / VALIDATION_BLOCKED**, official GĐ5.7
**BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**. Assessment luôn false.
Workspace không có Git repository; không tạo commit hoặc giả lập Git diff.

## Cập nhật 2026-10-18 — khảo sát synthetic temporal

Đã bổ sung analytic fixtures/reference và audit riêng cho profile mặc định,
không sửa thuật toán segmentation/reporting hoặc profile. 30 điều kiện (10 cases
×25/30/60FPS) chạy thành công:78/102 full reference cycles matched,24 không có
candidate,0 candidate thừa; phase availability82.70%, exact-frame phase
agreement73.73%. Bottom pause không được biểu diễn như interval: recall Bottom
ở case bottom_pauses chỉ2.44–5.56% dù tìm đủ cycles. Đây là số liệu synthetic,
không validation trên vận động viên. GĐ5.7 vẫn BLOCKED, GĐ6 IN_PROGRESS,
NOT_VALIDATED/official=false/assessment=false giữ nguyên.

[Báo cáo phương pháp, số liệu từng FPS, tests và giới hạn](STAGE6_SYNTHETIC_TEMPORAL_SENSITIVITY_REPORT.md).
Các số liệu và lệnh bên dưới là evidence của lượt prototype 2026-10-15.

## Phạm vi và mapping demo

Đã đọc README, technical spec, implementation plan, project state/tasks và kiểm
tra contracts/registry/quality/preprocessing/numeric0.4 hiện hành. Không đọc/khôi
phục implementation archive. User cho phép bắt đầu phần synthetic độc lập;
điều đó không thay acceptance của official5.7 hoặc DoD GĐ6.

Reference người dùng `result_feedback.mp4`: 426 frames, 30FPS, 640×360; xem 12
thời điểm phân bố 0–14.167s. Byte SHA:
`c7d7d3311ded2348ed848c192daa8b3f7f693ddb6e914f7693cf6bb9026ef874`.
Manifest/samples nằm trong run mới; chỉ tham khảo trải nghiệm, không Gym input
hoặc nguồn chuẩn kỹ thuật. Không có source nhánh chạy được tích hợp.

| Thành phần reference | Prototype Gym | Ranh giới |
|---|---|---|
| Header pha theo thời gian | Start/Top → Descent → Bottom → Ascent → Return/Top | PVP engineering candidates, S3 chưa validated; không dùng event môn chạy |
| Skeleton/overlay | Canvas native raw XY, observed cyan, low visibility amber, missing không vẽ | Không tô màu đúng/sai; không vẽ processed torso-normalized lên ảnh |
| Các số đo | Năm squat angles canonical, degrees, per-frame provenance/quality | Không dùng ngưỡng góc hoặc COM/launch/running rules trong reference |
| Panel cảnh báo/gợi ý | Data availability, reasons, experimental description, assessment blocked | Không sinh form faults, form severity, score hoặc lời sửa kỹ thuật |
| Playback/timeline | Scrubber, synthetic playback, PVP candidate timeline, signal graph | Real preview chỉ ba frame rời rạc, không giả lập video đã phân tích |

## Code và output contract

Không thêm dependency/framework. `segmentation/models.py` + `squat.py` thực hiện
temporal thuần stdlib ngoài contract Pydantic có sẵn. `reporting/squat_models.py`
định nghĩa feedback; `squat_builder.py` gọi registry adapters cho fixture synthetic
đã strict-validate Stage4, rồi qua **gate temporal riêng**. `renderer.py` chỉ
render snapshot đã validate, không tính geometry/segmentation. `demo.py` tạo
fixture, CLI, audit sidecars và blocked video preview, lazy-import backend.
`reporting.AuditHeader` và APIs/schemas GĐ0–5 giữ tương thích.

- Feedback schema0.1.0, `gym-squat-feedback-v1`; nguồn
  `squat-feedback-source-v1`, model/source/registry/segmentation content hashes.
- Temporal schema0.1.0, `squat-temporal-input-v1`, algorithm
  `squat-pvp-hysteresis-v1`; scope chỉ synthetic, official eligibility false.
- Registry `gym-feature-registry-v2` giữ hash
  `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72`.
- Feature IDs: `squat_hip_angle_left_deg`, `squat_hip_angle_right_deg`,
  `squat_knee_angle_left_deg`, `squat_knee_angle_right_deg`,
  `squat_torso_angle_vertical_deg`. Chỉ degrees; không dùng blocked ratios.
- Metric projection giữ value/null, definition/observation/input hashes, actual
  ordered ProcessedJoints/provenance/imputation refs, validation unvalidated,
  quality action, flags/reasons, exact evidence frame/time. Numeric0.4 vẫn
  **temporal_analysis_eligible=false**; không được truyền trực tiếp làm temporal input.
- Source giữ video ID, CFR/FPS/dimensions/count, synthetic Stage4/raw/quality
  hashes hoặc real video+CSV byte hashes. Flag/version/hash và nguồn có quy ước
  tại SCHEMA_CONVENTIONS; hashes không chứng minh provenance người xác nhận.
- Unavailable/NOT_RUN không điền góc, joint hoặc observation identity giả.
  Root giữ assessment false, form_score/technical_feedback null và official false.

## Temporal policy thử nghiệm

Profile `squat-temporal-s3-v1`, **S3 / NOT_VALIDATED**, hash
`5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`:
centered moving mean span0.2s, amplitude hysteresis5°, minimum half-cycle5/30s.
Đây là signal parameters, không chuẩn độ sâu/tempo kỹ thuật; chưa tune hoặc
freeze cho evaluation. Amplitude hysteresis không được gọi là prominence chuẩn.
Stage4 SG là xử lý upstream riêng có profile/hash; không sửa raw/processed.

Radius smoothing `floor(span*fps/2 + 1e-9)` capped run length, edge truncated.
Mỗi unresolved null hoặc đổi bên split runs; không signal interpolation, auto
side fallback hoặc fusion. Plateau dùng midpoint; first/last extrema unconfirmed
và flagged partial. Chỉ interior confirmed PVP đủ thời gian hai nửa là candidate.
Start/Bottom/Return hiện là event một frame, Descent/Ascent interval inclusive;
shared top hiển thị Return của chu kỳ trước, vẫn giữ start candidate tiếp theo.
Zero candidates khi unavailable không khẳng định true rep count=0.

Pure synthetic signals là unit fixtures, không có QualityReport. Fixtures từ
Stage4 phải có sequence allow, side viewpoint và exact **cả hai** knee metrics
allow với joints canonical. Không recompute/promote QualityReport trong gate.
Không có real temporal admission; chỉ đổi tên ID không chứng minh nguồn thật.

## Run và kết quả thực tế

Run mới: `outputs/gym_result_feedback/run-e1f46752326a49fea5696cc5ccc4e296`.
Bản dùng báo cáo/UI: **`demo-verified/`**; các attempt trước giữ nguyên riêng.

| Case | Frames/FPS | Sequence | Numeric observations | Temporal |
|---|---|---|---|---|
| synthetic | 181 /30 | allow, bilateral allow | 905 available | 2 PVP candidates:30→60→90 và90→120→150 |
| synthetic-gap (left knee missing58–62) |181 /30 |degrade, left knee degrade, right allow |543 available/362 unavailable |blocked; không đổi gate dù Stage4 imputed |
| real-blocked g25_c01 |103 /25 |NOT_RUN |15 preview projections unavailable, 0 numeric calls |NOT_RUN/blocked |

Synthetic là tọa độ được sinh bằng toán học (không chạy MediaPipe), dùng genuine
QualityGate → Stage4 → actual registry angles → separate synthetic temporal gate.
Viewpoint side là thuộc tính fixture, không annotation một người thật. Kết quả
không đo accuracy hoặc giải phẫu của vận động viên.

Real preview dùng đúng
`Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g25_c01.avi`, 320×240;
byte SHA `893f7b0c89d10030705eeadfca1f9681a81190540bff1b2ab77d0ab3b614261e`.
CSV gốc `Data/UCF-101_GYM/audit/UCF101_GYM_VIDEO_AUDIT.csv` SHA
`1334e5f43d2a060470eed8a4df28c6905a504099c34d9e7ce79c0d8783d5bd20`.
camera_view/person_visibility/occlusion **UNKNOWN**, không biến thành side.
Decoder mới đọc đủ103frames và chỉ lưu JPEG0/51/102 ở0/2.04/4.08s. Không
pose extraction, QualityGate, numeric adapter hoặc temporal trên clip thật.
Clip chọn để thể hiện blocked input đã biết, không chọn để ép gate pass.

Artifacts: feedback JSON+HTML mỗi case; full synthetic preprocessing và905
observations JSONL/case; original real audit row, measured decoder metadata,
preview JPEGs; schema, profile, registry snapshot; inventory chứa exact command,
environment và implementation/artifact SHA-256. `artifact_verification.json`
ghi strict readback, projection↔observation/source linkage và preservation check.
`reference_manifest.json` tách riêng UX reference khỏi nguồn Gym.

Preservation audit: baseline2324 files,2315 byte-identical,9 thay đổi có chủ đích,
0 missing. Trong đó119 dataset files (112 squat AVIs+7 metadata) và toàn bộ old
outputs/config trong baseline giữ hash. Hai spec và các báo cáo lịch sử giữ nguyên.
Không mở archive làm source; baseline không kiểm hash archive nên không tuyên bố
đã có archive byte audit. Reference video cũng giữ hash. Xem manifest cho phạm vi
inventory cụ thể, không coi preservation là scientific validation.

File mới: `segmentation/models.py`, `segmentation/squat.py`;
`reporting/squat_models.py`, `squat_builder.py`, `renderer.py`, `demo.py`,
`__main__.py`; `tests/test_temporal_squat.py`, `tests/test_squat_feedback.py` và
báo cáo này. File hiện có thay đổi: `_version.py`, `flags.py`, exports của
`reporting`/`segmentation`, README, IMPLEMENTATION_PLAN, PROJECT_STATE, TASKS,
SCHEMA_CONVENTIONS. Dependency manifest, registry và upstream implementation
không đổi. Run mới có separate inventory/test XML/screenshots, không ghi đè cũ.

## Lệnh chạy và test evidence

PowerShell tại project root; output mới được tạo exclusive:

```powershell
$run = 'outputs/gym_result_feedback/run-e1f46752326a49fea5696cc5ccc4e296'
& .venv/Scripts/python.exe -m gym_coach.reporting --output "$run/demo-verified" --video Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g25_c01.avi
& .venv/Scripts/python.exe -m pytest tests/test_temporal_squat.py tests/test_squat_feedback.py -q --junitxml="$run/feedback-final.xml"
& .venv/Scripts/python.exe -m pytest -q --deselect=tests/test_development_inventory.py::test_workspace_inventory_keeps_documentation_and_pose_eligibility_separate --junitxml="$run/suite-final.xml"
& .venv/Scripts/python.exe "$run/verify_artifacts.py"
```

Lệnh trên đã chạy; không chạy lại vào output/JUnit cũ. Dùng UUID mới cho lần mới.

| Run evidence | Pass | Fail | Error | Skip | Deselected | Warnings | Thời gian |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal-red.xml, trước implementation |0|0|1 collection|0|0|0|0.59s|
| temporal-attempt1.xml |30|0|0|0|0|0|1.01s|
| feedback-attempt1.xml |66|1|0|0|0|0|18.98s|
| feedback-attempt2.xml |67|0|0|0|0|0|18.60s|
| feedback-final.xml |71|0|0|0|0|0|24.83s|
| suite-attempt1.xml |1016|0|0|6|1|4|140.01s|
| suite-final.xml |1020|0|0|6|1|4|142.09s|

Failure đầu feedback là fixture gap2frames tại10FPS không được nội suy theo
Stage4 profile; sửa fixture thành1frame để kiểm tra đúng trường hợp đã impute
nhưng quality degrade vẫn block temporal. Không đổi Stage4 threshold/policy.

Suite chạy toàn collection với **một deselection tường minh**: inventory cũ quét
cả `Data/` ngoài GYM. Sáu saved exploratory audit readbacks skipped do không bật
`GYM_STAGE5_AUDIT_DIR`; không đưa historical exploratory input vào prototype.
Không tuyên bố toàn bộ collection đều pass. Các integration GĐ2/4 cũ có cả HSPU
được chạy để kiểm tra compatibility, đều dùng GYM; không mở rộng prototype sang
HSPU. Bốn warnings là protobuf GetPrototype deprecated từ các integrations này.

71 tests mới gồm FPS25/30/60 boundaries (tolerance1/FPS giây), empty/flat/jitter,
fast reversal/plateau/partial, all missing/long gap/side switch, invalid/nonfinite
input, serialization/hash tampering, exact bilateral/global/viewpoint gate,
unchanged upstream, unavailable null/reasons, no heavy public imports, safe JSON
embedding, exclusive output và real blocked preview không gọi numeric adapter.

Môi trường đo: Windows10 build26200, CPython3.11.9, Pydantic2.13.5, pytest9.1.1,
NumPy2.4.6/SciPy1.17.1 (existing Stage4), OpenCV contrib5.0.0.93 (preview),
MediaPipe0.10.14 installed (chỉ legacy compatibility integrations sử dụng).
Offline editable install0.7.0 đạt với --no-index --no-deps --no-build-isolation;
TEMP/TMP nằm trong run. Không cài model/framework hoặc dependency mới; chưa test
clean environment. Manifest dependency giữ nguyên.

Desktop UI kiểm tra bằng computer-use browser: scrub frame45 Descent evidence
30–60, frame60 Bottom,75 Ascent; nút Phát chuyển Dừng. Real preview frame51
giữ phaseUnavailable, năm metric null/reason, nút playback disabled. Screenshots
`ui-synthetic-descent.png`, `ui-real-blocked.png`. Browser instrumentation có
telemetry network403 ngoài trang; đây không phải warning pytest hoặc dependency
của HTML. Chưa chứng nhận cross-browser/mobile/performance hay timestamp VFR.
Playback synthetic đã tiến từ1.50s tới6.00s; gap UI hiển thị Unavailable với
temporal_gate_blocked. Kết quả UI lưu `ui-verification.json` cùng ba screenshots.

## Giới hạn và bước tiếp theo

Chưa có documented side input liên kết protocol/version/confirmer/date/video
path+ID+SHA+interval và camera/framing/occlusion. Chưa có fresh official sequence
và exact bilateral knee allow; visual/exploratory evidence cũ không thay gate.
Reference/count/phase ground truth, accuracy, coverage/failure evaluation,
group-isolated split và frozen config GĐ6/7 chưa có; GĐ6 chưa COMPLETE.
Technical assessment/feedback vẫn BLOCKED_PENDING_EXPERT_VALIDATION, không
severity/score/technical rules. Không auto promote synthetic thành official.

Bước nhỏ nhất độc lập: bổ sung fixture pause/variable tempo và phase reference
synthetic để khảo sát độ nhạy S3; trước real temporal integration cần vượt đúng
official5.7 và thiết kế reference protocol. Không yêu cầu người dùng cung cấp
thông tin trong lượt này và không tự tạo annotation để mở gate.
