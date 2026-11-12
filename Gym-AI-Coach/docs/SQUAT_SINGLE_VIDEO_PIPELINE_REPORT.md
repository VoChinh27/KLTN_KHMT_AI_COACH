# BodyWeightSquats — single-video descriptive pipeline

2026-10-21, Asia/Saigon. **IMPLEMENTED / REAL SMOKE PASSED** trong phạm vi
tích hợp mô tả framewise; không accuracy, athlete hoặc form validation.
NOT_VALIDATED, official_eligibility=false, assessment=false, REFERENCE_PENDING;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Workspace không Git repository; không tạo commit/diff giả.

## Sản phẩm mở được và một lệnh chạy

Run mới: `outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/`.
[Mở danh sách5 cases](../outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/START_HERE.html),
[demo exploratory có số đo](../outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/exploratory-allow/START_HERE.html),
[strict giữ null](../outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/strict-fresh/START_HERE.html),
[hướng dẫn](../outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/HUONG_DAN_SU_DUNG.md).
Giữ nguyên thư mục output/sidecars/video; không cần Python/server/mạng để xem.
Browser/native continuous playback QA **NOT_RUN**; sampled decoded image QA đã chạy.

Từ project root, CLI nhận video bất kỳ do caller khai báo BodyWeightSquats,
left/right cố định, output mới; không cần audit/run ID hoặc numerical baseline:

```powershell
# Strict mặc định; GUID bảo đảm tên output mới cho mỗi lần gọi.
& .venv/Scripts/python.exe -B -m gym_coach.reporting.single_video Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi --side left --output-dir ("outputs/squat-single-"+[guid]::NewGuid().ToString("N")) --cache-dir outputs/squat_pose_cache

# Exploratory cần cả mode và giả định side tường minh.
& .venv/Scripts/python.exe -B -m gym_coach.reporting.single_video Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi --side left --mode exploratory --assume-side --output-dir ("outputs/squat-single-"+[guid]::NewGuid().ToString("N")) --cache-dir outputs/squat_pose_cache
```

Bỏ --cache-dir để chạy pose mới. Đường video/side không bị hardcode trong
implementation; không action-recognition model. Strict chưa có interface nhận
annotation viewpoint chính thức: reader hiện không cung cấp metadata đó nên
raw/source viewpoint unknown/cameraUNKNOWN được giữ. Không tự tìm hoặc tạo
annotation. Exploratory chỉ là giả định caller, không expert/human confirmation.

## Implementation và contract tái sử dụng

Additive entry point `reporting/single_video.py`, envelope squat-single-video-v1;
không đổi primitive, numerical schema0.4.0, registryv2, legacy Linked factory,
session capability hoặc quality/preprocessing policy. Luồng thực:

video → OpenCVVideoReader/metadata → MediaPipePoseExtractor → raw pose unknown →
canonical QualityGate → preprocess_pose → FeatureContextSession → existing
observe_geometry_feature → saved result/JSONL → existing paint/encoder → report.

Giữ raw/source metadata; exploratory chỉ tạo processing copy khác viewpoint side
và lưu processing-metadata.json riêng. Status `caller_assumed_unverified`, source
`caller_cli_assume_side`, visual_assessment_performed=false; không dùng tên
visually_assessed_unverified để ngụ ý CLI đã xem video. Không documented/ground
truth. Quality source unknown và quality processing được lưu riêng. Canonical
requirements/viewpoints lấy từ registry cho BodyWeightSquats; selected metric
đúng ID và hip/knee/ankle bên đã chọn. Gate toàn clip giữ nguyên; frame sạch không
được nâng degrade/skip. Không đổi bên, fusion hoặc cắt đoạn khó.

PreprocessingProfile mặc định được đối chiếu exact với configs/preprocessing_s3_v1.toml;
quality-s3-v1 và preprocess-s3-v1/profile hashes giữ nguyên. 25FPS vẫn gap4/window9/
polyorder2. Session full strict intake một lần mỗi clip; mỗi frame select ordinary
context rồi adapter giữ validator/gate. Legacy factory/schema/API không đổi.
Session execution nằm receipt; source hash/reference-linkage/observation hash
ở per-frame records. Không yêu cầu exact baseline comparison trong sản phẩm.

Mỗi run có source.json/config/runtime/model bytes identity, raw-pose.jsonl,
quality-source.json/quality-processing.json, processing-metadata.json,
preprocessing.json, result.json/frames.jsonl, coverage.json, copied original,
overlay MP4, START_HERE.html, guide, video-verification.json, receipt/manifest.
Giá trị/null, reasons/flags, processed joint provenance, raw joints, fixed side,
video/frame/time/hash linkage được giữ. Geometry vẫn chỉ framewise projected2D;
temporal phase/repetition/point-turn/pause và assessment luôn unavailable/false.

Renderer dùng saved result và typed readback/raw/Stage4 linkage, không gọi
geometry/quality/preprocessing. Tái sử dụng real_descriptive_render.paint_frame,
font/colors/text, shared encode_saved_frames và verify_video. Old paint/HTML/
renderer defaults giữ nguyên; new opt-in presentation_mode/preserve_aspect hỗ trợ
strict và letterbox đúng aspect ratio cho input khác4:3. Overlay raw skeleton và
processed angle cùng frame/time; không dùng skeleton raw để tính lại số đo.
HTML static, escaping, CSP không network/scripts; nhãn Việt “Không đủ điều kiện
đo” và reason khi null. Màu skeleton nói về dữ liệu, không kỹ thuật đúng/sai.

## Cache identity, fail-closed và lỗi

Cache squat-source-pose-cache-v1 chỉ lưu raw unknown-viewpoint pose + metadata;
không cache giả định, QualityReport, preprocessing hoặc numerical eligibility.
Key SHA canonical gồm video byte SHA/size, full decoder metadata/path/videoID,
exercise declaration, backend/model asset/config, current quality/preprocessing
profiles/version/hash, registry hash, package version, installed core versions và
mọi package .py source hashes. Đây là invalidation bảo thủ: thay renderer/code
hoặc di chuyển cùng bytes sang path khác cũng có thể miss; không claim cache
reuse qua mọi environment/path. Không dùng global cache/session serialization.

Hit phải khớp identity, exactly2 payloads, từng hash/size, strict PoseFrame/
VideoMetadata và complete timeline/backend/config/unknown viewpoint. Cache không
phù hợp bị từ chối hoặc key khác thành miss có log; không fallback inference khi
entry hỏng. Cache được commit sau khi extraction/validation hoàn tất và source
bytes kiểm tra lại; payloads exclusive, manifest ghi cuối. Không automatic
repair/delete. Partial cache write được reject ở lần đọc sau; chưa atomic/concurrent
writer locking, cùng key bị race sẽ fail closed. Chọn cache directory mới hoặc bỏ
option khi cache hỏng, không sửa cache cũ. Hash không là chữ ký/authenticity;
không bảo vệ attacker sửa đồng thời cache/manifest hoặc private process code.

Quality/preprocessing/session/geometry chạy lại cả khi pose hit; receipt ghi
cache hit/miss/disabled/rejected/not_checked, inference_attempted và inference_run
riêng, stage timing/session. Không gọi hit là fresh inference. Profile/model/source
identity kiểm tra trước dùng; source video hash kiểm tra thêm trước copy/render.

Output tồn tại bị từ chối trước mọi ghi. Lỗi stage sau tạo run lưu failure.json/
FAILED.txt với reason/category/stage/S3/exception và upstream structured flag nếu
có; giữ partial artifacts, CLI exit1, không completed receipt/manifest. Preflight
errors trả CLI JSON/exit1; không sửa output cũ. Workflow reason envelope mới
không thay shared FlagCode/error semantics các stage trước.

## Smoke plan đã khóa, nguồn và kết quả thực chạy

smoke-plan.json lưu trước code execution/inference; tối đa2 source clips đã có
gate metadata, không chọn lại sau nhìn kết quả. Chỉ source videos trong
Data/UCF-101_GYM/BodyWeightSquats. Không dataset khác/archive/download/upload.

| Video | Byte SHA-256 | Mục đích |
|---|---|---|
| v_BodyWeightSquats_g01_c01.avi |1a199bf3ad385a66505fa0864a48fb9b9f9c1edfd1306d8345729ab06742c9e0|known left allow/right skip; strict missing viewpoint; cache equivalence|
| v_BodyWeightSquats_g04_c03.avi |4324177b7cda96aeab3e0b07e8692a5bcd9ca4724ebd2d91a9ecef60d530ec26|known right degrade; independent second real inference|

Exact source identities authoritative ở smoke-plan.json/source.json; không xem
known gates là reference accuracy. Mỗi clip25FPS320×240, CFR timestamp=i/25.
Không dùng viewpoint audit để ghi documented; giả định mới do CLI caller.

| Run | Pose | Bên | Mode | Sequence | Metric | Available / full frames | CLI elapsed s |
|---|---|---|---|---|---|---|---:|
|strict-fresh|fresh MediaPipe|left|strict|skip|skip|0/267|34.75|
|exploratory-allow|verified pose cache|left|explicit side|skip|allow|267/267|22.57|
|exploratory-skip|verified pose cache|right|explicit side|skip|skip|0/267|22.29|
|exploratory-degrade-fresh|fresh MediaPipe|right|explicit side|skip|degrade|0/88|14.95|
|strict-cache|verified pose cache|left|strict|skip|skip|0/267|21.77|

**2 real inference runs/355 frames**,3 cache hits. Không saved pose/baseline dùng
thay fresh2 runs. Strict fresh/cache complete result.json exact equal, không chỉ
counts; cùng raw source byte snapshot cho3 downstream hits. Sequence đều skip;
267 available chỉ là framewise experimental descriptive eligibility, không mở
official, bilateral, temporal hoặc accuracy. Right degrade/skip giữ null/reasons.

CLI commands đã thực chạy và stdout/stderr/elapsed giữ riêng từng case trong
`*-command.json`, không dùng source runID làm điều kiện CLI. Ví dụ lệnh thực:

```powershell
& .venv/Scripts/python.exe -u -B -m gym_coach.reporting.single_video Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi --side left --mode strict --output-dir outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/strict-fresh --cache-dir outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/cache
```

Đây là đường đã tồn tại, không chạy đè; dùng command GUID ở đầu khi chạy mới.
Model MediaPipe0.10.14/full complexity1, smooth_landmarks=false;
model SHA e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f,
pose config3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976.
runtime-before-smoke.json giữ all implementation/version/config hashes.

Phase timings (seconds; context/adapter là subsets của session_geometry,
không cộng hai lần). Tổng internal chưa gồm final manifest/startup:

| Run | Pose/cache | Quality | Preprocess | Session+geometry | Save/validate | Render/decode | Internal total |
|---|---:|---:|---:|---:|---:|---:|---:|
|strict-fresh|11.584|0.052|2.375|4.309|3.625|11.230|33.556|
|exploratory-allow|0.080|0.100|2.067|3.996|3.547|11.400|21.520|
|exploratory-skip|0.079|0.098|2.191|4.315|3.879|10.368|21.258|
|degrade-fresh|5.202|0.036|1.919|1.367|1.089|3.896|13.841|
|strict-cache|0.080|0.066|2.246|4.289|3.795|9.915|20.738|

Không pytest overlap các smoke timed runs. Không controlled benchmark/speedup
claim; cache giảm inference work nhưng render/serialization còn chi phí lớn.
Thời gian metadata/runtime verification0.328–0.381s đã tính trong internal total.

## Tests, failure cases, decode và image QA

Final related **981 passed,0 failed,0 errors,0 skipped,4 deselected,
0 pytest warnings in131.41s**, exit0.43 new tests/938 retained. tests-final.txt,
xml, command JSON và test-final-summary.json giữ evidence. Trước đó62 pass rồi
88 pass sau bổ sung edge cases; không failed pytest run trong milestone này.
Không full project suite hoặc other-source inference integrations;2 actual
MediaPipe runs là smoke CLI riêng, không cộng vào pytest count.

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_single_video_pipeline.py tests/test_squat_reference.py tests/test_feature_context_session.py tests/test_context_performance.py tests/test_real_batch_coverage.py tests/test_real_descriptive_demo.py tests/test_feature_context.py tests/test_geometry_features.py tests/test_feature_observation.py tests/test_right_knee_feature.py tests/test_preprocessing.py tests/test_quality.py tests/test_geometry.py tests/test_video.py tests/test_pose.py -m "not integration" -q --tb=short --basetemp outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/tests-final-temp --junitxml=outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/tests-final.xml
```

Tests dùng synthetic reader/backend nhưng genuine gates/preprocess/session/
adapter: default strict, explicit assumption, allow/degrade/skip/null/reasons,
fixed sides/time, exact fresh/cache và session/legacy equivalence, cache byte/
metadata/model/config/profile/version keys và payload tampering, same path/new
bytes miss, source changed mid-extraction không commit, phase failure receipts,
output collision, real decoder invalid-byte input, renderer saved-only/escaping,
portrait isotropic fit, CLI nonzero và no heavy import. Không numerical baseline
bắt buộc. Legacy renderer/batch regression đạt sau shared encoder refactor.

CLI negative checks thực chạy: output tồn tại, invalid synthetic AVI bytes và
cache copy bị append corruption đều exit1. Cache hỏng không chạy inference và
không có success manifest. Chỉ sửa bản copy trong run mới; cache gốc giữ nguyên.
Lỗi giữa pipeline chỉ fault-injection synthetic ở pose/quality/preprocessing/
session/render, chưa cố ý gây backend crash trên video thật. Receipts ghi rõ.

Independent verify-smoke.py: hashes từng run payload, source2 hashes và full
raw/numeric linkage, **1.156 output frames +355 original-copy frames decoded**,
frame/time map exact,25FPS1280×720 overlays. Tất cả decoded FOURCC h264; avc1
writers opened=true. native-diagnostics.json giữ5 probes và stderr identities.

Không ghi “0 warnings” chung cho integration:2 fresh subprocesses có tổng2
protobuf UserWarning emissions,4 MediaPipe W0000 messages và2 absl warning
headers.3 cache subprocesses có OpenH264/FFMPEG encoder-init error blocks;
writer vẫn reported opened=true, output đủ frames và decoder xác nhận h264.
Actual encoder backend không được record, không suy rằng diagnostic đã biến mất
hoặc tự tải codec.0 pytest warnings là kết quả riêng của unit/regression run.

Thực sự xem7 contact sheets bằng view_image(original), gồm15 overlay samples
(0/133/266 cho267-frame runs;0/44/87 cho88-frame run) và6 original samples.
Nhãn Việt, fixed side, values/null/reasons, mode và time rõ, không text clipping
trên mẫu đã xem. g04 còn original letterbox/lower-leg crop, raw skeleton frame0
lệch/không hợp lý so với hình ảnh; giữ nguyên raw/source, không sửa/chọn lại
hoặc kết luận form. visual-qa-review.json ghi hash/nhận xét. Đây là sampled image
inspection, **không browser/native playback/athlete validation**. Không server,
network hoặc đi vòng local-file policy; browser/playback QA NOT_RUN.

## Bảo toàn và giới hạn

Thêm reporting/single_video.py và tests/test_single_video_pipeline.py; renderer
cũ chỉ shared encoder và opt-in presentation/aspect. README/PROJECT_STATE/TASKS
cập nhật tối thiểu, report này mới. Không đổi dependencies, model/config/profile/
quality/numerical schema/geometry/legacy factory/session/temporal producer.
baseline-before.json kiểm kê2.471 files; preservation/final-integrity/inventory
kiểm tra source và mọi old batch/performance/reference artifacts, không xóa/move.

Môi trường hiện có: Python3.11.9, MediaPipe0.10.14, OpenCV-contrib5.0.0.93,
Pydantic2.13.5, pytest9.1.1, NumPy2.4.6, SciPy1.17.1, Pillow12.3.0.
Pillow renderer đã dùng từ milestone trước; không dependency/model download mới.
Chưa clean-environment compatibility, generic-video real smoke ngoài hai clips,
concurrent-cache behavior hoặc browser codec/playback acceptance.

Sẵn sàng: một CLI từ video đến per-frame JSON/overlay/offline report, strict
default và explicit exploratory, source-pose cache có kiểm tra, lỗi không báo
thành công giả. Chưa sẵn sàng: accuracy, documented side/bilateral gate, temporal
hoặc technical assessment. REFERENCE_PENDING giữ nguyên; không nhãn mới, không
form/score/rep/phase/coaching. Source hashes không xác nhận anatomical correctness.

Bước sản phẩm tiếp theo: xuất progress JSONL theo stage/frame cho CLI này, để
người dùng thấy tiến độ render/save kéo dài khoảng10–15s ở cache hits và xử lý
lỗi rõ trước khi xây giao diện gọi pipeline. Đây là bước usability, tách biệt
với việc vẫn cần nhãn độc lập và kiểm tra chất lượng reference để đo accuracy.
