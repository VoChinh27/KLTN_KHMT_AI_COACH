# BodyWeightSquats — local WEB demo

Cập nhật 2026-10-24 (Asia/Saigon). **IMPLEMENTED / REAL SMOKE AND BROWSER QA PASSED**
trong phạm vi desktop browser và một source clip đã khóa. Không accuracy/form validation.
NOT_VALIDATED, official_eligibility=false, assessment=false, REFERENCE_PENDING;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT; G6/toàn G5 IN_PROGRESS.
Workspace không Git repository; không commit/diff giả.

## Mở ứng dụng

Từ project root:

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.web
```

Mở **http://127.0.0.1:8765/**. Server chỉ bind địa chỉ này; `--port 8766` đổi port,
không có option mở LAN. `--output-root NEW_PARENT` tùy chọn nơi lưu; mỗi lần khởi
động tự tạo session mới, mỗi click được chấp nhận tạo job/run mới. API không nhận
filesystem path. Dừng bằng Ctrl+C và đợi job hoàn tất; không đóng cưỡng bức terminal.
Chưa có cancel. Tab đóng/refresh không dừng worker, không tạo job khác.

Server QA đã không còn listen ở lần kiểm tra bàn giao 24/10; chạy lệnh trên để mở lại.
Không cài service hoặc auto-start. Phiên UI mới không tự nạp/chạy lại các jobs của
server cũ; artifacts và logs trên đĩa vẫn giữ. Giới hạn này được ghi ngay README.

Run evidence mới:
`outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/`.
[Kết quả exploratory đã lưu](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/application/session-cdba40f4aa8d43d8b9377bbbbf8ce138/jobs/run-a3cf90b80dff4acbbab303ffc48fbce8/result/START_HERE.html),
[screenshot browser thực](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/browser-exploratory-paused.png),
[smoke integrity](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/smoke-integrity.json).
Các file report offline sinh trước QA giữ nguyên footer NOT_RUN tại thời điểm tạo;
receipt/screenshot QA ở run này ghi kiểm tra sau đó, không sửa bytes artifact/manifest.

## Implementation thực tế

`web.py` dùng Python stdlib ThreadingHTTPServer; `web_assets/` là HTML/CSS/JS không
build/framework/CDN. `job_controller.py` tách từ desktop supervisor, không import
Tk. Worker là subprocess dùng cùng Python, argv list, shell=false và CREATE_NO_WINDOW
trên Windows; hai reader threads drain stdout JSONL và stderr độc lập. Không viết lại
ingestion, MediaPipe, QualityGate, preprocessing, session, adapter hoặc renderer.
`pyproject.toml` chỉ thêm package-data cho ba web assets; không dependency mới.

Giao diện tiếng Việt: upload/file picker, metadata/hash, preview phụ thuộc codec,
gối trái/phải cố định, strict mặc định, exploratory cần checkbox xác nhận side chưa
kiểm chứng, cache tùy chọn, tiến độ/thời gian/log/lỗi, overlay/coverage, báo cáo và
tải từng artifact. Null không thành0°. Không nhận dạng bài tập: caller khai báo squat.
Skeleton raw và góc processed được ghi rõ; màu dữ liệu không chấm kỹ thuật.

Progress `squat-progress-v1` giữ nguyên schema và stage order: metadata → pose/cache →
quality → preprocessing → geometry → save → render → video_validation → finalize.
Mẫu số chỉ có ở công việc biết total; stage khác dùng indeterminate. Thời gian là
wall clock thật, không dùng để suy %; không có overall percentage. Cache status
được giữ xuyên event không mang cache field. UI polling700ms đọc state, không đo lại.

Một analysis active mỗi server instance, lock ở backend và disable ở frontend.
Request UUID hỗ trợ idempotency với cùng payload; UUID dùng lại khác payload bị
từ chối. Hash fragment giữ job đang xem; refresh hỏi state, không submit. Terminal
event chưa đủ để báo thành công: worker phải exit0, stream đúng version/order,
không protocol error, và full required manifest/hash/size/raw/processed/result/
source/side/mode/timeline/disabled gates/video receipt phải hợp lệ. Khi tất cả đủ
mới bật links và coverage. Execution complete với0 numeric khác execution failed.

## Phạm vi bảo vệ local

- Chỉ 127.0.0.1; exact Host gồm port, kiểm tra Origin/Sec-Fetch-Site. POST cần
  same-origin và token X-Gym-CSRF của phiên; không CORS, không endpoint mở path/lệnh.
- Upload raw binary có Content-Length, tối đa250MiB, type suffix allowlist và
  metadata+first-frame decoder check. Tên gốc chỉ metadata/textContent; lưu tên
  nội bộ UUID/input.ext, không ghép thành command/path. Pipeline kiểm tra đầy đủ
  timeline về sau. Lỗi giữ upload-failed.json/partial bytes, không tạo analysis job.
- Artifact route chỉ job ID của phiên và filename đơn trong manifest đã xác minh.
  Hash/size kiểm tra lại khi tải; traversal, ADS/path separators, symlink/reparse,
  ngoài application session và manifest bị đổi bị từ chối. Không phục vụ nguyên root.
- MP4 GET/HEAD, Content-Length, single byte Range/suffix-range,206/416 và
  Accept-Ranges hỗ trợ tua; browser ngắt range request được xử lý như lỗi transport.
- CSP self-only/no inline script, nosniff, frame deny, no-referrer; UI dùng
  textContent/DOM nodes, không innerHTML cho tên file/log/lỗi. Log không được execute.

Đây không phải Internet hardening hoặc authentication giữa các process/người dùng
cùng máy. Không database/tài khoản/quota tổng, không TLS/public/LAN/tunnel. Decoder
nhận file local do người dùng chọn; chưa fuzz codecs hoặc thử file sát250MiB. Hash
không phải chữ ký, không chống attacker cùng quyền sửa đồng thời code/manifest/file.

## Smoke đã khóa và kết quả đo

`smoke-plan.json` ghi trước inference: đúng
`Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi`, gối trái,
SHA-256 `1a199bf3ad385a66505fa0864a48fb9b9f9c1edfd1306d8345729ab06742c9e0`.
320×240,25FPS,267 frames; timestamp policy i/FPS. Uploaded bytes khớp source.
Input nội bộ tên input.avi; web-request.json liên kết original_name/upload hash với
source path/ID của pipeline. Source camera_view=UNKNOWN; assumption chỉ caller_assumed_unverified.

Final session `session-cdba40f4aa8d43d8b9377bbbbf8ce138`:

| Job prefix | Mode | Pose | Sequence / knee | Available / total | Pipeline elapsed |
|---|---|---|---|---:|---:|
|45a25f5b|strict|fresh/miss|skip / skip|0/267|28,92s|
|a3cf90b8|exploratory + acknowledgment|cache hit|skip / allow|267/267|18,91s|
|efa38770|strict|cache hit|skip / skip|0/267|20,35s|

Đây là elapsed thực ở final progress event, chưa cộng browser polling/verification
sau worker; không benchmark có đối chứng/speedup claim. Cùng path/bytes/model/config
identity mới có cache hit. Upload lại hoặc restart tạo path mới nên có thể miss.
Cache chỉ raw pose; quality/preprocess/session/numeric chạy lại. Strict fresh/cache
complete result JSON exact equal. Sequence skip reason low_pose_visibility; strict
metric skip reason viewpoint_unknown. Exploratory single metric allow không mở
bilateral/official/temporal. Mọi unavailable giữ null/reasons; không ép gate.

Final3 runs: **1 fresh inference/267 frames +2 cache hits**,801 output frames decode
độc lập,36 local href/src references hợp lệ, tải result.json qua browser hash khớp
`a9bb2063e312c6fc5e0e5917b09b89d1f330db36089c12f312b9137926f2c5e9`,14.089.648bytes.

Một Strict run trước sửa UI cũng completed0/267/fresh,36,08s, được giữ ở
session-1c0161f382f74837b911a71430d39625, job0fb987f5. Tổng milestone **2 fresh
inference trên cùng1 unique clip/534 frame extractions**,2 cache hits/4 analysis
runs. Không chọn thêm clip, không cả dataset. Báo riêng revision này, không trộn
với final3 hoặc gọi hai lần inference là hai video độc lập.

Model/config/profile kế thừa pipeline hiện có: MediaPipe0.10.14 full complexity1,
smooth_landmarks=false; quality-s3-v1/preprocess-s3-v1, registry/framewise policy và
numerical contracts không đổi. Source.json mỗi run chứa model bytes/config/profile/
source code identities. Không sửa model/threshold/gates để tăng coverage.

## Browser QA thực sự

Đã dùng IAB/browser API được phép trên URL127.0.0.1; không server Internet hoặc
workaround bảo mật. Desktop viewport1280, full-page screenshots1264px rộng.

| Thao tác | Evidence thực |
|---|---|
|File picker/upload|g01_c01 được decoder chấp nhận; tên/FPS/resolution/267/hash hiện trên UI|
|Strict + progress|nút Start khóa khi chạy; refresh vẫn job45a25f5b; pose110/267 và cache miss hiện thật|
|Duplicate submit|HTTP409 job_active khi initial real job active; concurrency/idempotency có unit tests riêng|
|Exploratory ack|chọn mode nhưng chưa tick thì Start disabled; tick mới chạy a3cf90b8/cache hit|
|Play/seek MP4|Space: paused=false/currentTime0,096/duration10,68/readyState4/errornull; ArrowRight có currentTime6,486; khi paused, ArrowLeft đổi10,68→10,5732|
|Download|browser tải result.json; bản nhận được copy vào run, hash/size khớp source artifact|
|Video lỗi|invalid synthetic AVI: decoder422, thông báo không tạo job, Start disabled;3 jobs final không tăng|
|Report/back/keyboard|Mở báo cáo cùng tab, quay lại đúng exploratory job; Tab từ side sang mode; không horizontal overflow ở viewport đã xem|

Screenshot và AX text: browser-refresh-running.txt, browser-strict-completed.png,
browser-exploratory-paused.png, browser-invalid-video.txt/png, browser-report.png.
browser-paused-seek.json là **attempt chưa chứng minh pause** vì video đã kết thúc rồi
Space phát lại; browser-paused-seek-confirmed.json mới ghi paused=true và seek delta.
Không giấu attempt, không gọi nó là bằng chứng pause thành công.

Đã sửa lỗi tái hiện: stdout Windows cp1252 không in được tiếng Việt; mất cache status
khi event tiếp theo không mang field; hint upload cũ sau refresh; chọn file lỗi để
lại video cũ sẵn sàng; browser abort byte range phát traceback; target=_blank không
đưa report tới tab được công cụ quan sát nên chuyển cùng tab/back. Không sửa artifact
cũ; chỉ assets/supervision/HTTP. Source result report footer NOT_RUN giữ lịch sử như trên.

Không QA Chrome/Edge riêng, mobile/touch, accessibility đầy đủ, đóng cưỡng bức process,
real backend crash giữa clip hoặc large upload. Shutdown chờ worker có unit test;
không ghi đã thử native Ctrl+C giữa real inference. AVI preview thực tế không hỗ trợ;
MP4 đã phát/tua được. UI QA không xác nhận anatomical correctness hoặc athlete accuracy.

## Tests và môi trường

- Initial sandbox: **122 passed,8 failed**,0 errors/skipped/warnings,31,15s. Tám HTTP
  tests bị WinError10013 do sandbox chặn loopback; giữ nguyên receipt, không xóa test.
- Rerun qua quyền localhost được phép: **130 passed**,0 fail/error/skip/warnings,
 29,48s (43 web +44 shared controller/progress +43 single-video).
- Sau cleanup và các sửa Python: **1566 passed,25 deselected,0 failed,0 errors,
  0 skipped,0 pytest warnings**,219,72s. Đây là toàn collection **trừ marker integration**;
  không gọi là đã chạy mọi integration hoặc inference trên dataset.45 web tests,
 45 shared progress/controller tests nằm trong con số1566, không cộng lặp.

```powershell
& .venv/Scripts/python.exe -B -m pytest -m "not integration" -q -p no:cacheprovider --tb=short --basetemp outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/tests-regression-final-temp --junitxml=outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/tests-regression-final.xml
```

Các path test receipt đã tồn tại, không chạy đè; chọn tên mới khi tái kiểm tra.
Command JSON/stdout/JUnit giữ exit code và timing. Tests dùng fake decoder/worker
được phân biệt với browser/inference3 final runs. Coverage: host/origin/CSRF,
path traversal/tampering, range/head/download, malformed events, stderr600KB,
nonzero/missing terminal, cache/source identity, strict/ack/fixed-side/null, exact
progress on/off numbers, active duplicate, unknown fields/types, Unicode/spaces,
import web không Tk/NumPy/OpenCV/MediaPipe, old CLI/renderer compatibility.

Final inference stderr không warning-free: fresh có1 protobuf UserWarning,
2 MediaPipe W0000 và1 absl header; mỗi cache run có1 OpenH264 initialization
diagnostic. Renderer receipt và full decode vẫn h264/267. Đây là native diagnostics
riêng, không nhập vào0 pytest warnings. Không download codec hoặc thay decoder policy.

Python3.11.9/Pydantic2.13.5/pytest9.1.1/MediaPipe0.10.14/OpenCV-contrib5.0.0.93/
NumPy2.4.6/SciPy1.17.1/Pillow12.3.0 hiện có; pip check exit0. Không dependency mới,
chưa clean-environment/wheel-install validation.

### Kiểm tra bàn giao cuối

Sau cập nhật tài liệu/assets và bổ sung 12 delivery tests: **145 passed,0 failed,
0 errors,0 skipped,0 deselected,0 pytest warnings**,31,95s,exit0. Gồm45 web,
12 delivery,45 shared job/progress và43 single-video tests. Không cộng145 vào1566
vì có overlap; full non-integration suite1566 ở revision trước delivery tests,
focused run này kiểm tra các thay đổi bàn giao sau đó. Không inference mới.

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_web.py tests/test_web_delivery.py tests/test_job_progress.py tests/test_single_video_pipeline.py -q -p no:cacheprovider --tb=short --basetemp outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/tests-delivery-corrected-temp --junitxml=outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/tests-delivery-corrected.xml
```

Lần delivery trước đó **141 passed,4 failed**,38,93s: test gọi sai bốn CLI cũ bằng
`python -m …cli` trong khi chúng dùng console entry points đã khai báo. Đã sửa test
gọi đúng `gym-video-info`, `gym-pose-extract`, `gym-quality-audit`,
`gym-pose-preprocess`; không thay CLI/pipeline để làm test pass. Cả receipt lỗi và
receipt sửa đều giữ nguyên. Web và single-video tiếp tục được kiểm tra bằng `-m`.

[Final integrity](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/final-integrity.json)
xác nhận885 baseline identities:877 unchanged,5 file sửa đúng phạm vi,3 originals
được archive nguyên bytes;0 unexpected changes. Khôi phục thử3 file vào thư mục
mới trong run, hash khớp; không đưa desktop trở lại cây active. Python implementation
khớp revision chạy final real smoke. Các thay đổi cuối chỉ presentation/docs/test;
delivery tests kiểm tra links, packaging assets và sáu CLI entry points hiện hành.
Sau phần bổ sung báo cáo, chạy riêng `tests/test_web_delivery.py`: **12 passed,
0 fail/error/skipped/deselected/warnings**,3,26s; `tests-docs-final` lưu command,
stdout và JUnit riêng. Đây là subset của145, không phải12 test độc lập cộng thêm.

## Bàn giao và bước tiếp theo

Sẵn sàng: local browser→single-video subprocess→verified descriptive result với
progress thật, explicit assumptions, zero-coverage handling và audit trail.
Tk đã retired có thể khôi phục; [cleanup report](PROJECT_CLEANUP_REPORT.md),
[active/history index](DOCUMENTATION_INDEX.md). CLI nghiên cứu và evidence cũ giữ nguyên.

Bước sản phẩm nhỏ tiếp theo: khôi phục lịch sử **read-only** từ persisted verified
sessions sau server restart; không tự chạy lại jobs, không đổi cache/gates. Chưa
triển khai trong milestone này. Nhu cầu khoa học độc lập vẫn là reference labels
và documented viewpoint/acceptance; không thay thế bằng UI/tests/coverage.
