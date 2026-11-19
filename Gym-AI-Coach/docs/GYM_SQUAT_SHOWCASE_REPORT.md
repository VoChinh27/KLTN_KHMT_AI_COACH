# BodyWeightSquats — saved-artifact offline showcase

2026-10-18, Asia/Saigon. Milestone trình bày đã triển khai; không phát triển
detector. **EXPERIMENTAL / NOT_VALIDATED**, official_eligibility=false,
assessment=false, G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5
IN_PROGRESS. Không đọc Data/video/metadata mới, inference, tính góc, chạy lại
producer, sửa profile/schema/domain API hoặc thêm dependency trong demo.

**Artifact giao:**
`outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/demo-final/index.html`.
Mở trực tiếp bằng browser thông thường, không cần server/network/framework.
JS/data nhúng trong HTML (~4.53MB); JSON/provenance links dùng các sidecars cùng
thư mục. Giữ cả folder khi sao chép nếu muốn dùng links; riêng HTML vẫn trình
bày từ dữ liệu đã nhúng. Không có Git repository, không tạo commit/diff giả.

## Nguồn đã đọc và quyết định không ghép frame

Đã đọc [prototype report](GYM_RESULT_FEEDBACK_PROTOTYPE_REPORT.md),
[v2 slice report](STAGE6_TEMPORAL_V2_SLICE_REPORT.md),
[point-resolution report](STAGE6_POINT_RESOLUTION_REPORT.md), renderer/demo
hiện hành và saved outputs synthetic. Nguồn JSON cố định:

- Prototype `outputs/gym_result_feedback/run-e1f46752326a49fea5696cc5ccc4e296/demo-verified/`:
  `synthetic/feedback.json` và `synthetic-gap/feedback.json`.
- Latest point run `outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/demo-final/`:
  manifest +9 case audit/input JSONs. Không đọc case real-blocked hoặc video.

Hai panel không có cùng provenance. A normal có video ID
`synthetic-squat-feedback-v1`, temporal input hash
`b8a04697e2f69dec6a52929f8db2ca9f27cec6bc7c07b85aa7c1601614d3503f`;
B smooth30 có ID `synthetic-point-resolution-demo`, input hash
`63c623f808f016d054d607f338af93768a5dc680a36d8952d921fd1debc9628c`.
A có pose/preprocessing/raw/quality identities; B là pure scalar fixture,
quality=null. Cùng30FPS/181frames không chứng minh cùng nguồn hoặc cùng geometry.

Vì vậy **hai panel độc lập, hai scrubbers/playback clocks**, không nối metric A
với event B, không gắn event temporal lên skeleton. Bắt đầu ở A normal và B
smooth25 còn giúp hiện rõ FPS độc lập. Dropdown B ghi đúng case name từ manifest.
Đây là luồng saved JSON → validation/projection → HTML, không phải video →
model → live feedback hoặc một sequence hợp nhất.

## Trải nghiệm và dữ liệu giữ nguyên

Panel A tái sử dụng cách vẽ/màu joints và cards metrics của prototype. Chỉ vẽ
raw image_normalized_xy joints đã lưu, observed cyan/low_visibility amber,
missing không vẽ. UI ghi rõ skeleton được tạo synthetic ở lượt trước, không
phải output inference MediaPipe ở milestone này. Không sinh joints/coordinates.
Tất cả source_frame_hash/frame/time/metric value/unit/provenance/observation/input
hash/evidence interval và quality action được chiếu từ saved records. UI chỉ
format góc1 chữ số; presentation JSON giữ exact numeric values. Không gọi
geometry, normalization hoặc feature extraction.

Hai A cases gồm362 frames/1.810 metric records:1.448 available,362 unavailable.
Gap case giữ sequence degrade, missing raw knee và null metric có reasons/flags;
không nâng availability vì processed joints đã imputed. Numeric temporal flag
vẫn false. Legacy v1 phase labels/messages/timeline không chiếu vào panel A để
tránh diễn giải chúng thành phase của nguồn B; source JSON gốc vẫn nguyên vẹn.

Panel B chỉ hiển thị stored PVP spans/top/bottom-anchor events, cycle roles và
raw necessary-condition checks. Timeline không phải motion-phase intervals;
không tạo event từ data reference. Shared-top roles đọc từ các cycle refs.
Point-turn, motion phase, arrival/departure/pause đều Unavailable/null như audit.
Không đổi raw-conditions pass thành point acceptance, không0s pause. Không có
full PVP thì ghi `no_full_pvp_candidate`; có PVP nhưng thiếu bounds thì ghi
`bottom_bounds_unresolved`. Original flags/run diagnostics và hashes có panel
provenance riêng, không giấu gap/side switch khi một candidate khác còn tồn tại.

9 B sources giữ1.659 samples,14 full PVP candidates,11 raw-condition passes,
**0 accepted point events**. Smooth/jitter noise ambiguity, plateau không strict,
imputed support bị từ chối; gap/side-switch không có candidate qua boundary,
clip-edge không có full PVP. Không đo accuracy/rep correctness trên người tập.

## Version/provenance và giới hạn nguồn

Presentation DTO `gym-squat-showcase-v1` là view riêng, không sửa frozen v1/v2
schemas/API. Typed saved records được deserialize/revalidate/detach trước render.
Selection config và source snapshots ghi byte SHA-256; point manifest summary,
audit/input equality và hashes được kiểm tra trước khi xuất. Input chỉ được
resolve dưới explicit workspace `outputs/`; Data, traversal, unknown versions,
nonfinite JSON và duplicate keys bị từ chối với structured reporting input error.
Output directory phải chưa tồn tại; file ghi exclusive, không overwrite.

- Profile S3 `squat-temporal-s3-v1`, hash
  `5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017` không đổi.
- Point policy/producer `squat-v2-point-resolution-v1`, policy hash
  `930d41199421d6544f775b4485ba2b347005498b836fbda6532ee16dce4e4642` không đổi.
- Conservative v2 `squat-v2-conservative-adapter-v1` và base policy/hash giữ nguyên.
- Registry `gym-feature-registry-v2`, hash
  `752305c17c7cf5dce2f7439af34d632fdd16e912a2a7d284276a90c07e8c9d72` giữ nguyên.

`demo-final/manifest.json` chứa source origins/snapshot paths/hashes, selection và
upstream manifest hash, output file hashes và environment. Sources được copy
byte-identical, không làm lại QualityReport hoặc numerical observations. Hash
không chứng minh measurement accuracy/viewpoint/annotation authenticity. Bundle
không đóng gói toàn bộ historical preprocessing/observation JSONL/model assets;
hash links tới upstream vẫn có trong feedback snapshots.

## Read-only noise/uncertainty review

Đã tìm theo noise/uncertainty/covariance/error-bound/confidence trên
`src/gym_coach`, `docs`, `configs`; đọc actual fields của JointObservation,
PoseExtractionConfig, ProcessedJoint, GeometryFeatureObservation và
SquatSignalSample. Kiểm tra selected2 feedback JSONs và1.659 saved signal samples
của9 point audit inputs. `noise-search.txt` giữ command output; `noise-review.json`
ghi scope, source hashes, field inventories, sample fields/counts và findings.

**Không tìm thấy independent calibrated noise/uncertainty evidence gắn từng
signal sample trong phạm vi này.** Visibility/presence là backend confidence/
availability, detection/tracking confidence là thresholds cấu hình; imputation,
smoothing status, quality coverage và content hashes là provenance/data quality.
Chúng không phải noise bound, covariance, repeated independent measurement hoặc
calibrated error interval cho geometry scalar. Các câu chữ/spec TODO về
uncertainty và point-policy abstention cũng không phải evidence độc lập.

Không kiểm kê Data/archive hoặc mọi artifact của các output runs không được
chọn; không tuyên bố đã chứng minh tuyệt đối rằng mọi file project không có
evidence. **Detector work dừng tại abstention**, không yêu cầu thêm thông tin,
thêm noise-free override hoặc phát triển rule mới trong milestone này.

## Tests và QA

30 tests mới đã chạy:30 passed/0 failed/error/skipped/warnings trong10.27s.
Related suite đầu:237 passed/0 failed/error/skipped/warnings/1 integration
deselected trong50.19s. Sau đó chỉ làm rõ label phase cho no-full-PVP và thêm
assertion tương ứng; tạo `demo-final`, giữ `demo` trước đó. Final regression
stdout/JUnit có tên riêng `final-related-tests` trong run này: **237 passed,
0 failed/error/skipped/warnings,1 integration deselected,49.17s**, exit0.

Tests bao phủ exact joints/metrics/values/null/reasons/evidence/hash, different
input IDs/hash despite matching FPS, no legacy phase projection, no producer/
geometry/Data I/O, malicious JSON/HTML escaping, unsafe links, tampered hashes/
manifest summary/path traversal/duplicate/nonfinite JSON, synthetic-only intake,
exclusive output/source byte copies. 30 tests mới không gọi feature/pose/temporal
producer để tạo fixture mới; chúng đọc saved synthetic artifacts. Related suite
còn chạy synthetic producer/geometry regression tests hiện có, không inference
hoặc real integration.

Node VM + DOM/Canvas stub thực thi actual embedded JS: independent A/B controls,
playback A không đổi clock/drawing B, event B không redraw skeleton A, displayed
metric values khớp saved JSON, missing joint không được vẽ, plateau/clip-edge
reasons và unavailable/null đúng. **Đây là local code QA, không browser QA**.
Không thực sự mở browser trong lượt này; chưa đánh giá visual layout,
cross-browser/mobile accessibility/performance. CSP chặn external connections,
không fetch/CDN/font/image/network dependency. Static/dynamic JSON links được
kiểm tra trong source manifest và evidence verifier.

Environment CPython3.11.9/Windows10.0.26200 x64, package0.7.0,
Pydantic2.13.5/pytest9.1.1, Node18.20.8 có sẵn. Không cài dependency mới hoặc
claim clean-environment compatibility; showcase tests cần saved source artifacts.
Một real-preview integration bị deselect; không chạy real integration. Full
project suite NOT_RUN, không gọi related suite là full suite.

Lệnh đã chạy, mỗi stdout/JUnit/temp/output riêng:

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.reporting.squat_showcase_demo --sources outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/selection.json --output-dir outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/demo-final

& .venv/Scripts/python.exe -B -m pytest tests/test_squat_showcase.py tests/test_point_resolution.py tests/test_temporal_v2.py tests/test_temporal_v2_renderer.py tests/test_temporal_squat.py tests/test_squat_feedback.py tests/test_temporal_sensitivity.py -m 'not integration' -q --basetemp outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/final-related-temp --junitxml=outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/final-related-tests.xml
```

Không chạy lại các lệnh tạo artifact vào output đã tồn tại; chọn directory và
test receipt names mới. Exact commands ở `commands.json`.

## Files và evidence cuối

Tạo `reporting/squat_showcase.py`, `reporting/squat_showcase_demo.py`,
`tests/test_squat_showcase.py` và báo cáo này. Cập nhật README/PROJECT_STATE/TASKS.
Không sửa domain schemas/flags/profile/producer/public exports/old renderers,
dependencies, old reports/receipts hoặc old outputs. Preservation checks theo
SHA-256 ở `preservation.json`; new artifact inventory tách riêng, không fake Git.

Baseline591 files:588 byte-identical, chỉ README/PROJECT_STATE/TASKS đổi;0 missing,
453 inventoried old artifact files giữ hash. Final bundle verify kiểm tra mọi
frame/joint/metric projection, saved event/check/status và snapshot bytes, links,
HTML embedded JSON equality, source/profile/policy identities và disabled gates.
Các outputs/test receipts trước của milestone được giữ, `demo-final` là bản giao.

Giới hạn còn lại: hai nguồn không thể liên kết như cùng một sequence, point/
phase/pause chưa available, synthetic measurement unvalidated, browser QA chưa
chạy và official input/temporal/assessment gate vẫn blocked. Milestone này hoàn
thành trải nghiệm saved-artifact presentation; không hoàn tất G5/G6 hoặc xác
thực trên người tập. Không tiếp tục detector work khi chưa có evidence độc lập.
