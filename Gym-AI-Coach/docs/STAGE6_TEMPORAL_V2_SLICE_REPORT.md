# GĐ6 — additive experimental temporal v2 slice

2026-10-18, Asia/Saigon. **Lát cắt contract/adapter/renderer đã triển khai;
không hoàn tất G6/G5, chưa có bounds detector.** NOT_VALIDATED,
official_eligibility=false, assessment.enabled=false. G5.7 vẫn
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT; G6 và toàn G5 IN_PROGRESS.

Run độc lập:
`outputs/squat_temporal_v2/run-7cae351170a441ba85b75e84b4d8e26d`.
Workspace không có Git repository; không tạo commit/diff giả. Không mở video/
metadata Data, chạy inference/real integration, dùng archive hoặc thay dependency.

## 1. Quyết định và phạm vi thực tế

Theo [proposal](SQUAT_TEMPORAL_CONTRACT_PROPOSAL.md), tách event khỏi motion phase,
anchor khỏi arrival/departure/pause, abstention từng component. Đã đóng băng
[contract v2](SQUAT_TEMPORAL_V2_FROZEN_CONTRACT.md), models, schema và43 acceptance
tests **trước khi tạo producer/renderer/demo**. `freeze-receipt.json` lưu thời điểm
UTC, hashes và xác nhận ba implementation files chưa tồn tại; snapshot models/
tests/document/schema được giữ trong run. Expected-red collection: thiếu module
producer, 1 error, không coi đây là test pass.

Producer `segment_squat_v2` nhận đúng `SquatTemporalInput` synthetic hiện hành,
chạy nguyên v1/default S3 rồi chuyển anchors/candidates sang v2. Không nhận
fixture definitions/reference labels hay `GeometryFeatureObservation` để nâng
temporal permission. Không tính góc; scalar fixtures là tín hiệu giả lập theo
contract input cũ, không phải góc đo từ joint hoặc vận động viên. Không chạy lại
30 điều kiện sensitivity hoặc chọn threshold để khớp dữ liệu đó.

**Quyết định bảo thủ:** proposal chưa có evidence rule cho bounds hoặc point-turn
resolution. Producer vì thế chỉ xuất `bounds_unresolved`: top/bottom-anchor và
cycle candidate có thể biết, còn arrival/departure/pause/motion unavailable.
Không phát hiện flatness, không lấy anchor ± window, không suy point-turn từ PVP,
không đọc ground truth để lấp bounds. Ngay cả input được generator đặt hold cũng
không được báo pause đã đo.

Positive `dwell_candidate`/`point_turn_at_resolution` hiện là **ví dụ output
contract thủ công trong tests**, gắn `producer_version=contract_acceptance_only`.
Chúng kiểm tra schema/half-open projection/renderer, không phải detector outputs
và không được đưa vào producer hoặc demo như dữ liệu đầu vào. Method/hash/support
record chỉ kiểm tra linkage/provenance, không chứng nhận tính đúng của bounds.

## 2. Semantics/schema đã chạy

- Schema `0.2.0`, contract `squat-temporal-components-v2`; đọc v1 `0.1.0` và v2
  bằng dispatch tường minh, không migration. Old classes/exports/serializers/UI
  và schema0.1 giữ nguyên.
- Policy `squat-v2-anchor-only-v1`, hash
  `6a9435df2b3f6d8ccc694820a20d3cf4d75f3ba7beeea12ae4339f0dc892e43f`.
- S3 `squat-temporal-s3-v1`, hash nguyên vẹn
  `5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`.
  Không thêm tham số profile hoặc custom-profile entry point cho adapter v2.
- Full legacy input/result + hashes giữ frame/timestamp, raw scalar, imputed/
  observed/missing provenance, missing mask, original flags và filter support.
  Event ID SHA-256(input hash/run/kind/frame); times đúng samples, không subframe.
- Events top/bottom_anchor/bottom_arrival/bottom_departure riêng khỏi intervals.
  Inclusive evidence support khác half-open motion/pause `[)`. Shared top cùng
  input/run/side có một ID, hai role refs; phase tại top thuộc cycle tiếp theo.
  Final top có event nhưng không kéo dài Ascent sau E.
- Dwell contract S<A<D<E, A≤B≤D: Descent[S,A), Bottom[A,D), Ascent[D,E);
  pause D−A, không cộng frame. Point contract: Descent[S,B), Ascent[B,E),
  physical bounds/pause null với `pause_unresolved_at_resolution`, không0s.
- Mỗi component candidate/unavailable/not_run/blocked; abstain bắt buộc null/
  empty và V2Reason. Category insufficient_data, stage segmentation, source S3,
  inclusive support; không thêm/đổi FlagCode cũ. Producer input lỗi dùng
  INVALID_TEMPORAL_INPUT/input_error/segmentation/S3 theo v1.
- Gap/side-switch split finite runs và event support. Count là full PVP
  **candidate count**, không true repetitions. Clip kết thúc trong hold: không
  tạo end/departure hoặc complete cycle; giữ partial-end/no-confirmed diagnostics.

Giới hạn schema slice: chưa lưu event của một partial cycle ngoài full PVP,
chưa hỗ trợ arrival độc lập khi departure chưa biết. Các partial cases abstain
cả bounds; đây là thiếu capability, không kết luận arrival/pause không tồn tại.

## 3. Demo thực tế: input → JSON → report → HTML

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.reporting.temporal_v2_demo --output-dir outputs/squat_temporal_v2/run-7cae351170a441ba85b75e84b4d8e26d/demo
```

Lệnh trên đã chạy exit0, tạo thư mục mới. Không chạy lại cùng đường dẫn; CLI
từ chối overwrite. Muốn chạy mới, chọn một output directory chưa tồn tại.
Mở `demo/index.html`, chọn case và tua frame. HTML độc lập/offline, không cần
framework/server/dependency mới. Motion phase và event overlay tách riêng;
pause chưa xác định hiển thị Unavailable/reason. Không hiển thị góc/score/form advice.

| Input generator | FPS | Frames | PVP candidates | Resolved phase/pause candidates |
|---|---:|---:|---:|---:|
| dwell_signal-25fps |25|151|2|0|
| dwell_signal-30fps |30|181|2|0|
| dwell_signal-60fps |60|361|2|0|
| point_turn_signal-30fps |30|181|2|0|
| gap-30fps |30|181|1|0|
| side_switch-30fps |30|181|0|0|
| clip_ends_in_pause-30fps |30|61|0|0|

Tên case mô tả generator, **không phải mode được detector xác nhận**. Tổng7
executions/1.297 samples/9 full PVP candidates; tất cả9 bounds_unresolved.
Không tính metric accuracy/recall từ bảng này. Unknown không được tính thành đúng
hoặc “không có pause”. Không có real/official eligibility.

Mỗi case lưu `input.json`, `result.json`, `index.html`; JSON readback so equality.
`demo/manifest.json` có byte SHA-256 cho input/result/HTML, profile/policy hashes,
frames/FPS/counts và environment. `demo/report.md`, `schema-v2.json` bổ sung audit.
Root run có commands/test stdout/JUnit/freeze/environment/preservation/inventory.

## 4. Kiểm thử đã chạy và kết quả

Environment: Windows10.0.26200 x64, CPython3.11.9, package0.7.0, Pydantic2.13.5,
pytest9.1.1; Node18.20.8 đã có sẵn, chỉ dùng local JS code test. Không cài thêm gì.
MediaPipe0.10.14/NumPy2.4.6/SciPy1.17.1/OpenCV5.0.0.93 chỉ ghi installed versions,
không load backend/inference cho lượt này. Không claim clean-environment compatibility.

| Run/evidence stem | Passed | Failed | Error | Skipped | Deselected | Warnings | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| acceptance-red (expected missing producer) |0|0|1|0|0|0|1.91s|
| acceptance-first |42|1|0|0|0|0|4.61s|
| acceptance-final |43|0|0|0|0|0|4.12s|
| related-tests (trước renderer supplement) |167|0|0|0|1|0|28.75s|
| renderer-first (test import sai) |0|0|1|0|0|0|1.92s|
| renderer-final |6|0|0|0|0|0|9.19s|
| **final-related-tests** |**173**|**0**|**0**|**0**|**1**|**0**|**37.37s**|

43 acceptance mới +6 renderer/guard tests mới +124 existing temporal/feedback/
sensitivity tests =173. Một test real-preview integration được deselect bởi
`-m 'not integration'`. Không gọi real integration. Full project suite **NOT_RUN**;
không trình bày related suite thành full suite. Các con số không cộng qua runs
để giả thành số tests duy nhất.

Acceptance-first fail vì test chặn nhầm cả lightweight contract modules
pose/video/features do old public imports nạp; sửa guard chặn heavy backends và
synthetic_fixtures, không đổi expected temporal semantics. Bản acceptance gốc
và `acceptance-harness-erratum.json` giữ cả hashes. Đây là hiệu chỉnh harness sau
freeze, không giấu như thể freeze test source không đổi. Models/document/schema
freeze giữ nguyên. Renderer-first error do sai import trong test bổ sung, sửa
thành `from gym_coach import geometry`; không sửa v1 để làm test pass.

Frozen acceptance đã kiểm tra dwell/point/unresolved, pause null/reason, shared
top/two roles/half-open ownership, gap/side at top, clip ends in pause, strict
serialization/version dispatch/tampering, default profile/v1 canonical output,
heavy import guard, fixture-wrapper rejection/source immutability, renderer
escaping và exclusive-output demo. Tolerance1e-9s cho duration/CFR arithmetic,
không phải accuracy bound. Tests không chứng minh detector chưa triển khai.

Renderer supplement thực thi embedded JS bằng Node VM với DOM stub; so mọi frame
với Python projection ở dwell/point/unresolved/gap/empty contract cases. Đồng thời
chặn `joint_angle`/axis calculation và `io.open` dưới Data. Đây là local code
tests, **không phải browser/visual QA**. Browser tool đã từ chối `file://` do URL
security policy; không dùng server/surface khác để đi vòng qua chặn, không có
screenshot hoặc tuyên bố đã xem UI trực quan.

Lệnh cuối đã chạy:

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_temporal_v2.py tests/test_temporal_v2_renderer.py tests/test_temporal_squat.py tests/test_squat_feedback.py tests/test_temporal_sensitivity.py -m 'not integration' -q --basetemp outputs/squat_temporal_v2/run-7cae351170a441ba85b75e84b4d8e26d/final-related-temp --junitxml=outputs/squat_temporal_v2/run-7cae351170a441ba85b75e84b4d8e26d/final-related-tests.xml
```

Lệnh acceptance riêng dùng `tests/test_temporal_v2.py -q` với stem
`acceptance-final`, renderer riêng dùng `tests/test_temporal_v2_renderer.py -q`
với stem `renderer-final`; cùng interpreter `-B`, mỗi run có basetemp/JUnit/stdout
riêng. Commands đầy đủ được lưu trong `commands.json` của run mới.

## 5. Files thay đổi và bảo toàn

Tạo4 implementation modules: `segmentation/temporal_v2_models.py`,
`segmentation/temporal_v2.py`, `reporting/temporal_v2_renderer.py`,
`reporting/temporal_v2_demo.py`. Tạo2 test files: `test_temporal_v2.py`,
`test_temporal_v2_renderer.py`. Tạo frozen-contract document và báo cáo này.
Cập nhật tối thiểu README, IMPLEMENTATION_PLAN, PROJECT_STATE, TASKS,
SCHEMA_CONVENTIONS. Proposal/sensitivity report và old artifacts giữ nguyên.

`preservation.json` so baseline421 files; chỉ5 docs trạng thái/conventions nêu
trên được phép đổi. Source/tests/configs/dependency manifest v1 và các related
artifact cũ được so byte SHA-256, không chỉ dựa test pass. Frozen schema/model/
document được đối chiếu receipt; test harness correction có receipt riêng.
`artifact-inventory.json` lưu hashes các artifact mới (trừ pytest temp và chính
inventory để tránh self-hash). Không kiểm kê/đọc Data để làm bằng chứng này.

## 6. Giới hạn và bước nhỏ kế tiếp

Chưa có evidence rule hay detector arrival/departure, point-turn resolution,
top dwell, partial-event retention hoặc noisy/imputed pause; v2 producer vì thế
chưa có motion phase available. Không so accuracy với30 sensitivity conditions
vì thay semantics/label closure không tự chứng minh cải thiện. Scalar synthetic
không đại diện người tập, camera 2D hoặc pose noise. Chưa browser layout QA,
clean-env check, real temporal admission, expert sources hoặc assessment.

**Bước nhỏ tiếp theo duy nhất:** thiết kế và duyệt evidence rule cho một detector
point-turn ở sample resolution (kèm abstention cho noisy/imputed/censored spans),
freeze acceptance độc lập rồi mới nối producer. Không chọn ngưỡng từ30 fixtures,
không gán physical pause=0. Dwell/partial bounds vẫn unavailable tới khi có rule
và tests riêng. G5.7 cần documented input/fresh sequence + bilateral allow và
temporal admission riêng; slice này không tháo gate đó hoặc xác thực vận động viên.
