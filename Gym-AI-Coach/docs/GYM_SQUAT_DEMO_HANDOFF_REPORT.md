# BodyWeightSquats — demo handoff

2026-10-18, Asia/Saigon. **HANDOFF_READY / OFFLINE_QA_ONLY**.
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Không có Git repository; không tạo commit/diff giả.

## Mở và trình bày

1. Giữ nguyên thư mục
   `outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle/`.
2. Mở `START_HERE.html` bằng trình duyệt desktop có sẵn, chọn **Mở demo**.
   Có thể mở trực tiếp `index.html` qua File Explorer hoặc Ctrl+O. Không cần
   Python/Internet/server/cài dependency hoặc thay security settings.
3. Tab/Shift+Tab chuyển điều khiển, Enter/Space bật nút; slider dùng phím mũi
   tên/Home/End khi có focus. A/B có clock riêng. Có link nhảy đến từng panel.

Giữ cạnh HTML: `manifest.json`, `presentation.json`, `noise-review.json`,
`selection.json`, `upstream-manifest.json`, `sources/` và toàn bộ JSON con.
`HUONG_DAN_MO_DEMO.txt` (UTF-8 BOM) giải thích mở file, keyboard, các trạng thái
và một luồng trình bày ngắn. Riêng index.html có embedded data/JS/CSS, nhưng
sidecar links cần cả thư mục; không tách/đổi tên từng file. Nếu chính sách máy
chặn local file, dừng; không hướng dẫn tắt bảo vệ hoặc dựng server để đi vòng.

Panel A: chọn normal/gap, tua gap frame60 để xem missing joint và metric null.
Panel B: chọn smooth/plateau/clip_edge để xem stored PVP/event/reasons.
Hai panel khác input hashes/provenance nên không đồng bộ hoặc nối metric A
với temporal B, kể cả cùng FPS/frame number.

Unavailable/null nghĩa là chưa có kết quả đủ evidence/quality, không phải0 hoặc
lỗi kỹ thuật của người tập. EXPERIMENTAL/NOT_VALIDATED là synthetic prototype
chưa validated trên người. PVP/raw-condition pass không phải accepted point-turn.
Point/phase/arrival/departure/pause vẫn unavailable; không0s pause. Không chấm
form vì chưa có validation/rules được xác nhận và official gate còn blocked;
không có score, lời sửa kỹ thuật hoặc kết luận đúng/sai.

## Nguồn và thay đổi

Nguồn duy nhất là bundle đã lưu của
[showcase milestone](GYM_SQUAT_SHOWCASE_REPORT.md):
`outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/demo-final/`.
Reader kiểm tra manifest/file hashes, disabled gates, independent linkage và
typed synthetic snapshots; rebuilt presentation phải bằng saved DTO. Không
resolve `selection.json` về upstream runs hoặc đọc Data. Existing noise review
được copy như metadata cũ; **không chạy lại audit noise** hoặc phân tích samples.
Không inference, feature calculation, detector, profile/schema/producer change.

Tạo `reporting/showcase_handoff.py` (CLI đóng gói),
`tests/test_showcase_handoff.py`, báo cáo này. Sửa presentation-only
`reporting/squat_showcase.py`; cập nhật README/PROJECT_STATE/TASKS.
Old outputs/reports/test receipts giữ nguyên.

Các usability changes có evidence offline:

- Hai play buttons trước đây không khai báo ARIA pressed state; giờ có initial
  false và cập nhật khi play/pause, scrub, đổi nguồn, đến cuối clip và replay.
- Slider công bố frame/time qua ariaValueText, liên kết output/help với control.
  Không thêm global key handler hoặc chặn native keyboard behavior.
- Navigation A/B/hướng dẫn, focus style cho links/summary; wrap hash dài trong
  status. Đây là DOM/CSS improvements, không tuyên bố đã quan sát layout desktop.
- Khi thêm handoff template marker, giữ chuỗi metadata trùng marker như JSON
  data; regression test kiểm tra marker + `</script>`/ampersand/Unicode separators.
  Không đưa untrusted strings vào innerHTML hoặc thay nội dung JSON bằng HTML.

New index.html chỉ đổi presentation. Các payload sidecars và source snapshots
copy byte-identical; upstream manifest lưu riêng để nhận diện nguồn. Manifest
mới là manifest của bản handoff, không dùng upstream index hash cho HTML mới.
Embedded presentation JSON bằng bản cũ; source/profile/policy hashes không đổi.

## Browser/desktop QA

**browser_qa=NOT_RUN; desktop_visual_qa=NOT_RUN;0 screenshots.**
Trong lượt này đã chọn in-app browser và đọc capability/documentation. File URL
navigation bị security policy từ chối ở lượt trước (chỉ http/https); native UI
APIs không được phép. Không có local-file path mới được cấp quyền, không retry
file navigation hoặc dùng alternate browser/UI surface để đi vòng. Không dựng
server, mở network hoặc tắt bảo mật. `browser-qa.json` ghi đúng thao tác hiện
tại và dẫn prior rejection evidence; không giả báo một lần mở file mới thất bại.

Vì không mở được demo, không có desktop screenshot/visual issue được xác nhận.
Native Tab/Space/Arrow/Home/End behavior, screen-reader, desktop layout và
cross-browser rendering còn **NOT_RUN**. Không tạo screenshot thay bằng mockup,
HTML rasterizer hoặc browser automation path khác.

Offline checks chỉ xác nhận native controls/labels/DOM order/no positive
tabindex/no key trap, focus CSS, app JS state và data/link invariants. Node VM
với DOM/Canvas stub không phải browser hoặc native keyboard QA.

## Test evidence

Environment: Windows10.0.26200 x64, CPython3.11.9, package0.7.0,
Pydantic2.13.5/pytest9.1.1, Node18.20.8 sẵn có; không dependency mới.

| Receipt | Pass | Fail | Error | Skipped | Deselected | Warnings | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| tests-first |46|0|0|0|0|0|19.54s|
| tests-final (thêm marker escaping) |47|0|0|0|0|0|22.13s|
| **tests-verified (thêm linkage tamper)** |**48**|**0**|**0**|**0**|**0**|**0**|**22.27s**|

Final48 =18 tests handoff mới +30 existing showcase tests, không cộng lặp runs.
Không chạy broader temporal/full suite vì domain producer/schema không đổi;
saved data equality và existing showcase coverage đã kiểm tra compatibility
cho presentation change. Không real-data integration/inference.

Tests mới: portable bundle-only read (original source paths không tồn tại trong
test workspace), no pipeline/geometry/noise audit/upstream reload, byte copies/
hashes/null/gates/linkage, local links/fragments/sidecars, UTF-8 guide, keyboard
DOM semantics/order/focus, embedded JSON escaping/marker, exclusive output,
tampered hash/path/presentation/gate/version/linkage bị từ chối.
Node state test xác nhận ariaPressed/ariaValueText, source/scrub/play/end/replay
và clock A/B độc lập. Existing30 giữ metrics/skeleton/events/unavailable behavior.

Lệnh đã chạy, các output directories/receipts đều mới:

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.reporting.showcase_handoff --bundle outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c/demo-final --output-dir outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle

& .venv/Scripts/python.exe -B -m pytest tests/test_showcase_handoff.py tests/test_squat_showcase.py -q --basetemp outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/verified-tests-temp --junitxml=outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/tests-verified.xml
```

Không chạy lại vào bundle/JUnit cũ; chọn đường dẫn mới. `commands.json` lưu mọi
command, `environment.json` ghi runtime; stdout/XML được giữ riêng từng run.
Preservation/inventory/equality/link receipt nằm trong run handoff mới, không
ghi đè receipts của showcase. Chưa claim clean-environment compatibility.

Baseline646 files:642 byte-identical; chỉ renderer showcase và3 docs
README/PROJECT_STATE/TASKS đổi,0 missing.504 inventoried old artifacts giữ hash.
`qa-offline.json` đối chiếu links/fragments, frozen copied JSON/embedded DTO,
ARIA issue trước/sau và disabled gates; `preservation.json`/inventory ghi hashes
và source identity mới. Noise-review payload giữ byte hash cũ, không re-audit.

Giới hạn còn lại là browser/native keyboard QA chưa chạy, hai nguồn không thể
liên kết như một sequence, temporal/assessment vẫn unavailable và synthetic
chưa validated. Handoff không thay G5.7/G6 hoặc tiếp tục detector work.
