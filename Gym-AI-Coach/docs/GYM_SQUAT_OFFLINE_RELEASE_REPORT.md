# BodyWeightSquats — offline ZIP release

2026-10-18, Asia/Saigon. **PACKAGED / PASSED_OFFLINE_CHECKS** chỉ xác nhận
đóng gói/integrity offline. **NOT_VALIDATED**, official_eligibility=false,
assessment=false; G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT,
G6/toàn G5 IN_PROGRESS. Không có Git repository, không tạo commit/diff giả.

## Nhận và mở demo

ZIP mới:
`outputs/squat_offline_release/run-0859ad662d114359a3c38b5298dfff97/release/BodyWeightSquats-offline-demo.zip`.

1. Dùng **Extract All / Giải nén tất cả** vào một thư mục mới. Không mở riêng
   HTML bên trong cửa sổ ZIP, không tách/đổi tên các sidecars hoặc `sources/`.
2. Mở `START_HERE.html` ngay tại gốc thư mục vừa giải nén, chọn **Mở demo**.
   Có thể đọc `HUONG_DAN_MO_DEMO.txt`; hướng dẫn UTF-8 BOM giữ nguyên bytes.
3. Không cần Python, dependency, mạng hoặc server để trình bày. Nếu chính sách
   máy chặn mở file local, dừng; không tắt bảo vệ hoặc dựng server để đi vòng.

Hai panel và scrubber vẫn độc lập. Unavailable/null là thiếu kết quả đủ evidence,
không phải0 hoặc lỗi người tập. Không mở point-turn, phase, arrival/departure,
pause, score hoặc lời khuyên kỹ thuật. Synthetic demo chưa validated trên người.

## Nguồn duy nhất và inventory

Chỉ lấy bytes từ
`outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle/`.
Không tái tạo HTML, render, resolve original paths trong selection/provenance,
đọc Data/video, inference, tính góc, chạy producer hoặc lặp noise audit.
Existing handoff tests dùng fixture saved showcase của chính tests đó;
**không** dùng fixture test làm nguồn ZIP phát hành.

| Thuộc tính đã đo | Giá trị |
|---|---|
| Release metadata version | `gym-squat-offline-release-v1` |
| Files trong ZIP, gồm manifest nguồn | **20** |
| Tổng bytes chưa nén | **16.679.343** |
| ZIP bytes | **1.221.944** |
| ZIP SHA-256 | `34f7f73fe4d50f33aa3a3e51ec507f756c62bbfebdac36c6412049cc7743362d` |
| Manifest nguồn SHA-256 | `f272b17658b4defc0e6f472bb559a87074bcf3d97a976a462237c3cc9a9717ef` |

ZIP giữ nguyên8 root files và12 snapshots trong `sources/` (11 feedback/point
snapshots và1 frozen noise-review snapshot). `START_HERE.html` ở gốc ZIP.
Nguồn và bản giải nén có cùng tên tương đối, bytes, SHA-256 và kích thước từng file.
Manifest handoff gốc được giữ nguyên, không thay bằng manifest release.

Sidecars phát hành **bên ngoài ZIP** trong `release/`:

- `release-manifest.json`: source path/hash, SHA-256/size cho mọi file, kể cả
  `manifest.json` nguồn; SHA-256/size/count của ZIP; trạng thái validation/gates.
- `SHA256SUMS.txt`:21 entries — ZIP và20 file trong `extracted/`.
- `verification.json`: source/extraction checks, links, null/gates và source unchanged.
- `extracted/`: thư mục giải nén thử mới, giữ lại để review; không ghi đè thư mục cũ.

ZIP chỉ chứa nguyên source bundle, không thêm manifest tự tham chiếu vào bên
trong. Nếu chuyển giao cần kiểm tra ZIP, gửi kèm `release-manifest.json` và
đối chiếu `Get-FileHash -Algorithm SHA256` với hash ở trên. Hash xác nhận bytes,
không chứng minh correctness/accuracy hoặc nguồn viewpoint chính thức.

## Verification thực tế

CLI dùng exclusive output/ZIP/file creation. Kiểm tra inventory và hashes nguồn
trước khi ghi, từ chối path traversal, đường dẫn ngoài outputs, Data path,
symlink/junction; không theo đường dẫn upstream được ghi trong JSON. ZIP được
giải nén vào đường dẫn chưa tồn tại, kiểm tra duplicate/case collision,
unexpected/missing entries, symlink/device, kích thước, CRC và SHA-256.

Kết quả trên **bản giải nén thực tế**, không phải kết quả dự kiến:

- **20/20** file hash/size khớp; source unchanged sau đóng gói.
- **2** HTML pages; **9** local links/fragments và **12** snapshot links hợp lệ.
- Embedded presentation JSON bằng sidecar, escaping/CSP offline giữ nguyên.
- **362** unavailable metrics giữ null/reasons; **9** point cases vẫn abstain,
  accepted_point_count=0, motion unavailable, arrival/departure/pause null.
- Independent panels và tất cả official/assessment gates giữ nguyên.
- **browser_qa=NOT_RUN**,0 screenshots. Lượt này không mở UI/browser; không
  dựng server/mở network/đi vòng security policy. Không xác nhận browser rendering,
  native keyboard, desktop layout hay compatibility trên máy người nhận.

`verification.json` ghi PASSED cho integrity offline, không nâng NOT_VALIDATED.
Module mới dùng thư viện chuẩn; package reporting vẫn có các import contract
Python sẵn có. Fresh import không kéo NumPy/SciPy/OpenCV/MediaPipe/Torch.
Không thêm dependency, thay API/schema/profile/producer/renderer.

## Tests và lệnh

Windows10.0.26200 x64, CPython3.11.9, pytest9.1.1/Pydantic2.13.5,
Node18.20.8 sẵn có cho test handoff JS. Không kiểm tra clean environment.

| Receipt | Pass | Fail | Error | Skipped | Deselected | Warnings | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| tests-first |53|2|0|0|0|0|21.61s|
| **tests-final** |**57**|**0**|**0**|**0**|**0**|**0**|**19.87s**|

Final57 =39 packaging tests mới +18 handoff tests hiện có, không cộng lặp runs.
Hai fail ban đầu là assertions: zipfile Windows chuẩn hóa backslash trước
validation, và reporting initializer đã import segmentation Python sẵn có.
Chỉnh test theo behavior thực tế; không sửa initializer hay domain semantics.
Thêm regression duplicate/device ZIP, giữ cả receipt đầu và cuối.

Tests kiểm tra byte-preserving tree/guide, manifests/checksums/CRC, source-only
reads/no producer, fresh output/extraction, tampered hashes/size/inventory,
unsafe paths, source refs, local fragments/network links, embedding/escaping,
null/gates và lightweight import/CLI dispatch. Handoff tests giữ coverage
DOM/JS state/keyboard structure/linkage; đây không phải browser QA.

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_offline_release.py tests/test_showcase_handoff.py -q --basetemp outputs/squat_offline_release/run-0859ad662d114359a3c38b5298dfff97/tests-final-temp --junitxml=outputs/squat_offline_release/run-0859ad662d114359a3c38b5298dfff97/tests-final.xml

& .venv/Scripts/python.exe -B -m gym_coach.reporting.offline_release --bundle outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle --output-dir outputs/squat_offline_release/run-0859ad662d114359a3c38b5298dfff97/release
```

Các lệnh trên là evidence đã chạy; khi chạy lại **phải chọn output, basetemp,
JUnit path mới**, không ghi đè artifacts/receipts. `build-command.json`,
`build-stdout.json`, `tests-*-command.json`, stdout/XML và `test-summary.json`
nằm trong run mới. Không chạy full suite/broader temporal tests vì không đổi
domain code; không real-data integration, browser QA hoặc inference.

## File thay đổi và giới hạn

Thêm `src/gym_coach/reporting/offline_release.py`,
`tests/test_offline_release.py` và báo cáo này; cập nhật tối thiểu
README/PROJECT_STATE/TASKS. Mọi writes artifact nằm trong run mới.
Baseline163 file gồm source bundle20 và code/tests/docs/config hiện có;
160 byte-identical, chỉ README/PROJECT_STATE/TASKS đổi,0 missing; toàn bộ20
file bundle nguồn giữ hash. `preservation.json` ghi kết quả so sánh. Không đọc nội dung
old outputs khác cho inventory hoặc sửa source bundle/report/receipt cũ.

Milestone đóng gói đã hoàn tất kiểm tra offline. Bước nhỏ tiếp theo: mở bản
giải nén trên desktop bằng cơ chế được phép để ghi browser/native keyboard QA.
G5.7 vẫn cần documented side-view protocol/annotation đúng video/hash và
sequence/bilateral knee allow; ZIP không giải quyết blocker đó. G6 IN_PROGRESS,
NOT_VALIDATED, assessment=false/official=false giữ nguyên, không xác thực trên người.
