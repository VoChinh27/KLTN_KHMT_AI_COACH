# Dọn project khi chuyển sang WEB

Cập nhật2026-10-24. Dọn có giới hạn, manifest trước từng nhóm thao tác; workspace
không Git. Không giả có commit hoặc Git restore. Không xóa dữ liệu/outputs/docs hàng loạt.

Evidence run: `outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/`.
[Manifest trước cleanup](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/cleanup-manifest.json),
[receipt sau cleanup](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/cleanup-receipt.json).

## Kiểm kê và quyết định

Đọc README, state/tasks, ground-truth spec, single-video/context-performance/
reference-readiness reports; kiểm tra source/tests/CLI entry points, pyproject,
scripts, docs links và manifest/source-code references. Baseline885 file identities
gồm source/tests/docs/config/scripts/rules, README/pyproject, manifest/inventory
JSON cũ và source clip đã khóa; không claim đã hash mọi bytes toàn Data/outputs.

| Nhóm | Evidence consumer / quyết định |
|---|---|
|Pipeline single-video, pose/cache/quality/preprocess/session/adapters|CLI, tests và new web worker sử dụng; giữ nguyên thuật toán/contracts/profiles|
|Progress/worker/verification|Tách từ desktop_controller thành job_controller,45 tests active; public JSONL dùng lại|
|Web assets/HTTP entry point|Giao diện chính mới, metadata/package-data rõ; không dependency/framework mới|
|desktop.py + desktop_controller.py|Consumer active chỉ Tk entry point và test gốc; web không import chúng; archive nguyên bytes sau130 tests pass|
|test_desktop_progress.py|44 shared tests migrate sang test_job_progress.py;1 Tk defaults/close guard retired cùng UI; không xóa test để che regression|
|Legacy/research demos, scripts, temporal prototypes|Còn module CLI/__main__, tests, báo cáo và manifest references; giữ active để truy vết, không suy unused từ import|
|Python/pytest caches|94 files chỉ trong src/tests/scripts __pycache__ và root .pytest_cache; có thể tái tạo, xóa từng file được liệt kê|
|Old outputs/test basetemps/pose caches|Có receipts/commands/source identities; giữ toàn bộ. Không xóa một phần evidence bundle|
|Data/model/config/profiles/reference/spec/archive cũ|Protected; không sửa/xóa/move|
|Historical reports/docx|Giữ nguyên; DOCUMENTATION_INDEX phân loại lịch sử, không dùng làm trạng thái hiện hành|

Không tìm thấy Git repository ở bước đầu. Old outputs có references đến desktop
source hashes và test paths; các bytes gốc được giữ tại retired và manifest map
original→destination, không sửa ngược manifests hoặc câu lệnh cũ.
Run desktop dc236431… đã có completed launcher receipt0/137 khi kiểm kê; giữ nguyên,
không diễn giải thành agent đã QA GUI end-to-end ở milestone trước.

## Đã lưu trữ, đã xóa, dung lượng

Thư mục mới:
`retired/desktop-658f0fc196254cb49a94a5ecc92bf376/`.
Giữ nguyên cây con/bytes và kèm cleanup-manifest.json/README hướng khôi phục:

- `src/gym_coach/desktop.py`
- `src/gym_coach/desktop_controller.py`
- `tests/test_desktop_progress.py`

**3 files/38.751 bytes chỉ chuyển vị trí, không phải dung lượng được giải phóng.**
Shared code được copy/tách trước archive; migrated tests đạt và web HTTP path hoạt
động trước cleanup. Module Tk không còn entry point active; đây là thay đổi giao diện
được user yêu cầu, không xóa source. One Tk-only test được phân loại rõ như trên.

**94 cache files/2.275.887 bytes logical đã xóa** (khoảng2,17MiB). Không đo allocation
NTFS/compression/free-space vật lý; không gọi đây là mức giảm dung lượng ổ đĩa ròng.
Tests/imports có thể tái tạo caches; final-integrity ghi paths tái xuất hiện/bytes
thực tại thời điểm kiểm tra. Output mới/evidence lớn hơn lượng cache đã xóa, không
claim project tổng đã nhỏ đi2,17MiB. Không empty-file hoặc output-fixture deletion
khác vì chưa đủ bằng chứng không còn references.

Script one-time cleanup.py resolve absolute path, xác nhận trong root, từ chối
symlink/reparse, không dùng root/drive/user-dir làm recursive target. Manifest ghi
path/type/reason/size/SHA/recovery **trước** unlink/move. Mỗi file được re-hash ngay
trước thao tác; destination phải chưa có; archive bytes được kiểm tra sau move.
Chỉ rmdir các cache directories sau khi file đã xóa, không rmtree outputs/archive.

## Khôi phục / tái tạo

Đọc retired README và cleanup-manifest.json. Với mỗi action=archive, copy file từ
destination về original path **chỉ khi original chưa tồn tại**, kiểm tra SHA-256
khớp manifest; không overwrite code hiện hành. Restore cả hai modules Tk trước khi
chạy archived test; reconcile shared-controller changes nếu muốn desktop active
trở lại. Gói này là historical source, không được import làm implementation web.
Không cần Git để khôi phục.

Bytecode tái tạo khi Python import không dùng -B; .pytest_cache tái tạo khi pytest
không dùng `-p no:cacheprovider`. Không cần phục hồi cache để chạy web/tests.
Không chạy lại cleanup.py: script từ chối khi manifest đã tồn tại. Cleanup mới cần
inventory/manifest mới vì evidence/references có thể đã thay đổi.

## Dọn thông tin

README viết lại thành một lệnh WEB/URL/dừng/cách dùng/giới hạn; bytes README cũ
được snapshot ở run/README.before-web.md. Links trong snapshot tính từ project
root, không dùng snapshot làm entry point. PROJECT_STATE/TASKS thêm trạng thái mới,
giữ các measurements lịch sử; mục progress chưa làm đã được thay bằng trạng thái
thực hiện. DOCUMENTATION_INDEX ghi active và lịch sử/superseded-as-main-interface.

Các đoạn Post-Temporal/MASTER_PROMPT cũ trong .agent/AGENT_RULES và bản docx tiến độ
được index cảnh báo là lịch sử, không lấy thay PROJECT_STATE/GYM_REBUILD_MASTER_PROMPT.
Không sửa spec, protocol, số liệu, receipts hoặc báo cáo Giai đoạn0–6 cũ.

## Verification và nội dung giữ lại cần quyết định sau

130 tests pass trước cleanup. Sau cleanup:1566 pass/25 integration deselected,
0 fail/error/skipped/pytest warnings; shared import/CLI/progress/pipeline/renderer/
web hoạt động. Browser/upload/real worker/cache/null/report/play/seek/download đã
chạy; [web report](SQUAT_WEB_DEMO_REPORT.md). Final preservation/link/entry-point
checks nằm final-integrity.json/delivery-check receipts; không chỉ dựa tests mock.

Giữ toàn bộ old run bundles, test basetemps, dataset duplicate sources, pose caches,
prototype renderers/CLIs, old report/docs/artifacts, docx, .venv và _archive có sẵn.
Nếu cần giảm dung lượng lớn, phải có retention/reproducibility decision theo **cả
bundle** và reference map; milestone này không tự quyết định xóa chúng.

NOT_VALIDATED, official_eligibility=false, assessment=false, REFERENCE_PENDING;
G5.7 blocked, G6/toàn G5 IN_PROGRESS. Dọn file/UI không xác nhận accuracy hoặc form.

## Kết quả kiểm tra bàn giao cuối

[Inventory thành phần](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/component-inventory.json)
và [preservation receipt](../outputs/squat_web_milestone/run-7d679fc9e7034e7d8ddf309b1bbd7903/final-integrity.json)
ghi885 baseline files:877 unchanged,5 modified đúng phạm vi và3 archived nguyên
bytes; không có unexpected changes. Baseline không hash toàn bộ mọi video/output
lịch sử, nên không suy rộng thành kiểm chứng mọi byte của cả workspace.

Restore-copy check đã tạo3 bản sao theo cây đường dẫn gốc trong run/restore-copy-check,
khớp SHA-256 baseline. Không overwrite file active. Python code khớp revision dùng
cho real smoke cuối. Delivery regression:145 passed,0 fail/error/skipped/deselected/
pytest warnings,31,95s; gồm imports/entry points/links/package assets và web/shared
worker/pipeline. Lần test gọi sai4 CLI trước đó được ghi trong web report và giữ
receipt lỗi; sửa cách test, không sửa CLI cũ.

Tại thời điểm final-integrity:55 cache files đã tái xuất hiện,485.400 bytes. Vì vậy
giảm **logical cache bytes ròng trong đúng phạm vi cleanup là1.790.487 bytes**
(2.275.887−485.400; khoảng1,71MiB). Đây không phải số NTFS free-space đo được, cũng
không phải mức giảm tổng project: run/evidence mới chiếm thêm dung lượng.38.751bytes
desktop chỉ chuyển nơi lưu và restore-copy cũng chiếm thêm bytes; không cộng chúng
vào freed space. Không xóa lại cache sau test để làm số liệu đẹp hơn.
