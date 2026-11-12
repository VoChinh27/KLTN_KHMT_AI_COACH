# Squat temporal sensitivity — synthetic references

Ngày2026-10-18, Asia/Saigon. **EXPERIMENTAL / NOT VALIDATED**.
GĐ6 IN_PROGRESS; `official_eligibility=false`, `assessment_enabled=false`;
GĐ5.7 **BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**, toàn GĐ5 IN_PROGRESS.
Không có inference, segmentation hoặc mở input video/metadata UCF trong audit.

## Câu hỏi, phạm vi và profile

Khảo sát độ nhạy **theo input và FPS**, không sweep/tune tham số: pause ở đáy,
thay đổi tốc độ giữa các chu kỳ, nhiều chu kỳ và vùng dữ liệu không hợp lệ tác
động thế nào tới candidate count, event timing và phase labels?
Hypothesis diagnostic: count có thể ổn định dù phase sai, vì Bottom hiện chỉ
là event một frame. Không có training/validation split hoặc fitting; đây là
deterministic unit-signal study, không external validation hoặc athlete sampling.

Đã kiểm tra code/tests segmentation, feedback contracts, `phase_at` và báo cáo
prototype trước sửa. **Không sửa core** `segmentation/models.py`, `squat.py`,
`reporting/squat_builder.py`, renderer hoặc bất cứ threshold/profile cũ nào.
Package vẫn0.7.0; hai module audit mới có version/hash riêng, không dependency mới.
Workspace không có Git repository; không tạo commit hoặc giả lập Git diff.

Baseline duy nhất: unchanged `squat-temporal-s3-v1`, S3/NOT_VALIDATED,
centered mean span0.2s, hysteresis5°, minimum half-cycle5/30s. Profile hash:
`5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`.
Audit từ chối profile drift. Không suy temporal permission từ numeric0.4;
input là contract `synthetic_signal_fixture`, không có QualityReport giả.

## Fixtures và reference độc lập

`squat-analytic-fixture-v1`: half-cosine descent/ascent, top170°, bottom bằng
top trừ excursion (mặc định60°), hold ở đáy constant. Đây là giá trị toán học
để thử signal, không chuẩn kỹ thuật hoặc góc đo từ pose. Guard trước/sau1s
(140→170→140°) giúp xác nhận extrema; guard không là rep. Không nội suy gap,
không chọn/fuse side; missing được giữ null/provenance missing.

Mỗi cycle khai báo `(descent, pause, ascent)` giây. Reference gồm top start,
bottom arrival/departure, top end; midpoint hold là **event proxy riêng**, còn
ground-truth phase Bottom bao gồm toàn hold. Không có hold thì Bottom là một
event ở CFR sample gần nhất. Event frame round-nearest ties-up, epsilon1e-9 cho
decimal arithmetic. Shared top gán Return/Top của cycle trước. Ground truth và
mask được sinh không gọi algorithm/`phase_at`, lưu **trước** segmentation.

| Case | Fixture definition |
|---|---|
| steady_three |3×(1,0,1)|
| bottom_pauses |(1,.24,1),(1,.6,1),(1,1.2,1)|
| asymmetric_pauses |(.6,1.2,1.4),(1.4,.8,.4),(.4,.4,1.8)|
| variable_tempo |(1.4,0,.6),(.4,0,.4),(1.8,0,1.2),(.24,0,.36)|
| eight_consecutive |8×(.6,0,.8), shared tops|
| fast_cycles |3×(.08,0,.08), stress temporal resolution|
| small_excursion |3×(1,0,1), excursion3°|
| missing_bottom |steady_three, null interval[1.9,2.1]s inclusive|
| explicit_side_switch |steady_three, declared left→right tại4s|
| partial_edges |steady_three, không guard, cycles1/3 chạm clip edges|

Mỗi case lấy mẫu25/30/60FPS: **30 điều kiện**. Không cherry-pick theo kết quả;
fast/small/gap cases không bị loại vì không có prediction.

## Evaluation protocol

Evaluator `squat-temporal-sensitivity-v1`, protocol hash:
`8141cb2ec2b4fca5e5f7a60ba4830d8a2ed9bbcab5f87ecf2691cf068096d7c7`.

- Match monotone one-to-one: maximize số cặp, rồi total temporal-interval IoU.
  Cặp cần IoU≥0.5 và predicted bottom trong reference cycle. Đây là association
  policy của evaluation, không threshold detection/form; không tolerance để
  biến event lệch thành exact match. Một reference không được tái dùng.
- Sai lệch signed=prediction−reference, lưu cả frame và giây. Timing MAE chỉ
  trên matched cycles; không match trả null MAE, không0. Misses/extras và execution
  failures được báo riêng để tránh chỉ nhìn timing của các ca thuận lợi.
- Count denominator giữ **mọi full reference cycles**, cả gap/side-switch và
  missed cases. Báo thêm source-safe coverage riêng, không dùng nó che main misses.
  Cycles chạm biên được liệt kê boundary-partial, không coi là full references.
- Phase denominator là union inclusive frame windows của full references,
  kể cả unresolved gaps và missed cycles. Availability khác agreement:
  Unavailable không được tính là đúng; lưu per-phase support/recall/confusion.
  Không dùng một-frame tolerance để che drift.
- Arrival/departure errors so **marker Bottom đơn** với bounds hold đã biết,
  không tuyên bố algorithm đã dự đoán hai pause boundaries. Duration error so
  Descent/Ascent event spans hiện có với physical durations do fixture định nghĩa.

## Kết quả đã đo

Run: `outputs/squat_temporal_sensitivity/run-87f31de96a654bc09ee816cd2f044f5e`.
Kết quả cuối trong **`audit-final/`**; `audit-attempt1/` giữ nguyên riêng.
30/30 executions thành công,0 execution failures. 108 cycle instances gồm
**102 full references và6 boundary-partial**. Matched78/102 (**76.4706%**),
24 không có candidate,0 candidate thừa. Count khớp ở18/30 conditions.

Phase có nhãn5780/6989 reference frames (**82.7014%**), đúng nhãn5153/6989
(**73.7301%**):1209 unavailable và627 đã có nhãn nhưng khác reference.
Các FPS là lặp lại cùng loại signal, không102 mẫu người tập độc lập hoặc
ước lượng xác suất tổng quát. Không dùng CI/p-value hoặc claim athlete accuracy.

Trong bảng, mỗi ô liệt kê **25 /30 /60FPS** theo thứ tự:

| Case | Matched/full refs | Phase availability % | Exact-frame phase agreement % |
|---|---|---|---|
| steady_three |3/3 ·3/3 ·3/3|100 ·100 ·100|100.00 ·98.90 ·99.45|
| bottom_pauses |3/3 ·3/3 ·3/3|100 ·100 ·100|74.75 ·74.79 ·74.74|
| asymmetric_pauses |3/3 ·3/3 ·3/3|100 ·100 ·99.80|71.09 ·71.15 ·70.89|
| variable_tempo |4/4 ·4/4 ·4/4|100 ·100 ·100|93.17 ·92.23 ·94.29|
| eight_consecutive |8/8 ·8/8 ·8/8|100 ·100 ·100|99.29 ·96.44 ·95.25|
| fast_cycles |0/3 ·0/3 ·0/3|0 ·0 ·0|0 ·0 ·0|
| small_excursion |0/3 ·0/3 ·0/3|0 ·0 ·0|0 ·0 ·0|
| missing_bottom |2/3 ·2/3 ·2/3|66.89 ·66.85 ·66.76|66.23 ·65.75 ·66.20|
| explicit_side_switch |2/3 ·2/3 ·2/3|67.55 ·67.40 ·67.04|66.89 ·65.75 ·66.20|
| partial_edges |1/1 ·1/1 ·1/1|100 ·100 ·100|98.04 ·98.36 ·99.17|

Timing trên **78 matched cycle instances**, so với analytic times, đơn vị ms:

| Event | Signed bias | MAE | Max absolute |
|---|---:|---:|---:|
| Top start |−10.214|15.684|80.000|
| Bottom midpoint proxy |−0.726|5.513|50.000|
| Top end |+0.855|17.009|80.000|

Max absolute **frame** deviation của start/bottom/end theo FPS:
25→2/1/2 frames;30→2/1/2;60→4/3/4. Timing thấp ở matched-only không
xóa24 misses. Tất cả signed per-cycle errors nằm trong `evaluation.json`.

### Giới hạn quan sát được và ca không phát hiện

**Pause:** count đủ không chứng minh phase đúng. `bottom_pauses` có Bottom
recall3/54=5.5556% ở25FPS,3/62=4.8387% ở30,3/123=2.4390% ở60. Remaining
hold frames bị gán Descent/Ascent, dù availability100%. `asymmetric_pauses`
Bottom recall4.7619/4.0000/2.0408%. Ví dụ hold1.2s tại30FPS: Bottom marker
lệch arrival+0.5933s và departure−0.6067s; descent/ascent duration errors
+0.6000/+0.6333s. Đây là thiếu interval model, không lỗi kỹ thuật người tập.

**Variable tempo/FPS:** bốn cycle vẫn đủ nhưng Bottom event exact recall tại
30FPS chỉ1/4. Các peak bị dịch do centered averaging và nhịp hai bên extrema
không đều; raw reference không được kéo theo smoothed prediction. Ngay steady
case,30FPS start đầu−1 frame/end cuối+1 frame; guards khác amplitude strokes
nên count đúng vẫn không đồng nghĩa mọi event đúng frame. Không giả định tăng
FPS luôn tăng phase agreement hoặc freeze profile cho evaluation thật.

**Misses:** fast_cycles bỏ cả3 tại từng FPS (9 misses), small_excursion cũng9.
Với fast case, halves0.08s ngắn hơn S3 spacing và smoothing ảnh hưởng tín hiệu;
30FPS có unstable_signal. Small excursion3° dưới hysteresis; do guards có
excursion30°, run không nhất thiết có low_signal_range flag, output30FPS chỉ
partial flags. Không tự thêm reason đặc hiệu mà thuật toán chưa xuất ra.

Gap first bottom từ chối cycle1, side-switch từ chối cycle2 (mỗi case3 misses
qua3FPS). Đây là abstention giữ policy an toàn; main coverage vẫn ghi thiếu,
source-safe references2/2 được tìm thấy. Không candidate nào bắc qua null hoặc
side change, không smoothing điền null. partial_edges chỉ match cycle2;
cycles1/3 được ghi partial, không đổi endpoint thành confirmed để tăng count.

## Code, artifacts và tests

File mới: `src/gym_coach/segmentation/synthetic_fixtures.py`,
`src/gym_coach/reporting/temporal_sensitivity.py`, `tests/test_temporal_sensitivity.py`
và báo cáo này. Cập nhật phần liên quan trong README, prototype report,
PROJECT_STATE, TASKS, IMPLEMENTATION_PLAN, SCHEMA_CONVENTIONS. Existing core,
profile, dependencies, package version/API exports và dataset không sửa.

Per case: `fixture.json`, `segmentation.json`, `evaluation.json`. Root:
`protocol.json`, `fixture_definitions.json`, `profile.json`, `summary.json`,
`results.json`, `summary.md`, `manifest.json` có commands/environment/code và
output byte SHA-256. Source-definition/input/result/profile content hashes
độc lập artifact byte hashes. Không ghi đè output có sẵn; baseline hash audit
ghi trong `preservation.json`. Hai attempts giữ cùng fixtures/predictions;
final bổ sung unit frame errors và metadata giải thích pause, không đổi profile.

Lệnh thực tế (PowerShell tại root; không chạy lại vào paths đã tồn tại):

```powershell
$run = 'outputs/squat_temporal_sensitivity/run-87f31de96a654bc09ee816cd2f044f5e'
& .venv/Scripts/python.exe -m gym_coach.reporting.temporal_sensitivity --output "$run/audit-final"
& .venv/Scripts/python.exe -m pytest tests/test_temporal_sensitivity.py tests/test_temporal_squat.py tests/test_squat_feedback.py -m 'not integration' -q --junitxml="$run/related-complete.xml"
& .venv/Scripts/python.exe -m pytest -m 'not integration' -q --junitxml="$run/unit-suite-complete.xml"
```

Test evidence cuối: **54 new tests** nằm trong related run124 passed/0
failed/error/skipped/1 deselected/0 warnings,28.59s (`related-complete.xml`).
Broader unit suite1056 passed/0 failed/error/skipped/25 deselected/0 warnings,
89.76s (`unit-suite-complete.xml`). Lượt trước:50 new passed/0 warnings
(5.58s), related120 passed/1 deselected/0 warnings (30.04s), broad unit1052
passed/25 deselected/0 warnings (97.15s). New regressions bổ sung fast-case
misses cả3FPS và variable-tempo exact Bottom mismatch; counts cuối ở XML mới.
Không chạy real-data integration hoặc gọi lại demo/video pipeline. Deselections
là tests đánh dấu integration; không ghi skipped thay deselected.

Tests mới kiểm tra reference độc lập detector/reporting, signed/count/phase
accounting, misses→null MAE, hold-vs-event mismatch, exact profile hash và
source/input integrity, invalid sampling/NaN, no gap bridging/side fusion,
exclusive output và strict JSON readback/hash. Subprocess audit chặn imports
NumPy/SciPy/OpenCV/MediaPipe/Torch và chặn Data/video file reads; toàn30 cases
vẫn chạy. GĐ0–5 compatibility unit tests không thay đổi.

Môi trường thực tế CPython3.11.9, Windows x64, package0.7.0,
Pydantic2.13.5, pytest9.1.1. Audit chỉ dùng stdlib+Pydantic đã có; không cài
model/dependency và không claim clean-environment compatibility.

## Bước nhỏ nhất tiếp theo

Thiết kế rõ contract **event Bottom và interval pause** cùng abstention reasons
trước khi cân nhắc thay thuật toán, version/profile hoặc acceptance tests.
Không tự sửa trong bước này. Khảo sát hiện chưa có noise/occlusion detector,
VFR, geometry/pose uncertainty hoặc reference thật; group-isolated validation
và expert form standards chưa có. Các số liệu chỉ mô tả generator/detector trên
matrix đã công bố; không xác nhận trên vận động viên, không bật assessment.
