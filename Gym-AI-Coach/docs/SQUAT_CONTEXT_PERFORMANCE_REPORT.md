# Squat feature-context performance và exact replay equivalence

Cập nhật 2026-10-21, Asia/Saigon. Receipt benchmark ghi ngày20/10;
timestamps đo gốc giữ nguyên. Tối ưu kỹ thuật trên saved development artifacts;
không phải thí nghiệm pose/angle accuracy hoặc form validation.
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Workspace không Git repository; không commit/diff giả.

**Milestone IMPLEMENTED / EQUIVALENCE PASSED**: session, regression,
paired benchmark và full40 replay đã hoàn tất; scientific gates giữ nguyên.

## Run, nguồn và kiểm tra trước sửa

Run mới: `outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/`.
Nguồn duy nhất cho replay:
`outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/`.
Baseline-before ghi1.194 hashes của source/tests/docs/config/scripts và toàn
baseline run. Đối chiếu root inventory1.033 files, batch manifest452 files,
selection/profile/registry/disabled gates; kiểm tra lại trước benchmark/batch.
Chỉ dùng saved poses/preprocessing/quality; không đọc video trong Data, không
MediaPipe hoặc render lại MP4. Hash saved artifacts gồm media cũ chỉ để kiểm tra
bytes, không mở/giải mã video hay gọi đó là visual QA mới.

Code và timing cũ xác nhận full source validation/hash O(N) bị lặp mỗi frame:
factory strict-intake một lần, LinkedFeatureInputContext after-validator một
lần nữa; thêm full-result hash mỗi context. Batch cũ bỏ full context sau khi
lấy frame/report/linkage, nên embedded source được tạo lại rồi bỏ ở từng frame.
767,79s của run cũ có pytest chạy chồng thời gian, **không dùng làm control hoặc
mẫu số speedup**. Các số speedup bên dưới chỉ dùng paired workload trong run này.

## Contract và bảo toàn validation/provenance

API additive `FeatureContextSession`, version
`validated-preprocessing-session-v1`, sở hữu đúng một clip/configuration:

1. Constructor chỉ nhận complete PreprocessingResult, dùng `_strict_json`
   Python-mode, warnings=error, allow_nan=false rồi strict full revalidation.
   `from_json(str|bytes)` luôn full strict Stage4 audit một lần ở external intake.
   Không nhận caller validated flag/token/hash hoặc cache theo object identity.
2. Full Stage4 audit hiện hành vẫn kiểm tra metadata/timeline, raw/frame/
   sequence/quality/profile hashes, provenance, imputation endpoints/runs và
   core quality consistency. Không chạy lại SG, quality eligibility hay geometry.
3. Session chỉ giữ bytes, tuple và scalar bất biến: encoded processed/raw frames,
   QualityReport, identity/hash projection. Không giữ Pydantic model/dict của
   caller hoặc giả định `frozen=True` làm nested dict bất biến.
4. `select(frame_index=...)` kiểm tra int/non-bool/range, xác định đúng frame/
   timestamp/raw/processed/profile/metadata/report hashes; kiểm tra frame/report
   và tính measurement-input hash mỗi selection. Không full-source validation
   hoặc full-result hash mỗi frame. Adapters giữ nguyên intake validators/gates.
5. Selection handle bất biến; `input_context()`, `raw_frame`, `reference_linkage()`
   trả fresh detached models/dict. Caller sửa bản trả ra không sửa ngược session;
   lần select/input tiếp theo vẫn nguyên snapshot. Nếu caller chủ động sửa input,
   không được gắn số đo mới với linkage cũ; replay bắt buộc check input hash.
6. Không global cache; mỗi worker tạo session riêng. Session không pickle/reload
   từ receipt. Serialized external data phải qua full intake; metadata không có
   thẩm quyền chứng nhận dữ liệu. Deliberate private memory modification hoặc
   monkeypatch trong cùng Python process không là security boundary.

LinkedFeatureInputContext/factory schema0.2.0 và validators trong context.py
giữ nguyên. Session không trả instance LinkedFeatureInputContext, không giả
model_construct hoặc bỏ after-validator. Nó trả ordinary FeatureInputContext
được adapter strict validate. API cũ còn nguyên làm reference.

`reference_linkage()` là **identity projection tương thích format reference**,
không phải serialized LinkedFeatureInputContext đầy đủ hoặc chứng nhận đã gọi
factory cũ. Hai context_schema/context_factory fields định danh format tham
chiếu; metadata thực thi session ở receipt riêng. So sánh giữ **tất cả** các
trường projection đó, mọi observation/reasons/flags/actual joints/units/tier/
validation và source/timeline fields. Không bỏ field hoặc dùng tolerance để
che khác biệt; không thêm session fields vào numerical schema0.4.0.

Session kế thừa phạm vi audit của Stage4; không biến hash thành chữ ký/authenticity,
không chứng minh origin của custom QualityReport hoặc tính đúng của landmark/
processed geometry bằng ground truth. Không thêm/làm yếu upstream quality policy.

## Cách dùng

```python
from gym_coach.features import FeatureContextSession, observe_geometry_feature

session = FeatureContextSession(processed_result)  # full strict snapshot audit
selected = session.select(frame_index=0)
context = selected.input_context()
observation = observe_geometry_feature("squat_knee_angle_left_deg", context)
assert observation.availability.input_hash == selected.measurement_input_hash
linkage = selected.reference_linkage()  # persist alongside numerical result
execution_receipt = session.metadata  # separate from numerical schema
```

External source: `FeatureContextSession.from_json(path.read_bytes())` thực hiện
full strict audit và tạo snapshot. Không deserialize session receipt thành
capability. Không truyền session giữa workers; tạo từ nguồn đầy đủ trong worker.
Hai bên dùng cùng session/frame, mỗi metric độc lập; không đổi side/gate.

Saved-baseline CLI (chọn thư mục output chưa tồn tại khi chạy lại):

```powershell
& .venv/Scripts/python.exe -u -B -m gym_coach.reporting.context_performance --plan outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/plan.json --output-dir outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/benchmark --action benchmark
& .venv/Scripts/python.exe -u -B -m gym_coach.reporting.context_performance --plan outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/plan.json --output-dir outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/optimized --action batch
```

CLI tái sử dụng frame-input helper, adapters và aggregate của real batch;
giữ full80 cặp kể cả degrade/skip. Không inference, preprocess lại hoặc render.
result rows gồm giá trị/null, provenance và source linkage; session execution
chỉ ở receipts. Summary coverage so exact với baseline, không chỉ available total.

## Đối chiếu đầy đủ development set

**10.628/10.628 complete records khớp exact canonical JSON,0 khác biệt**:
40 clips/80 pairs/5.314 frames, strict readback numerical models cả hai phía.
Mọi observation, actual joint inputs/provenance, values/null, reasons/flags,
measurement-input identity, timeline/video/source/profile/raw/processed hashes và
đầy đủ reference linkage cùng key set/value đều được so sánh. Không trường
baseline nào bị bỏ. JSON object-key order không có ý nghĩa; canonical comparison
vẫn phân biệt numerical representations, không dùng tolerance hoặc rounding.

Coverage summary **exact equal** với baseline:4.795 available,5.833 unavailable;
35 allow/10 degrade/35 skip pairs;35 sequence skip/5 degrade. Trái2.968/5.314;
phải1.827/5.314. Numeric micro45,12%, macro43,75%; raw independent-side61,93%.
16 imputed input slots giữ nguyên,0 vào available.0 bilateral allow/0 sequence
allow; không promote gate hoặc đổi bên.0 failure/unevaluated/incompatible clips.

Đường session trên đủ40 clips dùng4 workers; wall gồm replay + strict comparison
82.29s; internal total gồm input verification
88.16s trước final manifest hash; CLI elapsed
**90.71s**, exit0. Full-source validation đo được **80**:
40 common load audits +40 session-intake audits,**0 theo frame**.5.314 selections,
10.628 adapter calls. Không gọi đây là speedup full40 so với767,79s cũ, vì không
có full40 reference run cùng điều kiện trong lượt này.

`optimized/clips/Sxxx/left.jsonl/right.jsonl`: toàn bộ kết quả mới. Per-clip
receipt/stats/comparison; optimized/coverage.json và summary.json; manifests
hash/size cho mọi file. Session metadata nằm ở receipt, numerical rows giữ
nguyên schema/key set. Benchmark outputs/receipts/warmups cũng được giữ riêng.


## Benchmark có đối chứng

Plan được lưu trước khi đo: sort(frame_count,path), chọn shortest/lower median
index19/40/longest, ties path tăng. Short g05_c03/54 frames, medium g25_c03/116,
long g01_c01/267. Một process worker cho cả hai paths. Mỗi clip/path một warm-up
full pass không tính median; ba measured repeats/path, thứ tự reference/session
luân phiên. Cùng load+strict validation, hai adapters trên toàn clip, JSONL
serialization/write/close. Session setup gồm full validation/hash/snapshot, không
giấu chi phí này. Startup/import/output-dir creation/gc/memory_before nằm ngoài
pass timing; cùng cách cho hai paths. Baseline comparison/typed readback đo ngoài
scope replay. Không pytest cùng benchmark; không cold-cache/fsync/hardware/power
control, nên không claim benchmark nhiều máy hoặc speedup bất biến.

| Clip / frames | Reference median s | Session median s | Paired-workload speedup |
|---|---|---|---|
| short: v_BodyWeightSquats_g05_c03.avi /54 |7.5638 |1.0303 |7.34× |
| medium: v_BodyWeightSquats_g25_c03.avi /116 |36.1299 |2.8299 |12.77× |
| long: v_BodyWeightSquats_g01_c01.avi /267 |205.4439 |7.9015 |26.00× |

Medians từng phase (s); median của các phase không nhất thiết cộng bằng median total:

| Clip / mode | Load+validation | Session setup | Contexts | Adapters | Serialization/write/close | Total wall |
|---|---|---|---|---|---|---|
| short/reference |0.0375 |0.0000 |6.5827 |0.8393 |0.0989 |7.5638 |
| short/session |0.0383 |0.0584 |0.0476 |0.7883 |0.0974 |1.0303 |
| medium/reference |0.0840 |0.0000 |33.5040 |2.2577 |0.2805 |36.1299 |
| medium/session |0.0885 |0.1393 |0.1340 |2.1956 |0.2750 |2.8299 |
| long/reference |0.2193 |0.0000 |197.9887 |6.4747 |0.7945 |205.4439 |
| long/session |0.2146 |0.3869 |0.3841 |6.1528 |0.7780 |7.9015 |

| Clip | Snapshot payload bytes | Reference working set after load/setup MiB (range) | Session working set after setup MiB (range) |
|---|---|---|---|
|short |474556 |53.93–54.80 |55.40–56.19 |
|medium |1087082 |63.03–66.80 |67.43–68.66 |
|long |2639544 |79.46–80.52 |88.51–91.02 |

Benchmark process peak observed tối đa 124.84MiB, cumulative, không gán cho một mode riêng.
Toàn benchmark CLI 1073.59s, exit0; gồm6 warm-up passes,18 measured passes, ngoài-pass comparison và file hashes.
Per-pass actual full audit counts: reference109/233/535 cho54/116/267 frames; session2 cho cả ba, gồm common load1 + snapshot intake1. Không lấy tốc độ warm-up thay median.


RSS là Windows working set tại các mốc, peak là cumulative process peak bao gồm
warm-up và các path trước trong cùng worker. Không gọi đó là exact session
allocation hoặc chứng minh giảm bộ nhớ. snapshot_payload_bytes chỉ encoded
raw/frame/quality bytes, không gồm tuple/string/allocator overhead hoặc caller
PreprocessingResult. Không dùng tracemalloc trong timed path để tránh overhead.

## Tests, integrity và môi trường

Final related regression **712 passed,0 failed,0 errors,0 skipped,
3 deselected,0 warnings in99,49s**;63 tests mới (51 session +12 replay/integrity),
649 retained related tests. Lệnh/elapsed/stdout/JUnit:
tests-regression-1-command.json, tests-regression-1.txt, tests-regression-1.xml.

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_feature_context_session.py tests/test_context_performance.py tests/test_real_batch_coverage.py tests/test_real_descriptive_demo.py tests/test_feature_context.py tests/test_geometry_features.py tests/test_feature_observation.py tests/test_right_knee_feature.py tests/test_preprocessing.py tests/test_quality.py -m "not integration" -q --tb=short --basetemp outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/tests-regression-1-temp --junitxml=outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/tests-regression-1.xml
```

Lịch sử giữ nguyên: initial session49 pass/2 fail do fixture kỳ vọng raw flag
ở processed.flags[0] và gọi fixture với argument không tồn tại. Sửa đúng fixture,
không làm yếu validator: session+batch74 pass; cuối712 pass nêu trên.
Tests kết thúc trước benchmark; không pytest chạy chồng benchmark/full replay.
3 integration deselected, không chạy pose inference/other-source integrations.

Môi trường: Python 3.11.9, Windows-10-10.0.26200-SP0, 12 logical CPUs. Versions: pydantic=2.13.5, pytest=9.1.1, numpy=2.4.6, scipy=1.17.1, opencv-contrib-python=5.0.0.93, mediapipe=0.10.14.


Tests mới bao gồm exact observations/linkage cho available/degrade/skip/imputed/
missing/fallback/degenerate; invalid bool/index/timeline/source/profile/quality/
hash, nullable NaN/Inf/types và model_construct; nested mutation/flags/raw/
returned context isolation; khác clip/config/thread/process; no gate recompute;
full audit call count; no heavy imports; replay/mismatch detection không bỏ field;
hash/side/profile compatibility và output overwrite rejection.

Không full project suite hoặc inference integrations; không clean-environment
claim. Không dependency mới. Browser/visual QA NOT_RUN trong milestone này;
renderer và ba video cũ giữ nguyên, không tạo media mới.

Preservation receipt kiểm tra lại1.194 baseline-before files; mọi old
output/baseline media/report giữ byte hash,0 deleted files. Existing changes chỉ
features/__init__.py (additive export), reporting/real_batch_coverage.py (shared
frame-input helper, reference default unchanged), README/PROJECT_STATE/TASKS.
context.py, Linked validators, numerical schemas/adapters/geometry/quality/
preprocessing/profile/model config và renderer giữ nguyên. Không cần render
MP4 mới vì toàn bộ numerical records/provenance khớp và renderer không thay đổi.
New modules: features/session.py, reporting/context_performance.py; new tests:
test_feature_context_session.py, test_context_performance.py. New report này;
run manifest/inventory giữ hashes, commands, tests và mọi warmup/repeat/output.


## Quyết định và bước tiếp theo

**Chấp nhận session cho đường xử lý theo clip đã kiểm chứng.** CLI replay
`--action batch` mới dùng session mặc định và fail closed nếu source/profile/hash
hoặc comparison không khớp; lợi ích chỉ báo từ paired benchmark ở trên.
Public factory cũ/LinkedFeatureInputContext và CLI real_batch_coverage cũ vẫn
giữ reference default: session không thay được full linked serialization contract
của các consumer đó. Người gọi chọn API session tường minh theo hướng dẫn;
không tự động đổi return type, migration hoặc mặc định toàn project.

Bước nhỏ tiếp theo: thiết kế protocol reference2D độc lập với sampling khóa
trước để đo sai số landmark/góc. Hiệu năng và exact implementation equivalence
không thay bằng chứng accuracy hoặc mở official/temporal/form gates. Không
triển khai protocol/annotation mới hoặc mở milestone khác trong lượt này.


Đây là tiến bộ kỹ thuật về validation cost/throughput, không phải bằng chứng
pose/angle accuracy, calibrated confidence, athlete/generalization/form quality.
Camera_view=UNKNOWN và side visually_assessed_unverified giữ nguyên. Không
rep/phase/point-turn/pause/score/technical feedback; official gates chưa mở.
