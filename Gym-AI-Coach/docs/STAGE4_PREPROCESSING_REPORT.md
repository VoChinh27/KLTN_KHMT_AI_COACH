# Giai đoạn 4 — Preprocessing

Ngày đo: 2026-10-08, Asia/Saigon. **COMPLETE**, đủ DoD của yêu cầu và Prompt tổng.
Kết quả dưới đây đo từ implementation mới; không sử dụng archive hoặc thay
pose inference bằng artifact cũ. Không có geometry feature, rep/phase, scoring
hay feedback được thêm.

## Implementation và kiến trúc

Module `gym_coach.preprocessing` nhận full raw PoseFrame sequence, QualityReport,
VideoMetadata và versioned PreprocessingProfile. Trả `PreprocessingResult` gồm
raw snapshots, processed frames, normalization transforms, smoothing runs và
quality snapshot. Giữ module riêng để không lẫn dữ liệu backend với dữ liệu
biến đổi hoặc biến quality failure thành lỗi kỹ thuật.

- `models.py`: separate contracts, canonical hashes, interpolation endpoints,
  transform statuses, profile TOML loader/resolver, strict audit validation.
- `pipeline.py`: conservative interpolation → explicit normalization → finite-run
  smoothing; revalidate unsafe input và wrap failures thành structured reasons.
- `serialization.py`: self-contained UTF-8 JSON, exclusive write và strict read.
- `diagnostics.py`: provenance/usable coverage và transform/run counters.
- `cli.py`, `__main__.py`, public exports: full video → pose → quality →
  preprocessing audit, cả khi quality says skip. Không chạy assessment.

Raw được copy qua schema validation, giữ toàn bộ x/y, visibility/presence,
flags, backend/config và identities; không mutate input. Mỗi processed frame
lưu video_id/frame_index/timestamp chính xác và raw_frame_hash. Mỗi joint giữ
source_status, status, usable/reason, pre-normalization coords, unsmoothed coords,
smoothed coords và nguồn imputation nếu có. Unavailable processed coords là null.
Raw/quality/profile hashes kiểm tra lúc đọc/ghi; input dictionaries tách khỏi caller.

Quality report lưu nguyên quyết định/flags/metric eligibility, không tính lại
eligibility trên dữ liệu đã xử lý. Core counts/spans được đối chiếu raw để tránh
gắn report sai input. Scale eligibility chỉ là điều kiện measurement, chưa đủ
để dùng metric khi quality hoặc viewpoint không phù hợp.

## Dependency và môi trường thực tế

CPython 3.11.9, Windows x64, package `ai-coach-gym==0.4.0`. Runtime:

| Dependency | Version sử dụng | Vai trò |
|---|---|---|
| pydantic | 2.13.5 | Schema và JSON |
| numpy | 2.4.6 | MediaPipe/OpenCV/SciPy arrays |
| scipy | 1.17.1 | Savitzky–Golay, mới khai báo trong manifest GĐ4 |
| mediapipe | 0.10.14 | Pose backend GĐ2, không đổi model/config |
| opencv-contrib-python | 5.0.0.93 | Video decoder, cv2 5.0.0 |
| pytest | 9.1.1 | Dev tests |
| setuptools | 84.0.0 | Editable build |
| protobuf | 4.25.9 | Dependency hiện có của MediaPipe |

SciPy đã có trong `.venv`, không cài framework/model mới. Editable reinstall
offline thành công; `pip check` trả `No broken requirements found.` Venv legacy
vẫn có `opencv-python==5.0.0.93` cùng contrib; manifest chỉ chọn contrib. Chưa
kiểm tra clean environment, không tuyên bố compatibility trên môi trường sạch.
Các public imports vẫn không load NumPy/SciPy/OpenCV/MediaPipe; smoke test chặn
các backend này đạt. MediaPipe Solutions dùng full model đã có trong wheel,
mapping `mediapipe-pose-13-v1`, smooth_landmarks=false, presence=null.

## Profile, policy và giả định

File: `configs/preprocessing_s3_v1.toml`; schema `0.1.0`, version
`preprocessing-s3-v1`, source tier **S3 engineering baseline, chưa tối ưu/validated**.
Canonical content SHA-256:
`33edf08cdafea19ef0ae67b67ce02b9f53b1b1764641bc9df7af58259d387c33`.

| Setting | Giá trị | Ý nghĩa |
|---|---|---|
| max_gap_seconds | 5/30 s | Baseline 5 missing samples tại 30 FPS |
| interpolate_low_visibility | false | Low visibility là barrier, không endpoint/gap candidate mặc định |
| smoothing_span_seconds | 10/30 s | 11 samples gồm 10 intervals tại 30 FPS |
| smoothing_polyorder | 2 | Engineering baseline theo đặc tả |
| smoothing_method / mode | savitzky_golay / interp | Fit polynomial tại edge trong mỗi run |
| min_torso_scale | 1e-8 image-width units | Chặn scale zero/near-zero, chưa là ngưỡng chuyên môn |

Gap limit = floor(max_gap_seconds × FPS + 1e-9), đếm **số sample thiếu**.
Khoảng thời gian giữa hai endpoint là (gap_count+1)/FPS; không gọi khoảng đó
là max_gap_seconds. Smoothing: chọn số interval chẵn gần nhất với span×FPS,
ties up, cộng một sample; window odd ≥3 và >polyorder. Resolved parameters,
actual span và FPS đều được lưu. Mặc định:

| FPS | Max missing samples | SG window | Actual span |
|---|---:|---:|---:|
| 25 | 4 | 9 | 0.32 s |
| 30 | 5 | 11 | 1/3 s |
| 60 | 10 | 21 | 1/3 s |

Chỉ linear-interpolate candidate short gaps có hai original observed endpoints
nằm trong `[0,1]` x/y. Không dùng endpoint imputed, low visibility hoặc out-of-frame;
không nội suy đầu/cuối, gap dài hoặc observed ngoài ảnh. Policy opt-in low
visibility chỉ cho phép interior candidate thành imputed, vẫn giữ source_status
low_visibility; không biến thành observed. Mỗi imputation lưu exact endpoint
indices/timestamps/status; JSON validation kiểm tra các tham chiếu/policy này.

Working coordinates `x=x_raw, y=y_raw*height/width` có isotropic image-width
units; raw coordinate space không đổi. Origin = midpoint hai hips usable;
scale = Euclidean norm giữa midpoint shoulders và midpoint hips. Anchors có thể
imputed, source audit vẫn ở từng joint. Scale chỉ hợp lệ nếu hữu hạn và >1e-8.
Fallback bỏ translation/division tương ứng, origin/scale **null** và phát flag.
`distance_ratio_eligible=false` khi fallback scale; downstream phải thêm kiểm tra
joint usability, quality và viewpoint. Không có cm, camera/world calibration.

Smoothing chạy riêng từng finite joint run, tách tại unresolved gaps và chuyển
translation/scale statuses để không trộn units. Đoạn ngắn hơn window giữ nguyên
unsmoothed coords, status skipped_short_run và flag tại frame đầu, interval gồm
toàn run. Smoothing không đổi provenance và không clamp overshoot.
Behavior API đối chiếu [SciPy savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html);
version thực tế 1.17.1 được kiểm chứng bằng local inference/tests.

## Commands và kết quả kiểm thử thực tế

Tại project root, PowerShell:

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q tests/test_preprocessing.py
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage4/verification-final.xml
```

- Editable install: exit 0, ai-coach-gym 0.4.0; không tải dependencies.
- New unit tests: **55 passed in 4.07s**.
- Final full suite: **145 passed, 4 warnings in 31.37s**, exit 0;
  JUnit 145 tests, errors=0, failures=0, skipped=0.
- 4 warnings là protobuf `SymbolDatabase.GetPrototype()` deprecated khi
  chạy hai Stage 2 và hai Stage 4 fresh inference integrations; không suppress.
- Stage 4 có 55 unit + 2 fresh real-video integrations. Các test GĐ0–3 vẫn đạt.
- Test kiểm tra input/raw immutability; đủ bốn provenance; short/long/head/tail
  gaps; low visibility policy/barriers; out-of-frame endpoint; interpolation
  refs; mid-hip/aspect/torso; zero/near-zero/nonfinite scale; fallbacks/flags;
  imputed anchor audit; gap/unit run boundaries; short run; polynomial 30/60 FPS
  equality và sinusoid common times tolerance 0.0005 torso units; serialization;
  malformed/unordered/nonfinite inputs; quality mismatch; profile/SG/I/O errors;
  corrupted endpoint/provenance/run/hash audit rejection.

Artifact cuối: [JUnit](../outputs/stage4/verification-final.xml).
Snapshot portable, paths/hashes/counters per joint/source files:
[stage4_measured_results.json](artifacts/stage4_measured_results.json).
Workspace không có Git repository; source tree SHA-256 (src/configs/tests/manifest):
`34edb4b2ef8242732de2d41afe1339f12fca05d6575371ff454a54d62112f44c`.
Không tạo commit giả. Quét src/configs/manifest không có archive reference hoặc
drive path hard-code; không sửa archive/dataset.

## Hai real-video integrations

Final suite decode toàn video, chạy **MediaPipe mới → quality mới → preprocessing**,
không đọc Stage 2 artifact để thay inference. Reader xác nhận count; cả 345 frames
giữ đúng video ID/index/timestamp và raw/quality equality. JSON write/readback đạt.

| Bài / video ID | Relative input | FPS / frames |
|---|---|---|
| BodyWeightSquats / v_BodyWeightSquats_g01_c01.avi | Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi | 25 / 267 |
| HandstandPushups / v_HandStandPushups_g01_c01.avi | Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi | 25 / 78 |

Output directory `outputs/stage4/integration-7fba808fe3d54a57b5d132b037d574bb/`:
[Squat full JSON](../outputs/stage4/integration-7fba808fe3d54a57b5d132b037d574bb/BodyWeightSquats.processed.json),
[HSPU full JSON](../outputs/stage4/integration-7fba808fe3d54a57b5d132b037d574bb/HandstandPushups.processed.json).
Mỗi file có toàn bộ raw/processed joints và reasons/flags/identity/transforms.
Mỗi bài có `.summary.json` cùng thư mục; portable snapshot lưu summary cả hai.

| Measurement | Squat | HSPU |
|---|---:|---:|
| Total joint samples = N×13 | 3471 | 1014 |
| Raw observed / low_visibility / missing | 2515 / 956 / 0 | 871 / 143 / 0 |
| Processed observed / low_visibility / imputed / missing | 2515 / 956 / 0 / 0 | 871 / 143 / 0 / 0 |
| Raw observed coverage | 72.457505% | 85.897436% |
| Usable coverage before | 72.457505% | 69.033531% |
| Usable coverage after | 72.457505% | 69.033531% |
| Imputed points | 0 | 0 |
| Origin fallback frames | 0 | 0 |
| Scale fallback frames | 0 | 0 |
| Finite joint runs total | 33 | 14 |
| Runs smoothed | 14 | 11 |
| Short runs preserved | 19 | 3 |
| Quality sequence status before → after | skipped → skipped | skipped → skipped |

Observed coverage đếm provenance backend. Usable coverage đếm observed **in-frame**
trước, và processed usable (observed in-frame/imputed) sau; hai khái niệm không
đồng nhất. HSPU có 171 observed samples ngoài khung bị loại khỏi processed coords,
không đổi observed source status. Cả hai không có missing samples nên baseline
không tạo imputation; low visibility không được dùng để nâng coverage. Hai mẫu
không gặp fallback; các trường hợp fallback/imputation được test synthetic.

Per joint: O/L/M là raw observed/low_visibility/missing counts; U là processed
usable count. Processed provenance giữ nguyên O/L/M, imputed=0 ở tất cả rows.
Coverage raw=O/N, usable before=after=U/N; snapshot lưu ratios đầy đủ.

| Joint | Squat O/L/M | Squat U/267 | HSPU O/L/M | HSPU U/78 |
|---|---|---:|---|---:|
| nose | 267/0/0 | 267 | 78/0/0 | 78 |
| left_shoulder | 267/0/0 | 267 | 78/0/0 | 78 |
| right_shoulder | 267/0/0 | 267 | 78/0/0 | 78 |
| left_elbow | 267/0/0 | 267 | 78/0/0 | 78 |
| right_elbow | 0/267/0 | 0 | 78/0/0 | 78 |
| left_wrist | 267/0/0 | 267 | 78/0/0 | 3 |
| right_wrist | 2/265/0 | 2 | 78/0/0 | 0 |
| left_hip | 267/0/0 | 267 | 78/0/0 | 78 |
| right_hip | 267/0/0 | 267 | 78/0/0 | 78 |
| left_knee | 267/0/0 | 267 | 77/1/0 | 73 |
| right_knee | 8/259/0 | 8 | 78/0/0 | 78 |
| left_ankle | 267/0/0 | 267 | 13/65/0 | 0 |
| right_ankle | 102/165/0 | 102 | 1/77/0 | 0 |

Quality skip của squat liên quan right knee low visibility; HSPU liên quan
wrist ngoài khung. Đây là dữ liệu measurement, **không kết luận lỗi form**,
không tự gán nguyên nhân occlusion/inverted từ coverage.

## Files mới và cập nhật

Mới: preprocessing `models.py`, `pipeline.py`, `serialization.py`,
`diagnostics.py`, `cli.py`, `__main__.py`; `configs/preprocessing_s3_v1.toml`;
`tests/test_preprocessing.py`, `tests/test_preprocessing_integration.py`;
report này và `docs/artifacts/stage4_measured_results.json` (11 files).
Cập nhật preprocessing `__init__.py`, `_version.py`, `flags.py`, pyproject.toml,
README, IMPLEMENTATION_PLAN, SCHEMA_CONVENTIONS, PROJECT_STATE, TASKS (9 files).
Generated outputs/cache/egg-info được ignore, không là implementation source.

## Definition of Done

| Tiêu chí yêu cầu / Prompt tổng | Bằng chứng | Kết quả |
|---|---|---|
| Raw pose không sửa, input identity truy vết được | Detached snapshots/hashes, unit immutability, 345 real frames exact equality | PASS |
| Bốn provenance per joint/frame; observed hay imputed rõ | Separate source/status, nullable unusable coords, roundtrip/schema tests | PASS |
| Short gap interpolation đúng policy và endpoint nguồn | Boundary/limit/visibility/OOF tests; strict endpoint audit | PASS |
| Không điền giả head/tail/long gap | Null coords, explicit missing/reason, unit tests | PASS |
| Mid-hip/torso normalization và fallback flag rõ | Transform schema, zero/missing/nonfinite guard tests; flags + null metadata | PASS |
| Downstream biết scale validity | distance_ratio_eligible tied to scale_status, schema tests | PASS |
| Smoothing không cross unresolved gaps | Independent runs, gap/unit transition tests, run audit validation | PASS |
| Short run safe | unchanged coords/status + interval flag tests | PASS |
| FPS-aware profile version/hash | TOML/resolver, 25/30/60 conversion, polynomial/sinusoid tests | PASS |
| Quality information giữ nguyên | Snapshot/hash; custom metric and fresh integration equality | PASS |
| Serialization giữ provenance/transform/flags/timestamps | JSON readback tests + both full video artifacts | PASS |
| Invalid input/backend preprocessing/I/O structured reasons | Unsafe NaN/Inf/order/SG/config/quality mismatch/JSON tests | PASS |
| Unit tests và hai fresh real-video integrations đạt | 55 Stage 4 unit + 2 integration; toàn suite 145 passed | PASS |
| Không feature/rep/phase/assessment | Chỉ module preprocessing; namespaces downstream không đổi | PASS |
| Giả định/limitations ghi rõ | Sections policy, environment và limitations bên dưới | PASS |

## Limitations và bước tiếp theo

- Profile S3 là baseline thiết kế. FPS-aware không bảo đảm mọi signal khác sampling
  rate có output giống hệt; chỉ synthetic tests với tolerance công bố đã đạt.
- CFR metadata/geometry image dimension phải đúng; chưa VFR/resampling/crop/camera
  calibration, partial-sequence API, causal/streaming SG hoặc empirical tuning.
- Full pose sequence/result được giữ trong memory và JSON gồm raw+processed;
  chưa benchmark video dài hoặc compression. Không giữ decoded images.
- Conservative missing-only policy không phục hồi low visibility/OOF; real samples
  imputed=0/fallback=0 không chứng minh phục hồi chất lượng pose ngoài synthetic.
- Mid-hip centering triệt hip-y motion, vì thế pre-normalization coords/raw được
  lưu cho nhiệm vụ sau; chưa chọn primary segmentation signal.
- SG có thể overshoot và làm đổi geometry mô tả; không clamp hoặc suy ra accuracy.
  Normalization anchors có thể imputed; quality không được nâng.
- MediaPipe 2D single-subject không có multi-person enumeration/stable identity,
  perspective/depth/occlusion/inverted robustness chưa được đánh giá bằng labels.
- QualityReport GĐ3 chưa có raw-sequence hash; đối chiếu core counts/spans không
  chứng minh exact coordinate identity. GĐ4 lưu report nguyên và raw hash để audit.
- Frozen models chưa deep-immutable cho dicts; snapshots tách caller, hash revalidation
  khi persist/read bảo vệ audit. Caller không nên sửa nested dictionaries.
- Chưa clean-env test/transitive lock; dual OpenCV legacy và protobuf warnings còn.
  Không có Git; source manifest hash ghi scope tường minh.
- UCF101 labels chỉ action, không là pose/form accuracy ground truth. Cả hai
  quality sequences skipped; preprocessing execution pass không cho phép chấm form.

Nhiệm vụ nhỏ nhất tiếp theo **GĐ5.1**: định nghĩa FeatureDefinition/FeatureObservation
metadata và availability contract, kiểm thử gate joint/quality/viewpoint/fallback
scale trước khi thêm descriptive geometry. Chưa thực hiện GĐ5 hoặc mở scoring.
