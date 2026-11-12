# Giai đoạn 3 — Quality gate

Ngày: 2026-10-08 (Asia/Saigon). Trạng thái: **COMPLETE** theo Definition of Done
của Giai đoạn 3. Các ngưỡng là engineering assumptions S3, chưa được validated
trên development set.

## Thay đổi thực tế

- `QualityProfile quality-s3-v1`: minimum sequence 5 frame, missing ratio
  `>0.30`, low-visibility ratio `>0.30`, consecutive unusable run `>10` frame.
- `JointQuality`: provenance counts, observed coverage, missing/low-visibility,
  tọa độ ngoài khung, unusable ratio và gap spans inclusive theo frame ID.
- `MetricEligibility`: allow/degrade/skip theo required joints của từng metric;
  có machine-readable reason.
- `QualityReport`: profile hash, sequence status, metric eligibility, flags và
  coverage. Không đổi hoặc sửa dữ liệu pose đầu vào.
- `python -m gym_coach.quality <pose-jsonl>` xuất JSON report.
- Quality gate yêu cầu cùng video/exercise/viewpoint, frame index liên tiếp và
  timestamp tăng. Non-contiguous sequence bị trả structured input error.
- Out-of-frame được tính không dùng được cho metric liên quan. Ngưỡng `>0.30`
  dùng `max_missing_ratio` làm extension engineering S3; cần thí nghiệm xác nhận.
- Bilateral gate áp dụng giới hạn trên từng required joint, tránh joint lỗi bị
  làm loãng bởi trung bình với các joint còn tốt.

## Kiểm thử

Lệnh cuối:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

Kết quả: **88 passed, 2 warnings in 14.96s**, exit 0. Hai warning là protobuf
`SymbolDatabase.GetPrototype()` deprecation từ integration MediaPipe cũ; không
ảnh hưởng kết quả test.

Test quality bao gồm empty, short sequence, missing tại đầu/giữa/cuối, long gap,
low visibility, metric một bên, JSON roundtrip, unknown/wrong viewpoint,
out-of-frame, non-contiguous frames, CLI JSON và integration trên hai pose
sequence JSONL đã lưu từ Giai đoạn 2.

## Kết quả audit thực tế

Nguồn là pose JSONL của hai video Giai đoạn 2, không phải nhãn form:

| Bài / video | Frames | Pose found | Kết quả gate | Lý do chính |
|---|---:|---:|---|---|
| Squat `v_BodyWeightSquats_g01_c01.avi` | 267 | 267 | `skip` | Right knee low visibility 259/267; invalid run tối đa 125 frame |
| HSPU `v_HandStandPushups_g01_c01.avi` | 78 | 78 | `skip` | Tọa độ cổ tay ngoài ảnh: trái 75/78, phải 78/78 |

Số liệu đo được theo required joints:

| Bài | Joint | Observed | Low visibility | Missing | Out of frame | Max unusable run |
|---|---|---:|---:|---:|---:|---:|
| Squat | left knee | 267 | 0 | 0 | 0 | 0 |
| Squat | right knee | 8 | 259 | 0 | 0 | 125 |
| Squat | left ankle | 267 | 0 | 0 | 0 | 0 |
| Squat | right ankle | 102 | 165 | 0 | 0 | 29 |
| HSPU | left wrist | 78 | 0 | 0 | 75 | 44 |
| HSPU | right wrist | 78 | 0 | 0 | 78 | 78 |

`observed=78` không có nghĩa cổ tay HSPU dùng được trong ảnh: MediaPipe visibility
cao nhưng normalized coordinates nằm ngoài `[0,1]`. Gate nhận ra điều này và
skip metric phụ thuộc cổ tay. Hai kết quả skip là quyết định chất lượng dữ liệu,
không phải kết luận vận động viên làm sai.

## Definition of Done

| Tiêu chí | Bằng chứng | Kết quả |
|---|---|---|
| QualityReport và gap diagnostics per joint | schema, span tests, hai real sequences | PASS |
| Metric eligibility + skip/degrade reasons | synthetic per-metric cases và CLI | PASS |
| S3 profile version/hash; ngưỡng chưa validated | `quality-s3-v1` và test hash/schema | PASS |
| Empty/gap/visibility/side/viewpoint edge cases | `tests/test_quality.py` | PASS |
| Integration development videos | saved squat + HSPU pose JSONL | PASS |
| Gate độc lập assessment | chỉ coverage/eligibility/measurement flags; không score | PASS |
| Toàn bộ test đạt | 88 passed, 2 dependency deprecation warnings | PASS |

## Hạn chế

- Hai video mẫu không đại diện toàn dataset.
- `0.30`, 10 frame và 5 frame chưa được đánh giá/tune trên development set.
- Out-of-frame threshold là extension S3 mới; cần đánh giá riêng với camera crop,
  orientation và normalized coordinate semantics.
- Unknown viewpoint được ghi flag; chỉ gate metric có khai báo viewpoint.
- Quality eligibility không phải pose accuracy, rep-count accuracy hay form score.

## Bước tiếp theo

Giai đoạn 4 — preprocessing. Trước implementation, chốt contract cho raw và
processed pose, per-joint imputation provenance, normalization status và FPS-aware
smoothing profile. Không mutate raw pose hoặc làm mịn xuyên gap chưa xử lý.
