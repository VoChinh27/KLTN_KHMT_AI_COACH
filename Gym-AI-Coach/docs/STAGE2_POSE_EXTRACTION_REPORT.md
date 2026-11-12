# Giai đoạn 2 — MediaPipe Pose Extraction và Joint Provenance

Ngày: 2026-10-08 (Asia/Saigon). Trạng thái: **COMPLETE** sau full test và hai
real-video integrations. Không dùng implementation hoặc kết quả archive.

## Implementation thực tế

| File trong src/gym_coach/pose | Chức năng |
|---|---|
| mapping.py | 13 canonical joints, immutable/versioned mapping |
| models.py | JointObservation, PoseFrame, PoseSequenceSummary, backend/config contracts |
| extractor.py | Lazy MediaPipe adapter, per-joint provenance/flags, lifecycle/errors |
| coverage.py | Streaming per-joint/sequence counters, không quality gate |
| serialization.py | Exclusive-write UTF-8 JSONL và validated streaming readback |
| run.py | Ingestion → pose artifacts, model/input/config/output hashes |
| cli.py, __main__.py | gym-pose-extract / python -m gym_coach.pose |
| __init__.py | Public API, import không khởi tạo backend |

Thêm tests/test_pose.py và tests/test_pose_integration.py. Package 0.2.0,
pose_schema_version 0.1.0, joint_mapping_version `mediapipe-pose-13-v1`,
pose config `pose-extraction-v1`; foundation schema/config giữ 0.1.0.

VideoFrame chưa mang video_id; extractor bind `reader.metadata.video_id` và
sao chép nguyên frame_index/timestamp_seconds. Không recompute timestamp,
resample, skip frame hoặc đổi thứ tự. Một extractor dùng cho một video.

Coordinate space `image_normalized_xy`, units `dimensionless_image_width_height`,
axes `x_right_y_down`: x/y native do backend trả, chưa mid-hip/torso normalization,
không phải cm/m. Tọa độ hữu hạn ngoài [0,1] được giữ và flag out_of_frame.

Bundled full model SHA-256:
`e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f`.
Adapter kiểm tra asset trước khi khởi tạo, không tải lite/heavy model.
model_complexity=1, static_image_mode=false; smooth_landmarks,
smooth_segmentation và enable_segmentation đều false. Detection/tracking
confidence defaults 0.5 (S3). Backend dùng ROI tracking nội bộ; project không
interpolate/smooth/normalize pose. Không feature, repetition, phase hay scoring.

## Dependency và môi trường thực tế

| Thành phần | Phiên bản |
|---|---|
| CPython | 3.11.9, Windows x64 |
| MediaPipe | 0.10.14, import và CPU inference thành công |
| API | mp.solutions.pose.Pose |
| NumPy | 2.4.6 |
| OpenCV direct dependency | opencv-contrib-python 5.0.0.93 |
| cv2 runtime | 5.0.0 |
| Protobuf | 4.25.9 |
| Pydantic / pytest | 2.13.5 / 9.1.1 |

Manifest thêm MediaPipe và NumPy (adapter trực tiếp kiểm tra ndarray), đổi
opencv-python sang opencv-contrib-python vì MediaPipe yêu cầu contrib. Không
cài thêm model/framework; editable install dùng packages sẵn có với
`--no-index --no-deps`. Protobuf 4.25.9 phù hợp requirement >=4.25.3,<5.

Venv legacy vẫn có metadata của hai OpenCV providers 5.0.0.93 cùng cung cấp cv2.
Manifest chỉ chọn contrib. Không gỡ/reinstall shared files của venv đang chạy
khi chưa có lỗi runtime cần sửa. Pip check: exit 0, **No broken requirements found**;
import/inference/tests thành công. Pip check không chứng minh shared-file
collision không tồn tại. Chưa kiểm tra clean environment, không claim compatibility.

## Mapping, provenance và subject policy

| Joint | MediaPipe index |
|---|---:|
| nose | 0 |
| left_shoulder / right_shoulder | 11 / 12 |
| left_elbow / right_elbow | 13 / 14 |
| left_wrist / right_wrist | 15 / 16 |
| left_hip / right_hip | 23 / 24 |
| left_knee / right_knee | 25 / 26 |
| left_ankle / right_ankle | 27 / 28 |

Mapping test kiểm tra unique/immutable và đối chiếu từng index với PoseLandmark
enum của MediaPipe đã cài.

- Observed: x/y hữu hạn, visibility >= threshold (mặc định 0.5, S3).
- Low_visibility: x/y/visibility đo được vẫn giữ khi visibility < threshold.
- Missing: x=y=null và missing_reason bắt buộc; không thay bằng (0,0).
- Non-finite scalar, confidence ngoài miền, absent x/y/visibility hoặc joint
  thiếu trở thành missing có flag; không serialize NaN/Inf hoặc số giả.
- Frame không pose: giữ 13 missing joints, index/timestamp và pose_not_detected.
- Schema từ chối status trái visibility, missing có tọa độ và imputed ở GĐ2.
- Presence lưu khi backend cung cấp; Solutions Pose 0.10.14 xóa trường này
  trước khi trả output. Adapter dùng HasField, giữ null thay vì protobuf default
  0. Integration xác nhận presence null ở mọi joint.

Subject policy `backend_most_prominent_single_subject`: không enumerate nhiều
người, không explicit subject selection hoặc bảo đảm identity ổn định. Không
phát flag multi_person bằng suy đoán; limitations được lưu trong backend/manifest.

Đã xem trực tiếp frame đầu HSPU: người ở tư thế đảo ngược. Không rotate input,
đổi joint trái/phải hoặc sửa pose. Full video chạy đủ 78 frame. Inverted/occluded
input nếu missing/visibility thấp sẽ có các reason generic tương ứng. Chưa có
orientation/occlusion classifier, nên không tự kết luận nguyên nhân missingness.
Visibility thấp không phải lỗi kỹ thuật của vận động viên.

Đối chiếu API với [source MediaPipe 0.10.14 chính thức](https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.14/mediapipe/python/solutions/pose.py)
và source/docstrings trong package đã cài. Không áp dụng khả năng multi-pose
của Tasks API cho Solutions backend đang dùng.

## Integration và coverage đã đo

Chọn rõ hai development examples group g01; không dùng reference labels/tune
threshold. Chưa lập toàn bộ development/evaluation split, không claim generalization.

| Exercise | video_id / relative input path |
|---|---|
| BodyWeightSquats | v_BodyWeightSquats_g01_c01.avi; Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi |
| HandstandPushups | v_HandStandPushups_g01_c01.avi; Data/UCF-101_GYM/HandstandPushups/v_HandStandPushups_g01_c01.avi |

Cả hai: 25 FPS, 320x240, decoder opencv:FFMPEG,
timestamp policy fps_derived_cfr_assumption; chưa VFR support.

Coverage là observed_count / tổng observations mong đợi. Low_visibility/missing
không nằm trong numerator. required_joint_coverage dùng 8 joint bắt buộc mỗi
bài theo §1.2 đặc tả. N=0 có coverage null, counters 0 và structured reason.
Out-of-frame nhưng visibility đủ vẫn là observation backend có flag; coverage
không thay cho geometric eligibility/accuracy (quality gate ở GĐ3).

| Đại lượng | Squat | HSPU |
|---|---:|---:|
| Tổng frame | 267 | 78 |
| Frame có pose / không pose | 267 / 0 | 78 / 0 |
| Pose detection ratio | 100% | 100% |
| Observed observations | 2515 / 3471 | 871 / 1014 |
| Low-visibility observations | 956 | 143 |
| Missing observations | 0 | 0 |
| All-13 coverage | 72.45750504177471% | 85.89743589743589% |
| Required-joint coverage | 80.14981273408239% | 100% |

Coverage theo joint; O/L/M = observed/low_visibility/missing:

| Joint | Squat O/L/M | Squat coverage | HSPU O/L/M | HSPU coverage |
|---|---|---:|---|---:|
| nose | 267/0/0 | 100% | 78/0/0 | 100% |
| left_shoulder | 267/0/0 | 100% | 78/0/0 | 100% |
| right_shoulder | 267/0/0 | 100% | 78/0/0 | 100% |
| left_elbow | 267/0/0 | 100% | 78/0/0 | 100% |
| right_elbow | 0/267/0 | 0% | 78/0/0 | 100% |
| left_wrist | 267/0/0 | 100% | 78/0/0 | 100% |
| right_wrist | 2/265/0 | 0.7491% | 78/0/0 | 100% |
| left_hip | 267/0/0 | 100% | 78/0/0 | 100% |
| right_hip | 267/0/0 | 100% | 78/0/0 | 100% |
| left_knee | 267/0/0 | 100% | 77/1/0 | 98.7179% |
| right_knee | 8/259/0 | 2.9963% | 78/0/0 | 100% |
| left_ankle | 267/0/0 | 100% | 13/65/0 | 16.6667% |
| right_ankle | 102/165/0 | 38.2022% | 1/77/0 | 1.2821% |

Index từ 0, intervals inclusive. Missing thực tế trên hai mẫu là 0;
no-pose/missing/non-finite vẫn được test synthetic, không bịa trường hợp đã đo.

| Joint low visibility | Frame |
|---|---|
| Squat right_elbow | 0–266 |
| Squat right_wrist | 0–119, 121–183, 185–266 |
| Squat right_knee | 0–38, 40–98, 100, 102–226, 229, 232–249, 251–266 |
| Squat right_ankle | 0–20, 24–31, 44, 60–85, 89, 91, 94–96, 103, 106–116, 124–152, 154–157, 159–160, 185, 190–213, 215–223, 234–244, 255–266 |
| HSPU left_knee | 46 |
| HSPU left_ankle | 0–24, 38–77 |
| HSPU right_ankle | 0–76 |

Joint còn lại không low_visibility trên mẫu tương ứng. Không diễn giải các số
này thành lỗi form hoặc nguyên nhân che khuất chưa được kiểm chứng.

## Artifacts và identity

Final run: outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56.

- Squat: [summary](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/BodyWeightSquats/summary.json),
  [frames](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/BodyWeightSquats/frames.jsonl),
  [manifest/issue indices](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/BodyWeightSquats/manifest.json).
- HSPU: [summary](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/HandstandPushups/summary.json),
  [frames](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/HandstandPushups/frames.jsonl),
  [manifest/issue indices](../outputs/stage2/integration-0d9e95374797474e8f6468a17aa97c56/HandstandPushups/manifest.json).
- [Snapshot đo thực tế/source hashes](artifacts/stage2_measured_results.json):
  đủ summary/issue lists trong docs, độc lập với việc outputs bị ignore.
- [JUnit](../outputs/stage2/verification-final.xml).

Workspace không có Git; source/test/pyproject sorted SHA-256 manifest có hash
`0da64a2281b7c884f77d1a46228e500c9dc6d07d0c72c96bb4ac840dbcbbb68a`.
Pose config hash cả hai video:
`3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976`.

| Hash | Squat | HSPU |
|---|---|---|
| Input SHA-256 | 1a199bf3ad385a66505fa0864a48fb9b9f9c1edfd1306d8345729ab06742c9e0 | 05a3bc07db1baa3b64a63d987fe8691df13852b1ae1519d3eb6c0601a7825697 |
| Pose JSONL SHA-256 | 672ef2671137603eb817ecd7ac725c2ab775cada43b1aa66f0122a0cd7156695 | dec7902954d2a1fa908c07ce583520eae120d588dc45b0bcc192608b3140a1c4 |

## Test commands và kết quả thực tế

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage2/verification-final.xml
```

- Install exit 0: ai-coach-gym 0.2.0; không tải dependency.
- Pip check exit 0: No broken requirements found.
- Final pytest exit 0: **76 passed, 2 warnings in 14.75s**, không failed/skipped.
- 73 unit/contract cases và 3 integration cases (một GĐ1, hai GĐ2).
- GĐ2 thêm 46 unit cases và 2 full-video integration cases.
- Hai warnings Python: protobuf SymbolDatabase.GetPrototype deprecated, từ
  dependency trên hai real-video cases. Không suppress warning hoặc sửa
  site-packages. TFLite ghi native feedback warnings ra stderr, không phải
  inference failure; outputs vẫn được kiểm chứng.
- Readback đủ 345 PoseFrame: video_id/index/timestamp khớp từng frame;
  recalculated summary/issue indices/hashes khớp. Presence null được xác nhận.
- Kiểm toán cuối: JUnit xác nhận 76 tests, 0 failures/errors/skipped; source
  manifest không có hash mismatch; 23 local documentation links tồn tại;
  source/config/manifest không có archive reference hoặc hard-coded drive path.
- Tests bao phủ low/missing/no pose, invalid numeric/provenance, protobuf unset,
  empty coverage, malformed result/JSON, input shape/type/order/mixed identity,
  backend init/process/close errors, disabled smoothing/model policy và CLI.

Passing tests/detection 100% không chứng minh pose accuracy hoặc chất lượng form.

## Definition of Done

| Tiêu chí | Bằng chứng | Kết quả |
|---|---|---|
| Mapping version/test | 13-joint immutable/unique + actual enum check | PASS |
| Provenance per-joint/frame | contracts/extractor và synthetic tests | PASS |
| Low visibility khác missing | raw coords giữ; missing null/reason; negative schema | PASS |
| Frame/video/timestamp identity | exact check 345 frames + synthetic non-round timestamp | PASS |
| JSON save/load | JSONL frames + summary roundtrip/readback | PASS |
| Coverage joint/sequence | known synthetic counts/empty + integration recalculation | PASS |
| Squat/HSPU full integration | 267 + 78 frames, measured artifacts | PASS |
| Structured backend/input reasons | init/process/close, bad images/order/results/JSON/files/CLI | PASS |
| Không các bước ngoài GĐ2 | pose-only implementation; backend smoothing/masks off; policy tests | PASS |
| Toàn bộ test đạt | 76 passed, không skipped; dependency warnings được ghi | PASS |
| Limitations trung thực | single subject, unknown viewpoint, presence null, inverted/occlusion limits | PASS |

## Hạn chế và nhiệm vụ nhỏ nhất tiếp theo

- Chỉ hai mẫu; chưa pose accuracy/reference keypoints/toàn dataset hoặc unseen
  participant evaluation. Metadata action UCF101 không phải form ground truth.
- Visibility cao vẫn có thể đi cùng keypoint sai. Chưa viewpoint/occlusion
  classifier, stable identity hoặc multi-person enumeration.
- Chưa clean-environment/transitive lock/codec portability verification;
  legacy OpenCV provider metadata và protobuf deprecation còn tồn tại.
- Inference không giữ ảnh toàn video; issue indices trong manifest chiếm O(N*J).
  JSONL lặp backend/config để audit; tối ưu columnar storage chưa thuộc GĐ2.
- Partial JSONL sau failure không được coi là full completed run; chưa migration
  cho pose schema tương lai. Không sửa dataset/annotations/archive.
- Quality gate/preprocessing/features/segmentation/assessment chưa triển khai;
  không có form error hoặc score.

Tiếp theo nhỏ nhất: **GĐ3.1 — consecutive-missing spans theo joint và
QualityReport với eligibility/skip/degrade reasons, synthetic tests cho empty,
gap đầu/cuối và gap dài**. Chỉ ngưỡng engineering S3; không đổi pose failure
thành form error. Giai đoạn 3 chưa bắt đầu trong lượt này.
