# Giai đoạn 5.7A — Visual viewpoint audit và exploratory QualityGate

Ngày: 2026-10-12 (Asia/Saigon, theo ngày session). Người đánh giá: Codex agent.
Workspace không có Git repository; không giả commit/diff. Đã đọc report GĐ5.7,
current state/tasks/plan/schema/README và reports GĐ5.5–5.6, đối chiếu runtime
video/pose/quality/registry hiện hành. Không dùng archive implementation.

**GĐ5.7A COMPLETE trong phạm vi visual audit/exploratory. GĐ5.7 chính thức vẫn
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT. Toàn GĐ5 IN_PROGRESS.**
Exploratory candidate chưa đạt sequence/right-knee gate; chưa đo feature/góc.

## 1. Kết quả visual audit của agent

Chỉ kiểm tra video trong Data thuộc BodyWeightSquats. Class directory dùng để
xác định phạm vi, không là bằng chứng viewpoint hoặc ground truth về form.
224 paths có **112 byte SHA-256 khác nhau**; mỗi content chỉ review representative
trong UCF-101_GYM một lần. Hai duplicate paths được giữ trong inventory, không
giả định metadata/annotation độc lập hoặc kế thừa giữa các copies. SHA-256 chỉ
nhận diện bytes, không chứng minh viewpoint hay người tạo annotation.

Streaming decoder hiện hành đã decode đầy đủ từng representative. Agent thực sự
xem **560 hình mẫu**, năm hình/clip, trên **28 contact sheets**, bằng
view_image(detail=original). Không xem liên tục mọi frame của video. Chỉ số mẫu
floor((N−1)×f), f=0/0.25/0.5/0.75/1. Timestamp = frame_index/FPS theo
fps_derived_cfr_assumption; không khẳng định đây là container PTS. PNG lưu RGB từ
frame decoder, không xoay/lật/enhance; thumbnail giữ aspect ratio trong ô320×240,
caption ở ngoài ảnh. 112 decoder runs completed, không decode error.

| Viewpoint visual | Unique contents đã xem |
|---|---:|
| likely_side | 44 |
| oblique | 38 |
| front | 22 |
| uncertain | 8 |
| not_assessed | 0 |
| documented side-view | 0 |

Mỗi video có video ID, representative/duplicate relative paths, đầy đủ byte
SHA-256, decoder metadata, frame indices/timestamps/images đã xem, viewpoint,
camera/body orientation, thay đổi theo mẫu, joints in-frame/framing/occlusion,
confidence và lý do trong [visual_inventory.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/visual_inventory.json).
[agent_visual_reviews.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/agent_visual_reviews.json) giữ notes của agent;
catalog bên dưới giúp tra cứu từng review. Mọi record là
**visually_assessed_unverified**, documented=false. Confidence high/medium chỉ
thể hiện độ rõ của nhận xét visual trên mẫu; không là xác suất đã hiệu chuẩn,
không calibration camera, không measurement accuracy hay xác nhận form.

Input/sample inventories ban đầu giữ nguyên các trường chưa assessed; chúng là
snapshot trước review. visual_inventory.json là inventory đã nối kết notes và
viewed samples cuối cùng, không diễn giải snapshot ban đầu như kết quả sau audit.

### Candidate được chọn trước pose

- Review R106, video ID v_BodyWeightSquats_g25_c01.avi.
- Relative path: Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g25_c01.avi.
- Byte SHA-256: 893f7b0c89d10030705eeadfca1f9681a81190540bff1b2ab77d0ab3b614261e.
- Decoder đo: 103 frames, 25 FPS, 320×240, opencv:FFMPEG, timestamp CFR assumption.
- Đã xem frames0/25/51/76/102, timestamps0/1.00/2.04/3.04/4.08s.
- Visual likely_side, confidence high: thân và chân ở profile hướng image-left
  trong tư thế đứng, hạ và trung gian; camera/framing ổn định ở các mẫu. Hông/gối/
  cổ chân/bàn chân trong khung, đầu gần mép trên vùng ảnh, source letterboxed;
  nền tường/sàn đơn giản, một người hiện rõ, không vật cản foreground lớn.
  Hai chân phía gần/xa chồng nhau: visual in-frame không bảo đảm joint visibility.
- [candidate_selection.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/candidate_selection.json) đã lưu trước
  inference, SHA-256 ddde6bfd436efe716022cbcf7b351fbd4b9e6e123b901f85c2671d02f5137873.
  Không chọn từ filename/class label hoặc kết quả pose/quality. Không thử candidate
  khác để tìm allow sau khi candidate này fail gate.

Giả thuyết side trên toàn sequence chỉ phục vụ exploratory, ngoại suy từ năm mẫu;
các thời điểm chưa xem có thể khác. Không phải human protocol annotation.

## 2. Exploratory QualityGate dưới giả thuyết viewpoint

Hai pipeline fresh độc lập ingestion→MediaPipe pose→QualityGate:
fresh_unknown_v2 giữ raw viewpoint unknown; exploratory_side_v2 tạo mới pose với
side là giả thuyết agent. Không relabel raw unknown, không sửa source metadata,
CSV, annotations hoặc reports cũ. Manifest exploratory ghi documented=false,
source_viewpoint=unknown, run_viewpoint=side, assumption source/scope và selection
hash. Đây không phải documented eligibility. Không tái sử dụng saved quality report.

Requirements lấy trực tiếp từ registry hiện hành và assert đúng thứ tự:

| Metric key | Required joints | Viewpoint policy |
|---|---|---|
| squat_knee_angle_left_deg | left_hip, left_knee, left_ankle | side |
| squat_knee_angle_right_deg | right_hip, right_knee, right_ankle | side |

QualityGate tự có movement_analysis (tám shoulder/hip/knee/ankle joints).
Sequence action dùng policy hiện hành; allow ở một metric không nâng global gate.
Profile quality-s3-v1, source tier S3, min frames5, max missing ratio0.3, max low
visibility ratio0.3, max consecutive invalid frames10. Không relax hoặc đổi profile.

| Run | Sequence status/action; eligible | Left knee action; eligible; reasons | Right knee action; eligible; reasons |
|---|---|---|---|
| fresh_unknown_v2 | skipped/skip; false; low_pose_visibility | skip; false; viewpoint_unknown | skip; false; low_pose_visibility, viewpoint_unknown |
| exploratory_side_v2 | skipped/skip; false; low_pose_visibility | allow; true; [] | skip; false; low_pose_visibility |

Cả hai run: 103 frames có pose, 0 frames không pose; observed coverage toàn bộ
13 joints = 0.8700522778192681 (87.0052%). Pose detection100% không đồng nghĩa
joint usability100% hoặc accuracy. Joint counters đo được giống nhau ở hai run:

| Joint cho knee metrics | Observed | Low visibility | Missing | Out of frame | Observed coverage | Longest invalid span |
|---|---:|---:|---:|---:|---:|---:|
| left_hip | 103 | 0 | 0 | 0 | 100% | 0 |
| left_knee | 103 | 0 | 0 | 0 | 100% | 0 |
| left_ankle | 103 | 0 | 0 | 0 | 100% | 0 |
| right_hip | 103 | 0 | 0 | 0 | 100% | 0 |
| right_knee | 44 | 59 | 0 | 0 | 42.7184% | 29 |
| right_ankle | 92 | 11 | 0 | 0 | 89.3204% | 5 |

Right knee low visibility ở inclusive frame intervals3–31,44–46,51,60,62–66,
83–102; maxspan29 >10, ratio59/103=0.5728155339805825 >0.3. Right ankle low ở
14–15,18,20–22,85–89. Full by-joint counters/spans và per-frame provenance có
trong quality.json/frames.jsonl. Không chuyển low visibility thành observed.

Unknown report có 10 structured flags: 9 low_pose_visibility và 1 viewpoint_unknown.
Assumed-side report có 9 low_pose_visibility flags: right_elbow4, right_wrist2,
right_knee3; category insufficient_data, severity warning, stage quality, S3.
Right-knee flags trỏ spans3–31(length29),83–102(length20) và ratio59/103;
evidence ghi limits10/0.3. Elbow/wrist flags là report-wide diagnostics, không
được coi là required joints của hai knee metrics hoặc movement_analysis.

**Exploratory gate không allow toàn bộ. Measurement giữ blocked.** Left metric
allow không đủ khi sequence skip và right metric skip. Đây là measurement/input
limitation; không diễn giải thành lỗi kỹ thuật vận động viên.
observe_feature calls=0, joint_angle calls=0 (runtime call trace/assertion).
Không numeric adapters, preprocessing, interpolation, normalization, smoothing,
góc, repetition, phase, form assessment, score hoặc feedback trong audit này.

### Model/config/version và artifact identity

- Package ai-coach-gym0.5.4; MediaPipe0.10.14, backend mediapipe.solutions.pose.Pose.
- Bundled model complexity1, model asset SHA-256
  e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f.
- Mapping mediapipe-pose-13-v1, image_normalized_xy, dimensionless width/height,
  x_right_y_down; backend_most_prominent_single_subject, không multi-person guarantee.
- Config pose-extraction-v1: visibility/detection/tracking thresholds0.5,
  smoothing/segmentation disabled; hash
  3115cbcb75fbdc7b9421cb6936099b8ecf158c8abb1328047b93f52d7375b976.
- Profile quality-s3-v1 hash
  514eb022e774a64d9a1b01f85cde437d4befd81e657218a00306d80745357704.
- Registry gym-feature-registry-v1 hash
  bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a.
- Left definition hash875e6351320ea402ffcb377a00b95c0429d9008b9528f83b86b236d3f30bcce6;
  right definition hashc2a60690a7414f7827d20dcd80b7d6e69f8ab714637a75542e985699f13a4c93.

| Artifact (relative to run folder) | Byte SHA-256 |
|---|---|
| fresh_unknown_v2/frames.jsonl | c7d372d84a97d09c883592a7bc7a26ddd3adb0254649255f214b5512816bac9b |
| fresh_unknown_v2/quality.json | e7c465231febb9b18bba964f35ac1b56e88c5d8861b9204f7fcecdf787c4abb2 |
| exploratory_side_v2/frames.jsonl | 27e20ac2eee5878f47f8deff2f9d9a6cebbdaef2ca14f979ae61f07547f607ee |
| exploratory_side_v2/quality.json | a1299822dfabf0f7d9a26850f51f80dc57d24817020f72ca3c3daa40ee165c99 |
| exploratory_side_v2/manifest.json | 867d7f0f8cb9f8bf3e85418cb8009a19c27048675016993ff647970080adcb20 |

[exploratory_result_v2.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/exploratory_result_v2.json) ghi hai measured
runs, command, environment, warnings, config/model/profile và đủ tám pipeline
output hashes. [artifact_manifest.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/artifact_manifest.json) kiểm kê
mọi file audit mới với size/SHA-256, gồm sample PNG/contact sheets/scripts/notes,
failed attempt giữ nguyên và current report/docs identities; không tự hash chính nó.
Hash/content linkage là evidence audit, không chứng minh annotation authority.

## 3. Trạng thái chính thức của documented side-view input

**0 documented input; BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT.** Camera metadata
gốc vẫn UNKNOWN; agent visual notes không sửa hoặc thay formal annotation.
Report GĐ5.7/5.6 và dataset metadata giữ nguyên. GĐ5.7 chưa COMPLETE; GĐ5 toàn bộ
IN_PROGRESS. Ngay cả nếu người dùng xác nhận side cho R106, current exploratory
sequence/right-knee quality vẫn skip nên documentation một mình không mở measurement.
Chỉ official confirmed input và fresh gate sequence/cả hai metrics allow mới đủ
điều kiện cho bước measurement tiếp theo; không chứng minh accuracy hoặc form.

## 4. Người dùng còn cần xác nhận gì / bước nhỏ nhất

Xác nhận theo protocol cho exact video hiện có hoặc cung cấp input khác:

1. Exercise BodyWeightSquats; relative path/video ID và **đầy đủ byte SHA-256**.
2. Viewpoint side; protocol ID/version hoặc tài liệu nguồn; người xác nhận và ngày.
3. Toàn video hay inclusive frames/timestamps áp dụng; interval khớp nội dung hash.
4. Hướng camera so với mặt phẳng chuyển động, framing, joints in-frame, occlusion
   và hạn chế thay đổi viewpoint; không là đánh giá form.
5. FPS/resolution/timestamp policy nếu đã biết; giữ tách annotation con người với
   decoder-measured metadata (R106:25FPS/320×240/103frames/CFR assumption).

Bước nhỏ nhất là người dùng chọn/xác nhận input cùng các trường trên. Agent không
tự lập annotation chính thức. Sau verify linkage mới fresh official gate; cần
input có quality phù hợp cho cả hai bên, không thay limits để ép R106 allow.
Không bật adapters từ visual likely_side hoặc từ allow riêng ở gối trái.

## 5. Lệnh thực thi, kiểm chứng và môi trường

Các lệnh audit đã chạy (cwd project; artifacts mới, không rerun vào cùng paths):

```powershell
& .venv/Scripts/python.exe -B outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/prepare_visual_samples.py
& .venv/Scripts/python.exe -B outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/run_exploratory_gate.py
& .venv/Scripts/python.exe -B outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/run_exploratory_gate_v2.py
& .venv/Scripts/python.exe -B outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/write_audit_report.py
```

Sampling exit0, 112 full decodes/560 images/28 sheets. Agent đã mở tất cả sheets
để quan sát, không chạy classifier viewpoint. Attempt1 inference script exit1
(8.1689161s): audit caller để accumulator viewpoint unknown nhưng pose side,
GymError invalid_pose_sequence/input_error/stage pose/S3. Contract từ chối mismatch
đúng; không là backend failure hoặc lỗi người tập. [first_attempt.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/first_attempt.json)
giữ lỗi và năm partial/completed files, script gốc/fresh_unknown/exploratory_side
không ghi đè. Sửa caller truyền extractor.viewpoint trong helper mới; không sửa
runtime. Attempt2 exit0 (9.3955964s), hai fresh runs ở directories *_v2, cả hai103
frames. [second_attempt.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/second_attempt.json) ghi kết quả thực tế.

Attempt2 capture **412 Python UserWarning records**, cùng protobuf GetPrototype
deprecation, do warnings filter always ghi mọi lần lặp (không phải412 nguyên nhân
khác nhau). Native stderr riêng:4 feedback-manager warning lines,1 log-routing
warning notice,1 XNNPACK info. Python warning count của attempt1 không được lưu
trước exception, N/A; không suy count từ attempt2. Đây là inference warnings,
không phải pytest warnings. Không cài dependency/model/framework mới.

Windows build26200, Python3.11.9, ai-coach-gym0.5.4, MediaPipe0.10.14,
opencv-contrib-python5.0.0.93, NumPy2.4.6, Pydantic2.13.5, protobuf4.25.9,
Pillow12.3.0; pytest9.1.1 installed nhưng không chạy. Không clean-environment claim.

**Unit/integration pytest/full suite NOT_RUN. Passed/failed/errors/skipped/warnings:
N/A (JSON null)**. Chỉ audit dữ liệu/docs và helper trong artifacts, không đổi
application source/tests/config/policy, nên không chạy lại suite có MediaPipe
integrations. Fresh pipeline command exit0 không được trình bày là pytest pass.
Số713 passed/4 warnings trong GĐ5.6 là evidence lịch sử, không kết quả lượt này.

[verification.json](../outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c/verification.json) là audit assertions, không pytest:
112 unique content/review records,224 paths,560sample/28sheet hashes,source bytes,
canonical exact metric requirements, frame ID/index/CFR timestamp, raw unknown/
assumed side tách biệt, strict QualityReport JSON roundtrip, model/config/profile
identity và numeric call count0 đều đã kiểm chứng. Snapshot872existing files:
**868 unchanged,4 current docs changed,0 missing**. Tất cả224squat video files,
7Data metadata, source/tests/config/old reports/outputs trong snapshot giữ bytes.
Archive không mở/không sửa; snapshot scope loại environment/archive/cache và
non-squat action videos. Không claim đã hash mọi file ngoài scope.

Giới hạn: audit chỉ năm hình/clip, có thể bỏ sót thay đổi giữa mẫu; samples alias
theo chu kỳ có thể chỉ thấy tư thế đứng (đã ghi riêng trong notes); 2D phối cảnh
chưa calibration, limbs overlap, visibility không là accuracy/form. Backend chỉ
một prominent subject, không identity guarantee. UCF101 action labels không
technical ground truth. Chưa feature measurement, accuracy/form validation hoặc
official documented gate. Phạm vi GĐ5.7A đã xong với kết luận blocked rõ ràng.

## 6. Catalog review từng content

Các hash dưới đây là full SHA-256. Representative path cho mỗi video là
Data/UCF-101_GYM/BodyWeightSquats/<video ID>; inventory lưu riêng cả hai full
relative duplicate paths, metadata và image hashes. Timestamps là giây CFR.
Mọi dòng **visually_assessed_unverified**, documented=false. Các lý do đầy đủ
camera/body/change/framing/occlusion/confidence ở visual_inventory.json theo Review.

| Review | Video ID | Byte SHA-256 | Visual viewpoint | Confidence | Viewed frame indices | Timestamps (s) |
|---|---|---|---|---|---|---|
| R001 | v_BodyWeightSquats_g01_c01.avi | 1a199bf3ad385a66505fa0864a48fb9b9f9c1edfd1306d8345729ab06742c9e0 | likely_side | high | 0,66,133,199,266 | 0.00,2.64,5.32,7.96,10.64 |
| R002 | v_BodyWeightSquats_g01_c02.avi | 2a741ade440f9b072bce16584b76c15c346435fdf53e38d900620572a056901e | likely_side | high | 0,62,125,188,251 | 0.00,2.48,5.00,7.52,10.04 |
| R003 | v_BodyWeightSquats_g01_c03.avi | d82bcd9db5f82ae1bb746a5b86aefecbfa8761c2f71a91be6bf542677e6bfc3d | likely_side | high | 0,62,124,186,249 | 0.00,2.48,4.96,7.44,9.96 |
| R004 | v_BodyWeightSquats_g01_c04.avi | 1aaaade38d20cc28da71ba893f774e5383b107659555298a74dd3e8145b7ed00 | uncertain | medium | 0,43,86,129,173 | 0.00,1.72,3.44,5.16,6.92 |
| R005 | v_BodyWeightSquats_g02_c01.avi | 5146dc67db45583e42c6d92307923f03cb13aadae252626705c49b4585a0a23c | oblique | high | 0,25,50,75,101 | 0.00,1.00,2.00,3.00,4.04 |
| R006 | v_BodyWeightSquats_g02_c02.avi | 4125f79ecb7e05b83cada5efca210c78731de90a5f33b3495d174c788f33ffff | oblique | high | 0,14,28,42,56 | 0.00,0.56,1.12,1.68,2.24 |
| R007 | v_BodyWeightSquats_g02_c03.avi | 9144761adb133a9759fbb8563779dd09e8509334139b78e0978730911d40bca6 | likely_side | medium | 0,29,58,87,116 | 0.00,1.16,2.32,3.48,4.64 |
| R008 | v_BodyWeightSquats_g02_c04.avi | a701c056cc125e44ca23ee9f1b7dc47caaeedc4e84c8c5a744fcb319267dfc3b | front | high | 0,30,60,90,120 | 0.00,1.20,2.40,3.60,4.80 |
| R009 | v_BodyWeightSquats_g03_c01.avi | 5d718b5347ad531d5c20e5d0f1809c7d62bdf0b36d0498037786b04046a5df64 | oblique | medium | 0,32,65,98,131 | 0.00,1.28,2.60,3.92,5.24 |
| R010 | v_BodyWeightSquats_g03_c02.avi | c85756965c8f7d753409ff70e0e654ba960ee7d2c3dbba8f0eca3e163c156780 | oblique | medium | 0,12,25,38,51 | 0.00,0.48,1.00,1.52,2.04 |
| R011 | v_BodyWeightSquats_g03_c03.avi | 926b766c6e388372fb75c7212a6c7d04beb56ea9bc1ec0de325eacb40636cd0a | oblique | medium | 0,11,23,35,47 | 0.00,0.44,0.92,1.40,1.88 |
| R012 | v_BodyWeightSquats_g03_c04.avi | 20310b903b8597bf7a9c063a5240ddf6b41eb2c377ebd3c134d028d312d2ead9 | front | high | 0,25,50,75,100 | 0.00,1.00,2.00,3.00,4.00 |
| R013 | v_BodyWeightSquats_g03_c05.avi | 59fdbe5dbf02052864a8cb81c5071421aa5ad93d81c379130ed6acb5b7f7f1ab | front | high | 0,18,37,55,74 | 0.00,0.72,1.48,2.20,2.96 |
| R014 | v_BodyWeightSquats_g04_c01.avi | 7d6b58bf5633b9d0315aa61bf0fc89fc8de84f673d57592af586c4c04acd4930 | uncertain | medium | 0,32,64,96,129 | 0.00,1.28,2.56,3.84,5.16 |
| R015 | v_BodyWeightSquats_g04_c02.avi | b6a973fb6bfab0b88f36887157dc41a727c234f05e37f1d6ff207894812415ea | front | high | 0,14,29,43,58 | 0.00,0.56,1.16,1.72,2.32 |
| R016 | v_BodyWeightSquats_g04_c03.avi | 4324177b7cda96aeab3e0b07e8692a5bcd9ca4724ebd2d91a9ecef60d530ec26 | likely_side | high | 0,21,43,65,87 | 0.00,0.84,1.72,2.60,3.48 |
| R017 | v_BodyWeightSquats_g04_c04.avi | 2615311e2b068812f7e24aea8b0483dbd966368c136d0add577c6a366d4ac9f0 | likely_side | high | 0,15,31,46,62 | 0.00,0.60,1.24,1.84,2.48 |
| R018 | v_BodyWeightSquats_g05_c01.avi | 935ed78bdc8578d4da2347f2f9ab5e0e3ca456d090e1c473601e08f56582a1c0 | likely_side | medium | 0,17,35,52,70 | 0.00,0.68,1.40,2.08,2.80 |
| R019 | v_BodyWeightSquats_g05_c02.avi | 4e596e2c1d4ad2d2207e5db543deb961321f523d838e8dc97148928b034a4093 | likely_side | medium | 0,33,67,100,134 | 0.00,1.32,2.68,4.00,5.36 |
| R020 | v_BodyWeightSquats_g05_c03.avi | 808df884cbc19b70f1d234de5bbc322e095e8a5079d3f8fc606ffe5fb1d428ae | likely_side | medium | 0,13,26,39,53 | 0.00,0.52,1.04,1.56,2.12 |
| R021 | v_BodyWeightSquats_g05_c04.avi | eff18e8ad50d16e8ea3ae0353372a8722a6297186f055550f5fe1f7f0430cbe9 | uncertain | medium | 0,64,129,194,259 | 0.00,2.56,5.16,7.76,10.36 |
| R022 | v_BodyWeightSquats_g06_c01.avi | 4f699753bad8c8dbf7e95c866bea8d218726dae3d08ffa315034c84e80c91ad4 | oblique | high | 0,44,89,133,178 | 0.00,1.76,3.56,5.32,7.12 |
| R023 | v_BodyWeightSquats_g06_c02.avi | cef35cf4e978d7ec7ff10133c4a5e888de78c3dcf13dacb5b8aa4ec47862a24f | uncertain | medium | 0,30,60,90,121 | 0.00,1.20,2.40,3.60,4.84 |
| R024 | v_BodyWeightSquats_g06_c03.avi | 6b9432d44918f0ad0526083ee5672c2a42968ff3153083d80da39af89b15f443 | oblique | high | 0,37,75,113,151 | 0.00,1.48,3.00,4.52,6.04 |
| R025 | v_BodyWeightSquats_g06_c04.avi | bfdddb9fcd8c1826f00b2b5e577b9cfc32f850959383bd09fc873f1aae1652ec | oblique | high | 0,23,47,70,94 | 0.00,0.92,1.88,2.80,3.76 |
| R026 | v_BodyWeightSquats_g06_c05.avi | 92f55a393361c128fbd82c012e16c7c5587479b72e3f71177325c4fb16c358c8 | oblique | high | 0,44,88,132,176 | 0.00,1.76,3.52,5.28,7.04 |
| R027 | v_BodyWeightSquats_g07_c01.avi | 4c10e8d1dff20a18ba07c68caf77f0c893f457a422e5298f5804a601fe4f5a1b | likely_side | medium | 0,30,61,91,122 | 0.00,1.20,2.44,3.64,4.88 |
| R028 | v_BodyWeightSquats_g07_c02.avi | c94a160947d9575ae9b620f3fb182a417b6d63b85abe0b433ddcd824a4f78382 | likely_side | medium | 0,25,51,76,102 | 0.00,1.00,2.04,3.04,4.08 |
| R029 | v_BodyWeightSquats_g07_c03.avi | 021cb90bd45826f00772ff01b380643a80a5ba16c50ee4365fd8e06bae501816 | likely_side | medium | 0,20,41,61,82 | 0.00,0.80,1.64,2.44,3.28 |
| R030 | v_BodyWeightSquats_g07_c04.avi | 2da676f937f0a97c4dbc0e143d3b20e09755a4b1e8b796e59b55979368487dd4 | uncertain | medium | 0,13,26,39,53 | 0.00,0.52,1.04,1.56,2.12 |
| R031 | v_BodyWeightSquats_g08_c01.avi | 0cdb88565f2f407e3fab3c43d4af0cabf3227e392a4d664a53e48a6e01b2396b | oblique | high | 0,18,37,56,75 | 0.00,0.72,1.48,2.24,3.00 |
| R032 | v_BodyWeightSquats_g08_c02.avi | 63978796c32d51054f9cda7f2fe04c2a750f40c65444157444927f0ecd4d0989 | oblique | high | 0,26,52,78,105 | 0.00,1.04,2.08,3.12,4.20 |
| R033 | v_BodyWeightSquats_g08_c03.avi | 7b9a900c8e2a5a260692ae345fee890acb5b662d2bb8bda3714b7d853f1689ee | oblique | high | 0,12,24,36,48 | 0.00,0.48,0.96,1.44,1.92 |
| R034 | v_BodyWeightSquats_g08_c04.avi | fb7fe00c58d2cb24298e2b3fb577df405cb4244e70184f3f4bd55c8ab8f0c3fb | oblique | high | 0,25,51,77,103 | 0.00,1.00,2.04,3.08,4.12 |
| R035 | v_BodyWeightSquats_g09_c01.avi | 2bf5834cca16dba23d9ef2a5c20f51551e992046850e39fd4e2b17b369b3dc43 | oblique | high | 0,49,99,149,199 | 0.00,1.96,3.96,5.96,7.96 |
| R036 | v_BodyWeightSquats_g09_c02.avi | 4638b347ea970502f74d7b5f3b3cd6bb082df7201ad979a769afb6d587a0f545 | oblique | high | 0,50,100,150,200 | 0.00,2.00,4.00,6.00,8.00 |
| R037 | v_BodyWeightSquats_g09_c03.avi | 0025a8d6d83dc07d38072eedfad0e7af3509caa8ccb80d65ed405f8c97a82680 | oblique | high | 0,28,57,85,114 | 0.00,1.12,2.28,3.40,4.56 |
| R038 | v_BodyWeightSquats_g09_c04.avi | d4c0ca1b58ad54e504ce9d2ca9bbac22647e4a31aa6b40136fff2db8d151bcff | oblique | high | 0,30,61,91,122 | 0.00,1.20,2.44,3.64,4.88 |
| R039 | v_BodyWeightSquats_g09_c05.avi | 32887e3852ffadbb890902d7de5d2566bd46d3753b28cd4c1f1a7f1464deab49 | likely_side | high | 0,57,114,171,228 | 0.00,2.28,4.56,6.84,9.12 |
| R040 | v_BodyWeightSquats_g09_c06.avi | 3f96470cb1e6ec8b27e25419b296d6c6b7079b4125f2848f126b534be359cfe8 | likely_side | high | 0,64,129,194,259 | 0.00,2.56,5.16,7.76,10.36 |
| R041 | v_BodyWeightSquats_g09_c07.avi | fcd45be9dbc287e030d64f10219488a96061a4949406351b98b7475178291c88 | likely_side | high | 0,58,116,174,232 | 0.00,2.32,4.64,6.96,9.28 |
| R042 | v_BodyWeightSquats_g10_c01.avi | 768f35cdd34317d387df850534e915f80cdc26cbb6cb794b1b26a12479972b5c | front | high | 0,41,83,125,167 | 0.00,1.64,3.32,5.00,6.68 |
| R043 | v_BodyWeightSquats_g10_c02.avi | ab723a51ef1cdf86fc59759340bea20ea11da262c3a8b1cb619d47474bd5509b | front | high | 0,39,79,119,159 | 0.00,1.56,3.16,4.76,6.36 |
| R044 | v_BodyWeightSquats_g10_c03.avi | e486149dc7eb459aa1a832e47c8fd71a3e857e3d60ba277e27243b39dcd75137 | likely_side | high | 0,37,75,113,151 | 0.00,1.48,3.00,4.52,6.04 |
| R045 | v_BodyWeightSquats_g10_c04.avi | e35ca709aa707d92e172edfd20b2dd73ec324b1ff84dd113b4efe136e8b45824 | likely_side | high | 0,36,73,110,147 | 0.00,1.44,2.92,4.40,5.88 |
| R046 | v_BodyWeightSquats_g10_c05.avi | 94d9cf59d734d269a72fadc610acd2e1eae5c990d7abb2a8c4de1f4a207e9bf0 | likely_side | high | 0,39,78,117,157 | 0.00,1.56,3.12,4.68,6.28 |
| R047 | v_BodyWeightSquats_g11_c01.avi | 688b6cae58bbe82a22f367ab2203e7906434be8e5fa0dc7449182455349fb300 | likely_side | medium | 0,20,40,60,80 | 0.00,0.80,1.60,2.40,3.20 |
| R048 | v_BodyWeightSquats_g11_c02.avi | fa30b8948a78427dd21a88013111d33c82ec85ecbeffa70f434fdfb4d4adf7e8 | likely_side | medium | 0,23,47,70,94 | 0.00,0.92,1.88,2.80,3.76 |
| R049 | v_BodyWeightSquats_g11_c03.avi | 62c17f3ab510ee4828fa918080ccc833cf4248648358a22aaee2f06b09141bbc | likely_side | medium | 0,21,42,63,85 | 0.00,0.84,1.68,2.52,3.40 |
| R050 | v_BodyWeightSquats_g11_c04.avi | 571f987af0e7104888f9fe3816eb67b4aa19dee8acd97bd246a0f2c6e3047f4e | likely_side | medium | 0,19,38,57,76 | 0.00,0.76,1.52,2.28,3.04 |
| R051 | v_BodyWeightSquats_g12_c01.avi | 959e85d235ebff96b66680adddfb0178c4163466fcf9084357a6c179faf59c61 | likely_side | high | 0,59,118,177,236 | 0.00,2.36,4.72,7.08,9.44 |
| R052 | v_BodyWeightSquats_g12_c02.avi | 042c9a0caf5b0061f55c1d936d146cb7460b2cece6917780766cc90ce6e68b5e | likely_side | high | 0,35,71,106,142 | 0.00,1.40,2.84,4.24,5.68 |
| R053 | v_BodyWeightSquats_g12_c03.avi | 25f3e4bd3df23f9f00a9cba0b4c87de2402308ce9c839614f59734c9bff86275 | likely_side | high | 0,29,59,89,119 | 0.00,1.16,2.36,3.56,4.76 |
| R054 | v_BodyWeightSquats_g12_c04.avi | 82d1803e3f00e396b30e78a7f38c409aa2b552e3d2031e6efa9872c572d19363 | front | high | 0,48,96,144,192 | 0.00,1.92,3.84,5.76,7.68 |
| R055 | v_BodyWeightSquats_g13_c01.avi | 7ec322097f711a9b05261244b7c064755a66841c645c9731baa10bcca7ef006a | likely_side | high | 0,27,55,83,111 | 0.00,1.08,2.20,3.32,4.44 |
| R056 | v_BodyWeightSquats_g13_c02.avi | e81203075ab4a3c5a70a1f86287cb1f38d77dad9100fd0a01922d1e1dce7fe82 | likely_side | high | 0,15,30,45,61 | 0.00,0.60,1.20,1.80,2.44 |
| R057 | v_BodyWeightSquats_g13_c03.avi | 989a6d1c9667a92b245b72acc4c464154351d86ee2a64930852f38b596472a59 | likely_side | high | 0,28,57,85,114 | 0.00,1.12,2.28,3.40,4.56 |
| R058 | v_BodyWeightSquats_g13_c04.avi | 973f7546b4314ed2eb3e18dedfb5700642563b1f1d54586e3a0a3c9686eb409c | likely_side | high | 0,18,37,55,74 | 0.00,0.72,1.48,2.20,2.96 |
| R059 | v_BodyWeightSquats_g14_c01.avi | 97177e09ea4c96b98b96767ab7c0ca4831407a6c63751e8b71740cdfff7b4c22 | front | high | 0,27,54,81,109 | 0.00,1.08,2.16,3.24,4.36 |
| R060 | v_BodyWeightSquats_g14_c02.avi | c9731d35dd3c7ae30bf6b68f0547915829df15ee2e90d6acfa2942b3cf2290ac | front | high | 0,28,56,84,113 | 0.00,1.12,2.24,3.36,4.52 |
| R061 | v_BodyWeightSquats_g14_c03.avi | 96e8c5c0d6cc2efc838616fdce4c85a28eef2a7905416a34db113f4ab7aa59e8 | front | high | 0,28,56,84,113 | 0.00,1.12,2.24,3.36,4.52 |
| R062 | v_BodyWeightSquats_g14_c04.avi | 1a3e8de0b62042f96e54e46fc7f3fb11163c58345c5f36f1676bbb967f41092b | front | high | 0,27,55,82,110 | 0.00,1.08,2.20,3.28,4.40 |
| R063 | v_BodyWeightSquats_g15_c01.avi | db3cc50be948c2a99375f95190a96689fedace1a1df8ffe0816b0ba50204ed95 | likely_side | high | 0,35,70,105,141 | 0.00,1.40,2.80,4.20,5.64 |
| R064 | v_BodyWeightSquats_g15_c02.avi | fb3fd23c8d59c933a72c968ab9b8154687ef440648661e468436fc98e1d36db3 | likely_side | high | 0,30,60,90,121 | 0.00,1.20,2.40,3.60,4.84 |
| R065 | v_BodyWeightSquats_g15_c03.avi | 38c8500955b0a3abd565a5f8c593ec68b40c2e644a3a41bef0a3270290749880 | front | high | 0,34,68,102,136 | 0.00,1.36,2.72,4.08,5.44 |
| R066 | v_BodyWeightSquats_g15_c04.avi | 35d16fe010c21597ed4f3d6b0cf3fb4f26d8e8271c50af15d1f7996356c6a696 | front | high | 0,31,62,93,125 | 0.00,1.24,2.48,3.72,5.00 |
| R067 | v_BodyWeightSquats_g16_c01.avi | 0dad4950f8143b3101290ebd935f3d5e643fc2c17f807ae0976ef698bf045c85 | oblique | high | 0,18,36,54,73 | 0.00,0.72,1.44,2.16,2.92 |
| R068 | v_BodyWeightSquats_g16_c02.avi | 7c9cbbbdc538b80166dee13cb019e8c81d85325b6ec08c54190700450e8a9d54 | oblique | high | 0,18,37,55,74 | 0.00,0.72,1.48,2.20,2.96 |
| R069 | v_BodyWeightSquats_g16_c03.avi | 1577f817617c7b8071e766ec4f09fde3245548241d260c61430dec903e385d92 | oblique | high | 0,20,40,60,80 | 0.00,0.80,1.60,2.40,3.20 |
| R070 | v_BodyWeightSquats_g16_c04.avi | 2796833b804c10866180123bea5d21671b21a85c2e02a621f1083a9f4a5b6315 | oblique | high | 0,19,38,57,76 | 0.00,0.76,1.52,2.28,3.04 |
| R071 | v_BodyWeightSquats_g17_c01.avi | 76cf2fcda6c8d977a0f9d32f3205bba4b85adb2d2bf4e951ed2d00d9129ed932 | front | high | 0,44,89,134,179 | 0.00,1.76,3.56,5.36,7.16 |
| R072 | v_BodyWeightSquats_g17_c02.avi | c663d8f64fe1013135394f1cba3a9b71a69f993c833fa5a5ae1cc6b7002cb3d5 | uncertain | medium | 0,43,86,129,173 | 0.00,1.72,3.44,5.16,6.92 |
| R073 | v_BodyWeightSquats_g17_c03.avi | a51fd9987b4ac0fa9dfd3f96f65072a5696d100372ca322e928357e4cd2d0177 | uncertain | medium | 0,42,84,126,169 | 0.00,1.68,3.36,5.04,6.76 |
| R074 | v_BodyWeightSquats_g17_c04.avi | 2686d11a4f2949c133cd93c24f27072c2ab9c38c1307817bd8ffd56a489f1e11 | oblique | high | 0,35,71,107,143 | 0.00,1.40,2.84,4.28,5.72 |
| R075 | v_BodyWeightSquats_g18_c01.avi | 38f8926eb66a742d4af806b14735e9b7b69cd49dbab1af6bb6382b551db42b27 | oblique | medium | 0,50,101,151,202 | 0.00,2.00,4.04,6.04,8.08 |
| R076 | v_BodyWeightSquats_g18_c02.avi | 30f3542590988502062170883ce089391645049aa74f6ebaf2d6b9a9c278ef22 | oblique | medium | 0,53,106,159,213 | 0.00,2.12,4.24,6.36,8.52 |
| R077 | v_BodyWeightSquats_g18_c03.avi | 64de54a4848b1e0a5a1244e8b59b7e756d634f5acba76de912a251395be601a3 | oblique | medium | 0,52,105,157,210 | 0.00,2.08,4.20,6.28,8.40 |
| R078 | v_BodyWeightSquats_g18_c04.avi | 938ba708104b53fa181980808aebeb3a36db0cbc1ca020cd176b0200deeec50c | oblique | medium | 0,55,111,166,222 | 0.00,2.20,4.44,6.64,8.88 |
| R079 | v_BodyWeightSquats_g19_c01.avi | bb1de99bc18e98b419793368e26be06ec9671b9b7c062ffa66f2c3e1006cc167 | front | high | 0,33,67,100,134 | 0.00,1.32,2.68,4.00,5.36 |
| R080 | v_BodyWeightSquats_g19_c02.avi | 231bbe0c145db1fec7016007797db495894fdea0aa5b40563f1ebf8ad704a318 | front | high | 0,27,55,83,111 | 0.00,1.08,2.20,3.32,4.44 |
| R081 | v_BodyWeightSquats_g19_c03.avi | 2e7f3316e3b7067f9d34c07e44178f031368ec31de07db790566ac7ec9715f9e | likely_side | high | 0,35,71,107,143 | 0.00,1.40,2.84,4.28,5.72 |
| R082 | v_BodyWeightSquats_g19_c04.avi | b849a798d778ed7dfc75918beddb916d8bb72d948adb2036278be8da633468f2 | likely_side | high | 0,26,53,80,107 | 0.00,1.04,2.12,3.20,4.28 |
| R083 | v_BodyWeightSquats_g20_c01.avi | 6f1b0c65e755529d70ffafa233bf231f921323a6d93e02e43b5961ff56bd0979 | front | high | 0,24,49,74,99 | 0.00,0.96,1.96,2.96,3.96 |
| R084 | v_BodyWeightSquats_g20_c02.avi | 317d3ab158214684b0da2ca94ecb6015b7eeb5e3e48d21bb09ca9d88840af90d | front | high | 0,43,86,129,173 | 0.00,1.72,3.44,5.16,6.92 |
| R085 | v_BodyWeightSquats_g20_c03.avi | e244647909e6fee960414632647d94aaacb081b57a3d75cd7a7cdd304fa0df12 | oblique | high | 0,23,46,69,93 | 0.00,0.92,1.84,2.76,3.72 |
| R086 | v_BodyWeightSquats_g20_c04.avi | 23d1e9cadf53d9910cb0ec82ae313d2a1e92814fa0602884e0611f6180c9fb57 | oblique | high | 0,40,81,122,163 | 0.00,1.60,3.24,4.88,6.52 |
| R087 | v_BodyWeightSquats_g20_c05.avi | 5776da9db9fbf01521159bcb60c64e2f81973fd5b54a4a2fdd3e6b3a714a0938 | likely_side | high | 0,33,67,101,135 | 0.00,1.32,2.68,4.04,5.40 |
| R088 | v_BodyWeightSquats_g20_c06.avi | 727c5649b142dec6528f8dbc17619353231e07bb0587aef9818d8f209adb05b0 | likely_side | high | 0,41,83,124,166 | 0.00,1.64,3.32,4.96,6.64 |
| R089 | v_BodyWeightSquats_g21_c01.avi | 79bb3a503030acfc82ff1b3bc4a7a3e48269aed07c42c863e69406d6c53804b4 | front | high | 0,12,25,37,50 | 0.00,0.48,1.00,1.48,2.00 |
| R090 | v_BodyWeightSquats_g21_c02.avi | 7fa133291d5b3e1263b8f31e9450fc44f78cb15b86051814651d18576b3eeed5 | front | high | 0,15,31,47,63 | 0.00,0.60,1.24,1.88,2.52 |
| R091 | v_BodyWeightSquats_g21_c03.avi | 8101a40b621f941588af3df911da381ee828495458ea3255e62f3f52fe5c528b | oblique | high | 0,13,26,39,53 | 0.00,0.52,1.04,1.56,2.12 |
| R092 | v_BodyWeightSquats_g21_c04.avi | 56d6607277ecb3826a6fbe641054fd436ec8f3c24f5d6e2990a4768c66e8f814 | oblique | high | 0,21,43,64,86 | 0.00,0.84,1.72,2.56,3.44 |
| R093 | v_BodyWeightSquats_g22_c01.avi | 182acb5d4f26076759dc1cba1ff4248fa13f0de819b6bec51f66033095cbfe1f | oblique | medium | 0,22,44,66,89 | 0.00,0.88,1.76,2.64,3.56 |
| R094 | v_BodyWeightSquats_g22_c02.avi | 932444ce88a2b6523fe8f57273b0bbac97bb7e92a4cc497b527e4630c7f22438 | oblique | medium | 0,15,30,45,61 | 0.00,0.60,1.20,1.80,2.44 |
| R095 | v_BodyWeightSquats_g22_c03.avi | 3e18377f2df1e2b6edff558e687420ea2d2e912d379f3388bbe9f8cbb75883e5 | oblique | medium | 0,34,68,102,137 | 0.00,1.36,2.72,4.08,5.48 |
| R096 | v_BodyWeightSquats_g22_c04.avi | d5b2890875595fc88e67651d78f61fd25c75d6db16ff3afd7a710e3609b9b884 | oblique | medium | 0,15,30,45,61 | 0.00,0.60,1.20,1.80,2.44 |
| R097 | v_BodyWeightSquats_g23_c01.avi | 4ed66ce179adaa5123d5a773e97e99f4707ea7fe76eb633813d7efa6595d3641 | front | high | 0,12,25,38,51 | 0.00,0.48,1.00,1.52,2.04 |
| R098 | v_BodyWeightSquats_g23_c02.avi | 83dd766187c65eefc3ae3a44c915ab5941133c442018387862bf9e957284d40e | front | high | 0,18,37,56,75 | 0.00,0.72,1.48,2.24,3.00 |
| R099 | v_BodyWeightSquats_g23_c03.avi | 510ea1dc263e88f0a498f1e739628accdb1d0031ea49c69b0dae37c2bb6a54d0 | likely_side | high | 0,18,36,54,72 | 0.00,0.72,1.44,2.16,2.88 |
| R100 | v_BodyWeightSquats_g23_c04.avi | 90bf18d6d772d16ff94a37f3dc83e88590e9d0cceba726c0ab1af9f49991c3b4 | likely_side | high | 0,16,32,48,65 | 0.00,0.64,1.28,1.92,2.60 |
| R101 | v_BodyWeightSquats_g24_c01.avi | b83a5b76f72cdd4ee07cdabd46b15b7eb8731bc01f47717b1e04d47cda52030d | uncertain | medium | 0,57,114,171,229 | 0.00,2.28,4.56,6.84,9.16 |
| R102 | v_BodyWeightSquats_g24_c02.avi | 754c55538fd118fb8fd78d6c4d8b5896b8f3076916fd28ee673943df897bdda8 | oblique | high | 0,55,110,165,220 | 0.00,2.20,4.40,6.60,8.80 |
| R103 | v_BodyWeightSquats_g24_c03.avi | a1cc4a08b6d48e0db03fe8f8dfaaa41a2b9e18490e2ce4d24594d935068afa1c | oblique | high | 0,58,117,176,235 | 0.00,2.32,4.68,7.04,9.40 |
| R104 | v_BodyWeightSquats_g24_c04.avi | 33d9fe24ac87e45cc6cabe036673085521116a1597b6b5d101233cd6f3ea4d81 | oblique | high | 0,59,118,177,236 | 0.00,2.36,4.72,7.08,9.44 |
| R105 | v_BodyWeightSquats_g24_c05.avi | 860260ecca1657e7350d48883b0eae947208198153215c1910b61c582486889c | oblique | high | 0,19,38,57,77 | 0.00,0.76,1.52,2.28,3.08 |
| R106 | v_BodyWeightSquats_g25_c01.avi | 893f7b0c89d10030705eeadfca1f9681a81190540bff1b2ab77d0ab3b614261e | likely_side | high | 0,25,51,76,102 | 0.00,1.00,2.04,3.04,4.08 |
| R107 | v_BodyWeightSquats_g25_c02.avi | 7fa9f6f7eae4735daed83ef0c7bd654d79477d3c5cb8fd66ef789aaff9294643 | likely_side | high | 0,27,54,81,109 | 0.00,1.08,2.16,3.24,4.36 |
| R108 | v_BodyWeightSquats_g25_c03.avi | 7ee748e9e91a1170cb5469a4b4f851cf78022e64b59c5d51dbebfdd41475d5da | likely_side | high | 0,28,57,86,115 | 0.00,1.12,2.28,3.44,4.60 |
| R109 | v_BodyWeightSquats_g25_c04.avi | 2d3cebd4db8c3dbfe3ddc6743f68107032200252f6375ef9eac250b8dd641820 | likely_side | high | 0,26,53,80,107 | 0.00,1.04,2.12,3.20,4.28 |
| R110 | v_BodyWeightSquats_g25_c05.avi | f5ed32ee2ab7cc03c9c59b53e00ff78e324e491ac13c5d671a8cb9307543ed8b | likely_side | high | 0,26,52,78,105 | 0.00,1.04,2.08,3.12,4.20 |
| R111 | v_BodyWeightSquats_g25_c06.avi | 2cbc0a149ef6a8720e6e30b48a4a843bef5e47baf6c5035f97f262cc4bad5207 | likely_side | high | 0,25,50,75,100 | 0.00,1.00,2.00,3.00,4.00 |
| R112 | v_BodyWeightSquats_g25_c07.avi | 35a16fe63cfd3af9d270de87ac963c2dbd516d9779ea87af378269355b5bef97 | likely_side | high | 0,31,62,93,124 | 0.00,1.24,2.48,3.72,4.96 |
