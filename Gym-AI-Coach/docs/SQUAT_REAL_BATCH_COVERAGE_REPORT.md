# BodyWeightSquats — batch knee-angle coverage và representative demos

2026-10-20, Asia/Saigon. Development pool cố định, không holdout evaluation,
không measurement accuracy/generalization hoặc form validation.
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.
Không có Git repository; không tạo commit/diff giả.

**Hoàn tất implementation, xử lý40/40 clips và80/80 cặp clip–bên,
3 demos và offline/integrity checks; chưa scientific/official validation.**
704 regression tests pass;22 sampled PNG đã xem; browser/native playback NOT_RUN.

## Mở sản phẩm và artifact layout

Run mới:
`outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/`.
Mở `batch/START_HERE.html` để chọn allow/degrade/skip và xem bảng đủ40 clips.
Giữ nguyên cả thư mục; mỗi example có START_HERE.html, MP4 overlay, original.avi,
JSON/coverage/selection/guide/hash receipts. Không cần server/mạng/Python để xem.
Nếu chính sách máy chặn local file/codec, không tắt bảo vệ hoặc đi vòng.

`batch/clips/Sxxx/`: preprocessing dùng chung hai bên, saved pose/quality/source
snapshots, left.jsonl và right.jsonl, stats/timing và manifest riêng.
`batch/summary.json`: đầy đủ80 pairs,40 clips, micro/macro/by-side, provenance,
reasons và lỗi. `pairs.csv` xuất bảng80 cặp; HTML chung hiển thị40 clip và cả
hai bên. `selection.json` khóa mẫu/rule/config trước geometry/render.
`timing.json` ghi stage/worker/call counts. Root run có test/verification/QA/
preservation receipts và artifact inventory.

## Nguồn và ranh giới kế thừa

Chỉ dùng video trong `Data/UCF-101_GYM/BodyWeightSquats/`, đúng40 byte-distinct
contents của audit
`outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/`.
Demo trước:
`outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/`.
Không mở dataset khác hoặc archive; không chạy lại MediaPipe.

Tái sử dụng load_pool và validators hiện có: root/per-clip hashes, source bytes,
selection/visual audit linkage, raw/assumed timeline/identity, QualityReport/core
counts và current profile/registry identity. Raw unknown và assumed-side chỉ
khác viewpoint. Các artifact không tương thích phải fail closed, không chuyển
sang inferred allow. Source camera_view=UNKNOWN; side hypothesis luôn
visually_assessed_unverified/documented=false, không confirmed annotation.

Preprocessing g01_c01 được tái sử dụng sau hash + strict full-result readback và
so sánh exact raw frames/report/metadata/profile. Các clip khác preprocess mới
một lần mỗi clip bằng profile hiện hành. Không tái sử dụng số đo cũ: cả hai bên
đều đi qua adapter hiện hành cho mọi frame, kể cả metric degrade/skip.

Factory kiểm tra full source một lần mỗi frame rồi dùng **cùng frame/report
context** cho hai adapter độc lập. Không bypass validator hoặc sửa contract để
tăng tốc; scope reuse là trong một exact preprocessing result/frame. Full
factory linkage và observation hashes được lưu trong từng record JSONL.

Core giữ nguyên: registry v2/framewise-metric-only-v1, observation0.4.0,
factory0.2.0/preprocessing-context-v1, quality-s3-v1, preprocessing-s3-v1,
joint-angle-2d-v1 và saved MediaPipe0.10.14/full complexity1 identity.
Quality thresholds min5/missing0.30/low0.30/maxgap10;25FPS preprocessing vẫn
gap4/window9/polyorder2. Không tune model/threshold/profile/policy/schema.
Legacy sequence+metric gates không thay đổi. New batch_version là artifact
envelope, không đổi shared numerical schemas/API.

## Quy tắc chọn mẫu đã khóa

Trước geometry/render, trong mỗi metric action allow/degrade/skip: chọn cặp
đầu tiên theo relative path, rồi left trước right. Dùng nguyên clip, không đổi
bên, không chọn lại dựa vào số đo/độ đẹp/video đã xem. Nếu nhóm vắng ghi
GROUP_ABSENT; render lỗi ghi RENDER_ERROR, không tự thay mẫu khác.

| Nhóm | Clip | Bên |
|---|---|---|
| allow | v_BodyWeightSquats_g01_c01.avi | left |
| degrade | v_BodyWeightSquats_g04_c03.avi | right |
| skip | v_BodyWeightSquats_g01_c01.avi | right |

Đây là3 trường hợp/2 unique videos, không phải3 subject độc lập. Hai bên g01
được trình bày riêng, không fusion/fallback. Rule dùng audit/quality trước khi
đo; không đại diện ngẫu nhiên hoặc đánh giá ngoại suy. Hash/time evidence của
frozen selection nằm trong batch manifest và command receipts.

Renderer tái sử dụng real_descriptive renderer với tùy chọn dịch unavailable
sang “Không đủ điều kiện đo” và lý do Việt ngắn. Các reason codes/structured
flags trong JSON giữ nguyên; default renderer cũ không đổi. Video nói rõ raw
skeleton/processed angle, observed/low colors, unknown camera, unverified side.
Gate metric áp dụng toàn clip: clean raw frames trong clip degrade/skip vẫn
null; không diễn giải quality thành lỗi kỹ thuật người tập.

## Kết quả thực đo và mẫu số

40/40 clips,80/80 pairs,5.314 source frames /10.628 frame–bên đã xét đủ.
0 processing errors,0 unevaluated,0 incompatible artifacts, không loại clip/cặp
khỏi denominator.35 clips có số đo ở đúng một bên;5 clips không bên nào có;
0 clips có hai metric allow. Chỉ5 sequence degrade,35 skip,0 allow.

| Phạm vi | Metric allow/degrade/skip (cặp) | Available / frame–bên | Unavailable | Numeric micro | Numeric macro | Raw observed/in-frame | Raw micro | Raw macro |
|---|---|---|---|---|---|---|---|---|
| Cả hai bên độc lập |35/10/35 |4.795/10.628 |5.833 |45,12% |43,75% |6.582/10.628 |61,93% |65,60% |
| Trái |21/5/14 |2.968/5.314 |2.346 |55,85% |52,50% |3.607/5.314 |67,88% |70,12% |
| Phải |14/5/21 |1.827/5.314 |3.487 |34,38% |35,00% |2.975/5.314 |55,98% |61,07% |

Trong run này mọi cặp allow đều available toàn clip, mọi cặp degrade/skip đều
unavailable toàn clip. Vì vậy macro numeric35/80=43,75%; không có nghĩa thuật
toán/gate đo được43,75% vận động viên hoặc đúng43,75%. Raw bilateral của audit
cũ1.301/5.314 dùng điều kiện cả hai bên đồng thời, khác raw independent-side
6.582/10.628 ở bảng này. Không dùng chúng thay cho nhau.

31.884 input-joint slots =10.628 observations ×3 joints; không phải31.884
frame độc lập:

| Bên | Processed observed | Imputed | Low visibility | Missing | Available-input observed |
|---|---|---|---|---|---|
| Trái |13.649 |7 |2.284 |2 |8.904 |
| Phải |12.659 |9 |3.274 |0 |5.481 |
| Tổng |26.308 |16 |5.558 |2 |14.385 |

Source provenance:26.308 observed,5.558 low_visibility,18 missing. Stage4
tạo16 imputed từ source missing;2 còn missing. Không đổi chúng thành observed.
Toàn bộ14.385 slots của4.795 available observations là observed;0 imputed
được dùng cho số đo available. Observed status cũng không chứng minh landmark
đúng vị trí giải phẫu; raw in-frame chỉ kiểm tra coordinate bounds của ảnh.

Unavailable reasons (multi-label; mẫu số5.833 unavailable observations):
feature_quality_blocked5.833; feature_provenance_rejected4.041;
feature_joint_unusable4.041; missing_joint1. Underlying whole-clip metric
reasons:35 skip pairs có low_pose_visibility;10 degrade pairs có
partial_joint_coverage. Không phải35/10 frame và không phải form errors.
Full per-pair reasons/joint provenance/flags ở summary.json và JSONL.


Định nghĩa thống kê:

- Source frames: đếm mỗi frame video một lần. Frame–bên: một frame cho mỗi bên;
  đây là mẫu số numeric availability, không gọi là số frames video độc lập.
- Raw coverage: cùng lúc hip/knee/ankle **của đúng bên**, observed, finite,
  trong ảnh. Không phải numeric availability hay calibrated confidence.
- Numeric availability: adapter trả available; unavailable vẫn value=null và
  giữ flags/reasons. Full-clip metric gate không được promote theo frame.
- Micro: tổng available chia tổng frame–bên. Macro: mỗi clip có trọng số như
  nhau; lấy trung bình tỷ lệ trái/phải trong clip rồi trung bình40 clip.
  Đây là trung bình **coverage**, không trung bình hai góc.
- By-side:40 clip/side, mỗi bên dùng tổng source frame count làm mẫu số.
- Input status/source_status đếm3 input-joint slots cho mỗi observation, kể cả
  unavailable/unusable; available_input_status chỉ đếm inputs của số đo thật.
- Reasons là multi-label per frame–side; một unavailable có thể nhiều reasons,
  nên tổng counts reasons có thể lớn hơn số unavailable.
- Lỗi xử lý không bị loại: giữ denominator, ghi processing_error và số
  unevaluated riêng, không giả chúng thành numeric unavailable đã tính.
  Coverage available/full denominator là lower bound khi còn unevaluated.

Từng clip, raw/numeric độc lập theo bên; mọi mẫu số là full clip.

| Clip | Frames | Sequence | L gate | L numeric | L raw | R gate | R numeric | R raw |
|---|---|---|---|---|---|---|---|---|
| v_BodyWeightSquats_g01_c01.avi |267 |skip |allow |267/267 |267/267 |skip |0/267 |8/267 |
| v_BodyWeightSquats_g01_c02.avi |252 |skip |allow |252/252 |252/252 |skip |0/252 |14/252 |
| v_BodyWeightSquats_g01_c03.avi |250 |skip |allow |250/250 |250/250 |skip |0/250 |35/250 |
| v_BodyWeightSquats_g04_c03.avi |88 |skip |skip |0/88 |8/88 |degrade |0/88 |66/88 |
| v_BodyWeightSquats_g04_c04.avi |63 |skip |skip |0/63 |0/63 |degrade |0/63 |50/63 |
| v_BodyWeightSquats_g05_c01.avi |71 |degrade |degrade |0/71 |62/71 |allow |71/71 |71/71 |
| v_BodyWeightSquats_g05_c02.avi |135 |degrade |degrade |0/135 |133/135 |allow |135/135 |135/135 |
| v_BodyWeightSquats_g05_c03.avi |54 |degrade |degrade |0/54 |51/54 |allow |54/54 |54/54 |
| v_BodyWeightSquats_g07_c01.avi |123 |skip |allow |123/123 |123/123 |skip |0/123 |61/123 |
| v_BodyWeightSquats_g07_c02.avi |103 |skip |allow |103/103 |103/103 |skip |0/103 |45/103 |
| v_BodyWeightSquats_g07_c03.avi |83 |skip |allow |83/83 |83/83 |skip |0/83 |45/83 |
| v_BodyWeightSquats_g07_c04.avi |54 |skip |allow |54/54 |54/54 |skip |0/54 |32/54 |
| v_BodyWeightSquats_g09_c05.avi |229 |skip |skip |0/229 |8/229 |degrade |0/229 |227/229 |
| v_BodyWeightSquats_g09_c06.avi |260 |skip |skip |0/260 |3/260 |allow |260/260 |260/260 |
| v_BodyWeightSquats_g09_c07.avi |233 |skip |skip |0/233 |6/233 |allow |233/233 |233/233 |
| v_BodyWeightSquats_g10_c03.avi |152 |skip |skip |0/152 |4/152 |allow |152/152 |152/152 |
| v_BodyWeightSquats_g10_c04.avi |148 |skip |skip |0/148 |4/148 |allow |148/148 |148/148 |
| v_BodyWeightSquats_g10_c05.avi |158 |skip |skip |0/158 |9/158 |allow |158/158 |158/158 |
| v_BodyWeightSquats_g12_c01.avi |237 |skip |allow |237/237 |237/237 |skip |0/237 |0/237 |
| v_BodyWeightSquats_g12_c02.avi |143 |skip |allow |143/143 |143/143 |skip |0/143 |0/143 |
| v_BodyWeightSquats_g12_c03.avi |120 |skip |allow |120/120 |120/120 |skip |0/120 |0/120 |
| v_BodyWeightSquats_g13_c01.avi |112 |skip |skip |0/112 |67/112 |allow |112/112 |112/112 |
| v_BodyWeightSquats_g13_c02.avi |62 |skip |skip |0/62 |24/62 |allow |62/62 |62/62 |
| v_BodyWeightSquats_g13_c03.avi |115 |skip |skip |0/115 |46/115 |allow |115/115 |115/115 |
| v_BodyWeightSquats_g13_c04.avi |75 |skip |skip |0/75 |21/75 |allow |75/75 |75/75 |
| v_BodyWeightSquats_g15_c01.avi |142 |skip |allow |142/142 |142/142 |skip |0/142 |35/142 |
| v_BodyWeightSquats_g15_c02.avi |122 |skip |allow |122/122 |122/122 |skip |0/122 |30/122 |
| v_BodyWeightSquats_g19_c03.avi |144 |skip |skip |0/144 |44/144 |allow |144/144 |144/144 |
| v_BodyWeightSquats_g19_c04.avi |108 |skip |skip |0/108 |28/108 |allow |108/108 |108/108 |
| v_BodyWeightSquats_g20_c05.avi |136 |skip |allow |136/136 |136/136 |skip |0/136 |26/136 |
| v_BodyWeightSquats_g20_c06.avi |167 |skip |allow |167/167 |167/167 |skip |0/167 |41/167 |
| v_BodyWeightSquats_g23_c03.avi |73 |degrade |degrade |0/73 |62/73 |degrade |0/73 |72/73 |
| v_BodyWeightSquats_g23_c04.avi |66 |degrade |degrade |0/66 |59/66 |degrade |0/66 |65/66 |
| v_BodyWeightSquats_g25_c01.avi |103 |skip |allow |103/103 |103/103 |skip |0/103 |44/103 |
| v_BodyWeightSquats_g25_c02.avi |110 |skip |allow |110/110 |110/110 |skip |0/110 |30/110 |
| v_BodyWeightSquats_g25_c03.avi |116 |skip |allow |116/116 |116/116 |skip |0/116 |49/116 |
| v_BodyWeightSquats_g25_c04.avi |108 |skip |allow |108/108 |108/108 |skip |0/108 |35/108 |
| v_BodyWeightSquats_g25_c05.avi |106 |skip |allow |106/106 |106/106 |skip |0/106 |54/106 |
| v_BodyWeightSquats_g25_c06.avi |101 |skip |allow |101/101 |101/101 |skip |0/101 |38/101 |
| v_BodyWeightSquats_g25_c07.avi |125 |skip |allow |125/125 |125/125 |skip |0/125 |46/125 |

## Hiệu năng thực tế

CLI elapsed **767.79s** (~12 phút48 giây), exit0.
Validation/linkage 3.99s;4-worker batch wall
727.36s; render3 demos 34.46s.
Internal total 765.83s trước bước hash manifest cuối;
CLI elapsed còn gồm final hash/process shutdown.

Factory calls5.314; adapter calls10.628.39 preprocess mới,1 prior preprocessing
reuse đã kiểm chứng (g01_c01, dùng chung hai bên). Tổng elapsed trên các workers
(chồng thời gian, không cộng vào wall): preprocessing16,84s; factory2.655,87s;
adapters126,98s; worker elapsed2.818,88s. Factory chiếm khoảng94,2% tổng worker
elapsed, không gọi đây là CPU-time hoặc benchmark độc lập.
Independent readback/hash/counters/decode:120,36s, exit0.


4 process workers xử lý các clip, không agent/detector mới. Factory vẫn full
source revalidation O(N) mỗi frame nên tổng O(sum N_clip²). Một preprocessing
mỗi clip/context mỗi frame dùng cho2 adapter là reuse tại workflow, không sửa
core hoặc bỏ provenance checks. Times có validation/hash/read/write và process
overhead; regression tests chạy song song một phần, không coi đây là benchmark
máy độc lập hoặc speedup đã được đối chứng.

Nếu cần tối ưu đáng kể, đề xuất riêng capability/context session có chứng cứ
immutability/hash binding và acceptance tương đương trước khi thay core. Không
triển khai capability mới hoặc giảm validation trong milestone này.

## Kiểm thử, integrity và QA

Final regression: **704 passed,0 failed,0 errors,0 skipped,4 deselected,
0 pytest warnings in134,98s** (CLI137,17s), exit0.23 tests mới +681 tests hiện
có; các tests integration ngoài scope bị deselect bằng `-m "not integration"`.
`tests-final.txt`, `tests-final.xml`, `tests-final-command.json` là final receipt.
Lệnh cuối dùng basetemp `tests-final-temp`, JUnit `tests-final.xml` trong run;
khác đường dẫn của lần related trước ở block command bên dưới.

Lịch sử giữ nguyên, không ghi đè: first34 pass/4 fail (Counter dùng `.value`
với source_status vốn là str); sửa workflow thành `str`, không sửa shared
schema/flag. Sau sửa39 pass/0 fail; related698 pass/4 deselected; localization
39 pass; sau thêm6 hash-linkage cases chạy final704 pass nêu trên. Tất cả
receipt stdout/XML/command được giữ. Không gộp các lần chạy thành số tests unique.
Không full project suite, không real-data MediaPipe integrations.


Tests thêm cho đủ80 pairs, whole-clip/fixed-side scope, micro/macro/side
denominators, failure/missing/partial results không bị bỏ, null/reasons,
factory dùng chung đúng frame cho2 adapters, preprocessing reuse có validation,
renderer saved-only/no geometry, dịch reason và escaping/local links.
Existing numerical/legacy/factory/preprocessing/quality/feedback tests được giữ.

Independent readback **PASSED**:452 manifest files hash/size,40 source
video byte hashes,80 pairs /10.628 typed observations,5.314 raw timelines;
factory/profile/report/raw/processed/input/observation hashes và per-side
counts/provenance/reasons đều khớp.35 local HTML links tồn tại trong bundle;
source cameraUNKNOWN, unverified side và disabled gates giữ nguyên.
Readback/renderer không gọi primitive geometry hoặc inference.

Ba MP4 mới đều H.264 (VideoWriter avc1 opened=true, decoder FOURCC h264),
1280×720/25FPS. Decode hết267/88/267 output frames (622 tổng), cùng frame map
0..N−1 và timestamp i/25; decode cả original copies. Native libopenh264 log
3 backend initialization errors, nhưng VideoWriter mở và output decode thành
công; không fallback mp4v, không tải/cài codec. Đây là native encoder diagnostics,
khác0 pytest warnings; xem batch-stderr.txt và video-verification.json.

Đã thực sự xem22 PNG bằng tools.view_image:11 overlays +11 original samples.
Allow indices0/133/266; degrade0/1/44/87; skip0/39/133/266. Samples gồm
đầu–giữa–cuối và first selected-raw-valid/invalid; timestamps lần lượt index/25.
Allow trái267/267 available; degrade phải0/88 (raw66/88), skip phải0/267
(raw8/267). Skip frame39 dù raw observed/in-frame vẫn unavailable vì gate
toàn clip. Các nhãn Việt, fixed side, time/frame, nguồn raw/processed, temporal
unavailable và official disabled đọc được, không thấy text clipping trên samples.

Source limitation quan sát được: g04_c03 có letterboxing/cắt phần chân;
raw skeleton frame0/1 lệch rõ cơ thể, một số điểm ở dải đen. Giữ nguyên raw,
không xóa/cắt/chọn lại hoặc dùng observed/in-frame làm ground truth. Đây là
minh họa vì sao raw coverage không đo accuracy; không nhận xét kỹ thuật người tập.
Receipt **visual-qa-review.json** ghi đủ22 ảnh/hashes/nhận xét; capture receipt
visual-qa-samples.json và integrity.json giữ nguyên trạng thái trước khi xem,
review mới ghi nhận bước inspection đã hoàn tất, không sửa receipt cũ.

Baseline664 files:660 giữ nguyên,4 thay đổi đúng phạm vi (renderer opt-in,
README/state/tasks),0 mất file.112 allowed source videos và394 old output files
giữ hash, core policies/profiles/models/domain code giữ nguyên. Chi tiết tại
preservation.json; inventory mới bao gồm artifacts/code/docs hashes.


Browser/native-player/continuous playback QA **NOT_RUN**. Chỉ gọi sampled-frame
visual QA sau khi thực sự xem decoded PNG bằng view_image. Không dựng server,
mở network hoặc thử cách đi vòng local-file policy. Decode/metadata tests không
được gọi là browser/visual/athlete validation.

Các lệnh thực thi và receipts trong run mới:

```powershell
& .venv/Scripts/python.exe -u -B -m gym_coach.reporting.real_batch_coverage --audit-run outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf --prior-demo outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9 --output-dir outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/batch --workers 4

& .venv/Scripts/python.exe -B -m pytest tests/test_real_batch_coverage.py tests/test_real_descriptive_demo.py tests/test_geometry_features.py tests/test_feature_context.py tests/test_feature_observation.py tests/test_right_knee_feature.py tests/test_preprocessing.py tests/test_quality.py tests/test_squat_feedback.py tests/test_showcase_handoff.py -m "not integration" -q --tb=short --basetemp outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/tests-related-temp --junitxml=outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/tests-related.xml

& .venv/Scripts/python.exe -B outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/verify_batch.py
```

Khi chạy lại chọn output/basetemp/JUnit path mới, không ghi đè receipt cũ.
Không full suite/other-source integration hoặc claim clean-environment support.
Không thêm dependency; dùng Python3.11.9, Pydantic2.13.5, pytest9.1.1,
OpenCV5.0.0.93, Pillow12.3.0, NumPy2.4.6/SciPy1.17.1 hiện có.

## Thay đổi, giới hạn và kết luận

Thêm reporting/real_batch_coverage.py, tests/test_real_batch_coverage.py và
report này; renderer hiện có chỉ thêm tùy chọn localization mặc định tắt.
README/PROJECT_STATE/TASKS cập nhật tối thiểu. No core policy/profile/schema/
model changes, không xóa/move Data/archive/old outputs. Baseline/preservation
và SHA inventory mới ghi rõ mọi file thay đổi/source đã giữ nguyên.

**Milestone coverage +3 representative demos IMPLEMENTED / EXPLORATORY.**
Sẵn sàng dùng bảng40 clips/80 pairs, saved per-frame results và demo để mô tả
khả năng sẵn có của số đo, cùng các trường hợp bị chặn. Không sẵn sàng cho
accuracy/form/temporal/official claims. G5.7 vẫn blocked (không human documented
viewpoint,0 bilateral allow và0 sequence allow); G6/toàn G5 IN_PROGRESS.

Bước phát triển nhỏ tiếp theo được đề xuất: thiết kế protocol annotation tham
chiếu2D độc lập cho một tập nhỏ có sampling khóa trước, bao gồm ví dụ raw
observed nhưng landmark lệch; ghi vị trí joint, khả năng thấy joint và sai số
tham chiếu trước khi xem xét accuracy. Không dùng coverage/visibility thay
reference, không đổi gate hoặc triển khai detector trong milestone này.


Nguồn là development subset đã chọn theo visual audit của agent, các clip có
thể cùng người/cảnh gXX. Không holdout, không40 subject độc lập. Side chưa có
human protocol xác nhận;2D projection/occlusion và identity limitations của
single-subject backend vẫn tồn tại. Visibility/coverage không là accuracy hay
calibrated uncertainty. Không human keypoint/reference angle/form ground truth;
không claim model đúng, đo góc vật lý chính xác hoặc hệ thống chấm form đạt.
Official5.7 vẫn cần documented linkage và fresh sequence/bilateral allow;
G6 real acceptance chưa mở. Không rep/phase/point-turn/pause/score/tư vấn.
