# Rà soát UCF-101_GYM cho gate chính thức5.7 — 2026-10-15

**Kết luận: có5 ứng viên trực quan để kiểm tra tiếp, nhưng5.7 vẫn
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT. Toàn GĐ5 IN_PROGRESS.**

## Phạm vi nguồn

Chỉ video và metadata trong `Data/UCF-101_GYM`. Không tìm dữ liệu bên ngoài,
không dùng bản sao `Data/UCF-101`, không dùng old pose/QualityReports/exploratory
measurements làm data evidence. Specs/roadmap/reports trước chỉ dùng xác định
acceptance. Workspace không Git repository; không tạo commit/diff. Không đổi
code/policy/threshold, video, source audit CSV hoặc annotation.

Nguồn metadata đã kiểm tra toàn bộ7 files hiện có: DATASET_STATUS.md;
audit/UCF101_GYM_AUDIT_CONFIG.md, UCF101_GYM_AUDIT_SUMMARY.md,
UCF101_GYM_VIDEO_AUDIT.csv, UCF101_GYM_PROBLEM_VIDEOS.csv,
UCF101_GYM_CLASS_SUMMARY.csv và summary_data.json. CSV chính971 rows,112 squat
rows; `camera_view`, `person_visibility`, `occlusion`, `framing` và
`person_count_category` đều UNKNOWN ở112 squat rows. Config nói rõ automated
heuristics không đánh giá những trường này. GOOD/USABLE là heuristic blur/lighting,
không camera/pose ground truth. Không tìm được protocol/confirmer/date/interval
side-view có linkage path/ID/byte SHA trong các nguồn này.

## Quan sát trực tiếp

Subset có971 video; rà soát trực quan **112 squat clips/112 byte-distinct hashes**.
Từ chính các video này, decode5 frames/clip:
`floor((N-1)*[0,.25,.5,.75,1])`, tổng560 images trên28 contact sheets; đã xem cả28.
Chọn5 shortlist bằng profile/framing/occlusion quan sát được, rồi xem thêm16
frames/clip trải đều từ0 đếnN-1 (80 images trên5 sheets). Có sample indices,
timestamps, input/image hashes và nhận xét cả112 clips trong artifact mới.
Frame sampling không chứng minh toàn clip hoặc calibrated camera angle.

Mọi nhận xét mới: **visually_assessed_unverified; documented=false**. Reviewer là
agent, không phải annotator chính thức được dataset/protocol xác nhận. Confidence
high chỉ cho việc nhận dạng visual category trên mẫu đã xem, không xác suất đã
hiệu chuẩn hoặc độ tin cậy scientific. Không tạo ground truth từ review này.

Tất cả tên dưới đây ở `Data/UCF-101_GYM/BodyWeightSquats/`. Frames0-based;
timestamps=index/FPS với giả định CFR, không claim timestamps VFR. Decoder báo
25FPS,320x240 cho5 ứng viên. Hình cho thấy vùng hip/knee/ankle trong frame nhưng
**không chứng minh hai knee joints luôn tách biệt/usable**.

| File chính xác | Evidence frame indices (trong16 samples) | Lý do chọn qua hình ảnh | Giới hạn |
|---|---|---|---|
| v_BodyWeightSquats_g07_c01.avi | 0, 48, 65, 89, 105, 122 | Head, hips, knees and shoes stay in frame, camera fixed, body in near profile through lowering and rising. | Far knee and ankle overlap the nearer leg, especially frames48-105; cannot certify two independently visible knee joints throughout clip. |
| v_BodyWeightSquats_g12_c02.avi | 0, 18, 28, 37, 56, 94, 104, 113, 142 | Stable left-facing profile, full-body margin and uncluttered foreground in both lowering/rising samples. | Near/far knees and shoes project close together; far knee not continuously separable, shorts hide hip/knee landmarks. Poster is background, not another observed subject. |
| v_BodyWeightSquats_g15_c01.avi | 0, 37, 47, 56, 94, 112, 122, 141 | Stable lateral-looking camera, whole body visible through lower and upper positions; pillar stays behind subject. | Far leg is occluded/overlapped around frames47-56 and112-122; small 320x240 image; both knee observations not established independently. |
| v_BodyWeightSquats_g23_c03.avi | 0, 14, 19, 24, 38, 48, 52, 57, 72 | Right-facing near profile and head-to-feet framing retained through two apparent lowering/rising stretches in sampled images. | Far knee/shin overlaps; hands pass close to knee area at low positions and movement blur visible. No bilateral visibility guarantee. |
| v_BodyWeightSquats_g25_c01.avi | 0, 27, 40, 47, 54, 61, 68, 81, 102 | Plain background, fixed framing, stable left-facing profile; head, hip/knee region and shoes remain within bounds. | Far knee is largely hidden behind nearer leg at middle/low positions (e.g.frames47-68); full-body framing does not prove bilateral joint usability. |

S027123 frames; S052143; S063142; S09973; S106103. Các mẫu gồm standing,
lowering, low position và rising; đó là mô tả hình ảnh, không phase labels đã
validation. Với cả5 clips, đầu gối phía xa có chồng/che khuất theo hình chiếu,
đặc biệt khi hạ thấp; chưa clip nào được xác nhận continuous bilateral knee
visibility. Không suy low visibility hoặc QualityGate action từ quan sát mắt.
Các clip side-looking khác không mặc nhiên bị coi là không dùng được: chúng
không vào shortlist do crop, nhỏ/tối, chồng chân hoặc camera đổi hướng; lý do
sơ bộ lưu trong visual_audit.json. Đây không phải negative ground-truth labels.

| File | Byte SHA-256 |
|---|---|
| v_BodyWeightSquats_g07_c01.avi | `4c10e8d1dff20a18ba07c68caf77f0c893f457a422e5298f5804a601fe4f5a1b` |
| v_BodyWeightSquats_g12_c02.avi | `042c9a0caf5b0061f55c1d936d146cb7460b2cece6917780766cc90ce6e68b5e` |
| v_BodyWeightSquats_g15_c01.avi | `db3cc50be948c2a99375f95190a96689fedace1a1df8ffe0816b0ba50204ed95` |
| v_BodyWeightSquats_g23_c03.avi | `510ea1dc263e88f0a498f1e739628accdb1d0031ea49c69b0dae37c2bb6a54d0` |
| v_BodyWeightSquats_g25_c01.avi | `893f7b0c89d10030705eeadfca1f9681a81190540bff1b2ab77d0ab3b614261e` |

Hash chỉ nhận diện bytes, không chứng minh camera/viewpoint/annotator authority.
Không sửa nguồn hoặc gán official side metadata cho những file này.

## Đối chiếu acceptance5.7

| Điều kiện | Kết quả hiện tại |
|---|---|
| Exercise/scope phù hợp |5 candidate files trong BodyWeightSquats; không suy form từ class |
| Góc quay gần ngang/full framing qua nhiều mẫu | Có bằng chứng visual ở5 shortlist, vẫn unverified |
| Hai đầu gối quan sát độc lập qua interval cần thiết | Chưa xác nhận; far-side overlap/occlusion |
| Documented side với protocol/source, confirmer/date, interval, path/ID/hash | Thiếu trong7 metadata sources; agent review không thay thế |
| Fresh official sequence eligible=true/action=allow | NOT_RUN, chưa input chính thức |
| Exact left hip–knee–ankle metric allow | NOT_RUN |
| Exact right hip–knee–ankle metric allow | NOT_RUN |
| Official input gate5.7 | BLOCKED, không COMPLETE |

Sau documented linkage mới được chạy ingestion→pose→quality fresh, giữ
quality-s3-v1 và exact canonical IDs: `squat_knee_angle_left_deg` với
left_hip/left_knee/left_ankle; `squat_knee_angle_right_deg` với
right_hip/right_knee/right_ankle. **Sequence và cả hai metric phải eligible=true/
action=allow**, không key/joint/viewpoint/hash/interval mismatch. Any degrade/skip
vẫn blocked. Không thay gate5.7 bằng feature-local numeric policy v2 hoặc old
exploratory result. Allow cũng chưa chứng minh feature accuracy/form.

Không khởi tạo MediaPipe, không tạo raw pose/QualityReport mới hoặc số đo góc
real-video trong lượt này. Test-only QualityGate/synthetic fixtures được phân
biệt với official video gate trong phần kiểm thử dưới đây.

## Artifacts, thay đổi và bảo toàn

Run mới: `outputs/stage5_7/subset-visual-20261015-077a183104c94d28a301b4d538b69de8`; không overwrite old outputs.
`source_before.json` hashes112 source videos +7 metadata; sample_inventory.json,
metadata_review.json; visual_audit.json (112 reviews), shortlist.json (5 dense
reviews), samples/sheets/dense_samples và dense contact sheets. Inventory hashes
cuối lượt ở artifact_inventory.json; kiểm tra không đổi source và code ở
preservation_audit.json. Sử dụng OpenCV/Pillow hiện có chỉ để decode/lập sheets,
không quality heuristic, viewpoint classifier hoặc thêm dependency.

Chỉ cập nhật docs/PROJECT_STATE.md, docs/TASKS.md và tạo report này ngoài artifact
run mới. Các reports/specs/plan/code/tests/config cũ không sửa. Không xóa video,
metadata hoặc archive. Full suite không được chọn vì real integrations khác dùng
nguồn ngoài UCF-101_GYM; chỉ chạy tests liên quan bằng synthetic fixtures.

Bước tiếp theo nhỏ nhất: xác minh documented side-view/interval linkage cho một
ứng viên hiện có và bilateral knee visibility; sau đó fresh exact bilateral gate.
Nếu không có record xác nhận hợp lệ trong nguồn cho phép thì giữ blocked, không
bổ sung dữ liệu ngoài hoặc tự biến review này thành protocol/ground truth.

## Kiểm thử thực tế ngày2026-10-15

```powershell
& .venv/Scripts/python.exe -B -m pytest -q tests/test_quality.py tests/test_development_eligibility.py --junitxml=outputs/stage5_7/subset-visual-20261015-077a183104c94d28a301b4d538b69de8/related-tests.xml
```

Exit0: **49 passed in8.95s**,0 failed,0 errors,0 skipped,0 warnings; JUnit
elapsed8.947s. Existing tests kiểm tra actual synthetic sequence+bilateral allow,
wrong/missing key/joints/viewpoint, skip/degrade/imputation không được promote và
snapshot immutability. Không sửa tests/code. Đây là regression contract evidence,
không phải official5.7 real-video pass hoặc feature accuracy. Full suite và official
video integration **NOT_RUN**, counts N/A; không dùng956 pass của lượt trước làm
kết quả lần này. Môi trường và exact command/counts lưu test_evidence.json.
