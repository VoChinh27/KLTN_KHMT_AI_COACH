# Giai đoạn 5.7 — Documented side-view input và fresh QualityGate

Ngày kiểm tra: **2026-10-10, Asia/Saigon**. Trạng thái:
**BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**. **GĐ5.7 chưa COMPLETE**;
toàn GĐ5 **IN_PROGRESS**. Chưa có kết quả QualityGate mới.

## Kiểm tra input thực tế

Đã đọc README, PROJECT_STATE, TASKS, IMPLEMENTATION_PLAN, SCHEMA_CONVENTIONS,
reports GĐ5.5/5.6 và metadata dataset hiện hữu. Workspace không có Git repository:
`git status --short` báo fatal/exit 128; không tạo commit/diff giả.

Tìm video theo extensions avi/mp4/mov/mkv/webm/mpg/mpeg/m4v/wmv/flv/3gp/mts/
m2ts/ts/ogv, gồm hidden files, ngoài _archive/.venv/cache/.git. Kết quả mới đo
bằng file enumeration và byte SHA-256, không decoder/visual review/inference:

| Kiểm tra | Kết quả hiện tại |
|---|---:|
| Video paths toàn workspace trong scope | 14,291 |
| Data/UCF-101 / Data/UCF-101_GYM | 13,320 / 971 |
| Video paths ngoài Data | 0 |
| Squat paths / byte-distinct SHA-256 | 224 / 112 |
| Squat paths mới / thiếu / thay đổi bytes so với before-manifest GĐ5.6 | 0 / 0 / 0 |
| Metadata non-video trong Data / giữ nguyên byte SHA-256 | 7 / 7 |
| Squat CSV rows / camera_view UNKNOWN | 112 / 112 |
| Documented side-view input tìm được | 0 |

224 squat paths chỉ đối chiếu inventory, không xác nhận exercise/viewpoint/form
qua nội dung video. Không tự chọn một file UCF101 hoặc gán annotation cho bản sao.
Không suy viewpoint từ tên file, nhãn, pose/coverage hoặc heuristic ảnh.

Nguồn Data/UCF-101_GYM gồm DATASET_STATUS.md và audit/UCF101_GYM_AUDIT_CONFIG.md,
UCF101_GYM_AUDIT_SUMMARY.md, UCF101_GYM_VIDEO_AUDIT.csv,
UCF101_GYM_PROBLEM_VIDEOS.csv, UCF101_GYM_CLASS_SUMMARY.csv, summary_data.json.
Audit config nói rõ camera_view/occlusion UNKNOWN; CSV/aggregates không cung cấp
human side-view protocol. Reports pipeline và side metadata của synthetic fixture
không thay human annotation. Visual review lượt này: **not_performed**.

## Identity, artifacts và trạng thái gate

Không có video được tiếp nhận: **selected video path/ID/hash = null**; nguồn
annotation/protocol, người/ngày xác nhận và interval **chưa cung cấp**.
FPS/resolution/frame count/timestamp policy input mới chưa đo (**NOT_RUN**),
không lấy metadata lịch sử của video khác để điền.

| Scope | Eligibility / action / reasons |
|---|---|
| Sequence QualityGate | NOT_RUN; không có QualityReport mới |
| squat_knee_angle_left_deg | NOT_RUN; không có metric decision mới |
| squat_knee_angle_right_deg | NOT_RUN; không có metric decision mới |

Blocker là thiếu video mới và documented side annotation gắn đúng path/ID/byte
SHA-256. Đây là input gate, không là skip/degrade từ QualityGate hoặc lỗi kỹ thuật
người tập. Không dựng StructuredFlag/report giả hoặc tái sử dụng report cũ làm run mới.

Pipeline/model/config identities của run mới: **NOT_RUN**. Metadata package cục bộ
đọc qua importlib.metadata, không import backend: ai-coach-gym **0.5.4**,
MediaPipe **0.10.14**, Pydantic **2.13.5**, pytest **9.1.1**, CPython **3.11.9**,
Windows x64 10.0.26200. Không install/model/dependency, profile/threshold/registry/
schema changes. Không model initialization, raw pose, decoder metadata hoặc
QualityReport mới; pipeline command/output hashes **NOT_RUN**.

Artifact input check mới:
[stage5_7_input_check.json](artifacts/stage5_7_input_check.json), byte SHA-256
`e88626e5dcce70d7b00e434be210d6ad6863f2b4f3889479317be1bde0ac6886`.
JSON lưu scope, environment metadata, 224 hashes squat, 7 hashes metadata,
baseline identity và NOT_RUN states; **không phải pipeline manifest/result**.
Baseline so sánh:
`outputs/stage5_6/verification-67c29d1d3fff450e9ebf7822c598594c/before_manifest.json`,
byte SHA-256 `ef5755228922b510694c833ea1e5de61474af49387134b495de15899c1593f98`.
Hashes xác định bytes, không xác nhận viewpoint hoặc producer.

## Kiểm chứng ở lượt này

Các lệnh read-only đã chạy từ project root:

```powershell
git status --short
rg --files --hidden -g '!_archive/**' -g '!.venv/**' -g '!**/__pycache__/**' -g '!.pytest_cache/**' -g '!outputs/**' -g '!Data/**'
rg -l -i 'annotation|protocol|documented|side.view|camera_view|annotator|viewpoint' docs Data .agent configs scripts --glob '!*.avi' --glob '!*.mp4' --glob '!*.mov' --glob '!**/__pycache__/**'
Get-FileHash -LiteralPath docs/artifacts/stage5_7_input_check.json -Algorithm SHA256
```

Inline Python qua PowerShell here-string và `.venv/Scripts/python.exe -B -`
dùng os.walk/pathlib/hashlib/json/csv/importlib.metadata: enumerate files, hash
bytes theo blocks 1 MiB, so before-manifest, đếm camera_view CSV. Checks read-only
exit 0; Git exit 128 do không có repository. Không video/pose/quality/numeric API.

**Tests liên quan, integration và full suite: NOT_RUN.** Pass/fail/error/skipped/
warnings test run GĐ5.7 là **N/A**, không ghi 0 hoặc PASS cho tests chưa chạy.
Nhánh thiếu input yêu cầu dừng trước inference; full suite có MediaPipe inference
integrations. Không thêm regression code trước khi input gate được đáp ứng.
713 passed/4 warnings của GĐ5.6 là bằng chứng lượt trước, không test result GĐ5.7.

Chỉ thêm báo cáo/input-check JSON mới và cập nhật README/IMPLEMENTATION_PLAN/
PROJECT_STATE/TASKS. Không sửa code/tests/config/schema, dataset, archive hoặc
reports/artifacts cũ; không tạo thư mục inference run.

Final read-only preservation check: **870 existing files** trong scope audit,
**866 byte-unchanged**, đúng **4 current docs changed**, **0 missing/deleted**;
chỉ thêm report này và input-check JSON. Scope gồm non-Data files ngoài các
directories đã loại trừ, 224 squat videos và 7 Data metadata; không hash video
các action khác hoặc archive/venv/cache. JSON parse/counts và report→artifact
byte-hash reference khớp. Đây là kiểm tra file/audit, không pytest test run.

## Người dùng cần cung cấp

1. **Video squat mới**, source path trong workspace và video ID ổn định;
   exercise xác nhận **BodyWeightSquats**. Giữ nguyên file nguồn.
2. **Annotation/protocol viewpoint=side** liên kết đúng relative path, video ID
   và **byte SHA-256 của chính video**; protocol ID/version hoặc tài liệu nguồn,
   tên/ID người xác nhận và ngày xác nhận.
3. **Phạm vi**: ghi rõ toàn video, hoặc frame interval 0-based và timestamps
   giây tương ứng. Interval/hash không khớp tiếp tục blocked.
4. **Camera/framing**: hướng nhìn, orientation, joints trong khung và giới hạn
   occlusion/in-frame. Đây là ghi nhận input, không đánh giá form.
5. **FPS/resolution/timestamp policy nếu đã biết** từ metadata video; phân biệt
   với human annotation. Trường chưa biết sẽ được decoder đo/policy ghi rõ ở run
   sau, không tự suy hoặc điền số mặc định.

Không tạo annotation thay người dùng. Visual impression thiếu protocol chỉ là
visually_assessed_unverified, không documented side-view.

## Acceptance còn thiếu và bước tiếp theo

| Điều kiện GĐ5.7 | Trạng thái |
|---|---|
| Verify video mới + annotation/path/ID/hash/interval | BLOCKED — chưa input |
| Fresh video ingestion → pose → quality | NOT_RUN |
| Canonical left/right joints và side policies | PENDING — lấy từ registry khi có input |
| Sequence và đúng hai metrics allow, reasons đầy đủ | PENDING — chưa decisions |
| New-run raw/model/config/profile/command/output hashes | PENDING |
| Regression/integration và full suite | NOT_RUN |
| Báo cáo blocked và trạng thái | Đã ghi nhận; không hoàn tất GĐ5.7 |

Bước nhỏ nhất: người dùng cung cấp video và annotation theo checklist. Sau verify
mới tạo run riêng với canonical `squat_knee_angle_left_deg` (left_hip, left_knee,
left_ankle) và `squat_knee_angle_right_deg` (right_hip, right_knee, right_ankle),
side policies, giữ quality-s3-v1. Chỉ đánh giá gate; không preprocessing/numeric
adapters hoặc báo góc ở GĐ5.7. Sequence hoặc một metric skip/degrade, sai key/
joints/viewpoint/hash/interval vẫn blocked/unavailable. Cả hai allow chỉ đủ điều
kiện bước measurement sau, không xác nhận accuracy/form. Hạn chế 2D/occlusion và
ngưỡng S3 chưa validation còn nguyên; GĐ6–7 chưa bắt đầu, assessment blocked.
