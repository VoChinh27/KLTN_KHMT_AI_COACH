# PROMPT TỔNG ĐIỀU PHỐI — XÂY DỰNG LẠI AI COACH GYM

## 1. Cách sử dụng

Sử dụng **Prompt tổng** ở Mục 2 để bắt đầu hoặc tiếp tục toàn bộ project. Khi muốn thực hiện riêng một giai đoạn, nối Prompt tổng với prompt giai đoạn tương ứng tại Mục 3.

Mỗi lần làm việc, AI phải đọc trước:

1. `README.md`
2. `docs/GYM_TECHNICAL_GROUND_TRUTH_SPEC.md`
3. Tài liệu trạng thái/kế hoạch mới nhất nếu đã được tạo

Không sử dụng `_archive/` như mã nguồn đang hoạt động. Chỉ được mở một file trong archive khi nhiệm vụ hiện tại thật sự cần đối chiếu, và phải kiểm chứng lại trước khi tái sử dụng ý tưởng hoặc code.

---

## 2. Prompt tổng

```text
Bạn đang hỗ trợ xây dựng lại từ đầu đồ án “AI Personal Sports Coach — Gym Movement Analysis” trong workspace hiện tại.

NGUỒN YÊU CẦU CHÍNH
- Đọc toàn bộ README.md.
- Đọc toàn bộ docs/GYM_TECHNICAL_GROUND_TRUTH_SPEC.md.
- Nếu đã tồn tại PROJECT_STATE.md, IMPLEMENTATION_PLAN.md hoặc TASKS.md mới, đọc chúng trước khi hành động.
- Tài liệu trong _archive/ chỉ là lịch sử. Không lấy code cũ làm nền mặc định và không để kết quả cũ chi phối thiết kế mới nếu chưa được kiểm chứng.

PHẠM VI MVP
- Bài tập: BodyWeightSquats và HandstandPushups.
- Pose: video đơn camera, MediaPipe Pose 2D, không calibration.
- Bốn lớp chức năng phải tách biệt:
  1. kiểm tra chất lượng video/pose;
  2. trích xuất feature mô tả;
  3. đếm repetition và phân đoạn phase;
  4. đánh giá kỹ thuật, scoring và feedback.
- Trong giai đoạn chưa có ground truth chuyên gia, chỉ triển khai ba lớp đầu. Không tạo score đúng/sai giả định.

NGUYÊN TẮC KHOA HỌC
- Không dùng nhãn hành động UCF101 làm ground truth chất lượng kỹ thuật.
- Không gọi ngưỡng engineering là tiêu chuẩn chuyên môn.
- Mọi feature phải ghi rõ: công thức, input joints, đơn vị, source tier, measurement tier, viewpoint, quality gate và mục đích sử dụng.
- Mọi khoảng cách từ video 2D phải là relative/normalized; không báo cáo cm nếu không calibration.
- Tách technical_error, measurement_error, insufficient_data và segmentation_error.
- Nếu chưa có nguồn hoặc chuyên gia xác nhận, đánh dấu TBD-EXPERT và không đưa vào scoring.
- Không được tuyên bố thuật toán tốt chỉ vì chạy không lỗi. Phải tách execution success, coverage và accuracy.

NGUYÊN TẮC THIẾT KẾ
- Thiết kế mới, tối giản, module hóa và có type/schema rõ ràng.
- Không hard-code đường dẫn máy cá nhân.
- Cấu hình phải version được; ngưỡng không rải rác trong code.
- Xử lý FPS bằng thời gian hoặc chuyển đổi nhất quán từ giây sang frame.
- Không nội suy xuyên qua gap dài; không ghép một rep qua gap chưa giải quyết.
- Phải theo dõi provenance theo joint/frame: observed, low-visibility, imputed hoặc missing.
- Normalization fallback và side switching phải được ghi flag rõ ràng.
- Output phải audit được: input metadata, config version, quality flags, signal, event, phase và lý do không thể kết luận.

QUY TRÌNH BẮT BUỘC CHO MỖI NHIỆM VỤ
1. Kiểm tra trạng thái workspace và tài liệu yêu cầu liên quan.
2. Xác định chính xác mục tiêu, đầu vào, đầu ra và Definition of Done của nhiệm vụ hiện tại.
3. Nêu các giả định cần thiết; không tự mở rộng phạm vi sang scoring/form assessment.
4. Triển khai thay đổi nhỏ nhất nhưng hoàn chỉnh.
5. Viết hoặc cập nhật test cùng lúc với code.
6. Chạy test phù hợp và kiểm tra output thực tế.
7. Cập nhật PROJECT_STATE.md và TASKS.md bằng bằng chứng, không bằng tuyên bố chung chung.
8. Báo cáo: đã làm gì, file nào thay đổi, test nào chạy, kết quả nào đo được, hạn chế nào còn lại và bước tiếp theo hợp lý.

QUY TẮC DỪNG
- Dừng và yêu cầu quyết định nếu cần thay đổi phạm vi MVP, pose model hoặc nguồn ground truth.
- Dừng trước khi xóa dữ liệu, annotation hoặc kết quả nghiên cứu chưa có bản phục hồi.
- Dừng form scoring nếu thiếu nguồn chuyên môn/expert approval.
- Không tiếp tục sang giai đoạn sau nếu Definition of Done của giai đoạn hiện tại chưa đạt.

YÊU CẦU CHO LƯỢT LÀM VIỆC NÀY
[Điền nhiệm vụ cụ thể tại đây]

ĐẦU RA MONG MUỐN
- Hoàn thành trực tiếp trong workspace nếu nhiệm vụ cho phép.
- Giữ project chạy được và test được.
- Trình bày ngắn gọn bằng tiếng Việt, dẫn link đến các file chính.
```

---

## 3. Prompt theo giai đoạn

### Giai đoạn 0 — Khởi tạo kiến trúc và kế hoạch

```text
Áp dụng Prompt tổng.

Nhiệm vụ: khởi tạo nền tảng project mới, chưa triển khai thuật toán pose hoặc segmentation.

Hãy:
1. Kiểm tra workspace sạch hiện tại.
2. Đề xuất kiến trúc module tối giản cho video metadata, pose extraction, quality, preprocessing, feature, segmentation, assessment và reporting.
3. Tạo cấu trúc thư mục mới chỉ cho các module đã được phê duyệt.
4. Tạo pyproject/config dependency tối thiểu, không sao chép môi trường cũ một cách mù quáng.
5. Tạo schema/versioning conventions và quy ước lỗi/flag chung.
6. Tạo IMPLEMENTATION_PLAN.md, PROJECT_STATE.md và TASKS.md mới.
7. Tạo smoke test để xác nhận package import và cấu hình hoạt động.

Definition of Done:
- Cấu trúc mới không phụ thuộc code trong _archive/.
- Có dependency manifest, package import được và smoke test đạt.
- Mỗi giai đoạn tiếp theo có input, output, test và tiêu chí hoàn thành rõ ràng.
- Không có scoring hoặc ngưỡng kỹ thuật chưa được xác nhận.
```

### Giai đoạn 1 — Video ingestion và metadata

```text
Áp dụng Prompt tổng.

Nhiệm vụ: xây dựng module đọc video và metadata đáng tin cậy.

Hãy triển khai:
- schema VideoMetadata;
- đọc video theo iterator, không tải toàn bộ video vào RAM mặc định;
- frame_index và timestamp nhất quán;
- kiểm tra FPS, frame count, resolution và file unreadable;
- flag video_too_short, invalid_fps, unreadable_video;
- unit test bằng dữ liệu synthetic hoặc fixture nhỏ;
- CLI kiểm tra một video và xuất JSON metadata.

Không triển khai pose ở giai đoạn này.

Definition of Done:
- Metadata và timestamp được test.
- Lỗi đầu vào trả structured error, không crash mơ hồ.
- CLI không hard-code đường dẫn.
```

### Giai đoạn 2 — Pose extraction và provenance

```text
Áp dụng Prompt tổng.

Nhiệm vụ: tích hợp MediaPipe Pose 2D và lưu provenance đầy đủ.

Hãy triển khai:
- PoseFrame và JointObservation;
- x, y, visibility/presence nếu model cung cấp;
- trạng thái observed, low_visibility hoặc missing theo từng joint/frame;
- mapping joint được version hóa;
- lựa chọn một subject hoặc flag multi_person theo khả năng model;
- output không mất timestamp/frame_index;
- unit test mapping/schema và integration test trên một video development được chọn rõ ràng.

Không nội suy, normalize hoặc đánh giá kỹ thuật trong giai đoạn này.

Definition of Done:
- Có tỷ lệ pose coverage theo joint và video.
- Không biến low-visibility thành observed.
- Output pose có thể lưu/đọc lại mà không mất provenance.
```

### Giai đoạn 3 — Quality gate

```text
Áp dụng Prompt tổng.

Nhiệm vụ: xây quality gate độc lập trước preprocessing.

Hãy triển khai:
- coverage và consecutive-missing theo joint;
- flags cho excessive_missingness, low_pose_visibility, out_of_frame, viewpoint_unknown và insufficient_sequence_length;
- rule cấu hình riêng cho từng exercise khi cần;
- quyết định metric nào được phép chạy khi thiếu một bên;
- structured QualityReport;
- test cho chuỗi rỗng, thiếu đầu/cuối, gap ngắn/dài và một bên cơ thể.

Ngưỡng quality là engineering threshold và phải được ghi source tier S3.

Definition of Done:
- Quality gate không đưa ra kết luận kỹ thuật.
- Mỗi quyết định skip/degrade đều có reason code.
- Test edge case đạt.
```

### Giai đoạn 4 — Preprocessing

```text
Áp dụng Prompt tổng.

Nhiệm vụ: xây preprocessing giữ nguyên provenance và ranh giới gap.

Hãy triển khai:
- nội suy tuyến tính chỉ cho gap ngắn cấu hình theo thời gian/frame;
- imputation mask theo từng joint, không chỉ theo frame;
- normalization có trạng thái success/fallback;
- smoothing trên từng đoạn hữu hạn liên tục;
- cấu hình FPS-aware;
- cấm smoothing/nội suy xuyên gap dài;
- test không mutate input, không làm mất timestamp và không nối hai đoạn dữ liệu.

Definition of Done:
- Mỗi joint output biết nó observed hay imputed.
- Fallback normalization được flag.
- Không có rep/feature logic trong preprocessing.
```

### Giai đoạn 5 — Geometry và feature extraction

```text
Áp dụng Prompt tổng.

Nhiệm vụ: triển khai feature mô tả cho BodyWeightSquats và HandstandPushups.

Ưu tiên feature đã nêu trong đặc tả:
- Squat: knee angles, hip angles, torso angle; chưa bật knee-to-toe scoring.
- HSPU: elbow angles, shoulder angles, body-line candidate.

Mỗi feature phải khai báo metadata:
- công thức;
- required joints;
- unit;
- source tier;
- measurement tier;
- required viewpoint;
- quality requirements;
- role: segmentation, informational hoặc assessment-disabled.

Hãy viết test hình học synthetic cho góc chuẩn, điểm trùng, dữ liệu non-finite, thiếu joint và đối xứng trái/phải.

Definition of Done:
- Feature calculation và feature policy tách biệt.
- Missing input trả unavailable cùng reason, không trả số giả.
- Không có ngưỡng đúng/sai kỹ thuật.
```

### Giai đoạn 6 — Temporal signal và repetition segmentation

```text
Áp dụng Prompt tổng.

Nhiệm vụ: xây lại temporal processing, repetition detection và phase segmentation.

Hãy triển khai:
- signal registry có polarity rõ ràng;
- bilateral fusion không đổi bên âm thầm;
- smoothing/extrema theo thời gian và FPS;
- event top_start, bottom, top_end;
- full rep và partial rep;
- không nối chu kỳ qua gap hoặc side-switch discontinuity;
- phase Descent và Ascent, với Bottom là event/boundary rõ ràng;
- per-rep duration, ROM và quality flags;
- diagnostics lưu raw/smoothed signal, event và config version.

Hãy test bằng signal synthetic trước, sau đó mới chạy development videos.

Definition of Done:
- Synthetic edge cases đạt.
- Không đếm rep qua missing gap.
- Execution success, segmentation coverage và count accuracy được báo cáo riêng.
- Thuật toán chưa được gọi là validated nếu chưa đối chiếu reference labels.
```

### Giai đoạn 7 — Development validation và freeze

```text
Áp dụng Prompt tổng.

Nhiệm vụ: đánh giá thuật toán đếm/phân đoạn trên development set mà không làm rò rỉ evaluation set.

Hãy:
- xác minh manifest, provenance và group isolation;
- xác định reference-count protocol;
- chạy toàn bộ development set với output có version/hash;
- báo cáo MAE, exact match, signed error, coverage và failure reasons theo exercise/viewpoint;
- phân tích false positive, false negative, endpoint loss, low signal và pose failure;
- chỉ tune trên development set;
- ghi cấu hình freeze và hash trước evaluation.

Definition of Done:
- Không dùng evaluation labels để tune.
- Có frozen config/code identity.
- Hạn chế và failure coverage được báo cáo trung thực.
```

### Giai đoạn 8 — Ground truth kỹ thuật và form assessment

```text
Áp dụng Prompt tổng.

Nhiệm vụ: chuẩn bị form assessment; không triển khai scoring trước khi đủ điều kiện.

Hãy kiểm tra từng điều kiện:
- có nguồn chuyên môn S1 cho mỗi rule;
- có chuyên gia phê duyệt phase, feature, range, error type và severity;
- có protocol quay/viewpoint;
- có custom dataset đúng/sai;
- có annotation guide và inter-rater agreement;
- có split theo participant.

Nếu thiếu bất kỳ điều kiện cốt lõi nào, chỉ tạo schema/protocol và ghi BLOCKED; không tự đặt ngưỡng.

Khi đủ điều kiện mới triển khai:
- rule registry có citation;
- quality gate trước scoring;
- technical/measurement/data/segmentation error separation;
- calibrated score và feedback được kiểm chứng.
```

---

## 4. Prompt kiểm tra tiến độ

```text
Đọc README.md, GYM_TECHNICAL_GROUND_TRUTH_SPEC.md, IMPLEMENTATION_PLAN.md, PROJECT_STATE.md và TASKS.md hiện tại.

Hãy kiểm toán trạng thái project dựa trên bằng chứng thực tế trong code, test và artifact; không chỉ dựa trên trạng thái đánh dấu trong tài liệu.

Trả lời:
1. Giai đoạn nào thực sự đã đạt Definition of Done?
2. Giai đoạn nào được ghi hoàn thành nhưng thiếu test/bằng chứng?
3. Có vi phạm nào về ground truth, leakage, hard-code, FPS, provenance hoặc scoring không?
4. Blocker quan trọng nhất hiện tại là gì?
5. Nhiệm vụ nhỏ nhất tiếp theo giúp tiến gần mục tiêu nhất là gì?

Nếu phát hiện tài liệu sai trạng thái, hãy cập nhật nó sau khi đưa ra bằng chứng.
```

---

## 5. Prompt cho lượt tiếp theo được khuyến nghị

```text
Áp dụng Prompt tổng trong docs/GYM_REBUILD_MASTER_PROMPT.md và thực hiện Giai đoạn 0 — Khởi tạo kiến trúc và kế hoạch.

Hãy xây nền project mới từ đầu dựa trên docs/GYM_TECHNICAL_GROUND_TRUTH_SPEC.md. Không khôi phục code cũ từ _archive/. Tạo cấu trúc tối giản, dependency manifest, schema/versioning conventions, structured flags, IMPLEMENTATION_PLAN.md, PROJECT_STATE.md, TASKS.md và smoke test.

Chưa triển khai MediaPipe, feature, repetition detection hoặc scoring trong lượt này. Kết thúc bằng cách chạy smoke test và báo cáo các file đã tạo cùng Definition of Done của Giai đoạn 0.
```
