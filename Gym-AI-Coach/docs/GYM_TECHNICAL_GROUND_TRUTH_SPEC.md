# ĐẶC TẢ KỸ THUẬT GYM CHO AI COACH

## BodyWeightSquats và HandstandPushups — bản 1.0

**Hệ thống:** video đơn camera, MediaPipe Pose 2D, không hiệu chỉnh camera  
**Phạm vi:** đếm lần lặp, phân đoạn pha và nền tảng cho đánh giá kỹ thuật  
**Trạng thái:** đặc tả triển khai chi tiết; các tiêu chuẩn đúng/sai về kỹ thuật vẫn cần chuyên gia xác nhận

---

## 0. Mục đích và nguyên tắc sử dụng

Tài liệu này chuyển cấu trúc đặc tả chặt chẽ của tài liệu môn chạy sang hai bài gym thuộc MVP hiện tại. Tài liệu không sao chép ngưỡng của chạy và không coi nhãn hành động UCF101 là nhãn chất lượng kỹ thuật.

Ba nhiệm vụ phải được tách biệt:

1. **Nhận diện bài tập:** video có phải BodyWeightSquats hoặc HandstandPushups hay không.
2. **Đếm và phân đoạn:** xác định từng lần lặp và các pha đi xuống, đáy, đi lên.
3. **Đánh giá kỹ thuật:** xác định động tác đúng/sai, loại lỗi và mức độ lỗi.

UCF101_GYM hiện chỉ cung cấp nhãn hành động. Dataset này có thể phục vụ nhiệm vụ 1 và nghiên cứu nhiệm vụ 2, nhưng không phải ground truth cho nhiệm vụ 3.

### 0.1. Phân loại nguồn

Hai trục phân loại sau độc lập với nhau.

#### A. Cơ sở của tiêu chuẩn

| Mã | Ý nghĩa | Cách sử dụng |
|---|---|---|
| `S1` | Tiêu chuẩn trực tiếp từ tài liệu chuyên môn hoặc chuyên gia có thẩm quyền | Có thể dùng làm chuẩn kỹ thuật sau khi ghi rõ nguồn |
| `S2` | Quy đổi có lý giải từ `S1` sang đại lượng hệ thống đo được | Phải ghi công thức và giả định |
| `S3` | Giả định kỹ thuật/engineering do hệ thống đặt | Chỉ dùng thử nghiệm; không gọi là chuẩn kỹ thuật |
| `OBS` | Kết quả đã quan sát hoặc đo trên dữ liệu của project | Chỉ có hiệu lực trong phạm vi thí nghiệm đã thực hiện |
| `TBD-EXPERT` | Chưa có chuẩn; cần chuyên gia gym xác nhận | Không dùng để kết luận đúng/sai |

#### B. Khả năng đo bằng pose 2D

| Mã | Ý nghĩa | Ví dụ |
|---|---|---|
| `M1` | Đo trực tiếp tương đối tin cậy trong mặt phẳng ảnh | Góc khuỷu khi nhìn rõ vai–khuỷu–cổ tay |
| `M2` | Đo tương đối, phụ thuộc góc quay/chuẩn hóa | Độ lệch gối so với mũi chân |
| `M3` | Ước lượng yếu hoặc bị che khuất | Vị trí bàn chân/độ sâu khi camera không ở mặt phẳng phù hợp |
| `M0` | Không thể suy ra đáng tin từ video 2D hiện tại | Lực, đau, tải trọng, thăng bằng thật, kích hoạt cơ |

`S1` không đồng nghĩa với `M1`: một tiêu chuẩn chuyên môn có thể đúng nhưng camera 2D không đo được chính xác.

### 0.2. Quy ước góc và tọa độ

- `joint_angle(A, B, C)` là góc trong tại khớp `B`, miền `[0°, 180°]`.
- Góc duỗi thẳng của gối/khuỷu gần `180°`; góc giảm khi khớp gập.
- Trục ảnh dùng quy ước `x` tăng sang phải, `y` tăng xuống dưới.
- `torso_angle_vertical` là góc giữa vector hông giữa → vai giữa và phương thẳng đứng trong ảnh.
- Mọi khoảng cách chỉ được báo cáo dưới dạng pixel để debug hoặc dưới dạng ratio đã chuẩn hóa. Không quy đổi sang cm khi chưa calibration.
- Góc làm tròn 1 chữ số thập phân; ratio 3 chữ số; slope 4 chữ số; confidence và tỷ lệ hiển thị theo phần trăm nguyên khi đưa lên UI.

### 0.3. Ràng buộc camera

- Squat: ưu tiên mặt phẳng nghiêng bên để phân tích gập gối, gập hông và nghiêng thân; ưu tiên mặt phẳng chính diện để phân tích đối xứng trái/phải. Một góc quay không cung cấp đầy đủ cả hai nhóm thông tin.
- Handstand push-up: cần thấy rõ vai, khuỷu, cổ tay và hông; cạnh tường hoặc vật che có thể làm giảm visibility.
- Không tự động kết luận lỗi kỹ thuật từ metric `M2/M3` khi chưa xác định viewpoint phù hợp.
- Project hiện chưa có bộ phân loại viewpoint đáng tin cậy. Vì vậy report phải có trường `viewpoint: unknown` nếu không có metadata/người gán nhãn.

---

## 1. Đầu vào, pose và tiền xử lý chung

### 1.1. Đầu vào bắt buộc

```yaml
video:
  video_id: string
  fps: float_positive
  frame_count: integer_non_negative
  resolution: [width, height]
pose:
  model: MediaPipe
  dimensions: 2D
  per_joint: [x, y, visibility]
```

### 1.2. Các khớp bắt buộc

**BodyWeightSquats:** vai trái/phải, hông trái/phải, gối trái/phải, cổ chân trái/phải.  
**HandstandPushups:** vai trái/phải, khuỷu trái/phải, cổ tay trái/phải, hông trái/phải.

Các metric bổ sung liên quan bàn chân, thân toàn bộ hoặc alignment chân chỉ được bật khi các keypoint tương ứng tồn tại và vượt quality gate.

### 1.3. Chuỗi tiền xử lý bắt buộc

Thứ tự triển khai hiện tại:

1. Loại quan sát không hữu hạn hoặc có visibility dưới `0.5`.
2. Nội suy tuyến tính gap bị thiếu không quá `5` frame.
3. Giữ nguyên gap dài; không nối chu kỳ qua gap chưa giải quyết.
4. Tịnh tiến gốc tọa độ về mid-hip khi có đủ hai hông.
5. Chuẩn hóa scale bằng chiều dài torso khi có đủ vai và hông.
6. Làm mịn từng đoạn hữu hạn liên tục bằng Savitzky–Golay, mặc định `window=11`, `polyorder=2`.
7. Tính feature từ tọa độ đã xử lý.
8. Trích xuất temporal signal, làm mịn signal, tìm extrema, tạo chu kỳ và phân đoạn pha.

Các số `0.5`, `5`, `11`, `2` là ngưỡng engineering (`S3`), không phải chuẩn kỹ thuật gym. Window theo frame chưa bất biến theo FPS.

### 1.4. Quy tắc fallback chuẩn hóa

Hiện tại code có fallback ngầm: nếu thiếu mid-hip thì gốc `(0,0)`; nếu thiếu torso scale thì scale `1.0`. Hai fallback này không được coi là dữ liệu tương đương với frame chuẩn hóa đầy đủ.

Yêu cầu bản tiếp theo:

- Ghi `translation_status ∈ {mid_hip, fallback_origin}`.
- Ghi `scale_status ∈ {torso, fallback_unit}`.
- Không dùng feature khoảng cách/ratio để chấm kỹ thuật trên frame có `fallback_unit`.
- Vẫn có thể dùng góc khớp nếu ba điểm tạo góc hợp lệ.

---

## 2. Quality gate và phân loại độ tin cậy

### 2.1. Gate mức video/chuỗi

| Điều kiện | Trạng thái | Hành động |
|---|---|---|
| Không có frame | `insufficient_sequence_length` | Dừng pipeline |
| Missing frame ratio > `0.30` | `excessive_missingness` | Không đếm/không chấm |
| Gap thiếu liên tiếp > `10` frame | `excessive_missingness` | Không nối chu kỳ qua gap |
| Low-visibility frame ratio > `0.30` | `low_pose_visibility` | Không chấm metric bị ảnh hưởng |
| Signal không có chu kỳ hợp lệ | `segmentation_failed` | Trả về 0 chu kỳ cùng lý do; không suy diễn 0 rep thật |
| Viewpoint chưa biết | `viewpoint_unknown` | Cho phép đếm thử nghiệm; khóa nhận xét phụ thuộc viewpoint |

Các tỷ lệ trên đang là mặc định engineering của project, cần đánh giá lại trên development set trước khi freeze.

### 2.2. Gate mức feature

Một feature chỉ hợp lệ tại frame khi:

- tất cả keypoint cần thiết tồn tại;
- tọa độ hữu hạn;
- quan sát gốc vượt visibility threshold hoặc là giá trị nội suy trong gap hợp lệ;
- không dùng fallback normalization nếu feature cần distance/ratio;
- viewpoint phù hợp với feature;
- metric không nhảy bất thường do đổi bên quan sát.

### 2.3. Confidence

Không được dùng `valid_ratio` như “độ tin cậy khoa học” của kết quả. Tách riêng:

```yaml
confidence:
  pose_coverage: 0.0_to_1.0
  segmentation_confidence: unvalidated_proxy_or_model_output
  assessment_confidence: none_until_expert_validation
```

Trường `SegmentationResult.confidence` hiện dùng `valid_ratio` và đã được schema ghi là chưa được xác thực. UI/report phải gọi đúng là `pose_coverage_proxy` cho đến khi thay schema.

---

## 3. BodyWeightSquats

### 3.1. Phạm vi động tác

Một chu kỳ squat được mô hình hóa bằng:

`Start/Top → Descent → Bottom → Ascent → Return/Top`

Trong code hiện tại, output mới tạo hai đoạn chính `DESCENT` và `ASCENT`; `Start`, `Bottom` và `Return` mới là event/boundary, chưa phải các phase có khoảng thời gian riêng.

### 3.2. Feature đã triển khai

#### `squat_knee_angle_left_deg`, `squat_knee_angle_right_deg`

- Công thức: `angle(hip, knee, ankle)`.
- Nguồn: `S3 / M1`, là candidate kinematic feature.
- Ý nghĩa quan sát: góc thường giảm khi đi xuống và tăng khi đứng lên nếu mặt phẳng quay phù hợp.
- Dùng hiện tại: signal ứng viên cho segmentation.
- Không được kết luận: “đạt độ sâu chuẩn” chỉ từ một ngưỡng góc chưa được chuyên gia phê duyệt.

#### `squat_knee_angle_mean_deg`

- Chỉ được tính khi cả hai bên hợp lệ.
- Mean hai bên có thể che mất bất đối xứng; vì vậy chỉ dùng cho signal tổng quát, không dùng để đánh giá lệch trái/phải.
- Nếu chỉ một bên hợp lệ, signal extractor hiện có thể dùng bên còn lại, nhưng report phải ghi `side_source`.

#### `squat_hip_angle_left_deg`, `squat_hip_angle_right_deg`

- Công thức: `angle(shoulder, hip, knee)`.
- Nguồn: `S3 / M1`.
- Ý nghĩa quan sát: biểu diễn mức gập tại hông trong mặt phẳng ảnh.
- Dùng hiện tại: signal ứng viên, chưa phải tiêu chuẩn đúng/sai.

#### `squat_hip_y`

- Công thức hiện tại: trung bình `L_Hip_norm_y` và `R_Hip_norm_y`, fallback một bên.
- Cảnh báo quan trọng: normalization lấy mid-hip làm gốc ở từng frame có thể làm triệt tiêu chính chuyển động hip-y. Vì vậy metric này không được chọn làm primary signal nếu chưa sửa chiến lược normalization hoặc dùng tọa độ trước tịnh tiến.
- Trạng thái: `OBS + S3 / M2`, cần thí nghiệm lại.

### 3.3. Feature cần bổ sung trước form assessment

| Feature | Công thức dự kiến | Đo 2D | Mục đích | Trạng thái |
|---|---|---:|---|---|
| `torso_angle_vertical` | góc mid-hip → mid-shoulder với trục đứng | M1/M2 | độ nghiêng thân | `TBD-EXPERT` |
| `knee_angle_min` | min theo từng rep sau làm mịn | M1 | đại diện đáy squat | `S3`, chưa có ngưỡng chuẩn |
| `hip_angle_min` | min theo từng rep | M1 | mức gập hông | `S3` |
| `left_right_knee_asymmetry` | `abs(left-right)` | M2 | phát hiện lệch hai bên khi nhìn phù hợp | `TBD-EXPERT` |
| `descent_duration_s` | `(bottom-start)/fps` | M1 | tempo pha xuống | `S3` |
| `ascent_duration_s` | `(end-bottom)/fps` | M1 | tempo pha lên | `S3` |
| `tempo_ratio` | descent/ascent | M1 | mô tả nhịp | `S3` |
| `knee_to_ankle_horizontal_norm` | độ lệch x / scale hợp lệ | M2 | mô tả đường đi gối | `TBD-EXPERT`, phụ thuộc viewpoint |
| `rep_range_of_motion` | top angle − bottom angle | M1 | biên độ chu kỳ | `S3` |

Không dùng `knee_to_ankle_horizontal_norm` để gán lỗi “gối vượt mũi chân” khi không có chuẩn chuyên môn và không thấy mũi chân đáng tin cậy.

### 3.4. Event detection cho squat

#### Event A — `top_start_frame`

Frame cực đại hợp lệ của signal góc trước một valley, sau khi làm mịn. Đây là neo engineering, không nhất thiết là tư thế đứng hoàn toàn.

#### Event B — `bottom_frame`

Frame valley giữa hai peak trong pattern `PEAK–VALLEY–PEAK` của angle signal. Nếu dùng signal có chiều ngược lại, phải khai báo polarity và không tái sử dụng pattern một cách ngầm định.

#### Event C — `top_end_frame`

Frame peak hợp lệ sau valley. Một repetition hợp lệ hiện yêu cầu toàn bộ cửa sổ từ start đến end không chứa NaN chưa xử lý.

#### Điều kiện chu kỳ

- Thứ tự bắt buộc: `top_start < bottom < top_end`.
- Duration frame > 0.
- Không có gap chưa giải quyết trong chu kỳ.
- Prominence và distance phải được cấu hình theo signal/FPS, không dùng một bộ số cố định cho mọi video sau khi freeze.
- Các chu kỳ chạm đầu/cuối video phải được ghi `partial_rep` thay vì tự động coi là rep hoàn chỉnh.

### 3.5. Đánh giá theo pha — chưa được dùng để chấm điểm

#### Start/Top

Candidate observations: gối và hông ở vùng duỗi lớn hơn đáy, thân ổn định, signal gần peak. Không đặt ngưỡng “đứng thẳng” trước khi có `S1/S2`.

#### Descent

Candidate observations: góc gối/hip giảm có xu hướng liên tục; duration đo được; không có reversal lớn ngoài nhiễu. Reversal threshold là `S3` và cần hiệu chỉnh.

#### Bottom

Candidate observations: knee angle đạt local minimum; vận tốc góc đổi dấu; hip angle và torso angle được lấy trong cửa sổ quanh bottom để giảm sai số một frame.

#### Ascent

Candidate observations: góc gối/hip tăng; không có gap; kết thúc gần top peak.

#### Return/Top

Candidate observations: signal quay lại vùng đầu chu kỳ. “Lockout đúng” là `TBD-EXPERT`, không được suy ra chỉ vì có peak.

---

## 4. HandstandPushups

### 4.1. Phạm vi động tác

Một chu kỳ handstand push-up được mô hình hóa bằng:

`Start/Top → Descent → Bottom → Ascent → Return/Top`

Elbow angle là primary candidate hiện tại vì thí nghiệm của project quan sát thấy tín hiệu này rõ hơn shoulder-y trên mẫu đã thử. Đây là bằng chứng `OBS` trong phạm vi dataset, không phải tiêu chuẩn biomechanical toàn cục.

### 4.2. Feature đã triển khai

#### `hspu_elbow_angle_left_deg`, `hspu_elbow_angle_right_deg`

- Công thức: `angle(shoulder, elbow, wrist)`.
- Nguồn: `S3 + OBS / M1`.
- Ý nghĩa: góc thường giảm trong descent và tăng trong ascent khi keypoint rõ.
- Dùng hiện tại: primary temporal signal candidate.
- Rủi ro: bên gần tường hoặc bị thân che có thể mất cổ tay/khuỷu; đổi bên fallback giữa chuỗi có thể tạo discontinuity.

#### `hspu_shoulder_angle_left_deg`, `hspu_shoulder_angle_right_deg`

- Công thức: `angle(hip, shoulder, elbow)`.
- Nguồn: `S3 / M1-M2`.
- Dùng hiện tại: candidate signal phụ.
- Không được gọi là “arm flare đúng/sai” nếu chưa có viewpoint và chuẩn chuyên gia.

### 4.3. Feature cần bổ sung trước form assessment

| Feature | Công thức dự kiến | Đo 2D | Mục đích | Trạng thái |
|---|---|---:|---|---|
| `elbow_angle_mean` | mean hai bên khi cùng hợp lệ | M1 | signal tổng quát | `S3` |
| `elbow_asymmetry` | `abs(left-right)` | M2 | mô tả lệch hai tay | `TBD-EXPERT` |
| `body_line_angle` | trục vai–hông với trục đứng | M1/M2 | mô tả alignment thân | `TBD-EXPERT` |
| `hip_lateral_offset_norm` | lệch hông so với trục vai/cổ tay | M2 | mô tả thân gập/lệch | `TBD-EXPERT` |
| `bottom_elbow_angle` | min theo rep | M1 | đại diện độ sâu | `S3`, ngưỡng cần chuyên gia |
| `descent_duration_s` | frame delta / fps | M1 | tempo đi xuống | `S3` |
| `ascent_duration_s` | frame delta / fps | M1 | tempo đẩy lên | `S3` |
| `sticking_time_s` | thời gian vận tốc góc thấp trong ascent | M2 | mô tả điểm chậm | `S3` |
| `rep_range_of_motion` | top elbow − bottom elbow | M1 | biên độ chu kỳ | `S3` |

### 4.4. Event detection cho HSPU

#### Event A — `top_start_frame`

Peak của elbow-angle signal trước valley. Không đồng nhất “peak” với lockout kỹ thuật chuẩn.

#### Event B — `bottom_frame`

Valley hợp lệ giữa hai peak. Nên chọn trong tín hiệu đã làm mịn và lưu cả giá trị raw/smoothed để audit.

#### Event C — `top_end_frame`

Peak sau valley. Nếu video kết thúc trong ascent thì gắn `partial_rep_end`, không tự thêm peak.

### 4.5. Đánh giá theo pha — chưa được dùng để chấm điểm

#### Start/Top

Candidate observations: elbow angle ở vùng cao của chu kỳ, pose coverage đủ, body-line feature ổn định nếu viewpoint phù hợp.

#### Descent

Candidate observations: elbow angle giảm liên tục; hai bên không đổi nguồn đo bất thường; duration hữu hạn.

#### Bottom

Candidate observations: elbow angle đạt minimum; vận tốc đổi dấu. Quan hệ đầu–tay–tường không được kết luận từ bộ 13 khớp hiện tại nếu thiếu head/foot features.

#### Ascent

Candidate observations: elbow angle tăng về peak tiếp theo. Không suy ra lực hoặc kiểm soát cơ từ velocity 2D.

#### Return/Top

Candidate observations: angle trở lại vùng top. Tiêu chí khóa khuỷu, ROM tối thiểu và alignment đạt chuẩn đều là `TBD-EXPERT`.

---

## 5. Temporal processing và FPS

### 5.1. Nguyên tắc

- Không tính extrema, slope hoặc angular velocity trên signal thô khi nhiễu đáng kể.
- Không nội suy xuyên qua gap lớn hơn giới hạn.
- Không ghép peak/valley thuộc hai đoạn hữu hạn khác nhau thành một repetition.
- Tất cả duration phải báo cáo bằng giây qua FPS, không chỉ bằng frame.
- Window smoothing và extrema distance phải được quy đổi theo thời gian hoặc có profile riêng theo FPS.

### 5.2. Cấu hình hiện tại và hạn chế

```yaml
smoothing:
  method: savitzky_golay
  window_frames: 11
  polynomial_order: 2
extrema:
  distance_frames: 5
  prominence_degrees: 5.0
```

Đây là baseline `S3`. `window_frames=11` tương ứng khoảng `0.367 s` ở 30 fps nhưng chỉ `0.183 s` ở 60 fps. Bản production nên cấu hình `window_seconds` và chuyển sang số frame lẻ hợp lệ theo FPS.

### 5.3. Polarity của signal

Mỗi signal phải khai báo:

```yaml
temporal_signal:
  name: hspu_elbow_angle
  top_event: local_maximum
  bottom_event: local_minimum
  cycle_pattern: PEAK_VALLEY_PEAK
```

Không giả định mọi feature đều dùng `PEAK–VALLEY–PEAK`.

---

## 6. Chấm điểm và feedback

### 6.1. Trạng thái hiện tại

Project chưa có:

- ngưỡng kỹ thuật gym được chuyên gia xác nhận;
- dataset đúng/sai có nhãn chuyên gia;
- error taxonomy và severity rubric đã duyệt;
- công thức score đã hiệu chỉnh;
- độ tin cậy assessment đã được đánh giá.

Vì vậy hệ thống hiện chỉ được phép xuất:

- pose/quality diagnostics;
- signal và event;
- số rep dự đoán;
- phase boundaries;
- các metric mô tả không mang kết luận đúng/sai.

### 6.2. Điều kiện để bật form scoring

Mỗi rule phải có đủ:

```yaml
rule:
  rule_id: string
  exercise: string
  phase: string
  feature: string
  reference_range: [low, high]
  source_tier: S1_or_S2
  source_citation: required
  measurement_tier: M1_or_M2_or_M3
  required_viewpoint: string
  required_joints: [string]
  minimum_pose_coverage: float
  severity_mapping: expert_approved
  feedback_text: expert_approved
```

Không rule nào được đưa vào điểm tổng nếu thiếu `source_citation` hoặc chưa có expert approval.

### 6.3. Phân loại kết quả

| Nhóm | Ý nghĩa | Ví dụ |
|---|---|---|
| `technical_error` | Vận động viên thực hiện sai theo rule đã được phê duyệt | Chỉ bật sau expert validation |
| `measurement_error` | Pose/normalization/viewpoint làm metric không tin cậy | Khuỷu bị tường che |
| `insufficient_data` | Không đủ dữ liệu để kết luận | Gap dài qua bottom phase |
| `segmentation_error` | Không xác định đúng chu kỳ/pha | Ghép sai extrema hoặc bỏ rep |
| `informational_observation` | Metric mô tả, không phán đúng/sai | Tempo descent 1.2 s |

---

## 7. Edge cases bắt buộc

| # | Tình huống | Flag | Hành động |
|---:|---|---|---|
| 1 | Video rỗng/quá ngắn | `insufficient_sequence_length` | Không chạy segmentation |
| 2 | Pose thiếu quá ngưỡng | `excessive_missingness` | Không đếm qua đoạn thiếu |
| 3 | Visibility thấp | `low_pose_visibility` | Bỏ metric liên quan |
| 4 | Thiếu một bên cơ thể | `single_side_only` | Dùng một bên để mô tả nếu liên tục; khóa symmetry |
| 5 | Nguồn bên trái/phải thay đổi giữa chu kỳ | `side_switch_discontinuity` | Không chấm/không tạo rep qua điểm đổi |
| 6 | Thiếu mid-hip | `normalization_origin_fallback` | Khóa distance/ratio |
| 7 | Thiếu torso scale | `normalization_scale_fallback` | Khóa distance/ratio |
| 8 | Viewpoint chưa biết | `viewpoint_unknown` | Khóa feature phụ thuộc viewpoint |
| 9 | Rep bắt đầu trước video | `partial_rep_start` | Không tính full rep trừ khi protocol cho phép |
| 10 | Rep kết thúc sau video | `partial_rep_end` | Không tính full rep trừ khi protocol cho phép |
| 11 | Nhiều người trong frame | `multi_person` | Yêu cầu chọn subject/track |
| 12 | Người ra khỏi frame | `out_of_frame` | Chỉ xử lý đoạn còn đủ pose |
| 13 | Tín hiệu gần như phẳng | `low_signal_range` | Không tìm extrema; trả lý do |
| 14 | Extrema dày do jitter | `unstable_signal` | Điều chỉnh/làm mịn theo protocol, không tự tăng score |
| 15 | FPS thiếu hoặc sai | `invalid_fps` | Không xuất duration/velocity |

Mọi edge case phải xuất hiện trong report và không được làm pipeline crash.

---

## 8. Output contract đề xuất

```json
{
  "video_id": "example",
  "exercise": "BodyWeightSquats",
  "fps": 30.0,
  "viewpoint": "unknown",
  "quality": {
    "status": "VALID",
    "pose_coverage": 0.94,
    "flags": ["viewpoint_unknown"]
  },
  "normalization": {
    "translation_method": "mid_hip",
    "scale_method": "torso",
    "fallback_frame_ratio": 0.0
  },
  "segmentation": {
    "signal": "squat_knee_angle",
    "signal_polarity": "PEAK_VALLEY_PEAK",
    "predicted_rep_count": 3,
    "status": "CLEAR",
    "pose_coverage_proxy": 0.94,
    "parameters": {
      "smoothing_window_frames": 11,
      "smoothing_polyorder": 2,
      "extrema_distance_frames": 5,
      "extrema_prominence_deg": 5.0
    }
  },
  "repetitions": [
    {
      "rep_index": 1,
      "start_frame": 15,
      "bottom_frame": 28,
      "end_frame": 44,
      "duration_s": 0.967,
      "phases": {
        "descent": [15, 28],
        "ascent": [28, 44]
      },
      "features": {
        "knee_angle_min_deg": 82.4,
        "range_of_motion_deg": 91.7
      },
      "assessment": {
        "status": "NOT_AVAILABLE_PENDING_EXPERT_VALIDATION",
        "technical_errors": []
      },
      "flags": []
    }
  ],
  "error_classification": {
    "technical_error": 0,
    "measurement_error": 0,
    "insufficient_data": 0,
    "segmentation_error": 0
  }
}
```

Ví dụ trên minh họa schema, không xác nhận các giá trị là ground truth.

---

## 9. Đối chiếu với code hiện tại

| Hạng mục | Hiện trạng | Khoảng trống |
|---|---|---|
| Joint angle có clamp số học | Đã có | Cần test thêm non-finite input |
| Visibility filtering | Đã có | Cần provenance theo từng joint/frame |
| Nội suy gap ngắn | Đã có, tối đa 5 frame | `is_imputed` hiện ở mức frame, chưa theo joint |
| Mid-hip/torso normalization | Đã có | Fallback chưa được ghi explicit |
| Savitzky–Golay theo đoạn hữu hạn | Đã có | Window chưa FPS-aware |
| Squat knee/hip angles | Đã có | Chưa có per-rep summary và quality tier |
| HSPU elbow/shoulder angles | Đã có | Chưa kiểm soát side switching |
| Signal quality | Đã có finite ratio/statistics | Chưa đánh giá viewpoint và stability đầy đủ |
| Extrema và rep detection | Đã có PVP | Chưa xử lý partial rep/endpoint đầy đủ |
| Phase output | Có Descent/Ascent | Bottom chỉ là boundary; Start/Return chưa biểu diễn riêng |
| Form assessment | Chưa có | Đúng: cần expert ground truth trước khi triển khai |
| Error classification | Chưa có schema đầy đủ | Cần tách technical/measurement/data/segmentation |
| Output auditability | Diagnostics đã lưu trong Step 6 | Cần chuẩn hóa contract và version config |

---

## 10. Kế hoạch triển khai theo thứ tự

### P0 — Làm cho kết quả hiện tại trung thực và audit được

- [ ] Ghi rõ normalization fallback theo frame.
- [ ] Theo dõi provenance/visibility/imputation theo từng joint.
- [ ] Ghi side source của mọi bilateral signal và chặn cycle qua side switch.
- [ ] Đổi nhãn confidence hiện tại thành pose coverage proxy.
- [ ] Bổ sung flag viewpoint unknown và partial repetitions.
- [ ] Làm smoothing/extrema parameters theo FPS hoặc theo giây.

### P1 — Cải thiện đếm và phân đoạn trên development set

- [ ] So sánh có kiểm soát các signal squat: knee angle, hip angle và tọa độ hip trước centering.
- [ ] So sánh HSPU elbow mean, best continuous side và bilateral fusion.
- [ ] Phân tích endpoint/extrema loss trên 17/29 video đang dự đoán 0 rep.
- [ ] Định nghĩa tiêu chí `low_signal_range` và `unstable_signal` chỉ bằng development set.
- [ ] Freeze thuật toán trước khi chạy evaluation set.
- [ ] Báo cáo MAE, exact match, signed error và failure coverage tách theo bài tập/viewpoint.

### P2 — Xây dựng ground truth kỹ thuật gym

- [ ] Chọn tài liệu/chuyên gia nguồn `S1` cho từng bài.
- [ ] Xác định viewpoint quay chuẩn.
- [ ] Chuyên gia duyệt phase, feature, reference range, error type và severity.
- [ ] Thu custom dataset có correct/incorrect form và metadata người tập.
- [ ] Đánh giá liên chuyên gia (IAA) cho nhãn kỹ thuật.
- [ ] Chỉ sau đó mới bật rule-based scoring/feedback.

### P3 — Form assessment

- [ ] Cài rule có citation và measurement tier.
- [ ] Tách lỗi kỹ thuật khỏi lỗi đo lường/thiếu dữ liệu.
- [ ] Hiệu chỉnh score trên validation set; không tune trên evaluation set.
- [ ] Kiểm thử edge case và báo cáo uncertainty.

---

## 11. Definition of Done

### Đếm và phân đoạn

- Tất cả output có config version, signal source, quality flags và frame boundaries.
- Không có repetition vượt qua gap chưa xử lý hoặc side-switch discontinuity.
- Parameters theo FPS được freeze trước final evaluation.
- Development và evaluation tách theo `group_id`/participant để tránh leakage.
- Final report tách execution success, segmentation coverage và count accuracy.

### Đánh giá kỹ thuật

- Mỗi rule có nguồn `S1/S2`, citation và expert approval.
- Mỗi feature có measurement tier và required viewpoint.
- Report phân biệt technical error, measurement error, insufficient data và segmentation error.
- Không chấm khi quality gate không đạt.
- Score và feedback được kiểm chứng trên custom form dataset, không dựa vào nhãn hành động UCF101.

---

## 12. Quyết định áp dụng từ tài liệu chạy

Những nguyên tắc được áp dụng trực tiếp:

- tách nguồn chuẩn kỹ thuật khỏi khả năng đo 2D;
- ưu tiên metric tương đối/góc thay vì đơn vị vật lý tuyệt đối;
- đặc tả feature bằng công thức, nguồn, độ tin cậy và mục đích sử dụng;
- dùng temporal event cho chuyển động, không đánh giá chỉ bằng một frame;
- bắt buộc quality gate, fallback có flag và edge-case handling;
- tách lỗi kỹ thuật khỏi lỗi đo lường và thiếu dữ liệu;
- quy định output contract và Definition of Done.

Những nội dung không được chuyển nguyên xi:

- mọi ngưỡng góc/khoảng cách dành cho xuất phát chạy;
- shoulder-width normalization như quy tắc duy nhất cho gym;
- event hand-off, rear-foot-off và launch detection;
- giả định camera, phase và scoring riêng của chạy.

Tài liệu gym này là cầu nối giữa pipeline hiện tại và bộ ground truth kỹ thuật sẽ được chuyên gia xác nhận; nó không tự tạo ra ground truth còn thiếu.
