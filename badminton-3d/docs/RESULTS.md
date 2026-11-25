# Kết quả đo — 31 video, 10 video được chọn

Dữ liệu: **13 video HLV** (6 ready + 7 serve) và **18 video người tập** (8 ready + 10 serve, tập sai). Việc lọc và lý do từng video: `docs/SELECTION.md`. Số liệu bên dưới là của **10 video được chọn** (4 HLV làm chuẩn + 6 người tập) trong `Output/_tong_hop/`, trừ khi ghi khác. Chạy lại: `python -m badminton3d run`.

## 1. Những phát hiện quan trọng trên dữ liệu mới

1. **`coach_ready_07` không dùng làm chuẩn được.** Tư thế gần như trùng nhóm người tập sai (gối ~155°, thân ~9°). Đưa vào chuẩn thì **0/7** người tập sai bị bắt lỗi (87–100 điểm); chuẩn `coach_ready_01 + coach_ready_04` cho **7/7** người tập bị bắt (28–39 điểm). Chi tiết: `docs/SELECTION.md` mục 1.
2. **Bộ tách cú phát từng bỏ sót phần lớn video serve.** Luật "bắt buộc có bước chuẩn bị đứng yên ngay trước cú vung" làm mất các clip đã cắt sát: chỉ **8/17** video serve tách được ít nhất một lần phát. Đã sửa (đoạn tăng tốc của tay thuộc cú vung; clip cắt sát vẫn lấy cú vung mạnh, đánh dấu `no_prep`, bỏ pha "Chuẩn bị" giả): **17/17**.
3. **Tự chọn tay vợt sai** ở 2/10 video người tập serve (`02`, `03`). Đã khai báo tay vợt bằng `Input/<video>.json` sau khi xác nhận bằng hình; 2 người tập (`05`, `06`) thuận tay trái.
4. **Một lỗi nặng bị trung bình che mất.** Điểm là trung bình các góc nên cổ tay cao hơn chuẩn ~0,3–0,4 m chỉ mất khoảng 15–20 điểm và vẫn "Khá". Đã thêm **giới hạn điểm khi có lỗi nghiêm trọng** (mục 4).

## 2. Độ phủ (tỉ lệ frame dùng được, 10 video)

| Bước | ready (4 video) | serve (5 video) |
|---|---|---|
| Chỉ số cũ: 10 khớp cùng lúc (`visibility > 0,3`) | 94,2% | 99,6% |
| Valid theo từng góc → nội suy → bên/tay thay thế | **98,9%** | **100%** |

Theo video (cũ → sau xử lý): `coach_ready_01` 92,8% → 99,5%; `trainee_ready_04` 89,0% → 100%; `trainee_ready_08` 98,9% → 100%; `coach_ready_04` 96,1% → 96,1%. (`coach_ready_01` có hai người trong khung; bộ chọn người mới bám đúng HLV nên số liệu cũ 82,5% được cải thiện.) Các video serve mới quay **chính diện/chéo** nên cả hai tay thấy rõ (khác bộ HLV cũ quay ngang, chỉ đạt 61%): lợi ích xử lý mất khớp tập trung ở bài ready. Biểu đồ: `coverage_stages.png`, `coverage_heatmap_*.png`.

## 3. Điểm (chuẩn: ready từ `coach_ready_01` + `04`; serve từ `coach_serve_02` + `07`)

| Video | Cách chấm | Điểm | Lỗi chính hệ thống đo được |
|---|---|---|---|
| `coach_ready_01` | bỏ-một-video | 92,7 (Tốt) | chân hẹp hơn video còn lại |
| `coach_ready_04` | bỏ-một-video | 94,0 (Tốt) | chân rộng hơn video còn lại |
| `trainee_ready_04` | độc lập | 37,4 (Chưa đạt) | thân thiếu ~22°, gối duỗi hơn ~32°, hông duỗi hơn ~45° |
| `trainee_ready_08` | độc lập | 29,5 (Chưa đạt) | thân thiếu ~28°, gối duỗi hơn ~42° **và chân hẹp** (1,07 vai) |
| `coach_serve_02` | bỏ-một-video | 95,9 (Tốt) | — |
| `coach_serve_07` | bỏ-một-video | 95,9 (Tốt) | — |
| `trainee_serve_01` | độc lập | 65,0 (Cần cải thiện) | tay/cổ tay quá cao, vung xả đà nhẹ (−4); **nhấc gót chưa bắt được** (mục 6) |
| `trainee_serve_09` | độc lập | 59,0 (Cần cải thiện)† | cổ tay quá cao, **vung xả đà quá đầu (−10)** |
| `trainee_serve_05` | độc lập | 47,8 (Chưa đạt) | cổ tay quá thấp, gập thân |
| `trainee_serve_07` | độc lập | **89,0 (Tốt)** | thân đổ ra trước 17° |

† điểm thô 76,8 → bị giới hạn 69 vì cổ tay lệch quá xa chuẩn → trừ thêm 10 điểm vung xả đà.

Ngoài 9 video: `coach_ready_03` (đứng nghỉ) 37,8; `coach_ready_07` 35,0; 7/7 video ready của người tập 29,5–41,1; serve của người tập: 6/10 video dưới 70 điểm (`01` 65,0; `05` 47,8; `06` 69,0; `08`, `09`, `10` 59,0), 4/10 từ 71,4 trở lên (`02` 71,4; `03` 77,2; `07` 89,0; `04` 91,0).

**Điểm này có ý nghĩa gì và chưa có ý nghĩa gì** (mục 5 đo trước khi thêm các kiểm tra lỗi riêng của mục 6; các kiểm tra đó không làm HLV mất điểm)

- Có ý nghĩa — **ready:** phân tách rõ ràng: HLV 93–94 so với người tập 29–41; lỗi chung "đứng quá thẳng" được chỉ ra đúng với số đo và cách sửa.
- Có ý nghĩa — **serve:** các lỗi nặng (cổ tay quá cao/quá thấp, gập thân) bị bắt và có hướng sửa.
- **Chưa** — **serve:** lỗi nhẹ vẫn được điểm cao. `trainee_serve_07` (thân đổ 17°) được "Tốt" 89 điểm dù hệ thống đã liệt kê lỗi; `trainee_serve_04` 95 điểm chỉ có lỗi hông nhẹ. Nếu muốn nhóm lỗi này bị coi là "chưa đạt" cần thu hẹp dung sai `trunk_lean` của serve (hiện ±7,5°).
- **Chưa** — chưa có **nhãn lỗi có chủ đích** từng video nên chưa tính được độ chính xác/độ phủ lỗi; "nhóm lỗi" ở `SELECTION.md` là do hệ thống đo, không phải nhãn của người quay.
- **Chưa** — điểm "bỏ-một-video" của HLV chỉ chấm từ **một** video HLV còn lại, nên chỉ có tính tham khảo.

## 4. Thay đổi cách chấm khi có lỗi nghiêm trọng

Nếu một góc **PRIMARY** lệch quá `(1 + decay) × dung sai` (điểm của góc = 0) thì điểm của đoạn/lần phát tối đa **69** (dưới ngưỡng "đạt" 70), `scoring.critical_cap` trong cấu hình (`null` để tắt). Điểm thô vẫn được lưu (`raw_score`) và có cảnh báo trong báo cáo. Tác dụng trên dữ liệu: 5 video serve của người tập (`01`, `06`, `08`, `09`, `10`) từ 77–93 xuống 69; HLV bỏ-một-video không bị ảnh hưởng (95,9).

## 5. Hiệu chỉnh dung sai

Quy tắc: nhân dung sai tối thiểu (`floor`) với hệ số nhỏ nhất để **HLV (bỏ-một-video) đạt ≥ 85 điểm**, rồi nới **thêm 1 nấc (+0,25)** theo yêu cầu "bài tập cơ bản, không khắt khe". Số liệu quét hệ số (cột người tập: nhỏ nhất–lớn nhất, số video bị bắt <70):

| Hệ số | ready: HLV | ready: người tập (7) | serve: HLV (02/03/07) | serve: người tập (10) |
|---|---|---|---|---|
| ×1,0 | 80 / 78 | 28–39, 7/7 | 89 / 93 / 89 | 44–89, 7/10 |
| ×1,25 | 88 / 89 | 29–39, 7/7 | 96 / 97 / 96 | 48–95, 6/10 |
| **×1,5** | **93 / 94** | **30–41, 7/7** | 98 / 99 / 98 | 53–97, 6/10 |
| ×2,0 | 98 / 99 | 37–60, 7/7 | 100 | 61–99, 3/10 |
| ×2,5 | 100 | 58–77, 2/7 | 100 | 68–100, 3/10 |
| ×3,0 | 100 | 74–88, 0/7 | 100 | 69–100, 2/10 |

Chọn: **ready ×1,5** (gối ±12°, hông ±15°, khuỷu/vai ±18°, thân ±9°, chân ±0,30); **serve ×1,25** (khuỷu/vai ±15°, cổ tay ±0,10 m, gối ±10°, hông ±12,5°, thân ±7,5°). Từ ×2,5 (ready) hệ thống bắt đầu bỏ sót người tập (chỉ còn 2/7 bị bắt). Biểu đồ dải chuẩn: `reference_bands_*.png`.

## 6. Lỗi riêng của phát cầu: vung xả đà quá dài, nhấc gót

Hai kiểm tra nằm ngoài việc so góc với chuẩn (`badminton3d/faults.py`, ngưỡng và điểm trừ trong `config/badminton3d.yaml` mục `faults`; mỗi lỗi có mức nhẹ/nặng, hướng dẫn sửa, hiện trên báo cáo và trên overlay đúng các frame xảy ra lỗi).

**Vung xả đà quá dài (over follow-through) — bắt được.** Đo độ cao lớn nhất của cổ tay tay vợt so với vai từ lúc chạm cầu tới 0,3 giây sau pha kết thúc (m, > 0 là cao hơn vai). Lỗi nhẹ > 0,15 m (trừ 4 điểm), nặng ≥ 0,30 m ≈ quá đầu (trừ 10 điểm). Đây là lỗi hiệu quả (cầu dễ ra ngoài), không phải phạm luật.

| Video | Cổ tay so với vai sau cú chạm | Kết quả |
|---|---|---|
| HLV (7 video) | −0,41 … −0,01 m (`coach_serve_02` −0,08; `07` −0,01) | không báo |
| `trainee_serve_09` | **+0,44 m** | nặng (−10) |
| `03` / `10` / `08` | +0,39 / +0,35 / +0,33 m | nặng |
| `01` / `04` | +0,29 / +0,17 m | nhẹ |
| `07` | +0,05 m | không báo |
| `02` / `05` / `06` | −0,21 / −0,38 / −0,34 m | không báo |

**Nhấc gót — chưa bắt được một cách tin cậy; để thành tham số.** Đo mức gót cao hơn mũi chân (trong ảnh, so với lúc bàn chân phẳng nhất, chia chiều dài thân) trước cú chạm. Xem tận mắt thì `trainee_serve_01` thật sự nhấc gót chân trái từ ~0,6 s đến ~1,5 s, nhưng điểm đo MediaPipe của gót/mũi chân (sai số vài cm, người chỉ cao 20% khung) không tách được:

| Nhóm | Chỉ số nhấc gót |
|---|---|
| `trainee_serve_01` (lỗi thật) | 0,16 |
| HLV (7 video, lần phát đầu) | 0,06 … 0,20 (`coach_serve_02` 0,14; `03` 0,14; `01` 0,20) |
| người tập khác | 0,05 … 0,32 (`02` lần 2 = 0,32; `10` = 0,20) |

Thử dò lại bằng cách cắt vùng người (top-down) cũng không cải thiện (0,16 so với 0,14–0,15 của HLV). Vì vậy ngưỡng mặc định `faults.heel_lift.threshold = 0,30` (chỉ báo khi nhấc rất rõ, hiện báo `trainee_serve_02` lần 2) để không báo nhầm HLV. **Bạn có thể hạ xuống ~0,15 để bắt `trainee_serve_01`, đổi lại sẽ báo nhầm thêm vài video HLV.** Cách chắc chắn: quay riêng một camera cận cảnh bàn chân (người chiếm ≥ 50% khung) hoặc quay từ phía bên.

## 7. "Đứng yên = sẵn sàng": không cần bộ nhận diện tư thế

Theo yêu cầu của bạn, mục đích là chấm **tư thế phát cầu** (phạm luật / lỗi cơ bản), còn đứng yên nhặt cầu, đứng nghỉ, chuẩn bị vào tư thế **không chấm** mà chỉ ghi trạng thái. Hệ thống đã làm đúng như vậy nên bước "nhận diện tư thế đang tập" của bản trước được **bỏ**: bộ chấm chỉ xét từ bước chuẩn bị đứng yên ngay trước cú vung tới khi phát xong; các frame còn lại được gán nhãn hiển thị trên overlay: "Đang đứng chuẩn bị / sẵn sàng phát", "Đang di chuyển / nhặt cầu", "Sau cú phát (không chấm)", "Thiếu dữ liệu (không chấm)". Bài ready vẫn coi đứng yên ≥ 0,3 giây là sẵn sàng.

## 8. Lỗi nhẹ: nêu ra và trừ ít điểm

Cơ chế sẵn có đã làm đúng việc này: một góc chỉ mất điểm tuyến tính khi vượt dung sai, nên lệch nhẹ chỉ mất ít điểm. Phần mới: mỗi lỗi có **mức** (nhẹ ≥ 60 điểm góc · vừa 20–60 · nặng < 20) hiện ở bảng "Cần sửa gì" và ở overlay ("ĐẠT! Chú ý (nhẹ): …" khi vẫn đạt nhưng còn lỗi nhẹ, xương vẫn xanh). Ví dụ `trainee_serve_07` (thân đổ 17°): 89 điểm "Tốt" kèm lời nhắc.

## 9. DTW cho serve: đã thử, không đưa vào hệ thống

Thử so khớp chuỗi (khuỷu, cổ tay, thân, gối, vai; từ 0,2 giây trước cú vung đến khi phát xong) của từng video với 2 HLV (`docs/experiments/dtw_experiment.py`):
- Khoảng cách DTW **giữa hai HLV với nhau (1,54) lớn hơn** khoảng cách của nhiều người tập tới HLV (`trainee_serve_01` 0,76; `04` 0,86): nhiễu về nhịp giữa người đúng lớn hơn tín hiệu lỗi. Thời lượng cú phát của hai HLV đã là 2,9 giây và 1,4 giây.
- Tương quan hạng với điểm hiện tại chỉ −0,49 (bản chỉ lấy hình dạng: −0,16), tức không bổ sung thông tin có ích.
Kết luận: với 2 HLV khác nhịp, DTW không cải thiện chấm điểm; chấm theo pha đã đủ. Nên thử lại khi có nhiều HLV cùng nhịp hoặc có nhãn lỗi nhịp (dừng giữa cú phát).

## 10. FPS, làm mượt, rung máy

Công cụ: `python -m badminton3d check <video>` và mục "Chất lượng video" trong mỗi `report.html` (`badminton3d/videoqc.py`). Đo trên 31 video:

| Chỉ số | Kết quả | Kết luận |
|---|---|---|
| FPS | 30 FPS ở toàn bộ 20 video mới; HLV cũ 24–30 | đủ; cảnh báo nếu < 24 (`coach_ready_05`: 23,98) |
| Frame trùng lặp | ≤ 1,5% ở video mới; `coach_ready_01` 15,9%, `coach_serve_04` 6,3% | `coach_ready_01` bị nội suy FPS → nhiễu khớp thô cao (46% chiều dài thân) |
| Rung máy | video của bạn ≤ 0,13% bề rộng khung (máy trên chân đế); chỉ `coach_ready_05` 0,76%, `coach_serve_01` 0,83% đáng kể | không ảnh hưởng; góc đo trong tọa độ gốc hông nên miễn nhiễm rung tịnh tiến, chỉ tốc độ 2D bị ảnh hưởng |
| Nhiễu khớp (thô → sau làm mượt) | thường 2–10% → 0,5–2,4% chiều dài thân; giảm 56–84% | làm mượt có tác dụng rõ |
| Chiều cao người trong khung | video dọc của bạn 19–27% (≈ 370–520 px) | khớp kém chính xác nhất (gót, mũi chân, cổ tay); **quay gần để người chiếm ≥ 50% chiều cao** |

Đã sửa một điểm: cửa sổ làm mượt Savitzky-Golay trước đây cố định 7 frame (0,23 s ở 30 FPS, 0,12 s ở 60 FPS, làm méo cú vung nhanh ở FPS thấp/cao); nay tính theo thời gian (`quality.smooth_window_s = 0,23`): 7 frame ở 30 FPS, 15 frame ở 60 FPS. Kết quả ở 30 FPS không đổi. Khuyến nghị quay 30–60 FPS, máy cố định, cầm cú vung nhanh cần ánh sáng đủ để không mờ chuyển động.

## 11. Chọn đúng người: có cần YOLO (top-down) không?

Trả lời ngắn: **với một học viên đứng gần camera cố định thì không bắt buộc; cần khi có người khác trong khung hoặc người quá nhỏ.** Thử nghiệm: ghép hai người cạnh nhau (khung rộng 1920×540) — MediaPipe thường **chỉ bắt được người ở 38% số frame** và không thấy người thứ hai. Đã làm (không cần YOLO, mặc định bật):
- dò tối đa 3 người rồi chọn/bám người mục tiêu theo hộp lớn nhất (`pose.select`) hoặc vùng khai báo `person_roi` trong `Input/<video>.json` (ví dụ `[0, 0, 0.5, 1]` = nửa trái);
- khi mất dấu người đang theo dõi: cắt vùng quanh vị trí cũ rồi dò lại, nếu vẫn không thấy thì quét theo ô vuông (tiling).
Kết quả thử: bắt được người ở **100% frame**, thấy đủ 2 người, ROI trái/phải chọn đúng và **0 frame lẫn sang người kia**; video bình thường **không đổi kết quả** (sai lệch 0) và chỉ quét ô vuông ở frame đầu. Tùy chọn `pose.crop_refine` dò lại trên vùng cắt quanh người cho cả video (chậm gấp ~2; không cải thiện số đo nhấc gót nên mặc định tắt).

Cổng YOLO (`pose.detector: yolo_onnx`): dò hộp người bằng YOLOv8 ONNX (onnxruntime) rồi cắt vùng dò pose — **đã viết và kiểm thử phần giải mã/NMS bằng dữ liệu giả; chưa chạy với mô hình thật** vì môi trường này không có mạng/trọng số. Cách dùng: `models/README.md`. Khi nào nên bật: lớp học đông, người đi lại giữa khung, hoặc người chỉ chiếm < 15% chiều cao.

## 12. Quay video trực tiếp

`python -m badminton3d record serve` (hoặc `ready`): mở camera, **bấm nút xanh (hoặc phím SPACE) để quay, bấm lại để dừng**; lưu `Input/trainee_<bài>_liveNN.mp4` (+ file tay vợt), chấm ngay và tạo `Output/<video>/report.html`, `overlay.mp4`... như video có sẵn, mở báo cáo trên trình duyệt. Nguồn: webcam (`--source 0`), điện thoại làm webcam (`--source http://IP:cổng/video`), hoặc file để giả lập. FPS thực tế được đo trong lúc quay (webcam hay báo sai FPS). Đã kiểm thử bằng nguồn giả lập (quay → lưu → chấm → báo cáo cho ra đúng điểm của video gốc); **chưa thử với webcam thật** (môi trường này không có camera) nên cửa sổ xem trước/nút bấm cần bạn thử trên máy.

## 13. Hạn chế đã biết

1. **Nhấc gót chưa bắt được tin cậy** (mục 6): cần camera cận cảnh bàn chân hoặc chỉnh ngưỡng.
2. **Góc quay.** Góc 3D thay đổi theo góc quay; chuẩn chỉ đáng tin với video cùng góc (cảnh báo khi lệch > 30°). Trong bộ đã chọn, người tập lệch tối đa 8° (ready) và 20° (serve).
3. **Chuẩn dựng từ 2 video mỗi bài**; thiếu video `coach_ready` cùng bối cảnh với người tập (bạn sẽ quay thêm). Dung sai tối thiểu đã hiệu chỉnh ở mục 5 nhưng chỉ trên 2–3 video HLV.
4. **Người tập chỉ vài người**, hai người thuận tay trái (`trainee_serve_05`, `06`).
5. **Clip serve cắt sát** (`coach_serve_07`, `trainee_serve_05`): không có bước chuẩn bị đứng yên; pha "Chuẩn bị" không được chấm.
6. **Người nhỏ trong khung** (video dọc của bạn ~20% chiều cao) làm các khớp nhỏ (cổ tay, bàn chân) kém chính xác.
7. **Chưa có nhãn lỗi có chủ đích** của từng video nên chưa tính được tỉ lệ bắt lỗi theo nhãn.
8. MediaPipe chạy chế độ IMAGE (không làm mượt theo thời gian); phần làm mượt do bước làm sạch đảm nhiệm.
9. `trainee_serve_04_1.mp4` bạn gửi giống hệt `trainee_serve_04.mp4` (cùng nội dung từng byte), không tính là video mới.

## 14. Trạng thái các bước đã nêu ở bản trước và bước tiếp theo

| Bước | Tình trạng |
|---|---|
| 1. Thêm video | **Đã làm** (31 video; chọn 10). Còn thiếu `coach_ready` cùng bối cảnh — bạn sẽ quay |
| 2. Nhận diện tư thế | **Bỏ theo thiết kế** (mục 7): chỉ chấm cú phát, các giai đoạn khác gán nhãn |
| 3. DTW | **Đã thử, không dùng** (mục 9) |
| 4. Hiệu chỉnh dung sai | **Đã làm** (mục 5), làm lại khi có thêm video HLV |
| Mới: lỗi vung xả đà, nhấc gót | Vung xả đà **đã bắt**; nhấc gót **để tham số** (mục 6) |
| Mới: FPS/rung/làm mượt | **Đã kiểm tra, sửa cửa sổ làm mượt theo thời gian** (mục 10) |
| Mới: chọn đúng người / YOLO | **Đã làm** bản không cần YOLO; cổng YOLO ONNX chờ trọng số (mục 11) |
| Mới: quay trực tiếp | **Đã làm** (mục 12); cần thử trên webcam thật |

Bước tiếp theo: (1) quay `coach_ready` mới (gối gập 120–150°, thân đổ ≥ 10°, chân ≥ 1,2 vai) cùng góc quay với người tập rồi chạy `build-reference ready`; (2) quay cận cảnh bàn chân để bắt nhấc gót hoặc chỉnh ngưỡng; (3) gửi nhãn lỗi từng video để đo tỉ lệ bắt lỗi thật; (4) thử `record` trên webcam thật.
