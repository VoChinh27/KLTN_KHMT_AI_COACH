# Phương pháp

## 0. Tóm tắt thay đổi mới (xem chi tiết ở các mục dưới và `docs/RESULTS.md`)

- Chọn đúng người (nhiều người / người nhỏ): `PersonTracker` + quét ô vuông + cắt vùng (mục 2b); cổng YOLO ONNX tùy chọn.
- Lỗi riêng của phát cầu: vung xả đà quá dài, nhấc gót (mục 7b).
- Chất lượng video: FPS, rung máy, kích thước người, nhiễu pose; làm mượt theo thời gian (mục 2c).
- Chấm điểm: giới hạn điểm khi có lỗi nghiêm trọng, mức lỗi nhẹ/vừa/nặng (mục 7).
- Quay video trực tiếp: `python -m badminton3d record` (mục 9).

## 1. Luồng xử lý

```
Input/<video>.mp4
  └─ MediaPipe Pose Landmarker (IMAGE) ─► pose.csv   2D (px) + visibility + 3D world (m, gốc giữa hông, y XUỐNG)
       └─ làm sạch: mặt nạ visibility · nội suy khoảng mất NGẮN · Savitzky-Golay
            └─ góc 3D + độ tin cậy từng góc (Valid theo từng góc)
                 └─ tách đoạn cần chấm        ready: đoạn đã về tư thế sẵn sàng   |   serve: chuẩn bị → phát xong
                      └─ so từng góc với CHUẨN HLV (Reference/*.json)
                           └─ điểm · lỗi cần sửa · điểm lấy lại được · cảnh báo
                                └─ Output/<video>/ : report.html/json · biểu đồ · overlay.mp4
```

## 2. Valid theo từng góc

Chỉ số cũ (**%Valid**) yêu cầu 10 khớp cùng đạt `visibility > 0,3` trong cùng một frame, nên chỉ cần một khớp bị che là cả frame bị loại.
Chấm angle-by-angle không cần như vậy: một góc chỉ cần **các khớp tạo nên chính nó**.

- Khớp *dùng được* khi MediaPipe có phát hiện, `visibility ≥ 0,5` và toạ độ 3D hợp lệ.
- Góc *dùng được* khi cả ba khớp dùng được. **Độ tin cậy** của góc = độ tin cậy nhỏ nhất trong ba khớp (khớp đo trực tiếp = `visibility`).
- Khớp mất khi `visibility` thấp thì góc tương ứng bị bỏ riêng, các góc khác vẫn dùng. Ví dụ mất cổ tay trái không ảnh hưởng góc gối.

## 2b. Chọn đúng người

`extract_pose` dò tối đa `pose.num_poses` người mỗi frame; `PersonTracker` chọn người mục tiêu ở frame đầu theo `pose.select` (`largest` hộp lớn nhất · `center` gần tâm) hoặc `person_roi` trong `Input/<video>.json`, rồi bám bằng IoU + khoảng cách tâm; mất dấu quá `pose.lost_frames` thì chọn lại (ghi cảnh báo). Frame nào không thấy đúng người đang theo dõi: cắt vùng quanh vị trí cũ dò lại, rồi quét ô vuông (cạnh = cạnh ngắn của khung, chồng lấn 25%). `pose.detector = yolo_onnx`: YOLOv8 ONNX dò hộp người, cắt vùng, dò pose (xem `models/README.md`). `pose.crop_refine`: dò lại trên vùng cắt quanh người cho mọi frame (mặc định tắt). `pose.csv` lưu thêm gót/mũi chân, `n_people`, `switches`.

## 2c. Chất lượng video và làm mượt

`videoqc.py`: FPS, % frame trùng lặp, rung máy (luồng quang học trên nền, bỏ trôi chậm 1 giây, RMS theo % bề rộng khung), chiều cao người theo % khung, nhiễu pose (RMS đạo hàm bậc 2 theo chiều dài thân, thô và sau làm mượt). Ngưỡng cảnh báo ở `video_qc` trong cấu hình. Cửa sổ Savitzky-Golay tính theo thời gian: `quality.smooth_window_s` = 0,23 giây (7 frame ở 30 FPS, 15 frame ở 60 FPS).

## 3. Xử lý mất khớp (theo thứ tự áp dụng)

| Bước | Việc làm | Độ tin cậy |
|---|---|---|
| 1. Nội suy | Chỉ khoảng mất **≤ 0,25 giây** và có khớp hợp lệ ở **cả hai đầu** (nội suy tuyến tính 3D và 2D); không ngoại suy ở đầu/cuối video | × 0,7 |
| 2. Làm mượt | Savitzky-Golay (cửa sổ 7, bậc 2) riêng từng đoạn liên tục, không bắc cầu qua khoảng mất | — |
| 3. Bên còn lại (ready) | Tư thế sẵn sàng đối xứng: trung bình hai bên; thiếu một bên thì dùng bên còn lại | × 0,7 |
| 4. Cánh tay đo được (serve) | Xem bên dưới | — |
| 5. Không đánh giá | Độ tin cậy trung bình của góc trong đoạn chấm `< 0,4` → ghi **"không đánh giá"**, **không bị phạt** | — |

**Chọn cánh tay cho bài serve** (cố định cho cả đoạn, không đổi tay giữa chừng để so cùng loại với chuẩn):

1. Khai báo `racket_hand` (trong `Input/<video>.json` hoặc cấu hình) và tay đó thấy đủ rõ → dùng tay vợt. **Nên khai báo cho từng video:** `auto` từng chọn nhầm tay ở 2/10 video người tập (chọn tay thấy rõ nhất trong khi vợt ở tay kia).
2. `auto` và **cả hai tay đều thấy rõ** (≥ 0,7) → dùng tay **hoạt động nhiều nhất** (quãng đường cổ tay lớn nhất), nhiều khả năng là tay vợt.
3. `auto` và **chỉ một tay thấy rõ** → dùng tay camera thấy rõ nhất (ý tưởng "bắt góc khuỷu tay camera nhìn rõ nhất"), kèm cảnh báo.

> Lưu ý đã đo được: ở các video quay ngang, tay mà camera thấy rõ nhất có thể **đứng yên gần như suốt cú phát** (xem `docs/RESULTS.md`), nên đo bằng tay đó không phản ánh cú vung của tay vợt. Cách chắc chắn là quay từ phía tay vợt.

## 4. Các góc

Toạ độ MediaPipe world: x sang phải, **y xuống**, z hướng về camera (mét, gốc giữa hai hông).

| Bài | Góc / đặc trưng | Mức |
|---|---|---|
| ready | góc gối TB (hông–gối–cổ chân) · góc hông (vai–hông–gối) · góc khuỷu · góc nâng tay (hông–vai–khuỷu) · độ nghiêng thân so với phương thẳng đứng · độ rộng chân / vai | gối, nghiêng thân, độ rộng chân = PRIMARY; còn lại SECONDARY |
| serve | góc khuỷu, góc nâng tay, độ cao cổ tay so với hông, tầm với ngang của cổ tay (tay được chọn) · góc gối TB · góc hông · độ nghiêng thân | khuỷu, độ cao cổ tay, gối, nghiêng thân = PRIMARY; còn lại SECONDARY |

"Tầm với ngang" là khoảng cách trong mặt phẳng ngang (XZ) giữa hông và cổ tay, không đổi khi người xoay quanh trục đứng.

## 5. Tách đoạn cần chấm

Tốc độ tính theo **mét/giây**: toạ độ world gốc ở hông nên tốc độ thân đo từ ảnh rồi nhân hệ số *mét/pixel* = chiều dài thân 3D / chiều dài thân trong ảnh (trung vị). (Bản 2D trước phải giả định thân = 0,5 m.)

### Ready — chỉ chấm khi đã trở về tư thế sẵn sàng

| Trạng thái | Điều kiện | Chấm? |
|---|---|---|
| `MOVING` | tốc độ thân > 0,60 m/s (đang di chuyển đỡ cầu) | không |
| `SETTLING` | tốc độ < 0,35 m/s nhưng chưa đủ 0,30 giây liên tục | không |
| `READY` | tốc độ < 0,35 m/s liên tục ≥ 0,30 giây, đoạn dài ≥ 0,50 giây | **có** |

Mỗi đoạn `READY` là một đơn vị chấm; điểm tổng = trung bình theo thời lượng các đoạn.

> **Đứng yên chưa chắc là tư thế sẵn sàng.** `coach_ready_03` có đoạn đứng thẳng nghỉ vẫn được xếp `READY` (vì đứng yên), nhưng lệch xa chuẩn nên bị điểm thấp và có cảnh báo. Nhận diện *đang ở tư thế nào* là bước phát triển tiếp theo.

### Serve — một lần chuẩn = từ bước chuẩn bị đến phát xong

1. **Hoạt động của tay** = tốc độ lớn nhất của hai cổ tay (3D, so với hông). Một cú phát là một **đỉnh nổi bật** ≥ 0,60 m/s (độ nổi ≥ 0,30; hai cú cách nhau ≥ 0,8 giây).
2. **Cửa sổ cú phát** = vùng quanh đỉnh còn hoạt động ≥ max(0,20 m/s; 25% đỉnh); khoảng tụt ngắn hơn 0,15 giây (điểm quay đầu khi rút vợt) vẫn nằm trong cùng cú phát. Điểm **chạm cầu** là đỉnh cao nhất trong cửa sổ.
3. **Bước chuẩn bị** = đoạn **đứng yên** (tốc độ thân < 0,30 m/s, tay < 0,40 m/s) ngay trước cú vung, tối đa 1 giây, tối thiểu 0,30 giây. **Đoạn tăng tốc của tay** (từ lúc tay rời mức 0,40 m/s tới ngưỡng 25% đỉnh, tối đa `ramp_max_s` = 0,4 giây) thuộc cú vung, không bị coi là "chưa đứng yên". Những gì xảy ra trước đó (nhặt cầu, đi lại) **không được chấm**.
   - **Lượt 1 (nghiêm):** cửa sổ không có bước chuẩn bị bị loại (ví dụ động tác thả lỏng tay sau cú phát).
   - **Lượt 2 (clip cắt sát):** nếu lượt 1 không tìm ra cú nào và `serve.fallback_without_prep` = true thì lấy các cú vung có đỉnh ≥ 60% cú mạnh nhất (`fallback_min_rel_peak`), đánh dấu `no_prep`, **bỏ pha "Chuẩn bị"** (0,3 giây giả định không phải bước chuẩn bị thật, không chấm và không dùng dựng chuẩn) và ghi cảnh báo.
4. **Bốn pha:** chuẩn bị → rút vợt (đến lúc tay thu về sát thân nhất) → chạm cầu (đến đỉnh + 0,1 giây) → kết thúc (+0,2 giây). Trọng số 0,20 / 0,20 / 0,40 / 0,20.
5. Cú phát không tự phát hiện được thì đánh dấu tay trong `Input/<video>.json` (`serve_windows_s`).

## 6. Chuẩn từ HLV

Từ 1–3 video HLV tập đúng, mỗi video cho một giá trị cho từng góc (trung vị có trọng số độ tin cậy trên các đơn vị của video đó):

- **Giá trị chuẩn** = trung vị các video.
- **Dung sai** = `max(floor, 1,25 × nửa khoảng dao động giữa các video)` khi có ≥ 2 video; chỉ 1 video thì `floor × 1,5`. `floor` là dung sai tối thiểu từng góc trong `config/badminton3d.yaml`, **đã hiệu chỉnh bằng dữ liệu** (`docs/RESULTS.md` mục 5): ready gối ±12°, hông ±15°, khuỷu/vai ±18°, nghiêng thân ±9°, chân ±0,30; serve khuỷu/vai ±15°, cổ tay ±0,10 m, gối ±10°, hông ±12,5°, nghiêng thân ±7,5°.
- Với serve, chuẩn riêng cho **từng pha**.

Vì dao động giữa các HLV (đều đúng) có thể lớn — gối 127–149° giữa 5 HLV — chuẩn là một **khoảng chấp nhận** chứ không phải một khung xương duy nhất.

## 7. Chấm điểm angle-by-angle

- Một góc: **100 điểm** khi `|giá trị − chuẩn| ≤ dung sai`; vượt ra ngoài thì giảm tuyến tính: `100 × max(0, 1 − (độ lệch − dung sai) / (1,5 × dung sai))`.
- Điểm một đơn vị (đoạn `ready` / pha `serve`) = trung bình có trọng số `mức (PRIMARY 1,0 · SECONDARY 0,5) × độ tin cậy`.
- `serve`: điểm lần phát = trung bình có trọng số các pha; các pha chấm được phải chiếm ≥ 50% trọng số. Điểm video = trung bình các lần phát.
- **Lỗi nghiêm trọng (giới hạn điểm):** nếu một góc **PRIMARY** có điểm góc = 0 (lệch quá `(1 + decay) × dung sai`) thì điểm đoạn `ready` / pha `serve` / lần phát tối đa `scoring.critical_cap` = **69** (dưới ngưỡng "đạt" 70), để một lỗi nặng không bị trung bình các góc đúng che mất. Điểm thô vẫn lưu ở `raw_score`, danh sách góc ở `critical`, và có cảnh báo trong báo cáo.
- Xếp loại: ≥ 85 **Tốt** · ≥ 70 **Khá** · ≥ 50 **Cần cải thiện** · còn lại **Chưa đạt**.

**Lỗi cần sửa.** Mỗi góc vượt dung sai là một lỗi; xếp theo số điểm đang bị mất. *Điểm lấy lại được* là số điểm tăng thêm nếu góc đó về trong dung sai (ước tính từng lỗi độc lập). Mỗi lỗi có câu hướng dẫn theo hướng lệch (ví dụ "gối đang duỗi hơn chuẩn khoảng 20°: hạ trọng tâm, gập gối thêm").

**Cảnh báo tự động:** chuẩn chỉ từ < 3 video · video nằm trong tập dựng chuẩn (điểm lạc quan) · đo bằng tay không chắc là tay vợt · tay đo khác tay dựng chuẩn · **góc quay lệch chuẩn > 30°** · góc chỉ đo được < 60% số frame · đoạn lệch quá xa chuẩn (có thể không phải tư thế sẵn sàng) · cú phát bị cắt ở cuối video · clip cắt sát không có bước chuẩn bị (`no_prep`) · điểm bị giới hạn do lỗi nghiêm trọng.

## 8. Kiểm chứng không lạc quan (bỏ-một-video)

Video HLV nằm trong tập dựng chuẩn mà chấm bằng chính chuẩn đó thì điểm lạc quan. Lệnh `evaluate` chấm mỗi video đó bằng chuẩn dựng từ **các video còn lại**; video ngoài tập dựng chuẩn chấm độc lập. Cần ≥ 2 video HLV trong tập dựng chuẩn mới kiểm chứng chéo được.


## 7b. Lỗi riêng của phát cầu (`faults.py`)

- **Vung xả đà quá dài:** `max(vai_y − cổ_tay_y)` của tay vợt từ lúc chạm cầu đến `end + 0,3 s` (m, > 0 là cao hơn vai). Nhẹ > `warn_m` (0,15 m, trừ 4 điểm), nặng ≥ `high_m` (0,30 m, trừ 10 điểm). Lỗi hiệu quả, không phải phạm luật.
- **Nhấc gót:** mức gót cao hơn mũi chân trong ảnh (so với lúc bàn chân phẳng nhất, phân vị 5%) chia chiều dài thân, chân tệ nhất, từ `lookback_s` trước khi vung tới sát cú chạm; > `threshold` thì phạm luật, trừ `penalty` điểm. Số đo luôn được lưu trong báo cáo (`fault_metrics`) dù có vượt ngưỡng hay không để hiệu chỉnh. Độ chính xác hạn chế (xem `docs/RESULTS.md` mục 6).
- Điểm lỗi riêng trừ vào điểm lần phát sau khi áp giới hạn lỗi nghiêm trọng; hiện ở bảng lỗi, cảnh báo và trên overlay đúng các frame xảy ra lỗi. Tắt/bật và chỉnh ngưỡng ở mục `faults` của cấu hình.
- **Mức lỗi** của góc: nhẹ (điểm góc ≥ 60) · vừa (20–60) · nặng (< 20).

## 8b. Trạng thái không chấm

Chỉ chấm từ bước chuẩn bị đứng yên ngay trước cú vung đến khi phát xong. Các frame khác chỉ gán nhãn (không chấm): "Đang đứng chuẩn bị / sẵn sàng phát", "Đang di chuyển / nhặt cầu", "Sau cú phát", "Thiếu dữ liệu".

## 9. Quay trực tiếp (`record.py`)

`python -m badminton3d record <ready|serve> [--source 0|URL|file] [--role trainee|coach] [--racket-hand left|right]`: cửa sổ xem trước có nút bấm (hoặc SPACE) để quay/dừng; lưu `Input/<role>_<bài>_liveNN.mp4` + `.json`; FPS đo thực tế khi quay; dừng quay thì chạy `process_video` như `analyze` và mở `report.html`. `--no-preview` quay ngay đến hết nguồn (dùng cho kiểm thử/giả lập bằng file).
