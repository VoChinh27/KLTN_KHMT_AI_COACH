# Lọc video: chọn mẫu HLV và video người tập

Nguồn: 31 video (13 HLV + 18 người tập). Kết quả: **chọn 10** — 4 HLV làm chuẩn (2 video mỗi bài) và 6 video người tập; 21 video còn lại chuyển vào `Input/_luu_tru/` (không bị xoá, pose đã trích lưu ở `Output/_luu_tru/`). Bảng đầy đủ dạng máy đọc: `docs/selection_table.csv`.

`*` = HLV nằm trong tập dựng chuẩn nên điểm là **bỏ-một-video** (chuẩn dựng từ video HLV còn lại). Các điểm khác chấm bằng chuẩn `Reference/*.json` hiện tại.

## 1. Tiêu chí chọn HLV (không nhìn điểm của người tập để chọn)

1. **Đúng kỹ thuật cơ bản** theo số đo: ready = gối hạ (≈ 120–150°), thân đổ nhẹ ra trước (≥ 10°), chân mở ≥ 1,2 bề rộng vai; serve = phát đủ pha, có bước chuẩn bị đứng yên.
2. **Đo được tin cậy**: ≥ 90% khớp thấy rõ, tay vợt đo được, có đoạn đứng yên đủ dài để chấm.
3. **Góc quay gần với video người tập** (cảnh báo khi lệch quá 30°).
4. **Không trùng nhau**: hai video cùng HLV, cùng góc quay thì chỉ giữ một.
5. Mỗi bài tối đa **2 video**; dung sai được nới (mục 3) vì mục đích là dạy kỹ thuật cơ bản.

**Kết quả đáng chú ý — `coach_ready_07` bị loại khỏi chuẩn.** Tư thế của video này (gối trung vị 155°, thân đổ 9°) gần như trùng nhóm người tập (gối 152–167°, thân 5–11°) và nằm ngoài khoảng chuẩn gối 120–150° / thân ≥ 10° (ngưỡng trong video mẫu giao diện bạn gửi). Thử đưa vào chuẩn:

| Chuẩn dựng từ | Người tập sai (7 video) | HLV bỏ-một-video |
|---|---|---|
| `coach_ready_01` + `coach_ready_04` (đang dùng) | **28–39 điểm, 7/7 bị bắt lỗi** | 80 / 78 |
| `coach_ready_01` + `coach_ready_07` | 87–100 điểm, **0/7** bị bắt lỗi | — |
| `coach_ready_04` + `coach_ready_07` | 75–94 điểm, 0/7 | — |
| cả ba | 78–94 điểm, 0/7 | — |

(Số đo ở dung sai tối thiểu gốc ×1,0 để so sánh công bằng giữa các phương án; cấu hình hiện hành nới ×1,5 nên HLV đạt 93 / 94 — xem `docs/RESULTS.md` mục 5.) Vì vậy `coach_ready_07` **không** dùng làm chuẩn. Nếu bạn vẫn coi video này là đúng, cần quay lại với gối gập sâu hơn.

## 2. Tiêu chí chọn người tập

- **Ổn định**: độ tin cậy góc ≥ 0,9; ≥ 95% góc đo được; góc quay lệch chuẩn ≤ 30°; có ít nhất 1 đoạn/lần phát được chấm.
- **Phát hiện lỗi cơ bản**: hệ thống chỉ ra ít nhất một góc PRIMARY lệch rõ.
- **Không trùng lỗi**: nhóm theo loại lỗi hệ thống đo được, mỗi nhóm giữ video **ổn định nhất**.

Nhóm lỗi tìm được:

| Bài | Nhóm lỗi | Video thuộc nhóm | Giữ |
|---|---|---|---|
| ready | **Đứng quá thẳng** (gối duỗi +26…+42°, thân thiếu −22…−28°, hông +40…+52°, tay thấp) — *cả 7 video đều mắc* | 01, 03, 04, 05, 06, 07, 08 | `04` (ổn định nhất) |
| ready | Đứng thẳng **+ chân quá hẹp** (1,07 vai) | 08 (05, 01 nhẹ hơn) | `08` |
| serve | A — **tay/cổ tay quá cao + vung xả đà quá dài sau cú chạm** (cổ tay +0,3…0,4 m so với chuẩn; sau cú chạm vợt lên cao hơn vai 0,29–0,44 m) | 01, 03, 08, 09, 10 | `09` (ví dụ vung xả đà bạn chỉ định) |
| serve | **Nhấc gót** (bạn chỉ định) | 01 | `01` — hệ thống chưa bắt được, giữ để chỉnh ngưỡng |
| serve | B — **cổ tay quá thấp + gập thân** | 02, 05, 06 | `05` |
| serve | C — **thân đổ ra trước** (lỗi nhẹ, riêng biệt) | 07 | `07` |
| serve | D — khuỷu tay gập quá | 03 | — (góc quay lệch 30°, 86% góc đo được) |
| serve | không bắt được lỗi nào đáng kể (95 điểm) | 04 | — |

**Lưu ý trung thực:** nhóm lỗi trên do **hệ thống đo** được, mình chưa có nhãn "lỗi có chủ đích" của từng video từ bạn. Nếu bạn dự định lỗi khác (ví dụ chạm chân, vợt chạm cầu trên thắt lưng), hãy cho biết nhãn để đối chiếu.

## 3. Bảng toàn bộ video

### Bài ready
| Video | Góc quay | Đoạn chấm | Điểm | Lỗi chính (do hệ thống đo) | Quyết định | Lý do |
|---|---|---|---|---|---|---|
| `coach_ready_01` | 31° | 1 | 93* | Độ rộng chân / vai thấp | **CHỌN** | CHỌN làm chuẩn: tư thế sẵn sàng rõ (gối hạ, thân đổ trước, chân rộng), một đoạn ổn định dài, nhìn thấy đủ khớp |
| `coach_ready_02` | 24° | 0 | — | — | **LOẠI** | LOẠI: di chuyển liên tục, không có đoạn nào chấm được |
| `coach_ready_03` | 57° | 1 | 38 | Góc gối cao; Độ nghiêng thân thấp; Góc khuỷu tay thấp | **LOẠI** | LOẠI: đoạn 'sẵn sàng' thực ra là đứng thẳng nghỉ; dùng làm mẫu đối chứng 'không phải sẵn sàng' |
| `coach_ready_04` | 14° | 2 | 94* | Độ rộng chân / vai cao | **CHỌN** | CHỌN làm chuẩn: tư thế sẵn sàng rõ, chân mở rộng, góc quay gần chính diện; đoạn ổn định ngắn (video nguồn hơi mờ) |
| `coach_ready_05` | 63° | 0 | — | — | **LOẠI** | LOẠI: di chuyển liên tục, góc quay lệch 63°, không có đoạn chấm |
| `coach_ready_07` | 30° | 1 | 35 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **LOẠI** | LOẠI khỏi chuẩn: tư thế gần như trùng nhóm người tập (gối ~155°, thân ~9°); đưa vào chuẩn thì 7/7 người tập sai vẫn đạt 87–100 điểm |
| `trainee_ready_01` | 34° | 4 | 37 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **LOẠI** | LOẠI (trùng lỗi 'đứng thẳng', nhẹ nhất; độ phủ 87%) |
| `trainee_ready_02` | 25° | 0 | — | — | **LOẠI** | LOẠI: di chuyển liên tục, không có đoạn nào chấm được |
| `trainee_ready_03` | 36° | 2 | 41 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **LOẠI** | LOẠI (trùng lỗi, chỉ 1,9 s được chấm) |
| `trainee_ready_04` | 22° | 1 | 37 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **CHỌN** | CHỌN: lỗi 'đứng quá thẳng'; góc quay sát chuẩn nhất (lệch 1°), ổn định, 1 đoạn chấm 2,7 s |
| `trainee_ready_05` | 31° | 2 | 38 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **LOẠI** | LOẠI (trùng lỗi, chân hơi hẹp nhưng nhẹ hơn 08) |
| `trainee_ready_06` | 36° | 1 | 33 | Góc gối cao; Độ nghiêng thân thấp; Góc hông cao | **LOẠI** | LOẠI (trùng lỗi, chỉ 1,9 s được chấm) |
| `trainee_ready_07` | 39° | 2 | 34 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **LOẠI** | LOẠI (trùng lỗi, góc quay lệch 17°) |
| `trainee_ready_08` | 31° | 3 | 30 | Độ nghiêng thân thấp; Góc gối cao; Góc hông cao | **CHỌN** | CHỌN: lỗi 'đứng quá thẳng' NẶNG NHẤT + chân quá hẹp (1,07 vai); 99% khớp, 4,6 s được chấm |

### Bài serve
| Video | Góc quay | Đoạn chấm | Điểm | Lỗi chính (do hệ thống đo) | Quyết định | Lý do |
|---|---|---|---|---|---|---|
| `coach_serve_01` | 60° | 1 | 68 | Độ cao cổ tay so với hông thấp; Góc hông thấp; Góc nâng tay thấp | **LOẠI** | LOẠI: pha đánh trong trận, không phải phát cầu chuẩn |
| `coach_serve_02` | 11° | 1 | 96* | — | **CHỌN** | CHỌN làm chuẩn: phát cầu đủ 4 pha có bước chuẩn bị đứng yên, 97% khớp thấy rõ, tay vợt đo được |
| `coach_serve_03` | 6° | 1 | 96 | Độ nghiêng thân thấp; Độ cao cổ tay so với hông thấp | **LOẠI** | LOẠI: cùng HLV và góc quay với coach_serve_02 (trùng, không thêm thông tin), thiếu pha rút vợt, 93% khớp |
| `coach_serve_04` | 30° | 4 | 69 | Độ cao cổ tay so với hông thấp; Góc khuỷu tay cao; Góc nâng tay thấp | **LOẠI** | LOẠI: video dài 16 s nhiều cú, tay vợt chưa xác thực, quay xa |
| `coach_serve_05` | 81° | 1 | 83 | Độ cao cổ tay so với hông thấp; Góc nâng tay thấp; Độ cao cổ tay so với hông cao | **LOẠI** | LOẠI: quay ngang (~81°), tay vợt bị che |
| `coach_serve_06` | 77° | 1 | 81 | Độ cao cổ tay so với hông thấp; Góc nâng tay thấp; Góc gối cao | **LOẠI** | LOẠI: quay ngang (~77°), tay vợt bị che |
| `coach_serve_07` | 24° | 1 | 96* | — | **CHỌN** | CHỌN làm chuẩn: cùng bối cảnh quay với người tập (lệch ~24°), 100% khớp thấy rõ, tay vợt khai báo; clip cắt sát nên chỉ dùng 3 pha (bỏ 'Chuẩn bị') |
| `trainee_serve_01` | 20° | 1 | 65 | Độ cao cổ tay so với hông cao; Góc nâng tay cao; Góc khuỷu tay cao | **CHỌN** | CHỌN - ví dụ lỗi NHẤC GÓT do bạn chỉ định. Hệ thống CHƯA bắt được (chỉ số 0,16 lẫn với HLV 0,10–0,20): giữ để bạn chỉnh ngưỡng `faults.heel_lift`; cũng thuộc nhóm A (tay cao) |
| `trainee_serve_02` | 30° | 2 | 71 | Nhấc gót chân phải khi phát cao; Độ cao cổ tay so với hông thấp; Độ nghiêng thân cao | **LOẠI** | LOẠI (trùng nhóm B; 77% góc đo được, clip cắt sát) |
| `trainee_serve_03` | 48° | 2 | 77 | Góc khuỷu tay thấp; Độ cao cổ tay so với hông cao; Vung theo sau cú chạm cao | **LOẠI** | LOẠI: nhóm D 'khuỷu gập' nhưng góc quay lệch 30°, 86% góc đo được, clip cắt sát |
| `trainee_serve_04` | 20° | 1 | 91 | Vung theo sau cú chạm cao; Độ cao cổ tay so với hông cao; Góc hông thấp | **LOẠI** | LOẠI: hệ thống chỉ bắt được lỗi rất nhẹ (95 điểm) - không dùng làm ví dụ lỗi cơ bản |
| `trainee_serve_05` | 37° | 1 | 48 | Độ nghiêng thân cao; Độ cao cổ tay so với hông thấp; Góc hông thấp | **CHỌN** | CHỌN - nhóm lỗi B 'cổ tay quá thấp + gập thân': nặng và rõ nhất. Lưu ý: người tập thuận TAY TRÁI, clip cắt sát (không có pha Chuẩn bị) |
| `trainee_serve_06` | 40° | 1 | 69 | Độ cao cổ tay so với hông thấp; Độ nghiêng thân cao; Góc nâng tay thấp | **LOẠI** | LOẠI (trùng nhóm B, nhẹ hơn 05) |
| `trainee_serve_07` | 23° | 1 | 89 | Độ nghiêng thân cao | **CHỌN** | CHỌN - nhóm lỗi C 'thân đổ ra trước' (lỗi nhẹ, riêng biệt); 100% khớp, góc quay lệch 5° |
| `trainee_serve_08` | 18° | 1 | 59 | Độ cao cổ tay so với hông cao; Vung theo sau cú chạm cao; Góc nâng tay cao | **LOẠI** | LOẠI (trùng nhóm A) |
| `trainee_serve_09` | 17° | 1 | 59 | Độ cao cổ tay so với hông cao; Vung theo sau cú chạm cao; Góc nâng tay cao | **CHỌN** | CHỌN - nhóm lỗi A 'tay/cổ tay quá cao': góc quay lệch 0°, 100% khớp, tin cậy cao nhất |
| `trainee_serve_10` | 55° | 1 | 59 | Vung theo sau cú chạm cao; Độ cao cổ tay so với hông cao; Góc nâng tay cao | **LOẠI** | LOẠI: góc quay lệch 38° so với chuẩn, độ tin cậy thấp nhất (0,81) |

## 4. Việc kỹ thuật phát sinh khi lọc

- **Tay vợt:** tự động chọn nhầm tay ở `trainee_serve_02` và `trainee_serve_03` (chọn tay trái, vợt ở tay phải). Đã xác nhận bằng hình và khai báo `racket_hand` trong `Input/<video>.json`: `trainee_serve_05`, `06` thuận **tay trái**, các video còn lại thuận **tay phải**.
- **Bộ tách cú phát:** trước khi sửa chỉ **8/17** video serve tách được ít nhất một lần phát; sau khi sửa **17/17** (xem `docs/METHOD.md` mục 5).
