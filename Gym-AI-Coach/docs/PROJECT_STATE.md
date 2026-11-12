# Trạng thái project Gym

Ngày cập nhật: 2026-11-06 (Asia/Saigon). Clean restart; không kế thừa kết quả legacy.

## Trạng thái hiện hành

**FIXED-ORIGIN USER TRAINING VERIFIED / ORIGIN-SCALE CHAIN CLOSED / COHORT33 AUDITED.**
Đã đối chiếu run `user-b11ae5d244df4ebd985123f2a106739c` và comparison
`comparison-4acd5fdf4d104f6c9037e55d4f855bda`: sáu endpoint499 đúng identity/hash,
metrics từ saved predictions tái tính exact; reuse review3.000logs và132train/66held
replay đã khóa, không forward/training mới. Hai seed20261026/27: raw OOF
0,797626/0,682511; body0,604131/0,704033; fixed0,744501/0,595373;
duration0,552775. Fixed−body ngược hướng giữa seed; không chọn seed/model thắng.
Duration pooled OOF thấp hơn cả sáu GRU, không chứng minh hiểu động tác/form.

Audit numerical đủ33video/12groups,3728complete/3848frames; giữ120incomplete.
Đã xem51contact sheets chứa đủ3848frame liên tiếp và33overlay board/208frame
trọng điểm, thêm supplement g19 cũ. Không playback, không inference pose.
33label lineage khớp;6count đối chiếu ngữ nghĩa ở mức AI,27chưa xác minh lại
exact count;0independent/human labels mới. Ledger7confirmed (4pose,1framing,
2record differences),2suspected,3unresolved; không sửa pose/count/cache.
145tests pass (30mới),0fail/error/skipped/warnings reported; không full suite.

Ưu tiên duy nhất tiếp theo (đề xuất, chưa triển khai): sidecar reliability có
version cho các interval pose confirmed, dry-run tác động support/reset trên
cùng33clip. Không tự swap/interpolate, loại video hay đổi nhãn/gate; cần chốt
phương pháp trước khi dùng bản dữ liệu nghiên cứu mới. Chưa cần training mới.
[Báo cáo và evidence](../outputs/squat_representation_audit/run-fea39d75e8a84750805d31f94475a0ca/REPORT.md)
· [Audit33](../outputs/squat_representation_audit/run-fea39d75e8a84750805d31f94475a0ca/AUDIT_33.md).
Giữ NOT_VALIDATED, official_eligibility=false, assessment=false, G5.7 blocked,
G6 IN_PROGRESS. AI weak reference/adaptive development, không validation độc lập.

### Lịch sử — chuẩn bị fixed-origin trước user training (trạng thái pending bên dưới đã được thay thế)

**FIXED VIDEO HIP ORIGIN GRU500 PREPARED / USER TRAINING PENDING.** Chủ project
đã chốt thí nghiệm chỉ đổi H(t) sang H0=(median Hx,median Hy) toàn video;
scale giữ exact parent body-normalized. Giữ33clip/12groups, masks/gaps/time,
raw58, hai seed20261026/27, GRU16/SmoothL1/optimizer/sampler,500epochs/index499.
Condition/manifest/checkpoint family riêng; không sửa nguồn frozen hoặc pose/nhãn.

Đã xác minh3.470source hashes và evidence cha/review, audit đủ33clip:
3.728complete/3.848frames,120incomplete giữ nguyên. Independent geometry:
max new−old relation error3,0531e−16;198untrained forward-only hữu hạn,
same-seed initialization exact, weights không đổi.337tests liên quan pass,
0fail/error/skipped/warnings; không full suite. Verify manifest sau serialize đạt.
Không training/backward/step/fit, không chạy lại pose hoặc visual audit.

Đã bàn giao launcher một lệnh: verify→sáu fold tuần tự→aggregate/replay/report.
Resume kiểm latest/identity/hash, dùng attempt mới, giữ lineage; latest corrupt
không fallback, endpoint duplicate dừng. Orchestration đã mock-test; PowerShell
parse/hash-check đạt, **chưa chạy script training thật**. Aggregate so với body
cùng seed chính, raw/duration/constant bổ sung, equal-video AE và fold1g17/g19,
fold0g11/g20; không chọn seed/epoch. Chưa có kết quả condition mới.
[START_HERE và lệnh chạy/tiếp tục](../outputs/squat_count_fixed_origin/prepare-ade63d5e03c2453381bc53817286cc2d/START_HERE.md) ·
[Report/preflight/tests](../outputs/squat_count_fixed_origin/prepare-ade63d5e03c2453381bc53817286cc2d/REPORT.md).

Camera drift và pose issue cũ vẫn còn; whole-clip median chưa streaming. AI weak
labels, adaptive development; NOT_VALIDATED, official_eligibility=false,
assessment=false, G5.7 blocked, G6 IN_PROGRESS. Chủ project tự khởi chạy training.

### Lịch sử — chẩn đoán fold1 trước khi chốt fixed-origin

**BODY-NORMALIZED USER TRAINING COMPLETE / FOLD1 DIAGNOSIS COMPLETE.** Chủ project
đã chạy đủ hai seed × ba fold, 500 epochs. OOF33 raw→body: seed20261026
0,797626→0,604131; seed20261027 0,682511→0,704033. Cải thiện không nhất quán.
Fold0 cải thiện cả hai seed (1,331955→0,480026; 0,812585→0,507461), fold1
xấu đi (0,433094→0,827300; 0,593315→0,986196). Duration OOF33 vẫn0,552775.

Đã kiểm sáu endpoint499/3.000epoch logs, recompute saved comparison chính xác;
replay66/66 body OOF và26/26 raw fold1 OOF exact, weights không đổi. Geometry
độc lập33/33, float32 tensor exact; 198 input hashes đúng. Chưa thấy bug
normalization/checkpoint. 3.432 source files không đổi; không training/fit mới.
269 tests liên quan pass,0fail/error/skipped/warnings; không full suite.

g19 đóng góp78,05%/63,03% mức tăng MAE ròng fold1; hai clip có scale dưới range
train. Đã xem571 ảnh liên tiếp của4clip g17/g19 qua contact sheets (không playback
liên tục/count validation). g19_c02 có named-joint L/R inconsistency tại f90/93
trong pose lưu dù observed/complete; g19_c01 partial đầu A/B lệch1frame. Không
sửa pose/nhãn; chưa chứng minh quan hệ nhân quả. Centering và scale bị thay cùng
lúc, chưa tách được ảnh hưởng. Train/held summaries và đủ13video nằm trong báo cáo.

Đề xuất duy nhất, **chưa triển khai/chưa phê duyệt training**: thay origin từng
frame bằng median hip origin cố định theo video, giữ nguyên torso scale và mọi
điều kiện còn lại, để kiểm tra phần chuyển động hip bị loại. Nếu chốt, chủ project
tự training; không tự thêm epochs/chọn seed/đổi gate.
[Báo cáo, tables, visual scope và lệnh tái lập](../outputs/squat_body_normalized_fold1_diagnosis/run-c18372526a494a48868f3c57f9771cfc/REPORT.md).
AI weak labels, adaptive development; NOT_VALIDATED, official_eligibility=false,
assessment=false, G5.7 blocked, G6 IN_PROGRESS. Không có independent validation.

### Lịch sử — chuẩn bị body-normalized trước khi chủ project training

**BODY-NORMALIZED GRU500 PREPARED / USER TRAINING PENDING.** Khóa một thay đổi:
x/y complete lấy tâm hông từng frame, chia median chiều dài vai–hông của chính
video; epsilon số học 1e-8 chiều cao ảnh. Giữ nguyên frame incomplete, mask/reset,
các kênh khác, 33 video/12 nhóm, folds, hai seed, GRU16/SmoothL1 và endpoint499.
Module/manifest/checkpoint family riêng, không sửa mã hoặc artifact đã đóng băng.

Đã đối chiếu 381 file nguồn/đối chứng và rerun frame-order đã lưu. Đủ33 video có
scale hợp lệ: 3.728/3.848 frame complete; scale min/median/max
0,135489/0,212579/0,302224 chiều cao ảnh. 198 forward-only qua hai seed/các fold
cho loss hữu hạn, khớp raw initialization và weights không đổi. Verify đọc lại
manifest thành công. 540 test liên quan đạt (92 mới), 0 fail/error/skipped/warnings;
không phải full suite. Không training/backward/step/fitting trong lượt chuẩn bị.

Đã chuẩn bị aggregate đủ sáu checkpoint499, same-seed raw58 và saved baselines;
train mean2AE/video rồi mean33, OOF33, fold/group/count, g11/g20 và tổng hợp hai
seed theo cùng video. Chưa có kết quả body-normalized hoặc kết luận cải thiện.
Chủ project tự chạy tuần tự theo
[START_HERE](../outputs/squat_count_body_normalized/prepare-5b23ef3e8ed54cc5bacfe226db03f2c6/START_HERE.md) ·
[Báo cáo/preflight/tests](../outputs/squat_count_body_normalized/prepare-5b23ef3e8ed54cc5bacfe226db03f2c6/REPORT.md).
Nhãn AI weak, OOF đã định hướng thiết kế: adaptive development, chưa validation.
Scale toàn clip chưa streaming. NOT_VALIDATED, official_eligibility=false,
assessment=false, G5.7 blocked, G6 IN_PROGRESS.

### Lịch sử — hai seed raw58 và phép thử thứ tự frame

**GRU500 HAI SEED ĐÃ TRAIN / FRAME-ORDER SENSITIVITY COMPLETE.** Chủ project
đã hoàn tất seed 20261027. Tại index 499, train MAE trên 33 video là 0,138745
và pooled OOF MAE là 0,682511. Đối chứng seed 20261026 có OOF MAE 0,797626,
duration đã lưu có OOF MAE 0,552775. Fold 0 của seed 20261027 có train/held
MAE 0,129562/0,812585, signed error g11 +0,412295 và g20 −1,571337 trên
video gốc. [Comparison của chủ project](../outputs/squat_count_inner_cv_500_seed/user-cbac509784bc438ca22746fb802f7659/comparison-765a0e5322df4f96b65d9120285fc092/REPORT.md).

Đã tái lập chính xác 66/66 OOF và 66/66 identity. Hai model dùng cùng 20
phép xáo/video trong các đoạn complete của 33 video. Mean |Δ| tính đều theo
video là **0,251140 / 0,354867 lần squat** (seed 20261026 / 20261027). Mean Δ
có dấu là −0,071309 / −0,331456. Trong fold 0, mean |Δ| g11 là 0,070231 /
0,086442 (3 video). Với g20 là 0,242394 / 0,157046 (4 video), còn mean Δ
đổi hướng từ +0,201826 sang −0,157046. Hash của sáu bộ weights giữ nguyên.
[Báo cáo, bảng fold 0 và bằng chứng](../outputs/squat_frame_order_sensitivity/run-9f0b2c4b3383435985f0f40cf330047d/REPORT.md).

Các Δ đo thay đổi đầu ra trên dữ liệu đã xáo. Chỉ video gốc có nhãn AI weak
reference để tính sai số. Đây là adaptive development, chưa xác thực sự kiện
squat, nguyên nhân lỗi hay kỹ thuật động tác. NOT_VALIDATED,
official_eligibility=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS.

### Lịch sử — bàn giao seed 20261027 trước khi chủ project chạy

**GRU500 SEED20261027 PREPARED / USER TRAINING PENDING.** Đã khóa đúng một lần lặp
GRU500 với seed mới, cùng raw58, nhãn, 33 video/12 nhóm và ba fold. Index 499 là
endpoint chính, index 49 chỉ mô tả. Có lệnh chạy mới, resume và so sánh cùng video
với GRU500 seed 20261026, duration và constant đã lưu.

Đã đối chiếu 5.001 file nguồn/đối chứng, 99/99 hash đầu vào theo fold và khởi tạo
từng fold. Forward-only cho loss hữu hạn và giữ nguyên hash weights. 678 test liên
quan đạt, gồm 67 test mới, 0 fail/error/skipped/warnings ở lượt cuối. Chưa có
checkpoint đã học hoặc MAE cho seed 20261027. Chủ project tự chạy training.
[Hướng dẫn PowerShell](../outputs/squat_count_inner_cv_500_seed/prepare-a6c1b5ba79484854bbf0e419e15613d1/START_HERE.md) · [Báo cáo và bằng chứng](../outputs/squat_count_inner_cv_500_seed/prepare-a6c1b5ba79484854bbf0e419e15613d1/REPORT.md).

Ba quan sát sẽ báo riêng là chênh MAE held–train fold 0, signed error g11 và g20.
Hai seed dùng cùng cohort và nhãn AI weak reference. OOF đã được xem. Giữ
`NOT_VALIDATED`, official_eligibility=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS.

### Lịch sử — chẩn đoán seed 20261026 trước khi chốt lần lặp

**2026-10-30 — GRU500 INNER-CV COMPLETE / VERIFIED; FOLD0 DIAGNOSIS COMPLETE
với giới hạn visual lấy mẫu.** User đã training run
`outputs/squat_count_inner_cv_500/user-298d9a0a7e08444cb423d973c2a9ca05`;
comparison `comparison-d74d4c9bbe90404b9f2f0bd8549137b5` được đối chiếu với manifest,
review, 1.500 checkpoint byte hashes/1.500 epoch logs và saved predictions.
50epoch đầu cả3fold EXACT_MATCH trên các trường đã lưu và weights index49.
Train MAE33 **0,888301→0,198856**, pooled OOF33 **1,067322→0,797626**;
duration OOF0,552775 vẫn thấp hơn. Không chọn checkpoint/epoch mới.

Fold0: train20/8nhóm MAE0,954883→0,233756; held13/4nhóm1,150920→1,331955,
9clip xấu đi/4cải thiện. Duration tốt hơn trên12/13. g11 chiếm37,51% tổng AE;
g20 chiếm30,85% và cả532frame complete. Chưa xác nhận bug gây lỗi hoặc nguyên nhân
duy nhất; không thể giải thích tất cả bằng thiếu pose/reset. Raw→cache exact33clip;
decode13clip/1.255frame, xem312frame mẫu qua13contact sheets, **không playback liên tục**.
Nhãn A/B và reconciliation được giữ nguyên, không là human/independent ground truth.
Đã ghi hạn chế subject-selection, partial/hold và POS_MSEC của3clipg05 trong báo cáo.

[Báo cáo và bảng đủ13clip](../outputs/squat_cv500_fold0_diagnosis/run-1d407efc0b7144898feafe832d7c9bf6/REPORT.md),
[analysis/hash evidence](../outputs/squat_cv500_fold0_diagnosis/run-1d407efc0b7144898feafe832d7c9bf6/analysis-v1/verification.json).
252tests pass (21mới+231regression),0failed/error/skipped/warnings ở lượt cuối;
chưa full suite. Lượt unit đầu có1assertion message mismatch đã sửa kỳ vọngtest.
0model forward/pose inference/training/backward/step/fit trong chẩn đoán này.

**Một đề xuất chờ chốt:** lặp đúng GRU500 với seed20261027, giữ cohort/folds,
representation/loss/hyperparameters/budget và endpoint499; kiểm độ lặp lại mẫu lỗi
fold0 trước khi đổi cơ chế. Chưa triển khai/chuẩn bị/training condition mới; user tự
training sau khi chốt. AI weak reference, mộtseed hiện có, adaptive OOF development;
NOT_VALIDATED,official_eligibility=false,assessment=false,G5.7blocked,G6 IN_PROGRESS.

### Lịch sử chuẩn bị CV500 — đã user training; kết quả hiện hành ở trên

**2026-10-29 — TINY-SET1000 COMPLETE/CRITERION MET;
GRU 500-EPOCH INNER-CV PREPARED / USER TRAINING PENDING.**
Đã verify tiny1000 run `user-ef42d0274e4d4a7486176987838592ee`/report/review thật:
1000checkpoint hashes/history,6000records,report replay khớp. Epoch1000 MAE0.0165504751,
maxAE0.0872395262,20/20final epochs đạt;500epoch đầu EXACT_MATCH. Chỉ học khớp6train
clips, không suy đã học chuyển động đúng hoặc CV OOF chắc chắn cải thiện.

Condition riêng `counting/inner_cv_500` giữ33clips/12groups/folds20–13,20–13,26–7,
raw58/GRU/SmoothL1/seed/init/optimizer/masks/sampling/reset/pooling; chỉ config epochs50→500.
Fresh mỗi fold,1500/1500/2000steps; endpoints49/499, không best/earlystop/refit.
Resume identity/fold/config/hash riêng. Aggregate train mean2AE/video→mean33 và OOF33,
macro12groups/fold/count/signed/distribution/deltas với saved GRU50/duration/constant,
không baselinefit. Tái lập so exact50epoch shared fields/weights49; original không log
realized batchorder/gradient-update/train-prediction history nên không claim đã so trực tiếp.
Mismatch lưu evidence+exit3; không nới tolerance hoặc tự retrain.
590tests pass(63mới+527regression),0failed/error/skipped/warnings; chưa full suite.
Prepare/verify/6forward-only batches finite,input33khớp/init-weights không đổi;
6936file cũ bảo toàn, old verifiers/checkpoints vẫn dùng được. Không train/backward/step/fit.
[START_HERE — user commands](../outputs/squat_count_inner_cv_500/prepare-d37d29a45ee64f6d83f1f01c33ee30c4/START_HERE.md),
[báo cáo/evidence](../outputs/squat_count_inner_cv_500/prepare-d37d29a45ee64f6d83f1f01c33ee30c4/REPORT.md).
Manifest `frozen-v1/manifest.json`, canonical SHA-256
`50cfa5e095ab5688303362a8d853df647ad416492ac7e9fbe759190d5d94bbee`.
**MODEL TRAINING GATE: READY FOR USER-RUN CV500; RESULT PENDING.** User tự chạy3fold,
resume đúngcondition khi cần rồi gửi cả run.500CVepochs không tương đương500/1000tinysteps.
AI weak reference,mộtseed,adaptive OOF development; chưa independent final evaluation/
generalization/form validation. NOT_VALIDATED,official=false,assessment=false,G5.7blocked,G6 IN_PROGRESS.

### Lịch sử chuẩn bị tiny-set1000 (đã user training; kết quả hiện hành ở trên)

**2026-10-29 — TINY-SET500 DIAGNOSIS COMPLETE/CRITERION NOT MET;
TINY-SET 1000 PREPARED / USER TRAINING PENDING.**
Đã verify user500 run `user-5eeeff51f5e94cc6be52c8f8146c073d`, report và review:
500checkpoint byte hashes/history,3000video-epoch records, receipt/metrics/report khớp.
Epoch500 MAE0.14228733877340952,maxAE0.29584598541259766,0/20final epochs đạt đồng thời;
gradients hữu hạn/đến tham số,500steps có thay đổiweights,217/500steps norm>1.
Giữ nguyên kết luận cũ “chưa học khớp trong thiết lập và ngân sách này”; không quy nguyên nhân.

Condition mới `counting/tiny_set_1000`, chỉ tăng budget1000/1000steps; copy nguyên6IDs/
target/source-cache-label hashes/order từ manifest500, không chọn lại. Fresh init GRU/raw58/
SmoothL1/seed/AdamW/reset/pooling/sampling/setting8/actual6 giữ nguyên; không lấy weights500.
Criterion indices980..999, MAE≤0.10/maxAE≤0.25 đồng thời20/20; report49/499/999.
Identity/resume riêng; report so exact500epoch đầu và checkpoint499 thật với lượt500 cũ,
bỏ condition metadata/elapsed time; mismatch lưu evidence+exit3 để điều tra, không nới tolerance.
527tests pass(64mới+463regression),0failed/error/skipped/warnings ở lượt cuối; fixture fail
ban đầu đã sửa và giữ log. Chưa full suite/training1000. Forward-only6clip finite,init/weights
không đổi và khớp receipt chuẩn bị500;3887file cũ bảo toàn, old verifier/checkpoint vẫn đọc được.
[START_HERE](../outputs/squat_count_tiny_set_1000/prepare-b520074a9e884dcfab0635fa664b0130/START_HERE.md),
[báo cáo và evidence](../outputs/squat_count_tiny_set_1000/prepare-b520074a9e884dcfab0635fa664b0130/REPORT.md).
Manifest `frozen-v1/manifest.json`, canonical SHA-256
`290401a92c090f51c582113a150af53c7b77b66e17d48aa04aeb24f0168526e0`.
**MODEL TRAINING GATE: READY FOR USER-RUN diagnostic1000 đã duyệt; RESULT PENDING.**
Agent không train/backward/step/fit; user tự chạy từ đầu, resume chỉ cùng1000condition.
AI weak reference,mộtseed,adaptive development6train clips; không independent accuracy,
generalization/form validation. NOT_VALIDATED,official=false,assessment=false,G5.7blocked,G6 IN_PROGRESS.

### Lịch sử chuẩn bị tiny-set500 (đã user training; kết quả hiện hành ở trên)

**2026-10-29 — MSE COMPLETE/VERIFIED; TINY-SET DIAGNOSTIC PREPARED / USER TRAINING PENDING.**
Đã đối chiếu MSE user run `user-e18343207eb342f0967b52e3a31880f4`, comparison và
review thực tế:3final checkpoints,150epoch reports/hashes, saved predictions và metrics.
MSE train/OOF MAE0.8780918483/1.0737655380; SmoothL1 0.8883011359/1.0673224962;
duration0.4851842589/0.5527751946. Train mean2AE/video→mean33, không66mẫu độc lập.
Không kết luận MSE giải quyết vấn đề hoặc loss là nguyên nhân; không inference/fit mới
để xác minh MSE. [Bằng chứng](../outputs/squat_count_tiny_set/prepare-1e3e818d5699431eba32c0860622b1ce/mse-verification.json).

Nhánh riêng `counting/tiny_set` chọn đúng6video count0–5 từ train fold1(20clips),
sort sourceSHA256 rồi video_id,4groups, không chọn từ prediction/error/held-out.
Raw58/GRU16/SmoothL1/seed/AdamW/mask/reset/pooling/init giữ nguyên;500epochs,
batch setting8/actual6,1step/epoch; eval cùng6video sau mỗi epoch. Khóa MAE≤0.10
và maxAE≤0.25 đồng thời trong20epoch cuối(indices480..499), report riêng49/499;
không best selection/early stop. Không phải single-factor comparison với CV33video.
Identity/checkpoint/resume riêng, output mới; gradient/update logging thật chờ user chạy.
463tests pass(79mới+384regression),0failed/error/skipped/warnings; chưa full suite.
Prepare/verify/forward-only exit0, `[6,201,58]`→`[6]` finite, initial/reference/after weights
hash khớp;2341file cũ bảo toàn.0training/backward/step/fit; không sửa `KLTN_AI_COACH`.
[START_HERE — lệnh verify/user-run/resume/report](../outputs/squat_count_tiny_set/prepare-1e3e818d5699431eba32c0860622b1ce/START_HERE.md),
[báo cáo](../outputs/squat_count_tiny_set/prepare-1e3e818d5699431eba32c0860622b1ce/REPORT.md).
Tiny manifest `frozen-v1/manifest.json`, canonical SHA-256
`8aa58a66cb0c18a93cd2018b99a9345c9d66971af7a8052df9630468e62769d4`.
**MODEL TRAINING GATE: READY FOR USER-RUN diagnostic đã duyệt; tiny [RESULT PENDING].**
User tự training và gửi cả run; không mở thí nghiệm khác. Học khớp6video không chứng minh
accuracy/generalization hoặc học chuyển động đúng. AI weak reference,mộtseed,
development research; NOT_VALIDATED,official=false,assessment=false,G5.7blocked,G6 IN_PROGRESS.

### Lịch sử chuẩn bị MSE ngày2026-10-28 (đã được user training; xem kết quả hiện hành ở trên)

**2026-10-28 — MSE PREPARED / USER TRAINING PENDING:**
Chủ project đã duyệt đúng một thay đổi SmoothL1→MSE. Nhánh riêng `counting/mse`
dùng nguyên GRU/raw58/33clips/12groups/folds20–13,20–13,26–7; config chỉ khác loss.
Giữ masks/admission/normalization/reset/pooling/init/seed/AdamW/batch/gradclip/budget50;
endpoint49, không chọn lại epoch, không fine-tune SmoothL1 hoặc dùng zero/velocity inputs.
Manifest/checkpoint loss/config/fold identity riêng; resume chỉ cùng MSE condition,
output mới; evaluator tính train mean2AE/video→mean33 và pooled OOF33, count/fold/
signed errors/distribution/deltas với saved controls, không fit baseline lại.
[START_HERE — lệnh verify/train/resume/aggregate](../outputs/squat_count_mse/prepare-192696df332e402aa32f93b70987b2f1/START_HERE.md),
[báo cáo thiết kế và kiểm tra](../outputs/squat_count_mse/prepare-192696df332e402aa32f93b70987b2f1/REPORT.md).
Chỉ dùng `frozen-v2/manifest.json`, canonical SHA-256
`f3fd84a36e5b47f0bfcd112973e25c2d39e46bb06616ad1647e6200297d66c2f`.
384tests pass (56mới+328regression),0fail/error/skip/warnings; chưa full suite.
Verify nguồn/hashes/control replay đạt; raw input33clips khớp;6forward-only batches
finite,weights không đổi,initial hash khớp đối chứng.1829file cũ bảo toàn.
**MSE [RESULT PENDING]; agent không training/backward/optimizer step/fit.**
MODEL TRAINING GATE: READY FOR USER-RUN thí nghiệm đã duyệt; user tự chạy ba fold.
Chưa thực nghiệm training/resume có updates hoặc aggregate MSE thật; không model promotion.
Không chỉnh sửa/tích hợp `KLTN_AI_COACH`, không dependency mới hoặc UI.
AI weak reference,mộtseed,OOF dùng thiết kế: development research,không independent final
evaluation/form validation. NOT_VALIDATED,official=false,assessment=false,G5.7blocked,G6 IN_PROGRESS.

### Lịch sử chẩn đoán trước khi duyệt MSE

**2026-10-28 — VELOCITY COMPLETE/VERIFIED; TRAIN CAPACITY DIAGNOSIS COMPLETE:**
Đã xác minh ba user runs GRU/zero-x/y/velocity, manifest tìm từ receipt,9checkpoint
cuối index49 và450epoch reports/checkpoint hashes; không tạo lại split/config.
Trên cùng33video/12groups, pooled OOF MAE GRU1.0673224962,zero1.0702116851,
velocity1.0505837058,constant1.3030303030,duration0.5527751946.
198forward-only trên đúng train20/20/26 từng condition; eval/inference_mode,
9/9model hashes trước/sau không đổi,0held-out inference,0training/fit/optimizer restore.
Train MAE đều theo33video (mỗi video trung bình2AE, không66mẫu độc lập):
GRU0.888301,zero0.912936,velocity0.895093; duration saved coefficients0.485184.
Cả ba còn prediction co hẹp ngay trên train; count3–5 đều thiếu,0–1 đều thừa.
Gap không đồng nhất: fold1 held MAE thấp hơn train,fold2 cao nhất và shift count.
Không quy nguyên nhân cho pooling/model size/pose/epochs; online SmoothL1 không là MAE.
[Báo cáo, biểu đồ và một đề xuất loss-only](../outputs/squat_count_train_diagnosis/run-ccce5bdd15454961a94855e8061f75b9/REPORT.md),
[bảng495prediction/error records](../outputs/squat_count_train_diagnosis/run-ccce5bdd15454961a94855e8061f75b9/analysis-v1/PER_VIDEO_ERRORS.md).
Đề xuất tiếp theo **CHƯA DUYỆT/CHƯA TRIỂN KHAI/CHƯA TRAIN**: chỉ đổi SmoothL1→MSE
trên GRU gốc để kiểm sensitivity của fit count cực trị, giữ model/input/folds/seed/budget50.
Không mặc định loss gây lỗi hoặc MSE tốt hơn; user tự chạy mọi training sau khi chốt.
328tests pass (37mới+291regression),0fail/error/skip/warnings; chưa full repository suite.
1800file cũ bảo toàn; không Git repository. Tests không phải model accuracy.
AI weak reference,mộtseed,OOF đã dùng chọn thiết kế: development research,
không independent final evaluation/causality. Không dùng outer validation/development-test.
NOT_VALIDATED,official_eligibility=false,assessment=false,G5.7blocked,G6/G5 IN_PROGRESS.

### Lịch sử chuẩn bị velocity trước khi người dùng training

**2026-10-28 — ZERO-XY COMPLETE/VERIFIED; VELOCITY PREPARED / USER TRAINING PENDING:**
User đã hoàn tất zero-x/y tại `squat_count_xy_ablation/user-b1e4f75f31e4424fafc116d6a82a7ba8/comparison`.
Replay3checkpoint histories/hashes/saved controls khớp: pooledMAE zero-x/y1.0702116851,
GRU1.0673224962,constant1.3030303030,duration0.5527751946,33clips/12groups,index49.
Chênh nhỏ không chứng minh equivalence hoặc pose vô dụng.
Đã thêm nhánh riêng `counting/velocity`: thay16x/y bằng backward difference theo dt,
chiều cao ảnh/giây; cùng wrapper train/eval, không nối gap/future, zero ở biên giữmask.
Giữ58channels/config/GRU16/folds/sampler/init/reset/pooling,endpoint50/index49;
checkpoint family/manifest/hash riêng, resume và compare đủ33với4savedcontrols.
Input audit đủ33clips/3848frames:3643computed,205boundary placeholders;
range components−6.770966..6.533810,finite float64/float32; không clipping/scaler/tune.
291tests pass (79mới+212regression),0fail/error/skip/warnings;6forward-only batches
finite/weights unchanged,initialhash khớp GRU vàzero-x/y.1304files cũ/21modules không đổi.
[Lệnh PowerShell](../outputs/squat_count_velocity/prepare-1b1c2fb957ac4264a1cff3afc04682ad/START_HERE.md),
[report/design/input audit](../outputs/squat_count_velocity/prepare-1b1c2fb957ac4264a1cff3afc04682ad/REPORT.md).
**Velocity [RESULT PENDING]; agent không train/fit/backward/step. User tự training.**
Chưa full suite hoặc training/resume có updates; tests không phải model accuracy.
Nhãn AI weak,mộtseed,OOF đã xem/dùng thiết kế: development research,không independent
final evaluation. Không dùng outervalidation/development-test tối ưu,không refit/UI.
NOT_VALIDATED,official_eligibility=false,assessment=false,G5.7blocked,G6/G5 IN_PROGRESS.

### Lịch sử chuẩn bị zero-x/y trước khi người dùng training

**2026-10-28 — XY ABLATION PREPARED / USER TRAINING PENDING:**
Chủ project đã duyệt đúng một ablation zero16 x/y channels ở cả train/evaluation.
Đã thêm nhánh riêng `counting/xy_ablation`, giữ58channels/mask/provenance/time/duration,
GRU16/config/sampler/init và33clips/12groups/folds20–13,20–13,26–7.
Khóa endpoint49/epoch50, không minimum search hoặc fold-best stitching.
Manifest/checkpoint identity riêng, resume đúngfold/transform/config vào output mới;
aggregate đủ33pairs với GRU/constant/duration đã lưu, không fit lại baselines.
[Lệnh PowerShell verify/train/resume/compare](../outputs/squat_count_xy_ablation/prepare-8ef17e4792064e25b3013ef5361f3228/START_HERE.md),
[report và evidence](../outputs/squat_count_xy_ablation/prepare-8ef17e4792064e25b3013ef5361f3228/REPORT.md).
212tests pass (65mới+147regression),0fail/error/skip/warnings; chưa full suite.
Verify hash/code/dataset/controls đạt;6forward-only batches hữu hạn,weights không đổi;
initial state khớp3runs GRU gốc.810files cũ không đổi;14frozenmodules giữ nguyên.
**Agent không training/fit/backward/step; ablation [RESULT PENDING]**. Training thật
do user tự chạy; tests/forward không phải model accuracy hay readiness form.
AI weak reference; OOF đã xem/dùng chọn thiết kế nên vẫn development research,
không đánh giá cuối độc lập. Không đổi outer split/gate; development-test không dùng.
NOT_VALIDATED,official_eligibility=false,assessment=false,G5.7blocked,G6/G5 IN_PROGRESS.

### Lịch sử chẩn đoán trước khi duyệt ablation

**2026-10-28 — INNER-CV BASELINE COMPARISON COMPLETE / DIAGNOSED:**
Run user `squat_count_cv_comparison/user-b092e0a261dc435a9ce60670c980602e`
đã hoàn tất, kiểm input/hashes/checkpoint OOF/saved coefficients và replay metrics
khớp. Trên33clips/12gXX tại index49/epoch50: GRU pooled MAE1,067322,
constant1,303030,duration0,552775. Duration thấp hơn trên25/33clip và cả3fold;
không là independent final evaluation vì OOF đã dùng chọn epoch.
GRU range1,108927–2,740910,SD0,340679 so targetSD1,321573; count3–5 đều đếm thiếu,
count0–2 đều đếm thừa ở giá trị continuous. Pooled correlation thấp nhưng within-fold
ranking còn; chưa kết luận model không sử dụng motion. Support100% vẫn có gap
MAE0,318494/19clip, không đủ evidence quy toàn bộ lỗi cho reset/pooling.
[Báo cáo chẩn đoán mới và đúng một proposal](../outputs/squat_count_cv_diagnosis/run-50d00b64021042a9845508fc09ea8344/REPORT.md),
[bảng đủ33video](../outputs/squat_count_cv_diagnosis/run-50d00b64021042a9845508fc09ea8344/analysis-v1/PER_VIDEO_ERRORS.md).
Đề xuất: train ablation bỏ16 x/y channels ở cả train/eval, giữ các channels khác,
GRU16/config/folds và endpoint50; CHƯA duyệt/triển khai/chạy, user tự training sau
khi chốt. Không tự đổi pooling/epoch/gate hoặc refit. Agent chỉ saved-data diagnosis,
không fit/forward/backward/step;147related tests pass,0fail/error/skip/warnings,
chưa full suite.797 files cũ không đổi; không Git repository.
AI weak reference, NOT_VALIDATED,official=false,assessment=false,G5.7blocked,
G6/G5 IN_PROGRESS. Development-test đã exposed; không dùng tối ưu lượt này.

### Lịch sử chuẩn bị trước user chạy comparison

**2026-10-27 — INNER-CV GRU OOF VERIFIED / BASELINE COMPARISON PREPARED:**
User đã chạy3folds và aggregate tại `oof-7a3f559a10a345958de7b3e72df2acda`.
Kiểm lại manifest/dataset/code/config và3 checkpoint histories: OOF khớp, đủ
50epochs ×33clips/12groups; selected index49=epoch50 giữ nguyên. GRU pooled
OOF MAE1,0673224962, macro12-group1,0498952053; chưa là so sánh baseline mới.
Thêm công cụ riêng `counting/cv_baseline_comparison.py`, không sửa13 frozen
research/inner-CV modules;779 files cũ giữ nguyên hashes. Dùng đúng constant
median/OLS duration công thức cũ, fit riêng20/20/26 train clips rồi predict13/13/7.
**Agent không fit baseline thật, train/backward/step hoặc final refit**; chỉ44
tests synthetic mới và regression174pass,0fail/error/skip/warnings; chưa full suite.
[Lệnh user-run và evidence](../outputs/squat_count_cv_comparison/prepare-b69b56e1835149a2811620b7153290e1/START_HERE.md).
Baseline/comparison thật **[RESULT PENDING]**, chưa kết luận GRU tốt hơn baseline.
AI weak labels; OOF đã chọn epoch không là đánh giá độc lập cuối cùng.
Không dùng outer validation/development-test; development-test đã exposed.
NOT_VALIDATED,official=false,assessment=false,G5.7blocked,G6/G5 IN_PROGRESS.

### Lịch sử bàn giao trước user chạy inner-CV

**2026-10-27 — INNER 3-FOLD PREPARED / USER TRAINING PENDING:**
User đã phê duyệt thí nghiệm selection stability. Đã thêm nhánh `counting/inner_cv`
riêng, không sửa8 modules research cũ. Manifest SHA-256
`8f98e626657b64c038d07a8085d64d452fc2715c6f18ec49c02995ee335352bf`
khóa33 supervised train clips/12gXX bằng SHA-256(group), round-robin3.
Fold0/1/2 học–đánh giá20–13,20–13,26–7clips; mỗi fold8–4groups;
không group/source-hash/component leakage. Outer split67/21/24 giữ nguyên;
outer validation/development-test không vào loss, selection hoặc OOF.
GRU16/input/admission/loss/seed/50epochs/hyperparameters giữ nguyên.
CLI user-run, checkpoint/resume riêng, pooled same-epoch OOF/33 và macro12groups,
earliest minimum/all50, index5 comparison và fold-best dispersion đã kiểm thử.
**208 related tests passed,0 fail/error/skip/warnings; chưa full suite.**
Forward-only6 batches trên input train đã lưu: hữu hạn/không grad/weights không đổi.
320 files cũ không đổi; checkpoint thí nghiệm đầu đọc được bằng loader cũ.
**MODEL TRAINING GATE: READY FOR USER-RUN INNER-CV** chỉ nghĩa đã chuẩn bị code/
manifest và tests không học. Agent không train/backward/step/fit baseline;
OOF thật/selection CV vẫn **[EXPERIMENT NOT RUN YET]**, chưa refit.
[Bàn giao lệnh, resume và evidence](../outputs/squat_count_inner_cv/run-985d93a330b6492b94f16ec8d2332221/START_HERE.md).
AI weak references, NOT_VALIDATED, official=false, assessment=false, G5.7 blocked,
G6/G5 IN_PROGRESS. Development-test đã exposed, không dùng tối ưu inner-CV;
minimum OOF dùng chọn epoch không là independent evaluation.

### Lịch sử trước khi user phê duyệt inner-CV

**2026-10-27 — USER EXPERIMENT 1 DIAGNOSED / NEXT DESIGN PENDING:**
User đã chạy baseline và GRU50epochs/250steps; đã có saved development-test results.
Checkpoint selection đúng index5 (epoch thứ6), validation MAE0,709626 trên3 clips/g02.
Saved test GRU MAE1,145287 vs duration0,533840 trên11/24 clips;11 predictions rounded1.
Agent chỉ chẩn đoán: verify291 oldfiles,50 checkpoint states, replay train/validation,
39 raw/cache inputs và OOD probes; không fit/train hoặc test inference mới.
Chưa thấy lỗi implementation trong đường đã kiểm. Admission/validation selection
bias có evidence; shortcut/pooling causality chưa xác lập. Không chọn lại checkpoint.
76 related tests passed,0 fail/error/skip/warnings; chưa full suite. Core model và
mọi artifact cũ giữ nguyên. [Báo cáo chẩn đoán](SQUAT_COUNT_FIRST_EXPERIMENT_DIAGNOSIS.md).
Đúng một đề xuất tiếp: grouped3-fold nội bộ train để kiểm selection stability,
giữ model/input; CHƯA triển khai/chạy, user cần chốt thiết kế và tự training sau đó.
Development-test đã exposed, không là bằng chứng độc lập cho thiết kế kế tiếp.
AI weak reference, NOT_VALIDATED, official=false, assessment=false, G5.7 blocked,
G6/G5 IN_PROGRESS. Các đoạn CHƯA TRAIN phía dưới mô tả thời điểm bàn giao trước run user.

**2026-10-26 — COUNT-ONLY AI-REFERENCE / READY FOR USER-RUN EXPERIMENT:**
Chủ project đã duyệt nhánh raw-pose riêng; triển khai loader/mask/admission, GRU16
3.666 parameters, loss, baseline constant/duration, evaluator AI-reference,
training CLI + checkpoint/resume. **Không train/fit/backward/optimizer step**.
112 source/split/pose/AI label linkage đã kiểm; admission khóa trước batch kết quả.
50 input accepted/62 rejected,0 error/unchecked;104 targets available/8 null.
Supervised: train33 clips/12gXX, validation3/1gXX(g02), development-test11/4gXX.
Mẫu số gốc67/21/24 clips và15/5/5groups giữ nguyên. Validation rất nhỏ, selection
bias và không independent GT; READY chỉ nghĩa runnable experiment do user tự chạy.
219 related tests passed,0 fail/error/skip/warnings; chưa full suite. Forward-only
trọng số khởi tạo trên3 input train hữu hạn, trọng số không đổi, không checkpoint.
1.395 files trong3 inventory cũ đã verify;71 pose-only+41 preprocessing snapshots
giữ nguyên, không inference mới. 33 AI viewpoint chỉ thuộc subset33, không documented.
[Thiết kế, kết quả admission và lệnh thật](SQUAT_COUNT_TRAINING_PLAN.md),
[bàn giao](../outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/START_HERE.md).
Training/baseline performance NOT_RUN; chưa runtime training. Official/form gate
không đổi: NOT_VALIDATED, official=false, assessment=false, G5.7 blocked, G6/G5
IN_PROGRESS. Các đoạn NOT_READY/admission-chưa-duyệt dưới đây là lịch sử trước
quyết định mới, không phủ nhận quyền và implementation count-only hiện hành.

**2026-10-26 — AI ANNOTATION REVIEW EXECUTED / TRAINING NOT_READY:**
Theo quyết định mới của chủ project, đã thực sự xem ảnh nguồn toàn timeline của
112 clips/14.356 frames qua 203 contact sheets bằng `view_image`, rồi mở lại toàn bộ
trong lượt B. Có 224 checkpoints AI riêng; cùng assistant/context, không blind hoặc
hai người độc lập. Pilot 5 clip đã kiểm tra trước khi mở rộng, không đổi split/cohort.
104/112 clip có whole-count nhất quán (tổng 252 trên riêng 104 clip); 8 clip null,
không còn pass chưa xử lý. 24 clip khác ít nhất một trường; g17_c03 khác observed
lower bound 1/2, giữ unresolved. Exact exhaustive boundaries vẫn unavailable 112/112.
33 AI viewpoint reviews: 19 front, 13 oblique, 1 unknown/mixed; 0 side, 0 documented.
Tất cả `visually_assessed_unverified`, raw camera UNKNOWN/QualityReports không đổi.
Human templates/lịch sử giữ nguyên: 0 human-validated/independent ground truth mới.
159 related tests passed, 0 failed/error/skipped/warnings; chưa full suite. Verify
112 source hashes, 224 passes và 818 files trong ba inventory cũ không mismatch.
Không inference, counting baseline hoặc training. Nhãn AI count-only không tạo được
event targets cho GRU đề xuất; real admission chưa được duyệt/triển khai. **MODEL
TRAINING GATE: NOT_READY**. User tự training khi có config/lệnh hợp lệ sau này.
[Kết quả, bất đồng và cách kiểm tra](SQUAT_AI_ANNOTATION_REVIEW_REPORT.md).
NOT_VALIDATED, official=false, assessment=false; G5.7 blocked, G6/toàn G5 IN_PROGRESS.
Các trạng thái pending bên dưới là lịch sử/human branch, không phủ nhận nhánh AI
đã thực hiện và không yêu cầu user bố trí người trong milestone hiện hành.

**2026-10-25 — VIEWPOINT_REVIEW_PENDING / TEMPORAL PILOT HANDED OFF:**
Đã trích bằng code đúng 33 strict clips có sequence allow và canonical left-knee
skip chỉ vì viewpoint_unknown; worksheet giữ pending, người/time/viewpoint=null.
Pilot độc lập gồm g01_c01, g05_c01, g06_c01, g08_c01, g09_c01: clip đầu theo path
của 5 gXX train đầu, không dùng quality/count để chọn. Không đổi cohort/split;
hai templates gốc vẫn đủ 112 records. Chưa có nhãn mới: 0 completed, 0 final,
0 cặp đủ điều kiện agreement; 33 viewpoint reviews và 10 pilot passes pending.
Đã verify hashes 661 files run feasibility, 130 files dataset-final và 112 sources;
không decode/xem video, inference, preprocessing, counting hoặc training mới.
24 tests mới pass; suite liên quan 113 pass, 0 fail/error/skip/warnings. Chưa full suite.
[Hướng dẫn bàn giao riêng cho reviewer và hai rater](../outputs/squat_review_pilot/run-b967622c974d428a8c96c3bf03363d99/handoff/START_HERE.md).
Enum viewpoint không xác thực evidence: cần duyệt rubric/người xác nhận/quy trình
chấp nhận trước documented status; count-only admission là quyết định riêng chưa duyệt.
112 pose/quality evidence gồm 41 snapshot có preprocessing + 71 pose/quality-only.
Training NOT_READY, agent không train; user tự chạy sau MODEL TRAINING GATE.
NOT_VALIDATED, official=false, assessment=false; G5.7 blocked, G6/toàn G5 IN_PROGRESS.
Các mục milestone bên dưới giữ số liệu tại thời điểm thực hiện, không là status mới.

**2026-10-25 — MISSING-COHORT POSE/QUALITY EXECUTED / COUNTING STILL BLOCKED:**
Verify đủ112 sources,41 compatible snapshots; freeze71 missing trước inference.
Smoke g01_c04 rồi70 còn lại,71 unique new inference/8.905 frames/0 processing errors;
41 reuse, không re-infer smoke. Fresh strict sequence32 allow/13 degrade/26 skip;
left/right đều71 skip/viewpoint_unknown (một phần có thêm quality reasons).
Reused strict1:sequence allow,hai knees skip. Old40 exploratory:sequence0/5/35,
left21/5/14,right14/5/21 A/D/S; không gộp assumption cohorts thành quality rate chung.
112 hiện có verified pose/quality evidence; chỉ41 có verified preprocessing,
71 mới không preprocessing/geometry/count/overlay.33 strict sequence-allow clips
có left bị chặn chỉ bởi viewpoint_unknown, chưa documented/temporal eligible.
A official0/112; B proposed quality subset0/112, processed count-gap UNKNOWN,
method approval PENDING. Training NOT_READY/TEMPORAL_LABELS_PENDING.
152 related tests pass/4 deselected/0 fail/error/skip/pytest warnings; native
MediaPipe/protobuf diagnostics lưu riêng. Worker427,25s; execution497,42s, không
controlled benchmark. [Readiness + bảng112/evidence](SQUAT_COUNT_DATA_READINESS_REPORT.md).
Next: human protocol-linked viewpoint review cho33 candidate quality nêu trên,
giữ112 cohort và làm hai lượt temporal labels độc lập/adjudication. Không tự gán
viewpoint hoặc mở proposal/gate. Training user tự chạy sau MODEL TRAINING GATE.
NOT_VALIDATED/official=false/assessment=false, G5.7 blocked,G6/toàn G5 IN_PROGRESS.

**2026-10-25 — LABEL INTAKE CHECKED / TEMPORAL_LABELS_PENDING / TRAINING NOT_READY:**
Đã tìm metadata nhãn trong project và kiểm tra lại dataset-final bằng công cụ
hiện có; không rerun readiness.0/112 clip có completed independent temporal labels,
0 adjudicated,0 cặp đủ điều kiện agreement; agreement=null/NOT_EVALUATED. Chỉ thấy
pending templates/bundles cũ và pixel template riêng.329 artifacts run readiness
khớp hash;112 source/copy videos khớp; hai templates mỗi112 records nhập được.
Không lỗi cản trở import, không sửa code/schema/gate/split. Evidence mới và hướng
dẫn giao/lưu/nhập nhãn nằm ở mục0 của [report hiện có](SQUAT_COUNT_DATA_READINESS_REPORT.md).
78 counting tests pass,0 fail/error/skip/warnings; không full suite/inference hoặc
training. Đây là kiểm tra phần mềm, không accuracy/label agreement.
Baseline real chưa chạy: temporal input vẫn synthetic-only, geometry framewise
không cấp temporal eligibility. Count-only admission chỉ là proposal cần user
phê duyệt phương pháp; rule bảo thủ đề xuất vẫn có0/41 saved clips đủ điều kiện.
Hai người gán độc lập và người thứ ba phân xử cần thực hiện phần nhãn; user quyết
định phương pháp. **User tự chạy training** khi đủ điều kiện; agent chỉ bàn giao
MODEL TRAINING GATE/config/lệnh, không chạy training/fine-tuning/optimizer/smoke train.
Giữ NOT_VALIDATED/official=false/assessment=false/REFERENCE_PENDING, G5.7 blocked,
G6/toàn G5 IN_PROGRESS. UI/web và các môn khác tiếp tục tạm dừng phát triển.

**Lịch sử trước pose feasibility — COUNT DATA PREPARATION IMPLEMENTED / LABELS PENDING (2026-10-25):**
Ưu tiên BodyWeightSquats → baseline count → nhãn độc lập → đối chứng temporal
model → form sau. Web/desktop và đa bài tập tạm dừng phát triển, không xóa sản phẩm.
Kiểm kê toàn thư mục:112 paths/112 unique hashes/25 gXX,14.356 frames/574,24s;
112 decode hoàn chỉnh ở25FPS/320×240.41 saved snapshots đã xác minh,71 chưa có
snapshot tái sử dụng được xác minh tại thời điểm preparation; 71 này hiện đã có
pose/quality-only evidence như cập nhật phía trên. Saved sequence35 skip/5 degrade/1 allow;
clip allow g15_c03 vẫn có cả hai knee metrics skip vì viewpoint_unknown.
Không inference/đo góc mới hoặc real temporal baseline.0 temporal labels final,
evaluator đủ112 rows:112 not_run; MAE/exact/event metrics null/NOT_EVALUATED.
Protocol/schema/template mù hai lượt + adjudication log, group/duplicate split
67 train/21 validation/24 development-test đã khóa trước labels/performance.
Tất cả112 có prior visual exposure; không pristine holdout hoặc subject isolation.
423 related tests pass/5 integration deselected; delivery90 pass,0 fail/error/skip/
warnings (các suite chồng lắp).299 baseline identities:293 unchanged,6 docs cập
nhật đúng phạm vi,0 unexpected;112 video copies/hash và66 links đạt.
Receipts ở [readiness report](SQUAT_COUNT_DATA_READINESS_REPORT.md).
Training runs=0; chưa được phê duyệt. Tiếp theo: hai lượt nhãn video độc lập và
phân xử trên gói cố định; review count-only admission bằng proposal riêng trước
baseline real, sau đó trình MODEL TRAINING GATE trước bất kỳ fit nào.
REFERENCE_PENDING/NOT_VALIDATED/official_eligibility=false/assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS giữ nguyên.

## Sản phẩm web đã hoàn tất — evidence giữ nguyên, UI development tạm dừng

**2026-10-24 — WEB local IMPLEMENTED / browser + real smoke PASSED:**
Hướng sử dụng chính là `python -m gym_coach.web`, http://127.0.0.1:8765/.
Stdlib HTTP + HTML/JS, shared job controller/progress/subprocess pipeline; không
Tk/framework/dependency mới. Strict/acknowledged exploratory, upload decoder,
fixed side, one active job, manifest/hash verification, range MP4 và download.
Final3 cases trên g01_c01: strict fresh0/267; exploratory cache267/267;
strict cache0/267. Complete strict JSON exact equal,801 output frames decode,
36 report refs và browser download hash khớp. Browser thực: upload/progress/
refresh/ack/null/error/play/seek/download/report/back; screenshot/receipt đã lưu.
Một initial Strict revision fresh0/267 giữ riêng: toàn milestone2 fresh extractions
cùng1 unique clip/534 frames,2 cache hits; không accuracy/bilateral validation.
1566 tests passed/25 integration deselected,0 fail/error/skip/pytest warnings;
native MediaPipe/protobuf/OpenH264 diagnostics ghi riêng. Không full integration set.
Final delivery checks145 pass/0 fail/error/skip/warnings,31,95s (overlap với regression);
885 baseline identities:877 unchanged,5 expected edits,3 archived;0 unexpected.
3 Tk source/test files archived38.751bytes;94 reproducible cache files xóa2.275.887
logical bytes, không claim lượng archive là freed space hoặc net disk reduction.
[Báo cáo web](SQUAT_WEB_DEMO_REPORT.md), [cleanup/restore](PROJECT_CLEANUP_REPORT.md),
[mục lục active/lịch sử](DOCUMENTATION_INDEX.md). Giữ mọi old evidence/Data/archive.
REFERENCE_PENDING/NOT_VALIDATED/official=false/assessment=false; G5.7 blocked,
G6/toàn G5 IN_PROGRESS. Next product: read-only persisted job history qua restart;
chưa triển khai. Independent reference labels là nhiệm vụ khoa học riêng.

## Các milestone trước — evidence lịch sử, không phải giao diện chính

**2026-10-21 — single-video pipeline IMPLEMENTED / REAL SMOKE PASSED:**
Một CLI video→pose→quality→preprocess→session→geometry→saved JSON→overlay/report;
strict mặc định, exploratory cần --assume-side, source unknown không đổi.
Pose-only cache kiểm tra bytes/metadata/model/config/profile/code và linkage;
quality/preprocess/numeric không được reuse như quyền đo.2 fresh inference runs/
355 frames và3 hits; strict0/267, exploratory left267/267, right skip0/267,
degrade0/88. Strict fresh/cache complete results exact equal, không baseline cần thiết.
981 related tests pass/4 deselected/0 fail/error/skip/pytest warnings;1.156 output
frames decoded,21 samples đã xem; browser/playback NOT_RUN. Integration protobuf/
MediaPipe và encoder-init logs ghi riêng, không tuyên bố warning-free integration.
[Demo5 cases](../outputs/squat_single_video/run-011b5d3098204820b8b46bf1b35e0605/START_HERE.html),
[report/commands/limits](SQUAT_SINGLE_VIDEO_PIPELINE_REPORT.md).
REFERENCE_PENDING/NOT_VALIDATED/official=false/assessment=false;
G5.7 blocked, G6/toàn G5 IN_PROGRESS. Tiếp theo sản phẩm: progress events cho CLI;
nhãn reference độc lập vẫn là nhu cầu khoa học riêng, chưa được giải quyết.

**2026-10-21 — independent2D reference toolkit IMPLEMENTED / REFERENCE_PENDING:**
Khóa sampling trước extraction:12/40 development clips,12 gXX groups,60 native
PNG,24 clip-side pairs(9 allow/5 degrade/10 skip). Gói ảnh/template mù tách model;
protocol/schema/evaluator additive, không inference hoặc sửa upstream gates.
0 qualified independent reference documents/0 annotated frames; real error MAE/
median/p90=null,n=0. Raw292/360 observed/in-frame; processed45/120 available,
75 null, đây là coverage không accuracy.884 related tests pass/3 deselected,
0 fail/error/skip/warnings. Giữ NOT_VALIDATED/official=false/assessment=false,
G5.7 blocked, G6/toàn G5 IN_PROGRESS. Tiếp theo: annotation độc lập/mù và review
chất lượng nhãn trên frozen images, chưa thực hiện hoặc mở accuracy gate.
[Protocol, package và evidence](SQUAT_2D_REFERENCE_READINESS_REPORT.md).

**2026-10-21 — context session performance IMPLEMENTED / EXACT EQUIVALENCE:**
Clip-local session bytes/tuples bất biến, full strict audit một lần ở intake;
fresh detached context/raw/linkage, không caller validated token/global cache.
Factory cũ/Linked validator/schema và geometry/quality/preprocessing giữ nguyên.
40 clips/80 pairs/10.628 complete records exact match baseline,0 mismatch/error;
coverage/status/reasons/flags/provenance/source hashes và summary đều bằng nhau.
Optimized full replay CLI90.71s gồm strict comparison,
80 full audits(40 load+40 session),0 theo frame. Không dùng767,79s cũ làm control.
Paired54/116/267-frame benchmark,1 worker,1 warmup+3 repeats/mode: 7.34× / 12.77× / 26.00×;
712 regression pass/3 deselected/0 fail/error/skip/warnings, không pytest overlap.
Session default chỉ trong saved-clip replay CLI mới; old public/CLI default reference
không đổi. Không Data/inference/render mới, không accuracy/generalization evidence.
[API, benchmark, memory và evidence](SQUAT_CONTEXT_PERFORMANCE_REPORT.md).
NOT_VALIDATED, official=false, assessment=false; G5.7 blocked, G6/toàn G5 IN_PROGRESS.

**2026-10-20 — fixed40 batch coverage + representative demos IMPLEMENTED / EXPLORATORY:**
Saved pose/quality/source/config linkage verified;40 clips/80 independent knee
pairs,5.314 raw frames/10.628 observations,0 errors/unevaluated/incompatibilities.
4.795 available (45,12% micro/43,75% macro),5.833 null; raw independent-side
coverage6.582/10.628 (61,93%). Trái2.968/5.314; phải1.827/5.314.
35 metric allow/10 degrade/35 skip;35 sequence skip/5 degrade,0 bilateral allow.
16 processed imputed slots riêng;0 đi vào available; không đổi policy/schema.
Existing preprocessing/factory/adapters,1 validated prior preprocessing reuse,
context shared per-frame only; no MediaPipe/temporal detector, CLI767,79s.
Frozen path/side sampling cho allow(g01_c01,left), degrade(g04_c03,right),
skip(g01_c01,right);3 H.264 whole-clip demos.704 tests pass/4 deselected;
452 hash/size checks,35 links,622 output frames decoded;22 PNG thực sự xem.
Raw skeleton sample có sai lệch/framing giới hạn; observed/coverage không accuracy.
Browser/native playback QA NOT_RUN; source cameraUNKNOWN, side unverified.
Mở `outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f/batch/START_HERE.html`.
[Báo cáo và evidence](SQUAT_REAL_BATCH_COVERAGE_REPORT.md).
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS giữ nguyên.

**2026-10-20 — real descriptive fixed-side demo IMPLEMENTED / EXPLORATORY:**
Tái sử dụng audit40 clips; xếp80 clip-side pairs,35 metric allow, tie path rồi
left/right chọn g01_c01/gối trái.267/267 observations available,0 unavailable;
sequence vẫn skip, right metric skip. Dùng nguyên policy v2 framewise-metric-only-v1;
legacy vẫn blocked. Saved pose/QualityReports hash-validated, chỉ preprocessing/
geometry/video/report mới, không inference hoặc temporal detector. Raw source
cameraUNKNOWN, side visually_assessed_unverified/documented=false.
MP4 H.2641280×720/25FPS decode267 frames;6 decoded frames +2 original images
đã xem bằng view_image; browser/native-player QA NOT_RUN.681 tests pass,
0 fail/error/skip/warnings/4 integration deselected,93.82s. Coverage không accuracy.
Mở `outputs/squat_real_descriptive_demo/run-9f5a079d58e24bdeaebbfcaa34baf8a9/demo/START_HERE.html`.
[Báo cáo/source/test/QA](SQUAT_REAL_DESCRIPTIVE_DEMO_REPORT.md).
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS giữ nguyên.

**2026-10-18 — fresh real exploratory audit NO_BILATERAL_CANDIDATE:** pool40
unique likely_side của audit15/10, chỉ BodyWeightSquats trong UCF-101_GYM.
40/40 ingestion→pose→quality completed,5.314 frames;35 sequence skip/5 degrade,
không clip nào có left/right metrics cùng allow. Raw bilateral1.301/5.314;
bilateral allowed0/5.314, không đồng nhất hai coverage này. Không cắt clip,
nội suy, tính góc/temporal hoặc tạo demo bị gate khóa. Source camera_viewUNKNOWN,
raw viewpointunknown; side metadata copy chỉ visually_assessed_unverified.
67 tests passed/0 fail/error/skip/warnings/3 deselected,9.21s; độc lập kiểm tra
hashes, frame/timestamp/projection và80 QualityReports.112 source videos giữ hash.
Run `outputs/squat_real_exploratory/run-7900dcdac1b34c62bfe2228bfc439cdf/`.
[Kết quả40 clips và giới hạn](SQUAT_REAL_EXPLORATORY_CANDIDATE_REPORT.md).
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS giữ nguyên.

**2026-10-18 — ZIP offline PACKAGED / PASSED_OFFLINE_CHECKS:** chỉ copy bytes
từ bundle handoff đã chỉ định;20 file/16.679.343 bytes, ZIP1.221.944 bytes trong
`outputs/squat_offline_release/run-0859ad662d114359a3c38b5298dfff97/release/`.
Giải nén thử vào thư mục mới:20/20 hash/size,9 local links/fragments,
12 snapshot links, embedded JSON/null/disabled gates đều đạt; nguồn giữ nguyên.
57 tests passed (39 packaging +18 handoff),0 fail/error/skipped/warnings,19.87s.
Không đổi renderer/producer/schema/profile, không Data/video/inference/noise audit.
Browser QA **NOT_RUN**; NOT_VALIDATED, official_eligibility=false,
assessment=false, G5.7 blocked, G6/toàn G5 IN_PROGRESS giữ nguyên.
[ZIP, cách mở và evidence](GYM_SQUAT_OFFLINE_RELEASE_REPORT.md).

**2026-10-18 — demo handoff READY / OFFLINE_QA_ONLY:** bundle mới có
START_HERE.html/HUONG_DAN_MO_DEMO.txt và hướng dẫn ngay trong demo. Cải thiện
ARIA pressed state của play buttons, slider frame/time text, keyboard navigation
links/focus styles; giữ clock/scrubber A/B độc lập. Presentation/source JSONs
byte-identical từ saved showcase; không source reload từ Data/upstream runs,
noise re-audit, inference, detector/profile/schema/producer change. Browser
binding đã kiểm tra nhưng file navigation trước đó bị security policy chặn;
không server/network/alternate surface. **browser_qa=NOT_RUN**,0 screenshot,
native keyboard/desktop visual QA chưa chạy. Offline code checks có receipt riêng.
Mở `outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle/START_HERE.html`.
[Báo cáo handoff/test evidence](GYM_SQUAT_DEMO_HANDOFF_REPORT.md).
Final48 passed/0 failed/error/skipped/deselected/warnings,22.27s (18 mới +30
showcase).504 old artifacts giữ hash; browser/native keyboard QA vẫn NOT_RUN.
NOT_VALIDATED, official_eligibility=false, assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.

**2026-10-18 — saved-artifact demo showcase IMPLEMENTED:** HTML offline mới
cho BodyWeightSquats, panel A dùng2 saved synthetic feedbacks (skeleton/metric),
panel B dùng9 saved point-resolution audits. Input/source hashes/video IDs
không khớp, vì vậy clocks/scrubbers độc lập; không overlay B lên skeleton A,
không hiển thị legacy v1 phase như phase của B. Source JSON copied byte-identical,
không chạy lại góc/pose/producer. Point/phase/pause unavailable được giữ nguyên.
Read-only noise review trên contracts/config/docs và selected synthetic JSONs
không tìm thấy independent calibrated uncertainty/noise bound gắn signal sample;
visibility/presence/provenance không thay evidence đó. Detector work STOPPED,
không thay rule/profile/schema/gate. Không đọc Data hoặc thêm dependency.
Run: `outputs/squat_showcase/run-ae8d0517445c48feb91a566afdae9f8c`.
[Báo cáo/demo/test evidence](GYM_SQUAT_SHOWCASE_REPORT.md).
Tests cuối237 passed/0 failed/error/skipped/warnings,1 integration deselected,
49.17s; mới30 tests. Chưa browser QA hoặc full project suite. Bảo toàn453
inventoried old artifacts; toàn bộ source/domain code cũ giữ hash.
NOT_VALIDATED/official_eligibility=false/assessment=false;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.

**2026-10-18 — point-resolution milestone IMPLEMENTED / SAFE ABSTENTION:**
policy/producer opt-in `squat-v2-point-resolution-v1`, frozen acceptance trước
producer. Đã kiểm tra strict raw three-sample minimum, observed support, gap/
side/clip/UNSTABLE_SIGNAL và identifiability. Cặp latent reversal/noisy hold có
thể cho input hoàn toàn giống nhau; không có noise evidence độc lập trong
capability hiện tại. Producer vì thế accepted_point_count=0; raw necessary
conditions không trở thành point-turn event. Null bounds/pause/phase giữ nguyên.
Đây là hoàn tất milestone đánh giá rule + implementation abstention, không
phải accepting detector. Envelope mới lưu conservative v2 base/hash nguyên vẹn;
v1/v2 frozen models, old producer/API/renderer/profile và artifacts giữ nguyên.
Synthetic-only, không Data/inference/geometry/dependency mới. G5.7
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT; G6/toàn G5 IN_PROGRESS,
NOT_VALIDATED/official_eligibility=false/assessment.enabled=false.
34 tests mới; final related207 passed/0 failed/error/skipped/warnings,
1 integration deselected,39.95s. Demo9 inputs/14 PVP/11 raw-condition passes,
0 accepted points. Bảo toàn351 artifacts cũ; không chạy full project suite.
[Báo cáo point-resolution](STAGE6_POINT_RESOLUTION_REPORT.md).
Run: `outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48`.

**2026-10-18 — additive experimental temporal v2 slice IMPLEMENTED:** contract
`0.2.0`/`squat-temporal-components-v2` đã freeze trước producer; event/motion
tách biệt, bottom anchor/arrival/departure/pause và component abstention có
JSON roundtrip/explicit v1-v2 dispatch. Producer chỉ chuyển các PVP anchors của
v1 sang `bounds_unresolved`; chưa phát hiện dwell/point-turn hoặc bounds.
Positive dwell/point-turn là output-contract tests, không phải detector evidence.
Renderer v2 offline có scrubber/event roles/null pause; 7 demo cases synthetic,
25/30/60FPS. Không tính góc, đọc Data hoặc chạy inference/real integration.
V1/profile S3/API/dependencies và old outputs được giữ hash.
NOT_VALIDATED/official_eligibility=false/assessment.enabled=false;
G5.7 **BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**, toàn G5/G6 **IN_PROGRESS**.
[Báo cáo, test counts và giới hạn](STAGE6_TEMPORAL_V2_SLICE_REPORT.md).
Run: `outputs/squat_temporal_v2/run-7cae351170a441ba85b75e84b4d8e26d`.

Các mục bên dưới là evidence của các lượt trước; design-only mô tả thời điểm
trước slice additive v2, không phủ nhận implementation mới ở trên.

**2026-10-18 — temporal contract design, DESIGN_ONLY:** đã đề xuất duy nhất
events + optional evidence-backed dwell interval, tách event/motion phase và
component abstention. [Đề xuất](SQUAT_TEMPORAL_CONTRACT_PROPOSAL.md) phân tích
arrival/departure/pause, shared tops, version/migration và acceptance plan.
Chưa triển khai hoặc duyệt schema/algorithm mới; core/profile/API/UI/dependencies
và artifacts cũ giữ nguyên. GĐ6 vẫn IN_PROGRESS, G5.7 blocked, NOT_VALIDATED,
official=false/assessment=false. Các số sensitivity bên dưới là evidence trước,
không phải validation cho proposal. Lượt thiết kế: related124 passed/0 failed/
error/skipped/1 real integration deselected/0 warnings (24.76s); doc/evidence
checks đạt, source/profile/UI và288 old artifacts giữ hash. Chưa chạy future-v2
acceptance tests hoặc real integrations. Evidence ở
`outputs/squat_temporal_contract_design/run-dfbeeb29689144c2b0b7ceac283a648c`.

**2026-10-18 — synthetic temporal sensitivity:** 10 analytic cases ×25/30/60FPS,
30/30 executed. Profile `squat-temporal-s3-v1` và existing segmentation/reporting
không đổi. Matched78/102 full reference cycles (76.47%),24 missed,0 extra;
6 boundary-partial references báo riêng. Phase availability5780/6989 (82.70%),
agreement5153/6989 (73.73%); không tính unavailable như đúng. Event timing MAE
matched-only: start15.684ms, bottom midpoint5.513ms, end17.009ms. Các FPS là
biến thể cùng signal, không phải102 mẫu vận động viên độc lập.

Giới hạn rõ: Bottom chỉ event một frame, không giữ interval dừng đáy; tempo
thay đổi gây drift tới80ms ở top,50ms ở midpoint. Fast0.08s halves và excursion3°
đều không có candidates; gap/side-switch từ chối chu kỳ cắt qua vùng lỗi.
Không tune profile, nội suy hay chọn lại side để khớp reference. Synthetic
fixtures không đi qua pose/geometry/QualityGate; không claim real eligibility.
GĐ6 IN_PROGRESS, NOT_VALIDATED/official_eligibility=false/assessment=false,
GĐ5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT và toàn GĐ5 IN_PROGRESS.

Run mới: `outputs/squat_temporal_sensitivity/run-87f31de96a654bc09ee816cd2f044f5e/audit-final`.
54 tests mới đạt trong related124 passed/0 failed/error/skipped/1 real-data
integration deselected/0 warnings (28.59s). Broader synthetic/unit suite1056
passed/0 failed/error/skipped/25 integration deselected/0 warnings (89.76s).
Không chạy inference hoặc
segmentation trên dataset trong lượt này.
[Báo cáo sensitivity](STAGE6_SYNTHETIC_TEMPORAL_SENSITIVITY_REPORT.md).

Các phần bên dưới là trạng thái/evidence của các lượt trước.

**Lượt mới 2026-10-15 — Gym Result Feedback prototype, package 0.7.0.**
Đã có contract feedback JSON 0.1.0, renderer HTML/Canvas offline và temporal
squat PVP synthetic-only (`squat-temporal-input-v1`, profile S3
`squat-temporal-s3-v1`, NOT_VALIDATED). Hai chu kỳ ứng viên từ fixture synthetic
6 giây; fixture mất gối trái bị sequence degrade nên temporal blocked, kể cả khi
preprocessing nội suy. Năm angle IDs canonical dùng API geometry hiện hành;
numeric0.4 vẫn temporal_analysis_eligible=false. Đây là thử nghiệm độc lập do
người dùng yêu cầu, không đóng acceptance gate GĐ5/GĐ6.

Preview clip GYM g25_c01: 103 frames, 25 FPS, 320×240; chỉ decode và lưu frame
0/51/102. Metadata camera_view/person_visibility/occlusion UNKNOWN. Không chạy
pose/quality/numeric/temporal trên clip này. GĐ5.7 vẫn
**BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**, assessment.enabled=false.

Run: `outputs/gym_result_feedback/run-e1f46752326a49fea5696cc5ccc4e296`.
Test evidence lượt prototype: 71 mới passed/0 failed/error/skipped/warnings;
compatibility suite 1020 passed/0 failed/error/6 skipped/1 deselected/4 protobuf
warnings, 142.09s. Sáu saved exploratory audit readbacks không bật env fixture;
một inventory test quét cả Data bị deselect để giữ giới hạn GYM. Không tuyên bố
toàn bộ collection đều chạy. UI desktop đã xem/tua Descent/Bottom/Ascent và
real blocked với năm số đo null; artifact readback/hash được kiểm tra riêng.
[Báo cáo prototype](GYM_RESULT_FEEDBACK_PROTOTYPE_REPORT.md).

Các đoạn/số test GĐ5 bên dưới là bằng chứng lịch sử, không phải run prototype.

Rà soát mới ngày2026-10-15 chỉ dùng video/metadata trong `Data/UCF-101_GYM`:
112 squat clips/112 hashes, xem560 distributed samples và80 dense samples cho
5 ứng viên g07_c01/g12_c02/g15_c01/g23_c03/g25_c01. Đây đều là
`visually_assessed_unverified`, không documented. Camera_view/person_visibility/
occlusion của112 CSV rows đều UNKNOWN; knee phía xa chồng/che khuất nên chưa
xác nhận bilateral visibility liên tục. Official sequence/left/right gates
**NOT_RUN** vì thiếu documented input; GĐ5.7 vẫn BLOCKED, toàn GĐ5 IN_PROGRESS.
Không đổi code/policy/metadata/video hoặc dùng old exploratory làm gate chính thức.
[Bằng chứng và tests lượt rà soát](STAGE5_7_UCF101_GYM_REVIEW_2026_10_15.md).
Related synthetic regressions:49 passed/0 failed/error/skipped/warnings trong8.95s.
Full suite và official real-video pipeline NOT_RUN trong lượt này.

- Giai đoạn 0: COMPLETE — đạt cả bốn tiêu chí DoD của Prompt tổng, bằng chứng bên dưới.
- Giai đoạn 1: COMPLETE — metadata, streaming decoder, structured failures và CLI đã đạt DoD.
- Giai đoạn 2: COMPLETE — MediaPipe 0.10.14, mapping/schema/provenance/coverage và hai full-video integrations đạt DoD.
- Giai đoạn 3: COMPLETE — versioned quality profile, gap report và metric eligibility đạt DoD.
- Giai đoạn 4: COMPLETE — raw/processed provenance, interpolation, normalization,
  FPS-aware finite-run smoothing và hai fresh real-video integrations đạt DoD.
- Giai đoạn 5.1: COMPLETE — feature contracts/registry metadata và prerequisite
  policies đã kiểm thử; chưa có numerical feature calculation.
- Giai đoạn 5.2: COMPLETE — primitive góc 2D thuần, structured errors và 94 unit
  tests đạt; toàn suite 328 passed, chưa tích hợp tính feature.
- Giai đoạn 5.3: COMPLETE (một feature) — numeric schema/adapter left squat knee,
  120 new tests và 448 full-suite tests đạt; real-input samples unavailable.
- Giai đoạn 5.4: COMPLETE — factory linked context từ strict PreprocessingResult;
  109 new tests và 557 full-suite tests đạt; sáu real samples unavailable đúng policy.
- Giai đoạn 5.5: COMPLETE — right-knee adapter, schema mới 0.3.0 và compatibility
  bên trái 0.2.0; 118 new tests và 674 full-suite tests đạt; sáu saved samples unavailable.
- Giai đoạn 5.6: COMPLETE (eligibility/data audit) — 39 genuine gate/audit tests
  và full suite 713 tests đạt. Synthetic allow đạt, real documented side-view input chưa có.
- Giai đoạn 5.7: BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT — kiểm tra mới không tìm
  thấy annotation side-view chính thức phù hợp; visual audit không thay documentation.
- Giai đoạn 5.7A: COMPLETE (visual audit/exploratory) — 112 unique videos/560 samples
  đã xem; 44 likely_side, 38 oblique, 22 front, 8 uncertain, 0 documented.
  Candidate R106: fresh assumed-side sequence skip, left knee allow, right knee skip.
- Giai đoạn 5 implementation angles: COMPLETE — đủ 10 declared unblocked angles,
  registry v2/schema 0.4, framewise exact-metric policy, legacy knees tương thích.
- Exploratory real numeric integration: EXECUTED — 3 unique clips/6 fresh branches,
  558 frames/2.790 observations, 534 available và 2.256 unavailable; không official.
- Official numeric real-video validation: BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT.
- Toàn Giai đoạn 5: IN_PROGRESS (IMPLEMENTATION_COMPLETE / VALIDATION_BLOCKED).
  GĐ5.7 documented input/sequence+bilateral knee gate chưa đạt, không đóng gate này
  bằng visual assumptions. Chưa có public FeatureSequence hoặc temporal extractor.
- Giai đoạn 6: IN_PROGRESS — synthetic PVP/phase prototype; real acceptance BLOCKED
  bởi GĐ5.7 và chưa có reference/count/phase validation. Không COMPLETE.
- Giai đoạn 7: NOT_STARTED.
- Giai đoạn 8 assessment/scoring: BLOCKED_PENDING_EXPERT_VALIDATION.
- Model training: NOT_STARTED, ngoài phạm vi lượt này.

Tests cuối lượt: 243 mới passed/0 warnings, 627 liên quan passed/0 warnings;
full suite 956 passed/4 warnings trong126.30s, 0 failed/error/skipped.
Implementation mới dùng package 0.6.0, không thêm dependency. Registry v1/hash và
API legacy nguyên vẹn; registry v2 giữ tier/source/validation, chỉ đổi versioned
calculation status và framewise gate theo đặc tả §§2.1–2.2. QualityProfile,
thresholds, preprocessing, source/raw và reports5.6/5.7A không được sửa/promote.
Evidence/test counts cuối lượt được ghi trong
[STAGE5_FINAL_FEATURE_EXTRACTION_REPORT.md](STAGE5_FINAL_FEATURE_EXTRACTION_REPORT.md).
Các số test bên dưới là bằng chứng lịch sử của từng stage, không phải run mới.

Run mới: `outputs/stage5_final/run-5fe3a7fd1ffe40cc816c5e982bc30ffd`.
R106 assumed side: sequence skip, left hip/knee/torso allow; right hip/knee skip.
H004 assumed side: sequence skip, left elbow/shoulder/body-axis allow; right
elbow/shoulder skip. H002 assumed front: sequence allow nhưng cả 5 metrics skip
viewpoint_mismatch. Unknown branches: tất cả features unavailable. Mọi evidence
viewpoint vẫn visually_assessed_unverified, 0 documented, assessment blocked.

Bước nhỏ nhất cho validation thật: xác minh annotation/protocol side liên kết đúng
path/ID/byte SHA/interval/camera/framing/occlusion, rồi fresh official QualityGate
cho exact bilateral knees. Không tự tạo annotator hoặc sửa existing reports.
R106 hiện vẫn fail right/global quality; annotation riêng lẻ không đủ mở gate.
Nhánh prototype độc lập có thể bổ sung synthetic plateau/noise/gap fixtures;
không dùng kết quả đó làm bằng chứng accuracy hoặc thay protocol thật.

## Workspace và môi trường đã kiểm tra

- Ban đầu chỉ README, hai đặc tả trong docs, .gitignore, .agent, Data, .venv,
  _archive. Không có src/tests/manifest/kế hoạch/trạng thái/tasks mới.
- `git status --short`: workspace không có repository Git; không có commit/diff
  identity để ghi. Không tự khởi tạo Git hoặc sửa archive/dataset.
- `.venv` chạy được CPython 3.11.9, Windows x64; quan sát Pydantic 2.13.5,
  pytest 9.1.1, setuptools 84.0.0. Không cài thêm backend.
- `.agent/AGENT_RULES.md` còn ghi STEP1–6 completed và tên MASTER_PROMPT cũ;
  đây là trạng thái lịch sử, bị clean restart và yêu cầu hiện hành thay thế.
  Các nguyên tắc nghiên cứu phù hợp vẫn áp dụng; không sửa file hướng dẫn này.
- Đọc đầy đủ README, GYM_TECHNICAL_GROUND_TRUTH_SPEC, GYM_REBUILD_MASTER_PROMPT.
  Không mở/restore zip trong _archive; chỉ kiểm tra sự hiện diện thư mục.

## Thay đổi nền mới

- `pyproject.toml`: distribution ai-coach-gym; Python >=3.11; pin dependency tối
  thiểu đã quan sát. Layout src chỉ đóng gói gym_coach, loại archive khỏi discovery.
- `src/gym_coach`: config/versions/contracts/structured flags, 8 stage namespaces;
  reporting có AuditHeader. Domain namespaces chưa có thuật toán.
- `configs/default.toml`: portable paths, exercise/viewpoint, assessment disabled.
- `tests/test_foundation.py`: import thật từ installation, config, hash/roundtrip,
  bad config, boundary validation; backend imports bị chặn trong subprocess.
- IMPLEMENTATION_PLAN, SCHEMA_CONVENTIONS, TASKS và hướng dẫn chạy trong README.

## Bằng chứng kiểm thử

Commands đã chạy tại project root với interpreter `.venv`:

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pytest -q
```

- Install: exit 0, `Successfully installed ai-coach-gym-0.1.0`; build editable
  wheel thành công, không tải dependency từ mạng.
- Pytest: exit 0, `19 passed in 0.69s`; không có test failed/skipped.
- Tests gồm: import 8 namespaces và installed metadata; subprocess `-I` từ
  thư mục khác chặn MediaPipe/OpenCV/NumPy/SciPy/Torch; portable config và không
  tạo output khi load; config hash roundtrip/change; 8 invalid config cases;
  missing config; flags/header JSON roundtrip; 5 invalid boundary cases.
- Output thực tế từ `load_config(configs/default.toml)`: `foundation-v1`,
  `BodyWeightSquats`, `viewpoint=unknown`, `assessment.enabled=false`,
  `NOT_AVAILABLE_PENDING_EXPERT_VALIDATION`; resolved input directory tồn tại.
- Config hash trên môi trường Windows hiện tại:
  `e9e391d8a5ecd95e375c817e5bafb49f3a3a5c83864c8a3cc57c6d98ccff8dfa`.
- AuditHeader mẫu `video_id=not-analyzed`: schema/package 0.1.0, cùng config
  version/hash, mọi stage not_run, flags rỗng, không có score hoặc rep count.
  Đây chỉ là smoke output, không phải kết quả video.
- Quét `src/gym_coach`, `configs`, `pyproject.toml`: không có archive reference,
  backend imports hoặc absolute drive path hard-code.
- Không có execution/coverage/accuracy video trong GĐ0; chưa chạy real-data
  integration vì chưa có thuật toán. Không cần sửa test/code sau run: 19/19 đạt.

## Đối chiếu Definition of Done Giai đoạn 0

| Tiêu chí trong Prompt tổng | Bằng chứng | Kết quả |
|---|---|---|
| Cấu trúc mới không phụ thuộc code archive | package discovery chỉ gym_coach; module mới từ đầu; không mở archive; source scan | PASS |
| Dependency manifest, package import, smoke test | pyproject; offline editable install; installed metadata và isolated import; 19 tests đạt | PASS |
| Mỗi giai đoạn tiếp theo có input/output/test/tiêu chí hoàn thành | IMPLEMENTATION_PLAN GĐ1–8, backlog TASKS và gate assessment | PASS |
| Không scoring/ngưỡng kỹ thuật chưa xác nhận | config không threshold; bật assessment bị từ chối bằng test; features/segmentation/assessment chỉ namespaces | PASS |

## Danh sách file tạo mới

```text
pyproject.toml
configs/default.toml
src/gym_coach/__init__.py
src/gym_coach/_version.py
src/gym_coach/contracts.py
src/gym_coach/config.py
src/gym_coach/flags.py
src/gym_coach/video/__init__.py
src/gym_coach/pose/__init__.py
src/gym_coach/quality/__init__.py
src/gym_coach/preprocessing/__init__.py
src/gym_coach/features/__init__.py
src/gym_coach/segmentation/__init__.py
src/gym_coach/assessment/__init__.py
src/gym_coach/reporting/__init__.py
src/gym_coach/reporting/schema.py
tests/test_foundation.py
docs/IMPLEMENTATION_PLAN.md
docs/PROJECT_STATE.md
docs/TASKS.md
docs/SCHEMA_CONVENTIONS.md
```

21 file mới; cập nhật README.md (cách cài/chạy/config và docs links), .gitignore
(build/dist/egg-info/outputs). Installation sinh metadata egg-info và test sinh
cache được ignore, không phải source/experiment artifacts.

## Giả định và hạn chế

- Môi trường .venv chỉ dùng interpreter và dependency cần thiết; manifest không
  đại diện mọi package legacy. Chưa có clean-environment/transitive lock verification.
- Chưa kiểm kê dataset, chọn development fixture/reference labels, chia group.
- Chưa có rule S1/S2/citation/expert approval/custom form ground truth được cung
  cấp trong lượt này; assessment bị khóa. Không gọi số liệu cũ là kết quả mới.
- Đã có video/MediaPipe/quality/preprocessing GĐ1–4, feature metadata/policy
  GĐ5.1, primitive góc 2D GĐ5.2, hai knee adapters GĐ5.3/5.5 và factory GĐ5.4; chưa calculator
  cho các feature khác hoặc segmentation.
- Schema domain và full pipeline report được triển khai ở giai đoạn tương ứng;
  header smoke không phải output phân tích.
- Nguồn yêu cầu và config profile là engineering design chưa validated về thuật toán.

## Bước tiếp theo

Cung cấp một video squat với annotation side-view có protocol ID/version,
annotator/date/frame interval và input hash; sau đó tạo QualityGate report mới
cho hai knee IDs. Real measurement blocked; không sửa report thật để ép available.
Còn các feature MVP
khác và numeric development integration đủ quality/viewpoint; assessment vẫn
bị khóa. Xem [STAGE5_6_DEVELOPMENT_ELIGIBILITY_REPORT.md](STAGE5_6_DEVELOPMENT_ELIGIBILITY_REPORT.md).

## Giai đoạn 1 — hoàn thành 2026-10-08

- Thêm `VideoMetadata`, `VideoFrame`, `OpenCVVideoReader` và CLI JSON.
- Chọn decoder runtime `opencv-python==5.0.0.93` trong manifest.
- Timestamp explicit `fps_derived_cfr_assumption`; chưa claim VFR support.
- Synthetic video và video squat thật đều được kiểm tra.
- Test toàn bộ: `28 passed in 1.52s`.
- Integration CLI đo file squat g01_c01: 25 FPS, 267 frames, 320x240,
  backend FFMPEG.
- Không chạy pose hoặc tạo kết quả assessment.

## Giai đoạn 2 — hoàn thành 2026-10-08

- Đọc đầy đủ tám tài liệu được yêu cầu, kiểm tra source video/test/dependencies.
- MediaPipe 0.10.14 import + CPU inference thành công, không tải thêm model.
- Package 0.2.0; pose schema 0.1.0; mapping mediapipe-pose-13-v1; pose config
  pose-extraction-v1. Foundation schema/config không đổi.
- Thêm mapping, models, extractor, coverage, JSONL serialization, runner và CLI
  dưới src/gym_coach/pose; thêm test_pose.py và test_pose_integration.py.
- Manifest thêm MediaPipe/NumPy thực tế, đổi OpenCV direct provider sang contrib
  phù hợp dependency MediaPipe. Venv legacy vẫn có hai provider metadata,
  pip check không phát hiện broken requirements; chưa clean-environment claim.
- Final test command: `.venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage2/verification-final.xml`.
- Kết quả: **76 passed, 2 warnings in 14.75s**, exit 0; warnings protobuf deprecated.
- Squat g01_c01: 267 frames; 267 có pose, 0 không pose; all-13 observed coverage
  72.45750504177471%, required-joint coverage 80.14981273408239%.
- HSPU g01_c01: 78 frames; 78 có pose, 0 không pose; all-13 observed coverage
  85.89743589743589%, required-joint coverage 100%.
- Toàn bộ 345 frames đã readback JSON, index/timestamp/video_id chính xác;
  low_visibility giữ provenance, không biến thành observed hoặc missing.
- No-pose/missing/invalid input/backend failures được test với structured reasons.
- Raw input HSPU frame đầu đã xem: subject inverted; inference không rotate input.
- Coverage là diagnostic, chưa có pose accuracy/form ground truth; single subject
  backend không hỗ trợ enumeration/identity assurance; presence null.
- Không restore archive, interpolation, normalization, smoothing, geometry,
  rep/phase/technical assessment hoặc scoring trong GĐ2.
- DoD GĐ2: toàn bộ 11 tiêu chí PASS trong
  [STAGE2_POSE_EXTRACTION_REPORT.md](STAGE2_POSE_EXTRACTION_REPORT.md).
- Bằng chứng: [measured snapshot/source hashes](artifacts/stage2_measured_results.json)
  và artifact links trong báo cáo; không dùng số liệu lịch sử làm kết quả mới.

## Giai đoạn 3 — hoàn thành 2026-10-08

- `QualityProfile quality-s3-v1` ghi hash, source tier S3 và thresholds 5 / 0.30 / 0.30 / 10.
- `QualityReport` lưu coverage, low visibility, missingness, out-of-frame và
  consecutive invalid spans theo từng joint; metric eligibility tách biệt.
- Unknown viewpoint được ghi flag; chỉ chặn metric khi caller khai báo viewpoint.
- Out-of-frame count gate metric khi tỷ lệ >0.30; đây là extension S3 chưa validated.
- Synthetic empty/gap/visibility/side/viewpoint/out-of-frame/sequence tests đạt.
- Stage 2 saved pose artifacts cho cả hai bài chạy qua CLI.
- Toàn suite: 88 passed, 2 protobuf deprecation warnings in 14.96s.
- Squat sample 267 frame: sequence skipped vì required right knee có 259/267
  low visibility và invalid run tối đa 125 frame; không phải kết luận form sai.
- HSPU sample 78 frame: sequence skipped vì wrist coordinates ngoài khung
  (left 75/78, right 78/78), dù visibility cao; không phải lỗi kỹ thuật.
- CLI chỉ phân tích pose quality; không có geometry/form scoring.

## Giai đoạn 4 — hoàn thành 2026-10-08

- Đọc đủ 10 tài liệu hiện hành, kiểm tra GĐ1–3 source/tests/dependencies;
  workspace vẫn không có Git. Không mở/restore archive hoặc sửa dataset.
- Package 0.4.0; preprocessing schema/profile schema 0.1.0, S3 profile
  preprocessing-s3-v1 có hash `33edf08cdafea19ef0ae67b67ce02b9f53b1b1764641bc9df7af58259d387c33`.
- Raw/quality snapshots và hashes tách khỏi processed joints, giữ nguyên raw
  coords/confidence/provenance, video IDs/indices/timestamps, custom quality decisions.
- Linear short-gap missing-only interpolation mặc định, original observed
  in-frame endpoints; exact imputation refs và source_status. No head/tail/long gaps.
- Explicit mid-hip/torso statuses; fallback null metadata + flag, safe coords,
  distance_ratio_eligible=false khi scale không hợp lệ. Aspect ratio được ghi rõ.
- SG finite-run smoothing tách gaps/unit statuses; short runs giữ nguyên;
  FPS-aware gap/window: 25 FPS=4/9, 30=5/11, 60=10/21, chưa baseline tối ưu.
- Manifest thêm SciPy 1.17.1 đã có trong venv; offline editable reinstall đạt,
  pip check không broken requirements. Không clean-environment claim.
- Final command: `.venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage4/verification-final.xml`.
- **145 passed, 4 protobuf deprecation warnings in 31.37s**, no failures/skips.
  Stage 4 thêm 55 unit + 2 fresh full video integrations; import smoke vẫn đạt.
- Squat g01_c01: 267 frames/25 FPS; raw observed 2515, low 956, missing 0;
  usable before=after 72.457505%; imputed 0; origin/scale fallback 0/0;
  finite runs 33, smoothed 14, short 19.
- HSPU g01_c01: 78 frames/25 FPS; raw observed 871, low 143, missing 0;
  raw observed coverage 85.897436%, usable before=after 69.033531%; imputed 0;
  origin/scale fallback 0/0; runs 14, smoothed 11, short 3.
- Fresh inference từ cả hai video, không thay bằng saved GĐ2 output. Tất cả 345
  frame raw/quality equality và JSON readback đạt; quality giữ skipped cả hai.
- DoD GĐ4 PASS, full report/per-joint measurements/limitations:
  [STAGE4_PREPROCESSING_REPORT.md](STAGE4_PREPROCESSING_REPORT.md).
  Snapshot + code/input/artifact hashes:
  [stage4_measured_results.json](artifacts/stage4_measured_results.json).
- Không feature/rep/phase/assessment/score/feedback. Coverage là availability,
  chưa accuracy/form validation; GĐ5–7 chưa bắt đầu, GĐ8 vẫn BLOCKED.

## Giai đoạn 5.1 — hoàn thành 2026-10-08

- Đọc đủ 8 docs hiện hành; kiểm tra GĐ0–4 contracts/imports/flags/tests. Workspace
  không Git; lưu before_manifest hashes, không giả commit/diff, không mở archive.
- Package 0.5.0; feature/registry schema 0.1.0, registry gym-feature-registry-v1,
  hash `bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a`.
- FeatureDefinition ghi formula/method, ordered joints/provenance/usability,
  quantity/unit, source/measurement tiers, validation/viewpoint/quality/scale,
  roles, standard reasons và limitations; metadata thiếu/mâu thuẫn bị từ chối.
- 10 angle metadata unvalidated, 2 knee-to-ankle ratio candidates blocked.
  Side-view-only/allow-only policies là S3 conservative assumptions, chưa validated.
- FeatureInputContext snapshots và FeatureAvailability hashes/reasons/flags;
  FeatureObservation chỉ not_computed/unavailable, value null, không fake number.
- Missing/unusable/provenance, quality degrade/skip, absent/wrong metric joints,
  unknown/unsupported viewpoint và scale/origin fallback khóa đúng; không nâng quality.
- Không dependency mới; offline editable reinstall ai-coach-gym 0.5.0 đạt;
  pip check clean metadata requirements, chưa clean-environment compatibility test.
- New tests: **89 passed in 5.95s**, 0 warnings (87 unit + 2 real-input policy
  integrations trên Stage 4 artifacts, không geometry/inference accuracy claims).
- Full suite: **234 passed, 4 warnings in 39.55s**, failures/errors/skips=0.
  Warnings protobuf từ 4 MediaPipe inference integrations GĐ2/GĐ4 hiện có.
- JUnit lưu dưới outputs/stage5_1/verification-0400b5b2e23e4b8cb05c0171bf2afd9b;
  full command/test/code hashes/environment: [snapshot](artifacts/stage5_1_verification.json).
- Hash đối chiếu xác nhận video/pose/quality/preprocessing source, tests/configs
  GĐ0–4 và recorded Stage 4 outputs không bị sửa. Common flags chỉ bổ sung codes;
  package version cập nhật, không đổi upstream algorithms/schema/policies.
- [Registry snapshot](artifacts/stage5_1_feature_registry.json) và
  [report/DoD/limitations](STAGE5_1_FEATURE_REGISTRY_REPORT.md).
- GĐ5.1 COMPLETE; toàn GĐ5 IN_PROGRESS, không geometry calculation, segmentation,
  repetition/phase, form assessment, scores/feedback. GĐ8 vẫn BLOCKED.

## Giai đoạn 5.2 — hoàn thành 2026-10-08

- Đọc đủ 9 tài liệu hiện hành, inspect contracts/policies/tests GĐ0–5.1; workspace
  không Git. Before manifest hash 167 files, gồm source/docs/configs/tests và
  pre-existing outputs; không đọc archive hoặc sửa dataset.
- Thêm `gym_coach.geometry.Point2D` và `joint_angle(a,b,c) -> float`: góc trong
  tại B, `[0,180]` degrees, không làm tròn hoặc gắn semantics bài tập/hệ trục ảnh.
- Input tuple/list int/float hữu hạn; không bool/null/wrong dimension/NaN/Inf.
  Zero BA/BC có measurement error, A=C khác B cho 0°, không epsilon threshold.
- Arithmetic standard library: bounded directions, finite-positive denominator,
  cosine clamp và atan2 dot/cross. Có tests subnormal, overflow-safe differences,
  near-collinear và large integer translation không tạo vector zero giả.
- Additive reasons invalid_geometry_input / degenerate_geometry /
  geometry_numerical_error; stage features, S3. Không đổi upstream semantics.
- Package 0.5.1; offline editable reinstall và pip check exit 0, không dependency
  mới. Foundation/domain schemas và registry gym-feature-registry-v1/hash giữ nguyên.
- Final new tests: **94 passed in 1.00s**, errors/failures/skipped/warnings=0.
  Full suite: **328 passed, 4 warnings in 36.58s**, errors/failures/skipped=0.
  Warnings protobuf deprecation từ 4 fresh MediaPipe integrations GĐ2/GĐ4 hiện có.
- JUnit/before manifest nằm trong outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2;
  commands/environment/code hashes và preservation audit: [evidence](artifacts/stage5_2_verification.json).
- [Báo cáo API/error policy/DoD/limitations](STAGE5_2_JOINT_ANGLE_REPORT.md).
  Synthetic mathematical validation không chứng minh pose accuracy/form correctness.
- Không nối primitive vào registry/extractor, không numeric FeatureObservation/
  FeatureSequence, feature calculation, rep/phase, form/score/feedback. GĐ5.2
  COMPLETE; toàn GĐ5 IN_PROGRESS; GĐ5.3 NOT_STARTED và assessment BLOCKED.

## Giai đoạn 5.3 — hoàn thành 2026-10-08, phạm vi một feature

- Đọc/đối chiếu 10 docs hiện hành và source/tests GĐ0–5.2; before manifest
  lưu 197 file hashes. Không Git repository, không dùng archive hoặc giả diff.
- Package 0.5.2; NumericFeatureObservation schema 0.2.0 mới, không thay
  FeatureObservation/definition/registry schema 0.1.0 hoặc registry v1/hash.
- Chỉ `squat_knee_angle_left_deg` (left hip/knee/ankle) có numeric adapter.
  Primitive version joint-angle-2d-v1; geometry arithmetic functions không đổi.
- observe_feature dùng canonical registry/definition và check_feature_availability;
  không primitive trước gates, không recompute/promote quality, không scale gate
  ngoài definition, không đổi tọa độ/provenance. Structured input/unsupported errors.
- Available là float degrees/unvalidated; unavailable null + policy hoặc
  measurement reason/flag. Identity, hashes, tier, units, quality snapshot và
  ordered joint source_status/status/imputation references được lưu và kiểm tra.
- Context imputation refs phải bracket index/timestamp; output identity khớp
  snapshot chính xác. Detached context vẫn thiếu FPS/raw sequence để độc lập
  chứng minh caller đã ghép đúng frame/report; limitation được ghi, không đoán FPS.
- Final tests mới **120 passed in 9.20s**, failures/errors/skips/warnings=0
  (118 unit + 2 saved real-input integrations). Full suite **448 passed,
  4 warnings in 43.94s**, failures/errors/skips=0; warnings protobuf GĐ2/GĐ4.
- Initial run 120 passed/1 Pydantic serialization warning; sửa intake definition
  để serialization mismatch thành structured input error, không suppress warning.
- Ba sampled frames/video trên GĐ4 squat 267 và HSPU 78 frames đều unavailable,
  primitive calls=0, snapshot/JSON/input bytes giữ nguyên; HSPU dùng cùng squat ID
  để kiểm tra wrong-exercise, không tính feature HSPU thứ hai.
- Available measurements là synthetic (90/180/0/45°, tolerance 1e-10 degrees).
  Imputed available dùng fresh synthetic caller report contract fixture, không
  QualityGate output/real report promotion; actual pipeline imputation giữ degrade.
- Offline editable install/pip check exit 0, không dependency/model/framework mới.
- [Report/DoD/limitations](STAGE5_3_FEATURE_OBSERVATION_REPORT.md),
  [command/environment/hash/measurements evidence](artifacts/stage5_3_verification.json).
- GĐ5.3 scoped COMPLETE; toàn GĐ5 IN_PROGRESS. Chưa feature khác, full extractor,
  FeatureSequence, bilateral fusion, rep/phase/form/score/feedback/training.

## Giai đoạn 5.4 — hoàn thành 2026-10-09

- Đọc đủ 11 docs hiện hành, inspect GĐ0–5.3 contracts/tests và workspace không Git.
  Before manifest lưu 265 file hashes; không mở archive hoặc sửa dataset/output cũ.
- Package 0.5.3, không dependency mới; LinkedFeatureInputContext schema 0.2.0,
  factory preprocessing-context-v1. Legacy context, numeric 0.2.0 và registry v1/hash giữ nguyên.
- Factory revalidate full result từ Python-mode serialization + JSON allow_nan=false
  + strict=True; tránh serializer đổi NaN→null hoặc bool→float. Reject index/type/
  structural/audit failures có GymError flags features/input_error/S3.
- Exact processed/raw video/index/timestamp, raw frame/sequence hashes, quality
  snapshot/hash, profile/resolved profile, imputation/runs và result/metadata hashes.
  Snapshots tách caller nested state, không mutate coordinates/provenance/flags/report.
- Sửa tối thiểu validate_quality GĐ4 sang core-count/span audit, bỏ temporary
  QualityGate evaluation; không tính lại/promote eligibility. Geometry/interpolation/
  normalization/smoothing functions giữ nguyên. Registry/policy không đổi.
- Adapter chỉ thêm linked-context strict intake; legacy behavior và numeric output
  schema giữ nguyên. Context JSON lưu riêng cùng observation, bind measurement_input_hash.
- Tests initial: 89 passed/1 failed, 0 warnings; phát hiện JSON-mode bool→1.0.
  Repaired: 95 pass/0 warnings. Final: **109 passed in 22.99s**, 0 fail/error/skip/warnings.
- Full suite: **557 passed, 4 warnings in 60.60s**, 0 fail/error/skip; warnings
  protobuf SymbolDatabase.GetPrototype từ bốn fresh MediaPipe GĐ2/GĐ4 integrations.
- Real saved GĐ4: squat 267 frames, sampled 0/133/266; HSPU 78 frames, 0/39/77.
  Context JSON roundtrip và identity checks đạt; 0 available/6 unavailable, primitive
  calls=0, skip/unknown/missing metric giữ nguyên; HSPU vẫn negative squat-feature case.
- Offline editable reinstall/pip check exit 0; chưa clean-environment verification.
- Full result snapshot là O(N) mỗi context; chưa run identity/producer authentication
  hoặc chứng minh exact report origin khi raw core statistics trùng. Hash không chữ ký.
- [Report/DoD/limitations](STAGE5_4_CONTEXT_FACTORY_REPORT.md),
  [commands/environment/hash evidence](artifacts/stage5_4_verification.json).
- GĐ5.4 COMPLETE; toàn GĐ5 IN_PROGRESS, GĐ6–7 chưa bắt đầu, assessment BLOCKED.

## Giai đoạn 5.5 — hoàn thành 2026-10-09

- Đối chiếu 11 docs hiện hành và implementation/tests; trước sửa lưu 385 hashes.
  Workspace không Git; không giả commit/diff, không dùng archive/dataset code.
- Package 0.5.4, không thêm dependency; canonical right knee hip–knee–ankle dùng
  primitive joint-angle-2d-v1. Registry v1/hash, policy, factory và upstream giữ nguyên.
- KneeAngleFeatureObservation schema 0.3.0 nhận đúng hai knee IDs, right adapter
  squat-right-knee-v1. Left dispatch vẫn NumericFeatureObservation 0.2.0;
  legacy class/source/payload không đổi, from_legacy là migration tường minh.
- Readiness trước primitive, allow-only exact right metric/joints/side/exercise;
  không recompute/promote quality hoặc thêm scale gate. Unavailable null với
  nguyên policy flags hoặc measurement error đúng video/frame/feature/joint names.
- Strict audit/hash/provenance/unit/status/flag/zero-vector validation; serialized
  readback không tính góc. Imputed allow fixture là synthetic contract riêng,
  không QualityGate evidence; source report degrade vẫn giữ nguyên.
- Tests mới đạt ngay run đầu: **118 passed in 30.90s**, 0 fail/error/skip/warnings
  (116 unit + 2 saved real-input integrations). Full suite: **674 passed,
  4 warnings in 94.59s**, 0 fail/error/skip. Warnings protobuf GĐ2/GĐ4 như trước.
- Một parameter test cũ kỳ vọng right knee unsupported được cập nhật theo phạm
  vi mới; 556 retained tests và 118 mới đạt. Các calculators còn lại vẫn unsupported.
- Squat 267 frames, sampled 0/133/266; HSPU 78 frames, sampled 0/39/77:
  0 available/6 unavailable, primitive calls=0, skip/unknown/missing metric giữ
  nguyên. HSPU là wrong-exercise case cho squat ID; không tạo HSPU feature.
- Left payload/JSON khớp bảy golden GĐ5.3 (một synthetic available, sáu real
  unavailable). Saved Stage4 bytes và prior artifacts giữ nguyên qua hash audit.
- Offline editable install và pip check exit 0; chưa clean-environment verification.
- [Report/DoD/limitations](STAGE5_5_RIGHT_KNEE_FEATURE_REPORT.md),
  [commands/environment/hash evidence](artifacts/stage5_5_verification.json).
- GĐ5.5 COMPLETE theo phạm vi; toàn GĐ5 IN_PROGRESS. Chưa numeric real-input
  đủ điều kiện, accuracy/form validation hoặc tám angle calculators còn lại;
  hai ratio candidates blocked; chưa fusion/batch/rep/phase/assessment/training.

## Giai đoạn 5.6 — hoàn thành eligibility/data audit, 2026-10-10

- Đọc/đối chiếu 10 current docs và quality/registry/context/adapters/tests; workspace
  không Git. Before manifest 737 files gồm 224 squat videos/7 dataset metadata.
  Không archive restoration, không sửa/xóa dataset hoặc output/report cũ.
- Package 0.5.4 và runtime/schema/profile/registry/flags giữ nguyên; chỉ thêm
  test-only fixture, hai test files và read-only inventory script.
- Genuine QualityGate quality-s3-v1 tạo report từ 31 deterministic side/in-frame
  PoseFrames, nhất quán ID/timestamps ở 30/60 FPS, không sửa MetricEligibility.
  Sequence allow/eligible=true và exact left/right knee metric allow đạt.
- Linked preprocessing/factory/hai adapters cho 90° synthetic, unvalidated;
  không measurement người thật, không accuracy/form evidence. Global allow
  không thay exact metric allow, wrong joints bị readiness từ chối dù gate allow.
- Cases thiếu/sai metric key/joints, unknown/front/oblique, short/degrade/skip,
  low/missing/OOF và imputed gap không gọi primitive hoặc recompute/promote report.
- Inventory: 112 original + 112 subset files = 224 paths/112 content hashes;
  camera_view UNKNOWN toàn bộ, không documented side-view metadata/protocol.
  0 eligible, 1 path ineligible do recorded quality skip, 223 blocked chưa report
  trực tiếp. Không visual inspection/inference mới toàn dataset.
- Saved Stage4 report/hash revalidated; một source g01_c01 có skip/unknown,
  thiếu hai knee keys. Cả hai adapters trên samples0/133/266 unavailable/null,
  primitive calls=0; report/file bytes nguyên bản. Duplicate original không
  được gán report eligibility hoặc independent annotation từ subset copy.
- New run đạt ngay: **39 passed in 17.40 s**, 0 fail/error/skip/warnings
  (37 synthetic unit + 2 real audit/integrity tests). Full suite **713 passed,
  4 warnings in 100.53 s**, 0 fail/error/skip; 674 tests GĐ0–5.5 giữ nguyên.
  Warnings protobuf deprecated từ bốn fresh MediaPipe integrations cũ.
- Pip check exit0, no broken requirements; không dependency/install thay đổi,
  chưa clean-environment claim. Read-only CLI inventory run exit0, artifacts mới.
- [Report/DoD/user input requirements](STAGE5_6_DEVELOPMENT_ELIGIBILITY_REPORT.md),
  [environment/commands/preservation/test evidence](artifacts/stage5_6_verification.json).
- GĐ5.6 COMPLETE vì gate và audit có kết quả rõ ràng. Numeric real-video vẫn
  **BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**, toàn GĐ5 IN_PROGRESS, assessment blocked.

## Giai đoạn 5.7 — input check 2026-10-10, BLOCKED

- Đọc current state/plan/tasks/schema, README và reports GĐ5.5/5.6; kiểm tra
  metadata/protocol sources. Workspace vẫn không Git, không giả commit/diff.
- Fresh file enumeration: 14,291 video paths, tất cả trong Data; 224 squat
  paths/112 SHA-256 distinct. Không có squat path mới/thay đổi bytes so GĐ5.6.
  Seven dataset metadata giữ nguyên; 112 squat CSV rows đều camera_view UNKNOWN.
- Chưa có video squat mới cùng human side annotation liên kết path/ID/hash/
  protocol/confirmer/date/interval và camera/framing/occlusion limitations.
- Dừng trước inference: decoder/pose/QualityGate/numeric adapters NOT_RUN;
  không có sequence hoặc metric decisions mới. Tests/integration/full suite
  NOT_RUN vì full suite có MediaPipe inference; counts N/A, không claim pass.
- Chỉ thêm [báo cáo blocked](STAGE5_7_DOCUMENTED_SIDE_VIEW_GATE_REPORT.md) và
  [input-check evidence](artifacts/stage5_7_input_check.json), cập nhật bốn current
  docs; code/tests/config/schema, dataset, archive và old reports/artifacts nguyên bản.
- GĐ5.7 chưa COMPLETE, numeric real-video vẫn BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT.
  Bước nhỏ nhất: người dùng cung cấp video/annotation theo checklist báo cáo;
  sau verify mới fresh exact-two-knee QualityGate, chưa gọi numeric adapters.

## Giai đoạn 5.7A — visual audit 2026-10-12, COMPLETE theo phạm vi audit

- Người dùng cho phép xem squat hiện có và exploratory inference dù chưa human
  annotation. Chỉ review Data/BodyWeightSquats; không mở archive/nhãn form.
- 224 paths/112 SHA-256 distinct, mỗi content một representative; duplicate paths
  chỉ liên kết bytes, không suy annotation độc lập. Xem thật 560 PNG mẫu tại
  0/25/50/75/100% clip trên 28 sheets; mọi video decode thành công.
- 44 likely_side, 38 oblique, 22 front, 8 uncertain, 0 not_assessed. Mỗi video có
  path/hash/frame indices/timestamps, lý do camera/body/framing/occlusion và
  confidence định tính; tất cả visually_assessed_unverified, 0 documented.
- Chọn R106 g25_c01 trước inference: profile side ổn định ở các mẫu, lower joints
  trong khung, một người, nền ít vật cản. SHA-256 và selection hash lưu riêng.
- Hai fresh 103-frame runs: raw unknown giữ nguyên; exploratory side là giả thuyết
  agent trên toàn sequence dựa năm mẫu, không annotation chính thức. Registry
  exact left/right keys/joints/side policies, quality-s3-v1 và threshold không đổi.
- Raw unknown: sequence skip; left skip(viewpoint_unknown), right skip
  (low_pose_visibility, viewpoint_unknown). Assumed side: sequence skip
  (low_pose_visibility), left allow, right skip(low_pose_visibility).
  Right knee 59/103 low, longest invalid span29 > limit10; không promote decisions.
- Audit caller attempt1 exit1 vì coverage viewpoint mặc định unknown không khớp
  side; runtime contract từ chối đúng. Preserved partial outputs, corrected new
  script/output paths; attempt2 exit0, 9.3955964s. 412 protobuf UserWarning records
  với filter always, 4 native feedback warnings và một native log-routing notice.
- Numeric adapter/primitive calls=0; không preprocessing, interpolation, góc hoặc
  form/accuracy claim. Pytest/new tests/full suite NOT_RUN, counts N/A, vì chỉ audit
  dữ liệu/docs/artifact utilities; application code/tests/config không sửa.
- [Báo cáo](STAGE5_7A_VISUAL_VIEWPOINT_AUDIT.md) tách visual, exploratory gate,
  official documentation và pending human confirmation; inventory/artifact hashes
  lưu tại outputs/stage5_7A/visual-audit-1ccfeefe212e42c6b3efaf813033338c.
- GĐ5.7 vẫn BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, measurement còn fail quality
  trên R106, toàn GĐ5 IN_PROGRESS. Bước nhỏ nhất: xác nhận protocol/annotation cho
  exact input hash/path/ID/interval. Có documentation vẫn cần fresh official gate
  sequence và cả hai metric allow; không mở measurement từ visual confidence.
