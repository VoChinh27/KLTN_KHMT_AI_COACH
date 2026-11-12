# Chẩn đoán thí nghiệm count-only đầu tiên

2026-10-27. **AI weak-reference agreement; NOT_VALIDATED**, không independent
ground truth. Không sửa model, policy, nhãn, split hoặc checkpoint; không training,
backward, optimizer step hay fit baseline mới. Official=false, assessment=false,
G5.7 blocked, G6 IN_PROGRESS. Không có Git repository.

Evidence mới: [run-70efa9e2934c4557bfc0b82e7be78012](../outputs/squat_count_diagnosis/run-70efa9e2934c4557bfc0b82e7be78012/).
Đọc đúng5 run user chỉ định: preparation `run-6d45…`, baseline/training
`d34e179493d74e07ab3a7b13e2254aff`, baseline-test/model-test
`71b2e1ac019d46d1aa9a7bc4c41bea3d`. Đường dẫn đầy đủ và SHA từng file trong
`plan.json`; **291 files cũ không đổi**. Kiểm dataset/config/code/environment identity,
50 checkpoint SHA sidecars, training/validation IDs và evaluator/source/label linkage.

## 1. Kết quả được xác minh

| Phương pháp | Train MAE (33/67) | Validation MAE (3/21) | Development-test MAE (11/24) |
|---|---:|---:|---:|
|Constant train median=2|1,090909|1,000000|0,818182|
|Duration baseline đã fit bởi user|0,495618|0,804604|0,533840|
|GRU được chọn, epoch index5|1,555147|0,709626|1,145287|

Mẫu số trong ngoặc là comparable/full split. Development-test có13 rejected;
không coi MAE trên11 là kết quả toàn24. GRU signed error=-1,101294; exact rounded
match4/11. 11 predictions nằm trong **[0,726750;1,065443]**, tất cả làm tròn1.
Duration baseline exact7/11, signed error+0,212311. Không đổi quy tắc làm tròn.

Development-test chỉ **kiểm lại JSON đã lưu và metric/linkage**, không inference
bất kỳ checkpoint nào trên split này. Train/validation được replay inference-only.
Không chọn checkpoint khác dựa trên test. Coefficients baseline giữ nguyên
`-0,4373505967 + 0,6005234517 × duration_seconds`, clamp≥0; không fit lại.

## 2. Admission làm thay đổi phân bố ra sao

Giữ đủ112 records, train67/validation21/development-test24. Train có8 target null,
3 trong đó input accepted; chỉ33 clip/12gXX vào loss. Validation21 labels available,
nhưng chỉ3 clip/1gXX accepted. Bảng count chỉ đếm **nhãn known**, không biến null=0:

| Count | Train toàn bộ known (59) | Train supervised (33) | Train rejected known (26) | Validation toàn bộ (21) | Validation admitted (3) |
|---|---:|---:|---:|---:|---:|
|0|2|2|0|1|1|
|1|12|8|4|11|1|
|2|14|8|6|7|1|
|3|12|8|4|2|0|
|4|8|5|3|0|0|
|5|8|2|6|0|0|
|6|3|0|3|0|0|

Train rejected31 gồm26 known+5 null. Mean count known giảm2,814→2,364; nhóm
rejected mean3,385. Duration train toàn67 mean5,630s; supervised33 mean4,664s,
median4,48, range1,96–8,04; rejected31 mean6,801s, range2,48–10,68.
Đây là selection bias quan sát được, không bằng chứng admission sai hoặc cần hạ gate.
Train mất hoàn toàn g01/g18/g22; supervised giữ g05,g06,g08,g09,g10,g11,g12,
g17,g19,g20,g21,g23. GXX không là subject identity chắc chắn.

Validation mất toàn g03/g04/g07/g13 và g02_c03; reasons chính vẫn thiếu support/
long gap. Mean count21clip1,476→1,000 sau admission; duration toàn21 mean3,566s,
admitted3 mean3,733s, rejected18 mean3,538s. Ba clip còn lại:

| Video (`v_BodyWeightSquats_…avi`) | AI count | Duration(s) | Prediction index5 |
|---|---:|---:|---:|
|g02_c01|1|4,08|1,004773|
|g02_c02|0|2,28|1,089140|
|g02_c04|2|4,84|0,965035|

[distributions.json](../outputs/squat_count_diagnosis/run-70efa9e2934c4557bfc0b82e7be78012/distributions.json)
liệt kê từng clip bị loại, reason, group, count/null và duration; không loại âm thầm.
Tương quan mô tả duration/count trong33 supervised train là0,871; val3 là0,974
(n=3 quá nhỏ). Không là fit mới, independent accuracy hay quan hệ nhân quả.

## 3. Vì sao epoch thứ6 được chọn

`best_epoch=5` là **index0-based, tức epoch thứ6**, không off-by-one. Selection
đã khóa là validation continuous MAE nhỏ nhất, không training loss nhỏ nhất.
Index5 là minimum của cả50 logs; các lân cận index4/6 là0,748616/0,748677.

| Index | Logged train SmoothL1 | Train MAE | Validation MAE | SD prediction train/validation |
|---|---:|---:|---:|---:|
|0|1,585050|2,007518|0,889743|0,0694 /0,0413|
|5|1,220576|1,555147|0,709626|0,0926 /0,0517|
|6|1,140184|1,507182|0,748677|0,0918 /0,0504|
|19|0,692700|1,090305|1,122513|0,0621 /0,0138|
|49|0,316122|0,686292|1,174414|0,7293 /0,4763|

SD là population SD trên33/3 comparable clips. Ở index5, **33/33 train cũng
rounded1**, range0,786923–1,106773; hiện tượng không chỉ xuất hiện trên test.
Mean prediction train0,968/val1,020 trong khi target means2,364/1,000.
Tại index49, mean train2,327/val2,174: tăng mức dự đoán giúp train nhưng gây
overprediction trên phần lớn validation. Sai số val cho target1/0 tăng từ
0,0048/1,0891 lên1,4146/1,5094; target2 giảm từ1,0350 xuống0,5993 nhưng không bù
hai clip kia. Không có mâu thuẫn khi train loss giảm.

Với targets0,1,2, dự đoán hằng1 cho MAE toán học2/3. Đây là phân tích hình dạng
objective của3 labels, **không fit/đề xuất baseline từ validation**. Selection
hiện tại ưu tiên một trạng thái gần hằng1; chưa đo được khả năng phân biệt count
qua nhiều groups. Không tự suy rằng index49 hoặc thêm epochs sẽ tốt trên dữ liệu mới.

![Đường loss và phân bố predictions](../outputs/squat_count_diagnosis/run-70efa9e2934c4557bfc0b82e7be78012/training-validation-diagnostics.png)

Đủ50 epochs và predictions trong `epoch-diagnostics.json`. Logged training loss
là average minibatch ở các weights đang cập nhật; khác full-split loss tính lại
sau epoch (index5:1,220576 vs1,163749) là đúng cách ghi log, không lỗi checkpoint.
Run user lưu50epochs/250optimizer steps, elapsed29,485s trong log (không phải wall
time gồm startup/verify). Chẩn đoán này không tạo thêm steps.

## 4. Kiểm implementation và perturbation

**Lỗi implementation đã xác nhận: chưa phát hiện trong các đường đã kiểm.**

- Re-encode raw pose của39 accepted train/validation inputs, cache khớp toàn bộ
  values/mask/time/status/reasons.8 joints ×7channels +elapsed/dt=58;
  `(x−0.5)W/H, y−0.5`, không hip-centering, learned scaling hoặc góc khớp.
  Timestamps=`i/FPS`, duration=`N/FPS`, toàn cohort25FPS. Target0 vẫn là0;
  target null bị loại khỏi supervised loss. SmoothL1 beta1, output Softplus≥0.
- Batch-vs-single sai khác tối đa2,38×10⁻⁷; thay padding bằng12345 không đổi output.
  Hook xác nhận hidden state bằng0 sau incomplete sample/padding.14/33 supervised
  train có gaps:120/3848frames incomplete,85 valid runs; validation0/280 incomplete,
  3 runs. Các partial coordinates ở incomplete frames không góp hidden; binary
  provenance/mask channels tại complete frames là hằng. Đây là thiết kế đã khóa.
-50 checkpoint đều có `best_model` đúng bằng raw `model` của earliest validation
  minimum đến epoch đó; final `epoch-049.pt.best_model` đúng state index5. Replay
  validation lệch tối đa7,15×10⁻⁷. Source/target IDs và optimizer saved counters khớp;
  không nạp optimizer để chạy. Epoch0 có predictions rounded0, nên output không
  bị code ép tối thiểu1. Mọi trọng số sau inference/probes không đổi.

Perturbations chỉ trên supervised train33/validation3, tại **saved best5 và scheduled
final49**; chọn hai trạng thái này trước khi chạy probes, không chọn bằng test.
Giữ mask/provenance khi perturb tọa độ. Đây có thể là input ngoài phân bố,
**không đo accuracy và không chứng minh causality**. Mean absolute prediction delta:

| Probe | Index5 train/val | Index49 train/val |
|---|---:|---:|
|Freeze x/y ở pose complete đầu tiên|0,00314 /0,00415|0,04793 /0,04281|
|Đảo x/y trong từng valid run|0,00033 /0,00041|0,00728 /0,02450|
|Shuffle x/y trong valid run, seed cố định|0,00026 /0,00027|0,00578 /0,01691|
|Elapsed channel=0; giữ dt/duration|0,00979 /0,01021|0,97120 /0,90918|
|Duration head=1s; giữ sequence/time|0,18297 /0,14008|0,33392 /0,24502|
|Elapsed,dt,duration×2|0,21787 /0,18571|0,87048 /0,80165|

Direct duration-head coefficient tại5 là **−0,077711**, tại49 là+0,101510.
Kiểm tra công thức Softplus khi chỉ đổi duration khớp≤2,38×10⁻⁷. Tại5, hidden
contribution vào logit train SD chỉ0,0253, trong khi duration contribution SD0,1489:
phần head trực tiếp gán count thấp hơn cho clip dài hơn, khi giữ hidden cố định.
Không suy từ hệ số này rằng toàn model có quan hệ nhân quả với duration.

Mean pooling không phải bộ tích lũy count tường minh. Probe nhân đôi nguyên
sequence, chèn reset, **cố ý giữ nguyên time/duration**, cho output gần như y hệt
(max delta2,38×10⁻⁷): mean làm mất multiplicity của các hidden runs trùng nhau.
Probe này không phải video hợp lệ. GRU vẫn phụ thuộc order; elapsed/duration vẫn
có thể mang thông tin độ dài. Vì vậy chưa thể kết luận mean pooling gây toàn bộ
thất bại hoặc sum pooling chắc chắn tốt hơn.

**Vấn đề dữ liệu/evaluation có bằng chứng:** selection bias do admission;
validation chỉ1group/3labels; checkpoint5 gần hằng1 ngay trên train; nhạy với các
probe thời gian hơn probe pose đã thử; tất cả metrics chỉ so AI weak reference.

**Giả thuyết chưa kiểm chứng:** validation đơn nhóm khiến checkpoint selection
không đại diện; model dùng duration/elapsed shortcut; mean pooling/reset làm giảm
thông tin count. Chưa tách được đóng góp nhân quả của từng yếu tố, chưa chứng minh
model thiếu capacity hoặc thiếu epochs. Không đổi model từ các giả thuyết này.

## 5. Đúng một thí nghiệm tiếp theo — CHỈ ĐỀ XUẤT

**Kiểm tra độ ổn định checkpoint selection bằng grouped3-fold nội bộ trên33 train
clips/12gXX, giữ nguyên model/input/loss/50epochs/seed/admission.** Thay đổi duy nhất
ở protocol đánh giá/chọn epoch; chưa thay producer hoặc triển khai folds. Cần user
chốt thiết kế trước implementation; mọi training tiếp theo do user tự chạy.

Trước chạy: khóa3 inner folds bằng thứ tự SHA-256(`squat-count-inner-cv-v1|gXX`),
round-robin3 (4groups/fold); không stratify/chọn lại bằng count/quality. Giữ nguyên
outer split67/21/24, không chuyển clip giữa train/validation/development-test.
Ba runs fit trên8 train groups, infer trên4 inner-held-out groups; không fit trên
outer validation và không đọc thêm development-test. Không refit model cuối trong
thí nghiệm chẩn đoán đề xuất này.

Tiêu chí kiểm chứng: mỗi clip trong33 có OOF prediction đúng một lần/epoch, không
group/hash leakage, cùng33 denominator ở cả50epochs. Giữ continuous clip-MAE làm
metric chính: pooled OOF trên33clips; MAE từnggroup/fold và macro-MAE12groups là
diagnostics phụ. Ghi earliest minimum, phân tán fold-best epochs và chênh
OOF clip-MAE(index5)−minimum. Nếu index5 vẫn là minimum ổn định, chưa ủng hộ việc quy
vấn đề chỉ cho validation cũ; nếu khác, báo độ lệch và độ bất ổn, không tự tuyên bố
model mới tốt hơn. Không đặt accuracy target hoặc chọn hyperparameters từ test.

Rủi ro: chỉ12groups, inner training nhỏ hơn và số optimizer steps/epoch khác run
full-train, nên epoch minima không là phép so nhân quả trực tiếp. AI label bias/
subject leakage chưa giải quyết; OOF không là independent/unseen-subject accuracy.
**Development-test đã bị xem và đã ảnh hưởng bối cảnh chẩn đoán/đề xuất này.**
Không còn dùng kết quả đó như kiểm chứng độc lập cho thiết kế sau, không chọn lại
epoch bằng nó. Việc ưu tiên sửa protocol dựa chủ yếu evidence train/validation,
không làm mất sự tiếp xúc development-test đã xảy ra.

## 6. Thao tác, tests và bảo toàn

Thêm `scripts/diagnose_squat_count_experiment.py`, `tests/test_count_diagnosis.py`,
báo cáo này và output riêng. Core8 research modules, schemas/profiles/source/labels/
split/checkpoints giữ nguyên; không UI, inference pose, videos playback, train hoặc
baseline fit. Python3.11.9/torch2.14.0+cpu; chẩn đoán28,634s. Plot dùng matplotlib
3.11.2 có sẵn, không cài dependency; đã xem PNG bằng `view_image`, không browser QA.

```powershell
.venv/Scripts/python.exe scripts/diagnose_squat_count_experiment.py --output-dir outputs/squat_count_diagnosis/run-70efa9e2934c4557bfc0b82e7be78012
.venv/Scripts/python.exe -m pytest tests/test_count_diagnosis.py tests/test_count_research.py -q --junitxml=outputs/squat_count_diagnosis/run-70efa9e2934c4557bfc0b82e7be78012/tests-v1/junit.xml
```

Lệnh thứ nhất đã chạy; muốn tái chạy cần output-dir **mới**, không ghi đè run này.
**76 passed (16 diagnostic +60 research),0 failed/errors/skipped/warnings,6,63s**.
Trước bổ sung test report-tamper đã có15 diagnostic tests pass; không cộng lượt cũ
vào76. Chưa full suite. Tests và probes không là độ chính xác model trên người tập.
