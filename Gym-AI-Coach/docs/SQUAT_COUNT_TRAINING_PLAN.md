# Thí nghiệm count cấp video với AI weak reference

## Cập nhật hiện hành — baseline trên cùng inner-CV,2026-10-27

User đã chạy grouped3-fold; OOF `oof-7a3f559a10a345958de7b3e72df2acda`
khớp3 final checkpoints và50epochs×33clips. Giữ selection index49/epoch50;
pooled MAE1,0673224962 và macro12-group1,0498952053 chỉ là GRU OOF đã lưu.
Không dùng kết quả này chọn lại kiến trúc, epoch, hyperparameters hoặc gate.

[Bàn giao công cụ riêng, verification và lệnh đầy đủ](../outputs/squat_count_cv_comparison/prepare-b69b56e1835149a2811620b7153290e1/START_HERE.md).
Không sửa `research/` hoặc `inner_cv/`. Constant=median train targets;
duration=`frame_count/FPS`, OLS có intercept trên train_ids từngfold, variance0
dùng slope0/intercept mean target, clamp predictions≥0, không rounding metric.
Fold train20/20/26 và evaluation13/13/7 giữ nguyên; không dùng hệ số fit toàn33.

User-run tạo3fold parameter sets/2methods,33OOF predictions/method, source/label/
row/parameter/prediction hashes, train/eval IDs; đối chiếu GRU cùng33clips tại
index49. Báo pooled clip-MAE, macro12group-MAE, group/fold MAE và Δabsolute error
cùngvideo (GRU−baseline). Missing/duplicate/wrong linkage là error, không bỏ mẫu.
Verification đã kiểm inputs/checkpoint histories mà không gọi model hoặc fit.
174related tests pass (44mới),0fail/error/skip/warnings; chưa full suite.
779files cũ không đổi. **Agent không fit baseline thật/train/final refit**;
comparison thật **[RESULT PENDING]**, user tự chạy. Chưa kết luận phương pháp tốt hơn.
AI weak labels, không independent GT; OOF dùng chọn epoch không là đánh giá
độc lập cuối cùng. Không dùng outer validation/development-test; test exposure
cũ vẫn công khai. NOT_VALIDATED, official=false, assessment=false, G5.7blocked,
G6 IN_PROGRESS. Các phần CHƯA TRAIN dưới đây là lịch sử bàn giao.

## Lịch sử chuẩn bị inner-CV

**Cập nhật2026-10-27:** user đã phê duyệt triển khai grouped3-fold theo
[chẩn đoán thí nghiệm đầu](SQUAT_COUNT_FIRST_EXPERIMENT_DIAGNOSIS.md).
Đã chuẩn bị code, manifest và tests; **agent không train/fit/backward/step**.
Thí nghiệm CV thật chưa chạy; user tự chạy theo bàn giao dưới đây. Checkpoint5
và mọi kết quả thí nghiệm đầu giữ nguyên, không chọn lại bằng development-test.

## Grouped inner-CV — MODEL TRAINING GATE: READY FOR USER-RUN

[Hướng dẫn, lệnh từng fold/resume/aggregate và test evidence](../outputs/squat_count_inner_cv/run-985d93a330b6492b94f16ec8d2332221/START_HERE.md).
[Manifest khóa](../outputs/squat_count_inner_cv/run-985d93a330b6492b94f16ec8d2332221/frozen-v1/manifest.json)
SHA-256 canonical `8f98e626657b64c038d07a8085d64d452fc2715c6f18ec49c02995ee335352bf`.
Reused dataset byte SHA-256
`97f3619c18e95e9d7a1384596368e9b5fd5c4628a5d3be693b9549c09037502d`.

Câu hỏi duy nhất: checkpoint selection có ổn định hơn khi đánh giá nội bộ qua
nhiều gXX thay vì3clip/g02 ở validation cũ? Giữ33 supervised train clips/12groups,
không đưa34 train clips thiếu admission/target hoặc21val/24development-test vào
inner-CV. Kiểm hashes/linkage toàn112 để bảo toàn source, không là sử dụng outer
validation/test cho fitting hoặc selection. Không chạy readiness/inference lại.

Khóa SHA-256(`squat-count-inner-cv-v1|gXX`) sort tăng dần, round-robin3, không
stratify/chọn lại theo nhãn, chất lượng hoặc kết quả:

|Fold|Học: clips/groups|Nhóm học|Đánh giá: clips/groups|Nhóm đánh giá|
|---|---:|---|---:|---|
|0|20/8|g06,g08,g09,g10,g12,g17,g19,g21|13/4|g05,g11,g20,g23|
|1|20/8|g05,g08,g09,g10,g11,g12,g20,g23|13/4|g06,g17,g19,g21|
|2|26/8|g05,g06,g11,g17,g19,g20,g21,g23|7/4|g08,g09,g10,g12|

Giữ GRU16,58channels,normalization/admission/mask/reset/mean pooling,
SmoothL1 beta1,AdamW lr0,001/weight_decay0,0001,grad norm1,batch8,
seed20261026,50epochs,CPU2threads. Mỗi fold fresh model/optimizer, cùng seed
nên cùng initial values nhưng không chung storage/optimizer state. Resume chỉ
đúng manifest/fold/config/code/environment; lưu raw model,optimizer,RNG và toàn
epoch history. Không sửa8 research modules; checkpoint cũ vẫn đọc bằng loader cũ.
Không fit baseline mới; baseline lịch sử giữ nguyên, không dùng chọn epoch CV.

OOF chỉ từ held-out inner partition tại **cùng epoch**: mỗi clip đúng1prediction,
đủ33clips/epoch ×50epochs. Primary pooled continuous clip-MAE; thêm fold MAE
(13/13/7), group MAE và macro12-group MAE. Missing/failure/duplicate không giảm
mẫu số; pooled metric null và selection bị chặn nếu không đầy đủ/hợp lệ.
Chọn earliest minimum của pooled curve khi đủ toàn50; so index5 với min và báo
phân tán fold-best indices (range/mean/populationSD). Không trộn fold-best weights
để làm một epoch chung. Selector CV mới tách khỏi field `Config.selection` cũ.
Checkpoint mới `squat-count-inner-cv-checkpoint-v1`, CLI `gym_coach.counting.inner_cv.cli`.

**Đã chạy:** prepare/integrity112, forward-only6 batches (không grad/đổiweights),
208tests (54CV+60research+16diagnosis+78counting),0fail/error/skip/warnings,
16,24s.320 files cũ giữ hashes, old epoch049 checkpoint load thành công.
**Chưa chạy:** CV training/baseline fit, real OOF aggregation/selection, final refit,
full suite, training-continuous-vs-resume equivalence thực tế. Resume/aggregate đã
kiểm bằng fixtures không học; tests không là accuracy. Không dependency mới.

AI weak references, không independent GT; gXX không chắc là subjectID. Admission
selection bias còn nguyên, train subset nhỏ hơn nên scheduled steps/epoch khác
thí nghiệm đầu (3/3/4, chưa thực hiện). Development-test đã exposed và đã ảnh
hưởng bối cảnh chẩn đoán; không dùng để tối ưu thiết kế CV này, không xem là
holdout độc lập. OOF minimum dùng chọn epoch cũng không là independent evaluation.
NOT_VALIDATED/official=false/assessment=false,G5.7blocked,G6 IN_PROGRESS.
**[EXPERIMENT NOT RUN YET]**: bước tiếp duy nhất là user chạy3folds và aggregate;
không tự refit hoặc mở rộng phương pháp.

## Lịch sử bàn giao trước thí nghiệm đầu

Phần READY/CHƯA TRAIN dưới đây giữ nguyên lịch sử ngày2026-10-26.

## Quyết định hiện hành và MODEL TRAINING GATE — 2026-10-26

**READY FOR USER-RUN EXPERIMENT**, chỉ cho thí nghiệm count-only dưới đây.
Chủ project đã phê duyệt nhánh raw-pose/count-only riêng trong request SHA-256
`f8ab20eb281df77520e1f2ba0d2a218e003e815f6b2f5acee2251506b0c9b683`;
không phê duyệt agent chạy học tham số. **Training runs=0,
baseline fits=0**, không backward/optimizer step, không smoke training. Nhánh
ba event heads bên dưới được tạm gác vì không có target tương ứng; giữ làm lịch sử.
Không yêu cầu người dùng gán lại nhãn cho thí nghiệm AI-reference đã duyệt.

Câu hỏi: “Một mô hình nhỏ học từ chuỗi pose có thể dự đoán số rep cấp video phù
hợp đến mức nào với AI reference hiện có?” Chỉ giải quyết clip-level count;
không event localization, phase segmentation, pause hoặc form assessment.

### Dữ liệu và admission đã khóa trước kết quả

Run: [squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5](../outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/).
[Manifest 112 records](../outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/dataset-v1/dataset.json),
[frozen plan](../outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/frozen-plan.json).
Giữ split gXX/component/duplicate-byte đã khóa; không random lại, không đổi nhãn.
Đã kiểm 112 source hashes, 224 AI passes/source/protocol/evidence, 1.395 files trong
ba inventory cũ (604 AI, 661 feasibility, 130 dataset-final). 104 whole-count
available/8 null; 0 exact exhaustive event labels. Không suy events từ count/hints.
33 AI viewpoint records được kiểm tra riêng; không suy viewpoint của 79 clip còn lại.

Input contract `squat-raw-pose-count-research-v1`, S3/NOT_VALIDATED, hash
`e82627dcbc70d1d363a2afb6015c6622d92ce69b922269077957da73b47acd56`.
Order cố định: vai trái/phải, hông trái/phải, gối trái/phải, cổ chân trái/phải.
Mỗi joint: x, y, visibility, usable mask, one-hot observed/low_visibility/missing;
58 channels tổng cộng gồm elapsed seconds và dt seconds. Presence gốc vẫn ở saved
pose, không dùng làm feature. Raw chỉ có ba provenance trên; không tạo imputed.

- Tọa độ không học thống kê: `x'=(x-0.5)*width/height`, `y'=y-0.5`, trục phải/xuống,
  đơn vị chiều cao ảnh. Giữ chuyển động hông trong ảnh; không torso scaling,
  mid-hip centering, smoothing, interpolation hoặc geometry angle.
- Joint usable khi observed, finite và trong `[0,1]²`; threshold visibility0.5
  giữ theo pose config đã khóa. Tọa độ unusable ở tensor là zero **kèm mask=0**,
  không là measurement 0. Raw values/flags vẫn truy vết được. Không gộp/chọn lại bên.
- Giữ toàn bộ samples/timestamp `frame_index/FPS`, CFR assumption; không crop,
  resample hoặc truncate. Duration=`frame_count/FPS`; first dt=0, sau đó1/FPS.
- Nhận input có5..1500frames, tỷ lệ frame đủ cả8 joints ≥0.70 và longest incomplete
  support ≤10frames. 5/30%/10 kế thừa giới hạn S3; phép giao8 joints bảo thủ hơn
  marginal coverage. 1500 là resource cap. Không điều chỉnh sau khi xem kết quả.
  Giới hạn10frames chưa bất biến FPS; cohort hiện tại toàn25FPS.
- Frame thiếu bất kỳ joint nào reset hidden state và không góp hidden vào pooling.
  Gaps vẫn tồn tại trong timeline/mask; không tạo rep/events hoặc nối events qua gap.
  Model có thể ước lượng count toàn clip từ phần support còn lại; chưa kiểm chứng bias.
- Từ chối evidence flags `multi_person`, `side_switch_discontinuity`, `unstable_signal`;
  corrupt hash/source/timeline/config là input error, giữ record và chặn training gate.
  Backend không bảo đảm identity tracking; không có flag không chứng minh không đổi
  người/cut/camera. AI viewpoint chỉ33, unverified, không dùng làm admission selector.
- Original QualityReport được copy nguyên semantics vào cache và liên kết byte hash
  tới nguồn. `skip/degrade/allow`, UNKNOWN và temporal contracts cũ không đổi.
  Research admission này là quyền input riêng đã được user phê duyệt, không metric allow.

Đã xử lý saved pose/quality đủ112 (71 fresh từ milestone cũ +41 reuse); không
inference mới. Vẫn chỉ41 preprocessing snapshots cũ, không gọi112 là112 snapshots.
Các quyết định input được lưu trước khi mở targets trong
`dataset-v1/input-admission-before-targets.json`.

| Split | Clip/group gốc | Nhãn available/null | Input accepted/rejected | Supervised clip/group | Supervised frames |
|---|---:|---:|---:|---:|---:|
|train|67/15|59/8|36/31|33/12|3848|
|validation|21/5|21/0|3/18|3/1|280|
|development-test|24/5|24/0|11/13|11/4|1255|
|tổng|112/25|104/8|50/62|47/17|5383|

Toàn cohort14.356frames; input error=0, unchecked=0. 3 accepted train inputs có
target null, vẫn giữ nhưng không vào loss. 58 clip thiếu complete-frame coverage,
59 clip có long gap; các reasons có thể chồng nhau, union rejected62. Cộng thêm8
null labels:65 clip không đủ cả input+target. Không bỏ khỏi mẫu số hoặc đổi null=0.
**Validation chỉ3 clips thuộc g02**; checkpoint selection rất yếu, không ổn định.
Giữ policy/split; chỉ sẵn sàng chạy thí nghiệm nhỏ, không đủ claim generalization.

### Model, loss, baseline và evaluation cố định

Một GRUCell16, một tầng, 3.666 parameters; hidden reset tại incomplete support,
mean-pool hidden trên valid frames + duration, Linear→Softplus xuất một số ≥0.
Chọn model nhỏ do chỉ33 training clips, không architecture/hyperparameter search.
Không dùng duration đơn độc trong model; baseline duration kiểm tra shortcut nhưng
chỉ có thể kết luận sau khi user chạy. Không có pretrained weights/download.

Loss SmoothL1 beta1, trung bình theo video; null target bị mask, không thành target0.
AdamW lr0.001, weight_decay0.0001, gradient norm cap1.0, batch8,50epochs,
CPU2threads, seed20261026, deterministic torch algorithms, không augmentation/dropout.
Không learned normalization/scaler. Shuffle train theo seed+epoch; toàn clip cùng
source/group/component thuộc một split. Validation không cập nhật weights.

Hai baseline cùng supervised train33 và cùng admission/mẫu số khi evaluation:
constant=train median; duration=OLS intercept+slope trên train, clamp≥0. Nếu duration
train có variance0: slope0/intercept=train mean, ghi degenerate. **Chưa fit cả hai**.
Không thu thập/fitted statistics từ validation hoặc development-test.

Evaluator `agreement_with_AI_reference_not_independent_accuracy`:
continuous MAE, signed error=`prediction-target`; exact match sau
`floor(max(0,prediction)+0.5)` (không Python bankers rounding). Báo all-split cohort,
labels/input/predictions/comparable counts và coverage trên toàn cohort, comparable
trên labels; failed/unavailable/not_run không biến0 hoặc bị bỏ âm thầm. Metrics null
khi không có comparable pairs. Không tính event/timing F1 từ nhãn count.

Chọn checkpoint bằng **validation continuous MAE nhỏ nhất; hòa giữ epoch sớm nhất**.
Đủ cùng comparable denominator mới cho selection; model failures chặn selection.
Development-test chỉ trong lệnh riêng `--final-evaluation`, sau freeze config; không
tune từ đó. Không phải pristine/unseen-subject holdout: audit đã tiếp xúc dataset,
gXX không bảo đảm subject isolation, chưa có subject/capture IDs độc lập.

Checkpoint từng epoch gồm last model, optimizer state, torch RNG, best model/epoch/
score và training_steps; byte SHA sidecar, dataset/config/code/environment identities.
Resume giữ fixed50epoch budget, dùng directory mới; không sửa checkpoint cũ. Mọi thay
đổi code/data/env làm identity khác thì từ chối resume, không migration ngầm.

### Implementation và evidence không training

Package mới `src/gym_coach/counting/research/`: contract, data, network, baselines,
evaluation, experiment, cli. Optional extra `count-research=[torch==2.14.0]` chỉ
khai báo phiên bản; **không cài/thay environment**. Python3.11.9, torch2.14.0+cpu,
numpy2.4.6, pydantic2.13.5, pytest9.1.1; Windows10.0.26200,12 logical CPUs,
CUDA unavailable. CIM query RAM/CPU model không truy cập được; không claim đã đo RAM.
Input cache112files=16.616.544bytes; batch tensor tối đa8×1500×58×4≈2,8MB,
không phải peak memory của toàn training process. Chưa kiểm clean environment.

Prepare dùng saved artifacts:28,824s. Forward-only3 train inputs `[3,135,58]`:
0,0386s riêng model/collate, output hữu hạn, không grad/parameter change/checkpoint.
Không lưu predictions khởi tạo làm kết quả model; không accuracy claim.
Training runtime **NOT_MEASURED**, baseline/model performance **[EXPERIMENT NOT RUN YET]**.

Tests mới cuối:60; regression219 passed,0 failed,0 errors,0 skipped,0 warnings,
12,96s pytest. JUnit/stdout/command trong `regression-v1/`. Lệnh đã chạy:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_count_research.py tests/test_squat_counting.py tests/test_squat_ai_annotations.py tests/test_squat_review_pilot.py tests/test_pose_feasibility_batch.py -q
```

Receipt ghi thêm `--junitxml` ở directory run mới. Trước đó48 và59 tests mới đã
pass trong các lượt kiểm tra; giữ receipts lịch sử, không cộng thành test coverage.
Tests fixture kiểm admission/gaps/masks, target null, split leakage, hash, evaluator,
padding/reset, loss, checkpoint serialization/restore-only và guards; autouse block
backward/AdamW.step/baseline fit. Baseline fit chỉ kiểm code train-only tĩnh; chưa
kiểm fitted coefficients. Training loop/optimizer behavior thực nghiệm **chưa chạy**.
Không full suite/real-data inference/video integration, không browser/UI.
Không có Git repository; không commit/diff giả.

### Lệnh thật bàn giao — chỉ user chạy các lệnh có học tham số

Chạy từ root project. Hướng dẫn có thể mở trực tiếp trong
[START_HERE](../outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/START_HERE.md).

```powershell
$experimentRoot = 'outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5'
$countDataset = "$experimentRoot/dataset-v1/dataset.json"
.venv/Scripts/python.exe -m gym_coach.counting.research.cli environment
.venv/Scripts/python.exe -m gym_coach.counting.research.cli verify --dataset $countDataset

# USER ONLY: các lệnh bên dưới fit/train; agent chưa thực hiện.
$trialTag = [guid]::NewGuid().ToString('N')
$baselineRun = "outputs/squat_count_experiment/user-baseline-$trialTag"
$trainingRun = "outputs/squat_count_experiment/user-training-$trialTag"
.venv/Scripts/python.exe -m gym_coach.counting.research.cli baseline --user-run --dataset $countDataset --output-dir $baselineRun
.venv/Scripts/python.exe -m gym_coach.counting.research.cli train --user-run --dataset $countDataset --output-dir $trainingRun

# Chỉ sau khi training kết thúc và giữ nguyên config, không tune development-test.
.venv/Scripts/python.exe -m gym_coach.counting.research.cli evaluate --dataset $countDataset --baseline "$baselineRun/baselines.json" --role development_test --final-evaluation --output-dir "outputs/squat_count_experiment/user-baseline-test-$trialTag"
.venv/Scripts/python.exe -m gym_coach.counting.research.cli evaluate --dataset $countDataset --checkpoint "$trainingRun/epoch-049.pt" --role development_test --final-evaluation --output-dir "outputs/squat_count_experiment/user-model-test-$trialTag"
```

Evaluator lấy `best_model` được chọn bằng validation, không mặc định weights epoch49.
Nếu interrupted, user dùng `train ... --resume <epoch-NNN.pt>` với `--output-dir`
mới, giữ file `.sha256.json` cạnh checkpoint. Không restart seed hoặc fit baseline
trên val/test. Không chạy lại prepare/inference để dùng manifest đã tạo.

Giới hạn: AI weak labels cùng assistant/context, không blind/independent/human
validated; admission tạo selection bias; validation rất nhỏ; pose2D không calibrated,
không viewpoint-invariant, không bảo đảm subject tracking. Không có labels exhaustive
cho event heads. **NOT_VALIDATED, official_eligibility=false, assessment=false,
G5.7 blocked, G6/toàn G5 IN_PROGRESS.** Bước tiếp theo: user tự chạy một lần
baseline + cấu hình50epoch đã khóa, rồi so coverage/count agreement với AI reference;
không mở rộng detector hoặc claim accuracy trên vận động viên.

## Lịch sử — đề cương trước quyết định count-only (không là cấu hình hiện hành)

2026-10-24; quy định thực thi cập nhật2026-10-26. Preparation only. **User sẽ tự
chạy training**; agent không chạy training, fine-tuning, optimizer step hoặc smoke
training, kể cả khi data/gate đã sẵn sàng. Ưu tiên hiện hành:
BodyWeightSquats → baseline count → nhãn video độc lập → đối chứng temporal model
→ đánh giá kỹ thuật sau. Không huấn luyện lại MediaPipe, thêm dependency hoặc
chọn kiến trúc chỉ vì phức tạp. NOT_VALIDATED, official=false, assessment=false,
REFERENCE_PENDING; không đóng G5.7/G6 hoặc mở gate cũ bằng đề cương này.

**Quyết định mới — nhánh AI weak reference đã được thực hiện:** Chủ project cho
phép AI tự annotation/review, không yêu cầu người gán trong milestone này. Hai
lượt AI trên112 clips đã lưu riêng:104 whole-count nhất quán,8 unresolved/null,
0 exhaustive exact event labels; đây không human-validated/independent GT. Nhãn
AI có thể làm đầu vào thí nghiệm sau khi kiểm tra tương thích target/admission,
không thay thế đánh giá độc lập. 33 viewpoint AI đều unverified (19 front,13 oblique,
1 unknown); không mở G5.7. [Evidence](SQUAT_AI_ANNOTATION_REVIEW_REPORT.md).
Các yêu cầu human/adjudicated phía dưới mô tả nhánh nghiên cứu độc lập ban đầu;
không được giả hoàn tất hoặc dùng làm provenance cho nhãn AI.

## Câu hỏi và đối chứng

Trên các clip/gXX đã chia trước, một mô hình temporal nhỏ học từ nhãn video độc lập
có giảm count MAE so với PVP S3 giữ nguyên, đồng thời giữ coverage/abstention minh
bạch không? Đây là câu hỏi trên selected development data, chưa phải generalization
đến vận động viên mới. Không giả định neural model tốt hơn hoặc smoothing tăng accuracy.

Baseline đối chứng dự kiến là `squat-pvp-hysteresis-v1` + `squat-temporal-s3-v1`.
Code hiện chỉ nhận synthetic input; **chưa thể dùng trực tiếp cho real count**.
Anchor-only v2 giữ bounds unavailable; point-resolution policy abstain mọi point
khi không có noise discriminator. Không sửa ba producer để có baseline trong lượt này.
[Proposal count admission](SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md) ghi rõ khoảng
cách giữa allowance thử nghiệm của spec và acceptance/code hiện hành.

## Mô hình học đề xuất duy nhất

Một GRU nhỏ một tầng, hidden-size16, ba đầu ra xác suất event sample-resolution
start/bottom/end. Đây là cấu hình khởi đầu S3 để review, **chưa implementation,
chưa fit, chưa benchmark tài nguyên**. Không phải phase classifier hoặc form model.
Phần pairing events thành cycle/count cần freeze trước training và test độc lập
cho shared top, edges, gap, side switch; chưa coi đề cương là quyền thêm detector.

Input dự kiến: angle của **một bên cố định đã khai báo**, mask/provenance và dt;
đạo hàm thời gian chỉ trong đoạn hợp lệ. Không dùng scalar geometry framewise như
temporal authorization; phải có count-only admission được duyệt riêng hoặc đạt
gate hiện hành trước. Không chọn bên theo test performance, không fuse hai bên,
không nội suy qua gap. Sequence không đủ evidence phải giữ unavailable. Không
đưa GT vào quality/missing mask hoặc dùng nhãn để chọn cửa sổ đầu vào thuận lợi.

Target chỉ lấy từ final adjudicated video labels: start/end event tại sample đã
gán; Bottom point chỉ khi label xác định một sample, interval/unknown bottom được
mask khỏi point loss. Count-only labels không tạo được target event/phase; chỉ
dùng cho count evaluation, chưa dùng để supervised-train các heads này. Partial,
uncertain, cut và gap không tự biến thành negative examples. Cần freeze rõ negative
support và loss mask trước MODEL TRAINING GATE; chưa tạo pseudo-label.

Chưa đề xuất predicted phase ground truth từ interval giữa các events: đó là
một target khác cần protocol riêng. Endpoints được rater gán là mốc quan sát ở
sample resolution, không claim physical pause/reversal chính xác.

## Split và leakage

Proposal có seed `squat-count-groups-20261024-v1`, union gXX + duplicate bytes +
same-source links có bằng chứng; hash-order deterministic,60/20/20 theo component.
Manifest khóa trước labels/performance. Trên112 video/25 gXX:15 groups train
(67 clips),5 validation(21),5 development-test(24). Xem split manifest để biết
đúng từng ID/hash. Không chia random frame/window; mọi window cùng source/component
đi cùng split. Related source ngoài gXX chưa xác minh; không bảo đảm subject isolation.

Audit cũ đã xem112/112 videos;40 đã dùng pose/quality/batch/demo, có thêm saved run
ngoài40. **Không có pristine holdout được thiết lập.** Development-test chỉ là một
partition giữ lại từ thời điểm freeze, không được gọi unseen/human-independent
test. Muốn claim generalization cần một cohort độc lập với development và nguồn
subject/capture đáng tin; milestone này không tìm/thêm dữ liệu đó.

Fit normalization/feature selection trên train duy nhất; lưu fitted parameters
và train-ID hash. Validation chọn cấu hình/pairing; test chỉ chạy sau freeze, không
đổi split hoặc thresholds theo kết quả. Đổi protocol/split phải version mới và báo
contamination, không xóa kết quả cũ. Bộ60 ảnh pixel không thay temporal labels.

## Evaluation và chọn cấu hình

Count MAE là metric chọn cấu hình chính trên validation có labels; báo signed
error, exact match, prediction/label/comparable coverage cùng abstain/failure và
mẫu số toàn cohort. So sánh trên cùng cohort, cùng admission/mask, kèm bảng paired
intersection; không chỉ so MAE của các clip thuận lợi mà từng mô hình chọn khác nhau.
Chưa đặt accuracy target. Mô hình giảm MAE bằng cách tăng abstain không tự thắng:
phải trình tradeoff coverage và quyết định trước test, không tạo score tổng tùy ý.

Rep precision/recall và timing start/end/Bottom chỉ với exhaustive reference và
output đủ evidence, matching theo evaluator v1/tolerance0,20s S3 đã khóa. Count-only
hay bottom interval không đủ để chấm missing event chính xác. Không tune tolerance
trên test. Báo results theo clip và groups, errors/extras/misses/unavailable cùng
nhau. Không dùng pseudo-label làm independent reference.

## Tài nguyên và giới hạn

Hiện112 clips, tổng574,24s/14.356 samples ở25FPS. Nhánh AI có252 chu kỳ trên104
clip count available; đây không là tổng cohort hoặc số chu kỳ độc lập đã xác thực.
8 whole-count null; không chuyển observed lower bounds thành tổng clip.
Số samples/frame không là số observations độc lập. Khả năng đủ train GRU chưa được
xác nhận. CPU local là phương án khởi đầu; GPU optional chỉ sau profiling và approval.
Chưa cài/khóa training framework, chưa cam kết thời gian chạy, RAM hoặc ngày bắt đầu.
Chuẩn bị dependency/seed/determinism/checkpoint/export plan ở training gate, không
ngầm dùng framework legacy có sẵn. Visibility/coverage không phải accuracy nhãn.

## MODEL TRAINING GATE — bàn giao cho user tự chạy khi đủ điều kiện

Readiness hiện tại **NOT_READY**:224 AI passes đã được kiểm tra linkage, nhưng
0 human-validated/independent labels mới.104 cặp so sánh count là cùng AI, không
inter-rater agreement. Real baseline chưa chạy, admission chưa được phê duyệt.
GRU ba event heads hiện đề xuất không có target sample-resolution exhaustive:
không suy target từ count, cycle hints, extrema hoặc predictions. Thí nghiệm dùng
count-only AI labels cần target/method/input-admission được chốt riêng; quyền tạo
nhãn AI không tự cho phép đổi contract temporal hoặc triển khai real adapter.
Không tạo lệnh train giả cho implementation chưa có. Khi các blocker đã giải
quyết, trình hồ sơ dưới đây cùng hướng dẫn/config/lệnh thực sự chạy được để user
tự thực hiện; quyết định phương pháp cần user phê duyệt riêng. Agent không thực thi:

- Research question, model/input/targets và version config cuối.
- Nguồn target và giới hạn: nhánh độc lập cần hai lượt + adjudication/IAA thật;
  nhánh AI thử nghiệm phải khai weak labels/same-assistant consistency, không IAA
  hoặc accuracy độc lập. Hashes và positive/negative/masked support theo split phải
  phù hợp target, không tự chuyển null/hints/count thành event targets.
- Temporal admission được duyệt và baseline real runnable; giữ quyết định skip/
  degrade, không giả model input cho các clip bị gate chặn.
- Split/component/source hashes, exposure declaration, giới hạn subject identity;
  validation/test dùng đúng vai trò, không dùng test để tune.
- Target/loss mask, event pairing, metrics, selection rule và failure accounting.
- Dependency/resource plan, reproducibility seeds, output/checkpoint directory mới,
  baseline freeze và chính xác các lệnh train được đề nghị chạy.
- Bằng chứng tests/integration admission/evaluator đạt, quyết định phương pháp
  của user và hướng dẫn user tự chạy; tests không thay accuracy evidence.

Đã có AI count references, chưa exact event targets/independent validation,
chưa baseline real admission và chưa freeze training/decoder implementation.
**Training runs=0**. Chưa có lệnh train hợp lệ để bàn giao, không tạo lệnh giả.
Bước nhỏ tiếp theo: chốt tính tương thích target và input admission cho thí nghiệm
count-only AI-reference; ghi rõ proposal hiện hành chưa nhận được clip nào, không
tự bỏ viewpoint/quality gate để chạy. Chưa triển khai adapter, mở UI hoặc train.
