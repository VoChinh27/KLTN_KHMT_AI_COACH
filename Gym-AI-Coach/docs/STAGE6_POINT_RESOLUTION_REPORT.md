# GĐ6 — point-resolution milestone: safe abstention

2026-10-18 (Asia/Saigon). **Milestone rule/tests/opt-in producer/demo đã triển
khai với kết luận: input hiện chưa đủ evidence để chấp nhận point-turn.** Đây
không phải accepting detector hoặc hoàn tất G6/G5. NOT_VALIDATED,
official_eligibility=false, assessment.enabled=false; G5.7
BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT, G6/toàn G5 IN_PROGRESS.

Run mới: `outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48`.
Không có Git repository; không tạo commit/diff giả. Không đọc Data/video/
metadata, inference, real integration, archive, thêm dependency hay tune profile.

## Rule đã chốt trước implementation

[Frozen policy](SQUAT_POINT_RESOLUTION_POLICY.md): B phải là strict minimum
raw[B] < raw[B−1], raw[B] < raw[B+1]. Cả3 samples kề nhau, timestamp CFR hợp lệ,
cùng side/run, observed; toàn candidate/support không missing/imputed, chạm clip
edge hoặc có UNSTABLE_SIGNAL overlap. Overlap inclusive, unstable thiếu location
coi global. Không epsilon, anchor relocation, smoothing mới hoặc threshold mới.

Đây là **điều kiện cần**, chưa phân biệt được nhiễu. Raw observations có thể
được giải thích bởi một latent reversal không nhiễu hoặc latent hold cộng sai số
đúng bằng observed−hold. Test dựng hai histories cho **cùng full input bytes/
hash**, chỉ truyền observations vào producer. Cả hai có cùng strict minimum,
provenance observed, PVP và timestamps; producer không thể chấp nhận một và từ
chối một bằng dữ liệu giống nhau. `observed` không có nghĩa noise-free.

Vì capability hiện không chứa noise evidence độc lập, policy mới **không chấp
nhận point-turn nào**, kể cả khi raw conditions pass, với reason
`noise_not_identifiable`. Không thêm cờ noise-free từ fixture, không dùng công
thức generator/reference để cấp permission. Context smooth hoặc tên fixture
cũng không giải quyết identifiability. Kết luận này không nói mọi nghiên cứu
temporal là bất khả thi; nó chỉ giới hạn capability/rule và yêu cầu abstention
hiện tại. Không bịa noise bound hoặc threshold từ fixtures.

## Contract/versioning và producer thực tế

- Policy/producer `squat-v2-point-resolution-v1`; policy hash
  `930d41199421d6544f775b4485ba2b347005498b836fbda6532ee16dce4e4642`.
- New audit envelope `point_resolution_schema_version=0.1.0`, contract
  `squat-point-resolution-audit-v1`. Envelope liên kết nguyên vẹn conservative
  v2 `base_result`/`base_hash`; không thêm literal vào frozen schema v2 hoặc
  thay producer `squat-v2-conservative-adapter-v1`.
- `segment_squat_point_resolution(SquatTemporalInput)` chạy fresh unchanged
  conservative adapter/default S3, rồi kiểm tra raw/candidate support. Không
  nhận fixture references, GeometryFeatureObservation hoặc noise-free override.
- `inspect_point_support` là gate thuần kiểm tra raw samples và upstream flags;
  trực tiếp test missing/edge/unstable cases ngay cả khi v1 không tạo full PVP.
  Producer truyền exact flags từ base mới, không giả report/flag cũ thành run mới.
- Mỗi PVP có `strict_local_minimum`, `necessary_conditions_met`, exact raw B±1/
  candidate S..E supports, input hash, structured reasons. **Không phải** event
  được chấp nhận. `accepted_point_count=0`; point event/arrival/departure/pause
  luôn null, không0. Base giữ bounds_unresolved/unavailable motion.
- Reasons category insufficient_data, segmentation, S3 và inclusive support;
  original flags/provenance giữ trong base. Input/timestamp lỗi có
  INVALID_TEMPORAL_INPUT/input_error/segmentation/S3. Không đổi FlagCode cũ.
- New `read_temporal_experiment_json` dispatch envelope hoặc unchanged old
  segmentation readers; old readers từ chối envelope, không silent migration.
  Strict JSON/readback kiểm tra identity/hash/count/status/null/reasons;
  hash không authenticate hoặc chứng minh measurement accuracy. Không rerun
  segmentation khi deserialize.

Default S3 profile/hash giữ nguyên:
`5802a42e328578ca7ed43adb7bfdd2a21ede56fd9b61a7737587109c2085b017`.
Không chạy lại/tune trên30 sensitivity conditions. Không tính góc/joint geometry;
scalar synthetic là input giả lập theo contract hiện hành, không đo trên người.

## Freeze và acceptance evidence

`freeze-receipt.json` ghi UTC trước khi producer/demo files tồn tại. Snapshots
policy document, models, fixtures, tests và JSON schema có SHA-256. Tests được
freeze trước producer; không sửa frozen tests/models/policy sau để làm pass.
Expected-red run dừng vì module producer chưa tồn tại, receipt được giữ.

34 tests mới bao phủ:

- Smooth minimum tại25/30/60FPS: raw conditions pass nhưng noise abstention.
- Plateau không strict; jitter có thể strict nhưng vẫn unavailable.
- Identical-observation reversal/noisy hold: cùng input/hash → cùng abstention.
- Neighbor thiếu/gap, imputed neighbor và imputed ở phần khác của candidate,
  side switch; không nối qua discontinuity hoặc đổi sang side khác.
- UNSTABLE_SIGNAL overlap gồm hai biên inclusive, flag ngoài candidate và flag
  thiếu location; clip edges/neighbors thiếu, clip chưa có full PVP.
- Timestamp sai và reference-wrapper bị reject với structured input errors.
- Serialization/hash/unknown version/null/reason tampering; exact legacy v1 và
  conservative v2 canonical JSON; label đổi không mở permission.
- Heavy import guard, Data I/O/geometry guard, source immutability, exclusive
  demo output và artifact hashes/readback. Không có inference/real integration.

Environment: Windows10.0.26200 x64, CPython3.11.9, package0.7.0,
Pydantic2.13.5, pytest9.1.1. Không dependency mới hoặc claim clean-env support.
Related suite còn chạy Node renderer checks hiện có bằng runtime sẵn có.

| Run | Pass | Fail | Error | Skipped | Deselected | Warnings |
|---|---:|---:|---:|---:|---:|---:|
| acceptance-red (expected missing producer) |0|0|1|0|0|0|
| acceptance-first |34|0|0|0|0|0|
| related-tests |207|0|0|0|1|0|
| **final-related-tests** |**207**|**0**|**0**|**0**|**1**|**0**|

Acceptance34 đạt4.09s; related207 đạt39.52s; final207 đạt39.95s. Related =34 mới +173 tests
temporal/feedback/v2 hiện có, không cộng lặp các lần chạy. Một real-preview
integration bị deselect; không chạy real-data integration. Full project suite
NOT_RUN, không báo related suite như full suite.

Sau related run, chỉ thêm hiển thị input/run diagnostics và original flags vào
demo mới để gap/side-switch không bị che bởi summary của candidates còn lại;
đã tạo output mới `demo-final`, giữ `demo` ban đầu. Final regression receipt lưu
riêng `final-related-tests.xml/.txt` cho source cuối; số liệu cuối ghi ở phần
evidence cuối bên dưới.

Lệnh đã chạy (đổi basename/temp theo từng run, không overwrite):

```powershell
& .venv/Scripts/python.exe -B -m pytest tests/test_point_resolution.py -q --basetemp outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/acceptance-temp --junitxml=outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/acceptance-first.xml

& .venv/Scripts/python.exe -B -m pytest tests/test_point_resolution.py tests/test_temporal_v2.py tests/test_temporal_v2_renderer.py tests/test_temporal_squat.py tests/test_squat_feedback.py tests/test_temporal_sensitivity.py -m 'not integration' -q --basetemp outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/final-related-temp --junitxml=outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/final-related-tests.xml
```

Stdout/JUnit và commands đầy đủ ở run mới. Không browser/visual QA trong lượt
này; demo HTML static được kiểm tra links/escaping/artifact readback, old renderer
giữ nguyên và existing Node tests vẫn thuộc related suite. Không tuyên bố đã
xem UI trực quan hoặc bỏ qua browser restriction của lượt trước.

## Demo đã đo

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.reporting.point_resolution_demo --output-dir outputs/squat_point_resolution/run-32c3ab699ace418da80b808303c7df48/demo-final
```

Exit0. Mở `demo-final/index.html`; đường dẫn output phải chưa tồn tại. Mỗi case
lưu input/audit JSON, new summary HTML, unchanged base renderer HTML và byte
hashes trong manifest. Trang mới tách “Strict raw minimum / necessary conditions”
khỏi “Point-turn: Unavailable”, liệt kê input/run reasons và original flags.
Không hiển thị accepted point chỉ để có positive demo; điều kiện sufficient
không được fixture nào đáp ứng trong capability hiện tại.

| Case | FPS | Frames | Full PVP candidates | Raw conditions met | Accepted points | Reasons của checks |
|---|---:|---:|---:|---:|---:|---|
| smooth-25fps |25|151|2|2|0|noise_not_identifiable|
| smooth-30fps |30|181|2|2|0|noise_not_identifiable|
| smooth-60fps |60|361|2|2|0|noise_not_identifiable|
| plateau-30fps |30|181|2|0|0|raw_minimum_not_strict|
| jitter-30fps |30|181|2|2|0|noise_not_identifiable|
| gap-30fps |30|181|1|1|0|noise_not_identifiable trên candidate còn lại|
| imputed_support-30fps |30|181|2|1|0|imputed_support/nonobserved_support; candidate còn lại noise ambiguity|
| side_switch-30fps |30|181|1|1|0|noise_not_identifiable trên candidate còn lại|
| clip_edge-30fps |30|61|0|0|0|no_full_pvp_candidate|

Tổng9 inputs,1.659 frames,14 full PVP candidates;11 qua necessary conditions,
**0 accepted points**. Bảng không đo accuracy/recall trên người tập. Gap và
side switch loại candidate cắt boundary từ upstream; flags và run reasons giữ
riêng, không biến checks của candidate còn lại thành kết luận toàn clip sạch.
Jitter qua raw rule là bằng chứng cụ thể rằng strict three-sample minimum đơn
lẻ không phân biệt được nhiễu. Không dùng GT để sửa hoặc mở candidate đó.

## Files và bảo toàn

Mới: `segmentation/point_resolution_models.py`, `segmentation/point_resolution.py`,
`reporting/point_resolution_demo.py`, `tests/point_resolution_cases.py`,
`tests/test_point_resolution.py`, frozen policy document và báo cáo này.
Chỉ cập nhật README/PROJECT_STATE/TASKS/IMPLEMENTATION_PLAN/SCHEMA_CONVENTIONS.
Không sửa old v1/v2 producer, API, exports, renderer, profiles, tests, frozen
contracts, reports hoặc outputs. Baseline482 files được đối chiếu byte SHA-256
trong `preservation.json`; inventory mới ghi từng artifact hash, không dùng Git
diff giả. Old test receipts/output không overwrite; cả demo trước/sau diagnostics
được giữ. Không kiểm kê Data để thực hiện preservation checks.

### Evidence cuối

Final suite207 passed/0 failed/error/skipped/warnings/1 real integration
deselected,39.95s, exit0. Baseline482 files:477 byte-identical, chỉ5 docs đã
nêu thay đổi;0 missing,351 old artifacts giữ hash. New frozen models/policy/
tests/fixtures/schema giữ nguyên hashes, không cần erratum hoặc sửa test để pass.
`preservation.json`/`artifact-inventory.json` ghi document/link/readback/demo
hash checks và new source hashes. `commands.json` lưu commands từng lần;
`environment.json` ghi runtime. Các kết quả là synthetic/code verification,
không accuracy hoặc validation trên người tập.

## Giới hạn và bước nhỏ tiếp theo

Chưa có accepting point detector, noise model/reference protocol, dwell/bounds
detector, physical reversal timing, partial-event retention hoặc real temporal
validation. Status unavailable không có nghĩa người tập không đảo chiều hoặc
không pause; point_count0 không thay PVP count hay true rep count. Không cho
phase available từ một raw pattern chưa đủ evidence. Không0s pause, assessment,
score, form feedback hay kết luận đúng/sai kỹ thuật.

Bước nhỏ tiếp theo: xác định một capability/evidence độc lập cho noise
discrimination và tiêu chí kiểm chứng false positives trước accepting policy
mới. Nếu chưa có, giữ abstention hiện tại; không thêm “noise-free” từ tên
fixture/generator, không tune threshold trên30 sensitivity conditions. G5.7
vẫn cần documented input/fresh sequence+bilateral allow và temporal admission
riêng; milestone synthetic này không mở gate hoặc xác thực vận động viên.
