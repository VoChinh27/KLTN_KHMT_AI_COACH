# Đề xuất hợp đồng temporal squat: events và khoảng dừng đáy

Ngày2026-10-18, Asia/Saigon. Decision ID: `squat-temporal-contract-proposal-v1`.
Trạng thái: **DRAFT_RECOMMENDED / DESIGN_ONLY / NOT_IMPLEMENTED**.
Đây là một quyết định thiết kế đề xuất, chưa phải schema/API hoặc thuật toán mới.
GĐ6 giữ **IN_PROGRESS**, GĐ5.7 **BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT**;
`NOT_VALIDATED`, `official_eligibility=false`, `assessment.enabled=false` không đổi.

## 1. Quyết định đề xuất duy nhất

**Biểu diễn Bottom bằng các event riêng và một interval dwell/pause có điều kiện;
tách event overlay khỏi motion phase; abstain ở mức từng thành phần khi không đủ
bằng chứng.** Giữ một `bottom_anchor` cho PVP/provenance, không đồng nhất anchor
với thời điểm đến đáy, rời đáy, hoặc độ dài pause.

- `bottom_arrival`: candidate thời điểm kết thúc descent tại đáy signal.
- `bottom_departure`: candidate thời điểm bắt đầu ascent từ đáy signal.
- `bottom_anchor`: neo cực tiểu/plateau proxy theo method đã khai báo, có thể
  hữu ích cho PVP ngay khi hai bounds chưa xác định. Không mặc định là midpoint
  hold thật, không dùng tên "chạm sàn" hoặc "đủ độ sâu".
- `bottom_pause`: interval chỉ có khi hai bounds và evidence dwell đều được
  hỗ trợ. Khi đó duration tính từ hai bounds; unknown phải null, không0.
- Reversal không có dwell phân giải được vẫn có thể là point turn candidate
  theo resolution policy được duyệt; không tự kéo thành interval một frame.

Tất cả mô tả **signal hình học 2D**, không phải chuẩn kỹ thuật, contact event
vật lý hay đánh giá chất lượng người tập. Tài liệu không chọn threshold mới,
không chốt detector pause hoặc thay tham số S3 để khớp fixture.

## 2. Căn cứ hiện hành và đối chiếu số liệu

Nguồn chính: [sensitivity report](STAGE6_SYNTHETIC_TEMPORAL_SENSITIVITY_REPORT.md)
và saved run `outputs/squat_temporal_sensitivity/run-87f31de96a654bc09ee816cd2f044f5e/audit-final`.
Các số sau là **evidence đã đo ở lượt trước**, không phải kết quả của contract
đề xuất. Không đọc video/metadata `Data/` hoặc chạy audit algorithm mới để chọn thiết kế.

| Quan sát / fixture | Evidence đã lưu | Hệ quả thiết kế, tác động cần kiểm thử |
|---|---|---|
| `bottom_pauses` |3/3 cycles ở mọi FPS; Bottom recall5.56/4.84/2.44% tại25/30/60FPS dù availability100%|Cần interval cho hold; candidate count không được chứng nhận phase coverage hoặc pause accuracy|
| `asymmetric_pauses` |3/3 cycles; agreement khoảng70.89–71.15%; Bottom recall4.76/4.00/2.04%|Arrival/departure độc lập anchor; không lấy marker±nửa window làm hai bounds|
| Hold1.2s tại30FPS |Marker so arrival+0.5933s, departure−0.6067s; duration errors+0.6000/+0.6333s|Tách physical/operational phase duration khỏi legacy top↔marker spans|
| `steady_three` |3/3, nhưng30FPS first start−1frame/last end+1frame|Giữ signal/filter identity và resolution; không hứa sai số≤1frame chỉ vì FPS biết trước|
| `variable_tempo` |4/4; exact Bottom label đúng1/4 tại30FPS|Đánh giá event timing, interval bounds và phase labels riêng; không giấu drift bằng nhãn Bottom mở rộng tùy ý|
| `eight_consecutive` |8/8 tại mọi FPS; agreement99.29/96.44/95.25%|Một shared top event có hai cycle roles; không nhân đôi rep hoặc overlap phase|
| `fast_cycles` |0/3 mỗi FPS;9 misses|Cho phép abstention temporal-resolution/spacing; không hạ min-half-cycle hoặc time-warp|
| `small_excursion` |0/3 mỗi FPS;9 misses;30FPS chỉ partial flags vì guards có excursion30°|Reason cụ thể cần bằng chứng producer, không suy low-range chỉ từ tên fixture hoặc absence of candidates|
| `missing_bottom` |2/3, cycle1 không có candidate mỗi FPS|Gap là boundary của temporal run; pause qua gap unavailable, không nối bằng nội suy|
| `explicit_side_switch` |2/3, cycle2 không có candidate mỗi FPS|Event/interval/cycle không được vượt side transition, kể cả khi transition sát top|
| `partial_edges` |Chỉ cycle2 là full reference;6 partial instances qua3FPS|Giữ censoring và partial riêng; không tự tạo bounds ngoài clip|
| Toàn matrix |78/102 matched,24 misses,0 extra; phase availability82.70%, agreement73.73%|Tách execution/cycle availability/event accuracy/phase availability/label agreement; không gộp thành confidence hoặc form score|

MAE matched-only start/bottom-proxy/end là15.684/5.513/17.009ms; max80/50/80ms.
Max frame errors tương ứng25FPS2/1/2,30FPS2/1/2,60FPS4/3/4. Proxy khá gần
midpoint không chứng minh arrival/departure hoặc pause. Ba FPS là biến thể cùng
generator, không phải mẫu người tập độc lập. Số liệu không đặt acceptance tolerance mới.

Code hiện hành: `RepetitionCandidate` chỉ start/bottom/end; `_turns` chọn neo
plateau và xác nhận extrema; `phase_at` ưu tiên start/end/bottom event rồi
Descent/Ascent. Shared top hiện hiển thị Return của cycle trước. Khoảng không
thuộc candidates thường bị gán partial-start/end, chưa mô tả đủ lý do theo span.
Feedback v1 dùng một `phase` literal và inclusive `phase_evidence`; renderer đọc
label đó và timeline PVP. Không có hai bottom bounds, pause interval hoặc
independent phase availability. Các mô tả này là hiện trạng, không code đã sửa.

## 3. Records và provenance của hợp đồng tương lai

Các tên dưới đây là **khái niệm dự kiến**, không phải class/enum đã export.

| Record | Nội dung bắt buộc dự kiến |
|---|---|
| Source/capability |Video/synthetic ID, frame/time identity, FPS/CFR policy, feature ID/definition/registry hashes, raw/processed/input hashes khi có; explicit temporal gate snapshot/hash; không nâng numeric0.4 temporal=false|
| Temporal run |Run ID, side cố định, evidence span, source provenance/missing mask, filter support/config/profile/version/hash; không fusion hoặc auto fallback|
| Event |Stable ID, kind, candidate/unavailable/not_run/blocked, frame/time hoặc null, run/side, source basis (extremum/plateau proxy/arrival/departure), evidence và flags/reasons|
| Bottom model |Mode và availability riêng cho anchor, arrival, departure, pause; không yêu cầu đầy đủ bounds chỉ để lưu anchor|
| Phase interval |Descent/Bottom/Ascent, named boundary event refs, closure explicit, status/basis, run/side, duration semantics và evidence span|
| Cycle candidate |Top start/end và bottom anchor refs, completeness/censoring, countable-candidate status, phase availability riêng; không "valid form"|
| Abstention span |Location, scope (event/interval/phase/cycle), categorical reason, evidence/flags; null result values; không ép mọi failures thành partial-end|

Giữ source tier S3, NOT_VALIDATED và official=false. Hash nhận diện nội dung,
không chứng minh viewpoint hoặc độ chính xác. Số đo framewise vẫn giữ policy
của chính nó; temporal abstention không tự đổi QualityReport hay metric allow.
Contract này không cấp temporal eligibility mới. Capability synthetic unit
hiện hành vẫn riêng; đường thật tiếp tục cần documented viewpoint, sequence
allow và exact cả left/right knee metric allow theo gate5.7, rồi temporal
admission được duyệt riêng. Không dùng việc đếm/nhận diện pause để bỏ qua gate.

Event time phải khớp sample timestamp; với CFR hiện tại là frame_index/FPS.
Không nội suy timestamp/subframe ngầm. Analytic continuous reference time được
lưu ở evaluation, không ghi đè predicted sample time. `1/FPS` là resolution,
không phải error bound hoặc confidence đã hiệu chuẩn. Evidence span inclusive
chỉ mô tả samples hỗ trợ, không phải guaranteed uncertainty interval.

Giữ nguyên observed/imputed/low_visibility/missing. **Đề xuất cho producer tương
lai:** pause/bounds không được xác nhận chỉ bằng các đoạn imputed hoặc plateau
do filtering tạo ra. Cần audit provenance của cả filter support, không chỉ status
tại marker. Nếu chưa có validation cho hỗ trợ imputed, bounds/pause abstain;
không thay đổi imputation/readiness semantics v1 trong bước này.

## 4. Bottom modes và quy tắc biên

### 4.1. Ba modes, không fallback ngầm

1. **`dwell_candidate`**: hai bounds A,D được hỗ trợ, A<D trên sample grid;
   pause candidate là[A,D), duration=D−A. Anchor B nếu có phải nằm trong[A,D].
   Dwell là trạng thái ổn định theo **method S3 sẽ được đặc tả/duyệt riêng**;
   không yêu cầu tọa độ float bằng nhau tuyệt đối, cũng không tự đặt epsilon.
   Không mở rộng thành mọi vùng góc "gần đáy" để nâng recall.
2. **`point_turn_at_resolution`**: chỉ dùng khi resolution/evidence rule riêng
   đủ hỗ trợ một reversal ở B, không có dwell phân giải được. B là operational
   phase partition marker; **không sao chép B thành arrival và departure**.
   Hai physical-bound estimates chưa biết vẫn unavailable; pause interval và
   pause duration vẫn null với lý do not-resolved-at-resolution. Không suy rằng
   pause vật lý bằng0. Analytic no-hold ground truth không được truyền vào
   producer để chọn mode; detector phải quyết định từ input hợp lệ.
3. **`bounds_unresolved`**: có thể có PVP anchor nhưng thiếu bằng chứng để chọn
   một trong hai modes trên. Không mặc định point-turn, không bịa A/D hoặc0s.
   Chỉ gán phase trên vùng có bounds/evidence riêng được hỗ trợ; phần chuyển
   tiếp không xác định hiển thị Unavailable. Nếu không định vị được vùng ấy,
   phase toàn candidate unavailable; count candidate có thể vẫn tồn tại nếu
   PVP và gate của count đủ, với `phase_status` riêng.

Không có enum "pause chắc chắn không tồn tại" cho dữ liệu lấy mẫu hiện tại.
Các evidence rules chọn mode, xử lý plateau/noise và cách xác định A/D **chưa
được giải quyết bằng thuật toán trong đề xuất**; đó là điều kiện trước implementation.

### 4.2. Khi dwell được hỗ trợ

Đặt S=top-start, A=arrival, D=departure, E=top-end; S<A<D<E. Motion intervals:
**Descent[S,A), Bottom[A,D), Ascent[D,E)**. Event overlays không phải intervals.

| Frame/time boundary | Motion phase | Event overlay |
|---|---|---|
| S |Descent|Start/Top của cycle hiện tại|
| A |Bottom|Bottom arrival|
| Trong[A,D) |Bottom|Anchor B nếu frame này trùng B; không thêm pause mới|
| D |Ascent|Bottom departure|
| E=S của cycle sau |Descent của cycle sau nếu có evidence|Cùng một top event: Return/Top trước và Start/Top sau|
| E không có cycle sau |Không kéo dài Ascent; phase sau E unavailable/not in candidate|Return/Top tại E vẫn là event có thể biết|

Chọn half-open để không double-own D hoặc E và cộng duration nhất quán:
(A−S)+(D−A)+(E−D)=E−S. Frame spacing không được cộng thêm vào pause duration.
Evidence frame spans **vẫn inclusive và là record khác**, không đổi nghĩa v1.
Timestamp có thể bằng nhau sau quantization thì không được tạo dwell-positive;
phải point-turn đủ điều kiện hoặc unresolved, không tự cộng một frame.

Minh họa hình thức dưới đây lấy mốc synthetic đơn giản, **không phải output
algorithm hay serialization hợp lệ của API hiện hành**; không có hashes giả:

```json
{
  "illustration_only": true,
  "fps": 30.0,
  "events": {
    "S": {"frame_index": 30, "timestamp_seconds": 1.0},
    "A": {"frame_index": 60, "timestamp_seconds": 2.0},
    "B": {"frame_index": 69, "timestamp_seconds": 2.3},
    "D": {"frame_index": 78, "timestamp_seconds": 2.6},
    "E": {"frame_index": 108, "timestamp_seconds": 3.6}
  },
  "motion_intervals": [
    {"phase": "Descent", "start_event": "S", "end_event": "A", "closure": "[)"},
    {"phase": "Bottom", "start_event": "A", "end_event": "D", "closure": "[)"},
    {"phase": "Ascent", "start_event": "D", "end_event": "E", "closure": "[)"}
  ],
  "pause_duration_seconds": 0.6,
  "validation_status": "NOT_VALIDATED",
  "official_eligibility": false,
  "assessment_enabled": false
}
```

Ở30FPS frame60..77 thuộc Bottom,78 thuộc Ascent. Continuous duration0.6s
không trở thành19/30s vì inclusive evidence[60,78]. Không dùng midpoint2.3s
để khẳng định thuật toán đã tìm được A=2.0,D=2.6.

### 4.3. Point turn, unresolved và shared tops

Với point-turn **đã đủ điều kiện**, Descent[S,B), Ascent[B,E) là marker-based
phase partition, `phase_basis=point_turn_at_resolution`. Tại B, motion phase
thuộc Ascent theo closure nhưng UI đặt event badge "Bottom (event)"; không tô
một khoảng pause hoặc ghi0s. Duration marker spans phải mang tên/basis riêng,
không tráo thành estimated descent-to-arrival/ascent-from-departure durations.
Không đủ điều kiện thì dùng unresolved, không lấy legacy PVP làm bằng chứng đủ.

Mỗi physical shared top chỉ một stable event ID, có hai role references ở hai
cycle. Không đếm một frame thành hai phase; UI có thể ghi "Return/Top → Start/Top"
trong khi motion phase thuộc cycle sau. Chỉ share event khi cùng input/run/side
và không có invalid boundary. Side switch/gap ngay tại top vẫn split runs; không
share để bắc cầu. Có khoảng top giữa E_i và S_next thì **không ép chúng bằng nhau**:
chưa hỗ trợ top-dwell detector thì phase khoảng đó unavailable, không nối rep.

## 5. Abstention và lý do có cấu trúc

Availability per component: `candidate`, `unavailable`, `not_run`, `blocked`;
structural input/operational failure là lỗi có structured context riêng, không
"0 rep". Completeness/censoring độc lập: complete/left/right/both-partial.
Candidate count bằng0 không khẳng định true rep count0. Chỉ full candidate đủ
continuity/gate được count; partial chỉ diagnostics. Có cycle candidate không
bắt buộc có resolved pause hoặc đủ cả phase duration.

| Điều kiện | Hành vi hợp đồng tương lai | Flag hiện có / reason dự kiến |
|---|---|---|
| Chưa chạy |Events/interval/phase values null, status not_run|Reason stage_not_run dự kiến; không giả bộ measurement failure|
| Gate/viewpoint/official capability không đạt |Blocked, không inference/temporal count; không promote degrade/skip|Giữ TEMPORAL_GATE_BLOCKED/DOCUMENTED_INPUT_MISSING/VIEWPOINT_* khi có evidence|
| Missing/nonfinite/unusable |Giữ provenance/source reason, split run; không interval/cycle qua lỗi|MISSING_JOINT và upstream reasons; category measurement/insufficient_data theo nguồn|
| Side switch |Split, cấm cycle/interval/event support vượt transition|SIDE_SWITCH_DISCONTINUITY; không auto fallback|
| Chu kỳ chạm clip/run edge |Left/right/both censored; observed events giữ lại, unknown bounds/duration null|PARTIAL_REP_START/END; không invent unconfirmed extrema|
| Chỉ anchor, chưa đủ bounds/mode |Pause và phần phase không hỗ trợ unavailable; count/anchor độc lập nếu hợp lệ|BOTTOM_BOUNDS_UNRESOLVED dự kiến, không phải enum hiện hành|
| Không phân giải dwell từ sampling |Pause duration null, không0; point-turn chỉ khi rule riêng đủ|PAUSE_UNRESOLVED_AT_RESOLUTION dự kiến|
| Signal range nhỏ có đo/kiểm tra tại scope đã khai báo |Abstain cycle/phase tại scope đó; lưu actual excursion/unit/support|LOW_SIGNAL_RANGE chỉ khi producer thực sự có diagnostic; không suy từ fixture ID|
| Candidate turns quá sát theo profile đang dùng |Không full cycle; ghi temporal spacing và scope measured|UNSTABLE_SIGNAL hiện có; TEMPORAL_SPACING_UNRESOLVED là reason chi tiết dự kiến|
| Không có candidate nhưng diagnostic chưa phân biệt nguyên nhân |Unavailable với no_confirmed_cycle, không gán chắc "quá nhanh"/"biên độ nhỏ"|SEGMENTATION_FAILED hoặc reason generic mới phải version; không sửa old flags|
| Event/interval ordering hoặc source/hash sai |Reject structured input error, không render như đo thật|INVALID_TEMPORAL_INPUT/input_error, không technical_error|

Các reason mới trong bảng chỉ là **đề xuất**, chưa thêm FlagCode hoặc đổi
category/severity của stage cũ. Temporal failures không phải lỗi kỹ thuật người
tập. Cần lưu mọi reasons có bằng chứng; không dùng một priority reason để che
gap/side switch. Scope của reasons phải có span chính xác; frame giữa các
candidates không tự được gọi partial-end chỉ vì thứ tự loop hiện tại.

Đo được arrival nhưng clip kết thúc trong hold: giữ event arrival với
right-censoring, departure/pause duration unavailable; không lấy last frame làm
departure. Imputation tạo đoạn bằng phẳng không chứng minh có pause. Hai trường
hợp này chưa nằm trong matrix v1, cần future acceptance fixtures riêng.

## 6. Schema, renderer và backward compatibility

Đề xuất cần **contract version mới riêng**, dự kiến segmentation/feedback
schema0.2.0 và semantic identities mới; tên/version cuối chốt trong migration
review. Chưa tồn tại API v2 hoặc `phase_at_v2` trong project. Không thêm fields
vào v1 extra-forbid, không đổi literals, `RepetitionCandidate`, flags, profile,
public exports, dependencies hoặc v1 JSON hashes trong bước thiết kế.

| Thành phần | Tác động nếu triển khai sau này |
|---|---|
| Segmentation producer |Phải sản xuất/abstain arrival/departure/pause có source evidence; không chỉ đổi tên bottom_frame hoặc vẽ window quanh nó|
| Feedback schema |Tách motion phase, event refs/roles và pause status/duration; giữ metric/availability/source hashes độc lập; closure field explicit|
| Renderer/UI |Motion intervals tô theo thời gian thật; events là marker/badge; pause unavailable hiển thị "chưa xác định", không0s; shared top có hai roles; partial/unknown spans không bịa pha|
| Timeline |Một candidate cycle, một shared top ID; không dùng proportional rep blocks để khẳng định pause duration; count và phase completeness tách riêng|
| Legacy decoder |Dispatch theo version tường minh; old JSON tiếp tục đọc/render đúng old semantics. Không gọi old renderer để diễn dịch JSON v2|
| Adapter nếu cần |Giữ v1 anchors/spans, mark legacy_proxy; arrival/departure/pause unavailable với reason. Không suy dwell/no-dwell hoặc nâng temporal permission từ adapter|
| Audit/evaluation |Giữ protocol/hash/snapshots v1; thêm reference/protocol version mới cho interval closure và dual event/phase outputs; báo side-by-side, không sửa GT cũ|

V1 reference Bottom dùng **closed** arrival..departure và shared top label
Return của cycle trước; đề xuất motion intervals dùng half-open và events
overlay. Vì vậy raw frame label scores giữa hai protocols **không trực tiếp
so sánh**. Phải ghi các frame thay đổi chỉ do boundary convention, bảo toàn
v1 baseline và đánh giá cả event/boundary/interval overlap dưới đúng version.
Không coi việc đổi GT endpoint hoặc mở rộng nhãn Bottom là accuracy improvement.
Inclusive evidence/span conventions trong SCHEMA_CONVENTIONS hiện hành không bị sửa.

Không chạy detector/geometry trong renderer hoặc JSON readback. Event/pause
availability từ producer phải qua schema checks; UI không đoán khi thiếu fields.
V1 count candidate vẫn có thể dùng cho diagnostics độc lập, nhưng không là
chứng nhận contract v2, official eligibility hoặc feature/form accuracy.

## 7. Acceptance tests cần đặc tả trước implementation

Đây là **test plan**, chưa có tests/API cho v2 và chưa được thực thi. Các expected
invariants là semantics của hợp đồng, không ngưỡng biomechanical. Không lấy
observed80ms/50ms hoặc fixture thresholds làm pass tolerance mới.

| ID | Input / fixture | Acceptance cần có |
|---|---|---|
| AT01 |Dwell minh họa S1,A2,D2.6,E3.6 tại25/30/60FPS|Ordering, exact CFR identity, closure explicit; duration=D−A, không cộng1frame; A thuộc Bottom, D thuộc Ascent|
| AT02 |`bottom_pauses`, `asymmetric_pauses`|Arrival/departure errors riêng, pause duration error, Bottom interval overlap/recall riêng; count3/3 không làm test phase tự pass|
| AT03 |`steady_three` no-hold và fractional-frame turnaround|Point-turn cần evidence rule; không positive dwell tự tạo, không dùng known fixture labels để chọn mode; unknown pause không0|
| AT04 |`variable_tempo`|Tách event drift/frame/seconds khỏi count; preserve sample times; không time-warp hoặc widen labels để che1/4 Bottom exact hiện tại|
| AT05 |`eight_consecutive`|Shared event một ID/two roles; một motion phase mỗi frame; không duplicate count hoặc phase duration|
| AT06 |`missing_bottom`, imputed plateau mới|No cycle/pause across raw gap; lý do và support provenance đúng; imputation/filtering không tự chứng minh dwell|
| AT07 |`explicit_side_switch`, thêm switch đúng tại top|Run/side boundaries cứng; không reuse shared ID qua side switch, không chọn bên để khớp truth|
| AT08 |`partial_edges`, thêm clip kết thúc giữa hold|Không full count cho partial; known arrival giữ, unknown departure/duration null; không tạo boundary ở last frame|
| AT09 |`fast_cycles`, `small_excursion`|Abstention và misses còn trong denominator; specific reason chỉ khi có diagnostic; không thay S3 baseline parameters|
| AT10 |Empty/invalid/nonfinite/duplicate index, low visibility, missing scale khi metric cần|Structured operational/input/source reasons; không fake timestamp/value, không biến imputed/low thành observed|
| AT11 |Đủ PVP nhưng chưa hỗ trợ bottom mode|Cycle candidate và anchor có thể còn, phase/pause unavailable rõ; legacy proxy không tự thành hai bounds|
| AT12 |V1 JSON/roundtrip/corrupt hashes/new version|V1 unchanged; version dispatch explicit; invalid v2 ordering/source rejected; readback không chạy algorithm|
| AT13 |UI event đúng tại A,D,point B,shared E|Tách badge và motion color; null không hiển thị0s; không green/red form verdict; NOT_VALIDATED/assessment false luôn thấy|
| AT14 |Protocol v1 vs future v2|Baseline cũ không overwrite; quantify closure-only label changes; matched-only timing, misses/extras và unavailable denominators cùng báo|
| AT15 |Top dwell/noisy/shallow non-pause chưa có trong matrix|Không mặc định mọi turnaround là pause; nếu policy chưa phân giải thì abstain, không silently coi reference là supported|

Trước implementation, chốt độc lập reference protocol và prediction evidence
rules (including raw vs smoothed/filter support, minimum resolvable dwell,
diagnostic scope, point-turn evidence và censoring). Nếu cần numeric S3 settings,
phải có rationale/phê duyệt/freeze trước run và evaluation data mới; không tune
chúng trên chính30 conditions đã dùng để chọn thiết kế. Existing matrix chỉ là
development/regression diagnostics. Không hứa một-frame accuracy phổ quát.

## 8. Các lựa chọn bị loại và điều kiện đi tiếp

| Lựa chọn bị loại | Lý do |
|---|---|
| Chỉ giữ Bottom event và tiếp tục gọi nó là whole Bottom phase |Lặp lại2.44–5.56% hold recall và duration bias~0.6s; không diễn đạt pause|
| Chỉ dùng interval, mọi cycle phải có pause dương |Bịa dwell ở smooth reversal; mất point marker/PVP linkage và không xử lý sampling ambiguity|
| B→arrival/departure bằng padding hoặc ±half smoothing window |Proxy MAE thấp không xác định hai bounds; tạo measured duration không có evidence, có nguy cơ fit fixture|
| Dùng một enum chung cho motion/event, hoặc nới/tune profile để tăng recall |Vẫn mơ hồ shared tops, boundary ownership và unavailable; trái phạm vi và không chứng minh semantics đúng|

**Điều kiện trước khi xem xét implementation:** (1) thống nhất quyết định dual
event/qualified interval và modes/abstention ở trên; (2) giải quyết evidence rule
còn mở, không chọn threshold từ performance trên fixture; (3) freeze reference
v2/closure/migration và acceptance expectations độc lập producer; (4) thiết kế
schema version mới, adapters/readback và UI opt-in không phá v1; (5) thêm fixtures
censored/quantized/noisy/top dwell và phân tách development/evaluation phù hợp;
(6) giữ nguyên G5.7 blocked/G6 IN_PROGRESS/assessment false, không mở real-data
admission từ thiết kế này. Chưa có đủ các điều kiện trên để triển khai detector mới.

Một quyết định được khuyến nghị duy nhất là **events + optional evidence-backed
dwell interval, event/motion separation, component-level abstention**; không có
phương án fallback ngầm để biến missing evidence thành pause hoặc no-pause.

## 9. Kiểm tra và bảo toàn trong lượt thiết kế

Chỉ tạo tài liệu này và ghi link/trạng thái thiết kế tối thiểu ở PROJECT_STATE/
TASKS; không sửa sensitivity report hoặc source/tests/schema/UI hiện hành.
Evidence mới: `outputs/squat_temporal_contract_design/run-dfbeeb29689144c2b0b7ceac283a648c`.
Số liệu dẫn ở§2 được đối chiếu saved JSON/artifact hashes, không phải đo mới.

Tests hiện hành chạy synthetic-only:

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_temporal_sensitivity.py tests/test_temporal_squat.py tests/test_squat_feedback.py -m 'not integration' -q --basetemp outputs/squat_temporal_contract_design/run-dfbeeb29689144c2b0b7ceac283a648c/pytest-temp --junitxml=outputs/squat_temporal_contract_design/run-dfbeeb29689144c2b0b7ceac283a648c/related-tests.xml
```

Kết quả: **124 passed,0 failed/error/skipped,1 deselected,0 warnings**,24.76s;
test deselected là real-data integration. Document checks đạt: local links,
balanced code fences, draft labels/3 modes/AT01–AT15, JSON illustration/CFR/
duration arithmetic, saved evidence hashes và đối chiếu số liệu trích dẫn.
Baseline410 files:408 byte-identical, chỉ PROJECT_STATE/TASKS được sửa;0 missing,
288 old related artifact files giữ hash. Tài liệu mới không nằm trong baseline.
Core, profile, schema/API, UI, dependencies và tests source đều giữ nguyên.
Receipt hashes cuối: `document_checks-final.json`, `preservation-final.json`,
`source_evidence-final.json` trong run riêng; không ghi đè receipt sơ bộ.

Lệnh kiểm tra tài liệu/evidence độc lập:

```powershell
& .venv/Scripts/python.exe outputs/squat_temporal_contract_design/run-dfbeeb29689144c2b0b7ceac283a648c/check_design.py -final
```

Tests cũ kiểm tra compatibility/baseline, **không validate proposal hoặc detector
v2 chưa tồn tại**. V2 acceptance plan chưa chạy; không chạy full suite/integration
dữ liệu thật, inference, UI browser QA hoặc thay threshold/profile/dependencies.
Không có Git repository; không tạo commit hoặc giả lập diff.
