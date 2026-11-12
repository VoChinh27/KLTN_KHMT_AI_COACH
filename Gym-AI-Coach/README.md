# Gym AI Coach — Nghiên cứu đếm squat

**2026-10-27 — Run đầu đã được user chạy và chẩn đoán.** Saved development-test
MAE: GRU1,145287, duration baseline0,533840 trên11/24 clips. Selection index5 đúng
rule validation3clips/g02; không đổi checkpoint hoặc train lại. Chẩn đoán train/val
và76 tests hoàn tất; AI weak reference, chưa independent validation.
[Báo cáo và một đề xuất tiếp theo](docs/SQUAT_COUNT_FIRST_EXPERIMENT_DIAGNOSIS.md).
Các thông tin “chưa training” bên dưới là lịch sử lúc bàn giao; gate official không đổi.

**2026-10-26 — MODEL TRAINING GATE: READY FOR USER-RUN EXPERIMENT.**
Nhánh riêng count cấp video từ raw pose và AI weak labels đã có implementation,
frozen admission, dataset112 records và lệnh baseline/train/evaluate. Input accepted50;
supervised train33/validation3/development-test11. Validation chỉg02, không independent
GT/holdout. **Chưa fit hoặc training**; user tự chạy. 219 related tests pass,
forward-only không đổi weights. NOT_VALIDATED, official=false, assessment=false;
G5.7 blocked, G6 IN_PROGRESS. Không phải event/phase/form model.
[Mở bàn giao và lệnh chạy](outputs/squat_count_experiment/run-6d45f415095746f9add7fa32044f72b5/START_HERE.md)
· [Thiết kế, số liệu và giới hạn](docs/SQUAT_COUNT_TRAINING_PLAN.md).

Phần chuẩn bị bên dưới là lịch sử: hiện112 pose/quality evidence gồm41 preprocessing
snapshots +71 pose/quality-only, không còn thiếu71 pose records. Public research CLI:
`python -m gym_coach.counting.research.cli --help`; `train`/`baseline` chỉ user chạy
với `--user-run`. Không thay các CLI/gates cũ.

Ưu tiên từ 2026-10-25: **BodyWeightSquats → baseline đếm lần → nhãn video độc lập
→ đối chứng temporal model → đánh giá kỹ thuật sau**. Tạm dừng phát triển web/
desktop, đa bài tập và chấm form; sản phẩm hiện có được giữ nguyên bên dưới.

Milestone chuẩn bị đã kiểm kê đủ **112 clip / 25 gXX / 14.356 frame**, có **41 clip**
với saved pose/quality/preprocessing tương thích. Đã tạo gói nhãn mù, split theo
group và evaluator. **0 nhãn temporal độc lập hoàn tất; accuracy null/NOT_EVALUATED;
chưa training hoặc baseline trên video thật**. gXX không phải danh tính người.

- [Kết quả, manifest, tests và lệnh tái chạy](docs/SQUAT_COUNT_DATA_READINESS_REPORT.md).
- [Protocol nhãn temporal](docs/SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md).
- [Đề xuất contract count riêng](docs/SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md).
- [Đề cương training và MODEL TRAINING GATE](docs/SQUAT_COUNT_TRAINING_PLAN.md).
- [Gói bàn giao và hướng dẫn dùng](outputs/squat_count_readiness/run-4c26d8ba677f42fb9d0751a11688b25d/HUONG_DAN_SU_DUNG.md).

Public CLI chuẩn bị: `python -m gym_coach.counting.readiness --help`;
hai lượt nhãn: `python -m gym_coach.counting.annotations --help`;
evaluator: `python -m gym_coach.counting.evaluation --help`. Không CLI nào train.
Training đầu tiên cần hồ sơ cụ thể và phê duyệt mới của người dùng.

## Sản phẩm web hiện có — bảo toàn, tạm dừng phát triển

Giao diện chính hiện tại là **WEB tiếng Việt**, chỉ cho `BodyWeightSquats`.
Backend chỉ bind `127.0.0.1`; không LAN, cloud, tunnel hoặc gửi video ra ngoài.

## Khởi động

Từ `D:\KLTN_K28_TKD`, dùng môi trường `.venv` hiện có:

```powershell
& .venv/Scripts/python.exe -B -m gym_coach.web
```

Mở **http://127.0.0.1:8765/**. Nếu port bận, thêm `--port 8766` rồi mở URL được in.
Dừng bằng **Ctrl+C** tại terminal và đợi job hiện hành hoàn tất. Chưa hỗ trợ hủy;
không đóng cưỡng bức terminal trong lúc phân tích. Đóng/refresh tab không hủy job.
Mỗi lần khởi động server tạo session mới trong `outputs/squat_web/`; lịch sử phiên
cũ giữ trên đĩa, chưa được tự nạp vào UI sau restart. Không tự chạy lại job cũ.

## Sử dụng

1. Chọn video local (AVI/MP4/MOV/MKV/WebM, tối đa 250 MiB). Upload chỉ tới backend
   trên máy. Decoder đọc metadata/frame đầu; pipeline kiểm tra toàn timeline.
2. Chọn gối trái hoặc phải, cố định suốt run. **Strict mặc định** giữ viewpoint unknown.
3. Exploratory cần chọn riêng và tick xác nhận giả định side-view chưa kiểm chứng.
   Giả định không phải annotation đã xác nhận; raw/source camera_view vẫn UNKNOWN.
4. Bắt đầu; xem stage, frame count thật và thời gian. Một job tại một thời điểm.
   Chọn lại video đã tiếp nhận để dùng pose cache; upload lại tạo path mới và có thể miss.
5. Chỉ sau xác minh manifest/artifacts mới có overlay, coverage, báo cáo và tải file.
   AVI không preview được trong browser chưa phải lỗi pipeline. Overlay MP4 hỗ trợ byte-range.

**Chưa kiểm chứng accuracy — không chấm đúng/sai động tác.** Squat do người dùng khai báo,
chưa có action recognition. Null/“Không đủ điều kiện đo” không phải 0° hay lỗi người tập.
Skeleton là raw pose, góc là processed projected 2D. Không rep/phase/pause/score/coaching.
NOT_VALIDATED, official_eligibility=false, assessment=false, REFERENCE_PENDING;
G5.7 BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT; G6/toàn G5 IN_PROGRESS.

## Tài liệu và code

- [Báo cáo web, smoke, browser QA và giới hạn](docs/SQUAT_WEB_DEMO_REPORT.md).
- [Báo cáo dọn và khôi phục desktop](docs/PROJECT_CLEANUP_REPORT.md).
- [Mục lục active / lịch sử](docs/DOCUMENTATION_INDEX.md).
- [Trạng thái hiện hành](docs/PROJECT_STATE.md) · [Tasks](docs/TASKS.md).
- Pipeline CLI vẫn dùng được: `python -m gym_coach.reporting.single_video --help`.
  [Contract, cache và usage CLI](docs/SQUAT_SINGLE_VIDEO_PIPELINE_REPORT.md).
- `web.py` + `web_assets/` → `job_controller.py` → pipeline/progress/renderer hiện có.
  Không framework/dependency mới. Tk được lưu trữ, không phải entry point hiện hành.

Chỉ thiết kế cho local single-user demo. Không coi CSRF/Host checks là tài khoản,
authentication giữa người dùng máy hoặc hardening cho Internet. Không deploy public.
Môi trường hiện có đã dùng để test; chưa kiểm chứng cài đặt trên môi trường sạch.
