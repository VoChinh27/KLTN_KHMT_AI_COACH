# Reference/ — chuẩn HLV

`ready.json`, `serve.json` được dựng từ **2 video HLV tập đúng mỗi bài** bằng:

```bash
python -m badminton3d build-reference ready Input/coach_ready_01.mp4 Input/coach_ready_04.mp4
python -m badminton3d build-reference serve Input/coach_serve_02.mp4 Input/coach_serve_07.mp4
```

| Chuẩn | Dựng từ | Ghi chú |
|---|---|---|
| `ready.json` | `coach_ready_01`, `coach_ready_04` | Gối ≈ 126° ±12°, nghiêng thân ≈ 32° ±9°, chân ≈ 1,7 vai ±0,43. **Loại `coach_ready_07`:** tư thế (gối ~155°, thân ~9°) gần như trùng nhóm người tập; đưa vào chuẩn thì không bắt được lỗi nào của người tập sai (`docs/SELECTION.md`). Loại `coach_ready_03` (đứng nghỉ), `02` và `05` (di chuyển liên tục) |
| `serve.json` | `coach_serve_02`, `coach_serve_07` | Phát đủ pha + clip cùng bối cảnh quay với người tập. `coach_serve_07` là clip cắt sát nên chỉ đóng góp 3 pha (rút vợt, chạm cầu, kết thúc). Loại `coach_serve_03` (trùng `02`), `01` (pha đánh trong trận), `04`, `05`, `06` (quay xa/ngang, tay vợt bị che) |

Mỗi góc có `center` (giá trị chuẩn), `tol` (dung sai ±), `spread` (dao động giữa các video HLV) và giá trị từng video. Dung sai tối thiểu (`floor`) đã được hiệu chỉnh (`docs/RESULTS.md` mục 5).
Khi có thêm video HLV tập đúng, chạy lại `build-reference` — tất cả điểm sẽ được tính lại ở lần `run` sau.
