# gui_app — AI Personal Sports Coach (PyQt5)

## Chạy
```bash
# tại thư mục gốc KLTN_AI_COACH/
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements_gui.txt   # requirements.txt = của pipeline
python -m gui_app            # hoặc: python run_gui.py | ./run_gui.sh | run_gui.bat
```
Cần có: `config/system_config.yaml`, `weights/rtmpose-m.onnx`, `models/phase_segmenter/registry.json`
(thiếu mô hình pha vẫn chạy được ở chế độ **đánh dấu thủ công**).

## Kiểm thử
```bash
python -m pytest gui_app/tests -q      # có pipeline giả -> chạy được cả khi chưa có weights/GPU
```

## Thêm `mo_ta` cho từng pha (tab ② đọc từ YAML, thiếu thì dùng văn bản mặc định)
```yaml
technique_phases:
  vao_cho:      { ten_day_du: "...", engine_phase: null,         mo_ta: "Hai tay rộng bằng vai, gối sau chạm đất." }
  san_sang:     { ten_day_du: "...", engine_phase: static,       mo_ta: "Hông cao hơn vai 10–20cm; gối trước ~90°." }
```
