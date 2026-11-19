"""Hằng số dùng chung cho GUI. Không chứa logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DATA = ROOT / "app_data"
SESSIONS_DIR = APP_DATA / "sessions"
DB_PATH = APP_DATA / "ai_coach.db"
SYSTEM_CONFIG = ROOT / "config" / "system_config.yaml"
SPORT_CONFIG_DIR = ROOT / "config" / "sport_configs"
PHASE_MODEL_DIR = ROOT / "models" / "phase_segmenter"
PHASE_FEEDBACK_LOG = ROOT / "data" / "phase_feedback" / "log.jsonl"
QSS_PATH = Path(__file__).parent / "resources" / "styles" / "dark_theme.qss"
FONT_PATH = Path(__file__).parent / "resources" / "fonts" / "NotoSans-Regular.ttf"

GOOD_THRESHOLD = 85.0   # khớp feedback_generator.GOOD_THRESHOLD
WARN_THRESHOLD = 60.0   # khớp feedback_generator.WARN_THRESHOLD
CONF_HIGH, CONF_MID = 0.85, 0.60   # độ tin cậy biên pha (PhaseSegmenter.level_thresholds)
PAUSE_REPEAT_SECONDS = 2.0         # lặp khung khi xuất video có "dừng"
VIDEO_FILTER = "Video (*.mp4 *.mov *.avi *.mkv)"
POSE_BACKENDS = ["rtmpose", "mediapipe", "yolo"]
APP_NAME = "AI Personal Sports Coach"

# môn trên UI -> mã môn trong config/sport_configs/<mã>.yaml
SPORT_ALIAS = {"Chạy bộ": "sprint_start"}
COMING_SOON = ["Cầu lông", "Nâng tạ"]   # hiện thẻ mờ "Chưa có mô hình"

PHASE_ORDER = ("vao_cho", "static", "sprint_start", "drive_phase")
PHASE_VI = {"vao_cho": "Vào chỗ", "static": "Sẵn sàng",
            "sprint_start": "Bộc phát", "drive_phase": "Chạy lao"}
# render_overlay dùng BGR -> đổi sang RGB cho Qt
PHASE_RGB = {"vao_cho": (180, 180, 180), "static": (255, 220, 0),
             "sprint_start": (255, 140, 0), "drive_phase": (80, 220, 80)}


def score_color(score: float) -> str:
    if score != score:  # NaN
        return "#888888"
    return "#2ecc71" if score >= GOOD_THRESHOLD else "#f1c40f" if score >= WARN_THRESHOLD else "#e74c3c"


def conf_color(c: float) -> str:
    return "#2ecc71" if c >= CONF_HIGH else "#f1c40f" if c >= CONF_MID else "#e74c3c"
