"""AppContext: dịch vụ dùng chung toàn app (một Pipeline duy nhất)."""
from gui_app import constants as C
from gui_app.services.db_service import DbService
from gui_app.services.pipeline_service import PipelineService
from gui_app.services.sport_registry import SportRegistry


class AppContext:
    def __init__(self):
        C.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.db = DbService(C.DB_PATH)
        self.pipeline = PipelineService(C.SYSTEM_CONFIG, C.PHASE_MODEL_DIR)
        self.registry = SportRegistry(C.SPORT_CONFIG_DIR, C.PHASE_MODEL_DIR)
