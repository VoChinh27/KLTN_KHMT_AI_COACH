"""SessionContext: toàn bộ trạng thái 1 lượt chấm. Đổi video => reset từ bước 2."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Optional


class Step(IntEnum):
    UPLOAD = 0
    PHASE = 1
    RESULT = 2
    REPORT = 3


@dataclass
class PauseEvent:
    frame: int
    phase: str
    severity: str                 # "error" | "warn"
    items: List[dict] = field(default_factory=list)  # [{name, message_error, message_correct}]


@dataclass
class SessionContext:
    sport: str = "sprint_start"
    student_code: str = ""
    student_name: str = ""
    note: str = ""
    recorded_at: datetime = field(default_factory=datetime.now)
    source_video: Optional[Path] = None
    session_dir: Optional[Path] = None
    input_video: Optional[Path] = None     # bản sao tên duy nhất
    yaml_path: Optional[Path] = None
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    frames: List[Any] = field(default_factory=list)        # List[PoseFrame]
    proposal: Any = None                                    # PhaseProposal
    phase_set: Any = None                                   # PhaseSet
    result: Any = None                                      # SprintStartFullResult
    pauses: List[PauseEvent] = field(default_factory=list)
    quality: List[str] = field(default_factory=list)       # cảnh báo chất lượng (không dừng)
    manual_mode: bool = False                               # đánh dấu pha thủ công (thiếu mô hình)
    db_session_id: Optional[int] = None
    pose_path: Optional[Path] = None
    auto_phases_path: Optional[Path] = None
    phases_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    report_txt: Optional[Path] = None
    report_pdf: Optional[Path] = None

    def reset_from(self, step: Step) -> None:
        """Vô hiệu dây chuyền (mục 8 đặc tả)."""
        if step <= Step.PHASE:
            self.frames, self.proposal, self.phase_set, self.manual_mode = [], None, None, False
            self.pose_path = self.auto_phases_path = self.phases_path = None
        if step <= Step.RESULT:
            self.result, self.pauses, self.overlay_path, self.quality = None, [], None, []
            self.db_session_id = None
        if step <= Step.REPORT:
            self.report_txt = self.report_pdf = None
