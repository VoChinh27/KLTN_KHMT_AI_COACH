"""Backend-neutral contracts for video ingestion."""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator

from ..contracts import ContractModel


class TimestampPolicy(StrEnum):
    """How frame timestamps are produced.

    OpenCV does not expose a dependable VFR audit. Phase 1 therefore makes the
    CFR assumption explicit instead of presenting derived timestamps as source
    presentation timestamps.
    """

    FPS_DERIVED_CFR_ASSUMPTION = "fps_derived_cfr_assumption"


class VideoMetadata(ContractModel):
    video_id: str = Field(min_length=1)
    source_path: Path
    fps: float = Field(gt=0.0)
    frame_count: NonNegativeInt = Field(strict=True)
    width: PositiveInt = Field(strict=True)
    height: PositiveInt = Field(strict=True)
    timestamp_policy: TimestampPolicy = TimestampPolicy.FPS_DERIVED_CFR_ASSUMPTION
    decoder: str = Field(min_length=1)

    @field_validator("fps")
    @classmethod
    def validate_finite_fps(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fps must be finite")
        return value

    @property
    def duration_seconds(self) -> float:
        """Nominal CFR duration; this is not a measured VFR presentation time."""
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded frame without coupling public imports to NumPy/OpenCV."""

    frame_index: int
    timestamp_seconds: float
    image: Any

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.image is None:
            raise ValueError("image must not be None")
