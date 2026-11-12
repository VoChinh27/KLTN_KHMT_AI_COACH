"""Streaming UTF-8 JSON Lines pose storage with validated readback."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from ..contracts import ErrorCategory
from ..flags import FlagCode, GymError, Severity
from .extractor import pose_flag
from .models import PoseFrame


def write_pose_frames(path: str | Path, frames: Iterable[PoseFrame]) -> int:
    count = 0
    try:
        # Exclusive creation avoids overwriting prior research artifacts.
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            for frame in frames:
                stream.write(frame.model_dump_json() + "\n")
                count += 1
    except OSError as exc:
        raise GymError(pose_flag(
            FlagCode.POSE_SERIALIZATION_ERROR, "Could not write pose JSONL", "unknown",
            category=ErrorCategory.OPERATIONAL_ERROR, severity=Severity.ERROR,
            path=str(path), detail=str(exc), frames_written=count,
        )) from exc
    return count


def read_pose_frames(path: str | Path) -> Iterator[PoseFrame]:
    line_number = 0
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                yield PoseFrame.model_validate_json(line)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise GymError(pose_flag(
            FlagCode.POSE_SERIALIZATION_ERROR, "Could not read/validate pose JSONL", "unknown",
            category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
            path=str(path), line_number=line_number, detail=str(exc),
        )) from exc
