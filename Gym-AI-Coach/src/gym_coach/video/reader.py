"""OpenCV-backed metadata inspection and streaming frame decoding."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import ErrorCategory, SourceTier, Stage
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from .models import VideoFrame, VideoMetadata


def _cv2() -> Any:
    """Import the selected decoder only when video work is requested."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "opencv-python is required for video ingestion; install project dependencies"
        ) from exc
    return cv2


def _input_error(
    code: FlagCode,
    message: str,
    path: Path,
    **evidence: str | int | float | bool | None,
) -> GymError:
    return GymError(StructuredFlag(
        code=code,
        category=ErrorCategory.INPUT_ERROR,
        severity=Severity.ERROR,
        stage=Stage.VIDEO,
        message=message,
        source_tier=SourceTier.S3,
        location=FlagLocation(video_id=path.name or None),
        evidence={"source_path": str(path), **evidence},
    ))


def inspect_video(path: str | Path, *, min_frames: int = 1) -> VideoMetadata:
    """Read container metadata without decoding the full video.

    `min_frames` is an engineering input gate, not a movement-quality rule.
    """
    source = Path(path).expanduser().resolve()
    if isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames < 1:
        raise ValueError("min_frames must be an integer >= 1")
    if not source.is_file():
        raise _input_error(FlagCode.UNREADABLE_VIDEO, "Video file does not exist", source)

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise _input_error(FlagCode.UNREADABLE_VIDEO, "Decoder could not open video", source)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "OpenCV"
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise _input_error(FlagCode.INVALID_FPS, "FPS must be finite and positive", source, fps=fps)
    if not math.isfinite(raw_count) or raw_count < 0 or not raw_count.is_integer():
        raise _input_error(
            FlagCode.INVALID_FRAME_COUNT,
            "Container frame count must be a finite non-negative integer",
            source,
            frame_count=raw_count,
        )
    frame_count = int(raw_count)
    if frame_count < min_frames:
        raise _input_error(
            FlagCode.VIDEO_TOO_SHORT,
            "Video does not meet the minimum frame requirement",
            source,
            frame_count=frame_count,
            minimum_frames=min_frames,
        )
    if (
        not math.isfinite(raw_width)
        or not math.isfinite(raw_height)
        or raw_width <= 0
        or raw_height <= 0
        or not raw_width.is_integer()
        or not raw_height.is_integer()
    ):
        raise _input_error(
            FlagCode.INVALID_VIDEO_RESOLUTION,
            "Video resolution must contain positive integer dimensions",
            source,
            width=raw_width,
            height=raw_height,
        )

    try:
        return VideoMetadata(
            video_id=source.name,
            source_path=source,
            fps=fps,
            frame_count=frame_count,
            width=int(raw_width),
            height=int(raw_height),
            decoder=f"opencv:{backend}",
        )
    except ValidationError as exc:  # Defensive boundary around decoder metadata.
        raise _input_error(
            FlagCode.UNREADABLE_VIDEO,
            "Decoder returned invalid metadata",
            source,
            validation_error=str(exc),
        ) from exc


class OpenCVVideoReader:
    """Validated video source whose frames are decoded lazily during iteration."""

    def __init__(self, path: str | Path, *, min_frames: int = 1) -> None:
        self.metadata = inspect_video(path, min_frames=min_frames)

    def frames(self) -> Iterator[VideoFrame]:
        cv2 = _cv2()
        capture = cv2.VideoCapture(str(self.metadata.source_path))
        if not capture.isOpened():
            capture.release()
            raise _input_error(
                FlagCode.UNREADABLE_VIDEO,
                "Decoder could not reopen video for streaming",
                self.metadata.source_path,
            )

        def iterate() -> Iterator[VideoFrame]:
            frame_index = 0
            try:
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    yield VideoFrame(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / self.metadata.fps,
                        image=image,
                    )
                    frame_index += 1
            finally:
                capture.release()

            if frame_index == 0:
                raise _input_error(
                    FlagCode.UNREADABLE_VIDEO,
                    "Video opened but no frame could be decoded",
                    self.metadata.source_path,
                )
            if frame_index != self.metadata.frame_count:
                raise _input_error(
                    FlagCode.INVALID_FRAME_COUNT,
                    "Decoded frame count differs from container metadata",
                    self.metadata.source_path,
                    metadata_frame_count=self.metadata.frame_count,
                    decoded_frame_count=frame_index,
                )

        return iterate()
