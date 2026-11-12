"""Single-video MediaPipe adapter with explicit missingness and lazy imports."""

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from ..contracts import ErrorCategory, Exercise, JointProvenance, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..video.models import VideoFrame
from .mapping import JOINT_INDEX
from .models import JointObservation, PoseBackendIdentity, PoseExtractionConfig, PoseFrame


def pose_flag(code: FlagCode, message: str, video_id: str, *,
              category: ErrorCategory = ErrorCategory.MEASUREMENT_ERROR,
              frame_index: int | None = None, joint: str | None = None,
              severity: Severity = Severity.WARNING, **evidence: Any) -> StructuredFlag:
    return StructuredFlag(
        code=code, category=category, severity=severity, stage=Stage.POSE,
        source_tier=SourceTier.S3, message=message,
        location=FlagLocation(video_id=video_id, frame_start=frame_index,
                              frame_end=frame_index, joint=joint), evidence=evidence,
    )


def _create_backend(config: PoseExtractionConfig) -> tuple[PoseBackendIdentity, Any]:
    import mediapipe as mp

    if mp.__version__ != "0.10.14":
        raise RuntimeError(f"This adapter was tested with MediaPipe 0.10.14, got {mp.__version__}")
    model = Path(mp.__file__).parent / "modules/pose_landmark/pose_landmark_full.tflite"
    if not model.is_file():
        raise RuntimeError("Bundled full pose model missing; automatic download is disabled")
    identity = PoseBackendIdentity(
        mediapipe_version=mp.__version__,
        model_asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    backend = mp.solutions.pose.Pose(
        static_image_mode=config.static_image_mode, model_complexity=config.model_complexity,
        smooth_landmarks=config.smooth_landmarks, enable_segmentation=config.enable_segmentation,
        smooth_segmentation=config.smooth_segmentation,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return identity, backend


def _field(landmark: Any, name: str) -> float | None:
    # Protobuf scalars default to zero even when absent; HasField prevents fake data.
    if hasattr(landmark, "HasField") and not landmark.HasField(name):
        return None
    value = getattr(landmark, name, None)
    return None if value is None else float(value)


class MediaPipePoseExtractor:
    """One instance per video. Video identity comes from the reader's metadata.

    The backend factory is injectable for boundary tests, not a second model.
    Native normalized x/y are retained; no image rotation or pose smoothing.
    """

    def __init__(self, video_id: str, exercise: Exercise | str, *,
                 viewpoint: Viewpoint | str = Viewpoint.UNKNOWN,
                 config: PoseExtractionConfig | None = None,
                 backend_factory: Callable[[PoseExtractionConfig], tuple[PoseBackendIdentity, Any]] | None = None):
        try:
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("video_id must be a non-empty string")
            self.video_id = video_id
            self.exercise = Exercise(exercise)
            self.viewpoint = Viewpoint(viewpoint)
            self.config = config if config is not None else PoseExtractionConfig()
            if not isinstance(self.config, PoseExtractionConfig):
                raise TypeError("config must be PoseExtractionConfig")
        except (ValueError, TypeError) as exc:
            raise GymError(pose_flag(FlagCode.INVALID_POSE_INPUT, str(exc), str(video_id),
                                    category=ErrorCategory.INPUT_ERROR,
                                    severity=Severity.ERROR)) from exc
        try:
            self.identity, self._backend = (backend_factory or _create_backend)(self.config)
        except Exception as exc:  # Only the foreign backend construction boundary.
            raise GymError(pose_flag(
                FlagCode.POSE_BACKEND_ERROR, "Unable to initialize pose backend", self.video_id,
                category=ErrorCategory.OPERATIONAL_ERROR, severity=Severity.ERROR,
                operation="initialize", exception_type=type(exc).__name__, detail=str(exc),
            )) from exc
        self._closed = False
        self._last_index: int | None = None
        self._last_timestamp: float | None = None

    def extract(self, frame: VideoFrame) -> PoseFrame:
        import cv2
        import numpy as np

        if self._closed:
            raise GymError(pose_flag(FlagCode.POSE_BACKEND_ERROR, "Extractor is closed",
                                    self.video_id, category=ErrorCategory.OPERATIONAL_ERROR,
                                    severity=Severity.ERROR, operation="process"))
        if not isinstance(frame, VideoFrame):
            raise GymError(pose_flag(FlagCode.INVALID_POSE_INPUT, "Expected VideoFrame",
                                    self.video_id, category=ErrorCategory.INPUT_ERROR,
                                    severity=Severity.ERROR))
        image = frame.image
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise GymError(pose_flag(
                FlagCode.INVALID_POSE_INPUT, "Expected a non-empty uint8 HxWx3 BGR image",
                self.video_id, frame_index=frame.frame_index,
                category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
            ))
        if self._last_index is not None and (
            frame.frame_index <= self._last_index or frame.timestamp_seconds <= self._last_timestamp
        ):
            raise GymError(pose_flag(
                FlagCode.INVALID_POSE_SEQUENCE, "Frame indices/timestamps must strictly increase",
                self.video_id, frame_index=frame.frame_index,
                category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
            ))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            result = self._backend.process(rgb)
            if not hasattr(result, "pose_landmarks"):
                raise ValueError("Backend result has no pose_landmarks field")
        except Exception as exc:  # Only the foreign backend process boundary.
            raise GymError(pose_flag(
                FlagCode.POSE_BACKEND_ERROR, "Pose backend processing failed", self.video_id,
                frame_index=frame.frame_index, category=ErrorCategory.OPERATIONAL_ERROR,
                severity=Severity.ERROR, operation="process",
                exception_type=type(exc).__name__, detail=str(exc),
            )) from exc
        pose_list = result.pose_landmarks
        if pose_list is not None and not hasattr(pose_list, "landmark"):
            raise GymError(pose_flag(
                FlagCode.POSE_BACKEND_ERROR, "Backend returned a malformed landmark list",
                self.video_id, frame_index=frame.frame_index,
                category=ErrorCategory.OPERATIONAL_ERROR, severity=Severity.ERROR,
            ))
        has_pose = pose_list is not None and len(pose_list.landmark) > 0
        landmarks = pose_list.landmark if has_pose else ()
        flags = []
        observations = []
        if not has_pose:
            flags.append(pose_flag(FlagCode.POSE_NOT_DETECTED,
                                   "Backend returned no pose; all mapped joints are missing",
                                   self.video_id, frame_index=frame.frame_index,
                                   category=ErrorCategory.INSUFFICIENT_DATA))
        for joint, index in JOINT_INDEX.items():
            reason = None
            values = {name: None for name in ("x", "y", "visibility", "presence")}
            if not has_pose:
                reason = FlagCode.POSE_NOT_DETECTED
            elif index >= len(landmarks) or landmarks[index] is None:
                reason = FlagCode.MISSING_JOINT
            else:
                try:
                    values = {name: _field(landmarks[index], name) for name in values}
                except (ValueError, TypeError, AttributeError, OverflowError):
                    reason = FlagCode.MISSING_LANDMARK_FIELD
                if reason is None:
                    if any(values[name] is None for name in ("x", "y", "visibility")):
                        reason = FlagCode.MISSING_LANDMARK_FIELD
                    elif any(value is not None and not math.isfinite(value) for value in values.values()):
                        reason = FlagCode.NON_FINITE_LANDMARK
                    elif any(values[name] is not None and not 0 <= values[name] <= 1
                             for name in ("visibility", "presence")):
                        reason = FlagCode.INVALID_LANDMARK_CONFIDENCE
            if reason is not None:
                # Invalid/missing coordinates are null, never zero substitutes.
                observation = JointObservation(
                    joint=joint, status=JointProvenance.MISSING, missing_reason=reason,
                    visibility_threshold=self.config.visibility_threshold,
                )
                if has_pose:
                    flags.append(pose_flag(reason, "Joint observation unavailable", self.video_id,
                                           frame_index=frame.frame_index, joint=joint.value))
            else:
                status = (JointProvenance.OBSERVED
                          if values["visibility"] >= self.config.visibility_threshold
                          else JointProvenance.LOW_VISIBILITY)
                observation = JointObservation(
                    joint=joint, status=status, visibility_threshold=self.config.visibility_threshold,
                    **values,
                )
                if status == JointProvenance.LOW_VISIBILITY:
                    flags.append(pose_flag(
                        FlagCode.LOW_POSE_VISIBILITY, "Measured visibility is below S3 threshold",
                        self.video_id, frame_index=frame.frame_index, joint=joint.value,
                        visibility=values["visibility"], threshold=self.config.visibility_threshold,
                    ))
                if not (0 <= values["x"] <= 1 and 0 <= values["y"] <= 1):
                    flags.append(pose_flag(
                        FlagCode.OUT_OF_FRAME, "Estimated joint is outside image bounds; coordinates retained",
                        self.video_id, frame_index=frame.frame_index, joint=joint.value,
                        x=values["x"], y=values["y"],
                    ))
            observations.append(observation)
        output = PoseFrame(
            video_id=self.video_id, exercise=self.exercise, viewpoint=self.viewpoint,
            frame_index=frame.frame_index, timestamp_seconds=frame.timestamp_seconds,
            has_pose=has_pose, backend=self.identity, config=self.config,
            joints=tuple(observations), flags=tuple(flags),
        )
        self._last_index, self._last_timestamp = frame.frame_index, frame.timestamp_seconds
        return output

    def extract_frames(self, frames: Iterable[VideoFrame]) -> Iterator[PoseFrame]:
        for frame in frames:
            yield self.extract(frame)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._backend.close()
            except Exception as exc:  # Foreign resource cleanup boundary.
                raise GymError(pose_flag(
                    FlagCode.POSE_BACKEND_ERROR, "Pose backend cleanup failed", self.video_id,
                    category=ErrorCategory.OPERATIONAL_ERROR, severity=Severity.ERROR,
                    operation="close", exception_type=type(exc).__name__, detail=str(exc),
                )) from exc

    def __enter__(self) -> "MediaPipePoseExtractor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except GymError as close_error:
            if exc is None:
                raise
            exc.add_note(str(close_error))
