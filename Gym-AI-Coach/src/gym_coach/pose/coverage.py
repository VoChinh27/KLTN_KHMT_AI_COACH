"""Observed coverage counters, without gating or technical conclusions."""

from collections.abc import Iterable

from ..contracts import ErrorCategory, Exercise, JointProvenance, Viewpoint
from ..flags import FlagCode, GymError, Severity
from .extractor import pose_flag
from .mapping import REQUIRED_JOINTS, JointName
from .models import JointCoverage, PoseBackendIdentity, PoseExtractionConfig, PoseFrame, PoseSequenceSummary


class PoseCoverageAccumulator:
    def __init__(self, video_id: str, exercise: Exercise, backend: PoseBackendIdentity,
                 config: PoseExtractionConfig, viewpoint: Viewpoint = Viewpoint.UNKNOWN):
        self.video_id, self.exercise = video_id, Exercise(exercise)
        self.backend, self.config, self.viewpoint = backend, config, Viewpoint(viewpoint)
        self.total_frames = self.frames_with_pose = 0
        self.counts = {joint: {status: 0 for status in (
            JointProvenance.OBSERVED, JointProvenance.LOW_VISIBILITY, JointProvenance.MISSING
        )} for joint in JointName}
        self._last_index: int | None = None
        self._last_timestamp: float | None = None

    def add(self, frame: PoseFrame) -> None:
        if ((frame.video_id, frame.exercise, frame.backend, frame.config, frame.viewpoint) !=
            (self.video_id, self.exercise, self.backend, self.config, self.viewpoint) or
            (self._last_index is not None and (frame.frame_index <= self._last_index or
                                               frame.timestamp_seconds <= self._last_timestamp))):
            raise GymError(pose_flag(
                FlagCode.INVALID_POSE_SEQUENCE, "Mixed identity/config or unordered sequence",
                self.video_id, frame_index=frame.frame_index,
                category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR,
            ))
        self.total_frames += 1
        self.frames_with_pose += int(frame.has_pose)
        for joint in frame.joints:
            self.counts[joint.joint][joint.status] += 1
        self._last_index, self._last_timestamp = frame.frame_index, frame.timestamp_seconds

    def summary(self) -> PoseSequenceSummary:
        total = self.total_frames
        by_joint = {
            joint: JointCoverage(
                observed=counts[JointProvenance.OBSERVED],
                low_visibility=counts[JointProvenance.LOW_VISIBILITY],
                missing=counts[JointProvenance.MISSING],
                pose_coverage=counts[JointProvenance.OBSERVED] / total if total else None,
            ) for joint, counts in self.counts.items()
        }
        required = REQUIRED_JOINTS[self.exercise]
        flags = () if total else (pose_flag(
            FlagCode.INSUFFICIENT_SEQUENCE_LENGTH, "Empty pose sequence; coverage is undefined",
            self.video_id, category=ErrorCategory.INSUFFICIENT_DATA,
        ),)
        return PoseSequenceSummary(
            video_id=self.video_id, exercise=self.exercise, viewpoint=self.viewpoint,
            backend=self.backend, config=self.config, config_hash=self.config.content_hash(),
            total_frames=total, frames_with_pose=self.frames_with_pose,
            frames_without_pose=total - self.frames_with_pose,
            pose_detection_ratio=self.frames_with_pose / total if total else None,
            pose_coverage=sum(c.observed for c in by_joint.values()) / (len(JointName) * total) if total else None,
            required_joint_coverage=sum(by_joint[j].observed for j in required) / (len(required) * total) if total else None,
            by_joint=by_joint, flags=flags,
        )


def summarize_pose_sequence(frames: Iterable[PoseFrame], *, video_id: str, exercise: Exercise,
                            backend: PoseBackendIdentity, config: PoseExtractionConfig,
                            viewpoint: Viewpoint = Viewpoint.UNKNOWN) -> PoseSequenceSummary:
    accumulator = PoseCoverageAccumulator(video_id, exercise, backend, config, viewpoint)
    for frame in frames:
        accumulator.add(frame)
    return accumulator.summary()
