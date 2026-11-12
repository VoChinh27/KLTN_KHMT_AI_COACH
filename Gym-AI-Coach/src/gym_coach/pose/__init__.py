"""Pose extraction contracts. Backends load only when an extractor is created."""

from .coverage import PoseCoverageAccumulator, summarize_pose_sequence
from .extractor import MediaPipePoseExtractor
from .mapping import JOINT_INDEX, JOINT_MAPPING_VERSION, REQUIRED_JOINTS, JointName
from .models import (JointObservation, PoseBackendIdentity, PoseExtractionConfig,
                     PoseFrame, PoseSequenceSummary)
from .serialization import read_pose_frames, write_pose_frames

__all__ = [
    "MediaPipePoseExtractor", "JointName", "JOINT_INDEX", "JOINT_MAPPING_VERSION",
    "REQUIRED_JOINTS", "JointObservation", "PoseFrame", "PoseSequenceSummary",
    "PoseBackendIdentity", "PoseExtractionConfig", "PoseCoverageAccumulator",
    "summarize_pose_sequence", "read_pose_frames", "write_pose_frames",
]
