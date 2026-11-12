"""Auditable preprocessing with separate raw/processed pose contracts."""

from .models import (
    InterpolationSource, NormalizationTransform, PreprocessingProfile,
    PreprocessingResult, ProcessedJoint, ProcessedPoseFrame, ResolvedProfile,
    SmoothingRun, load_preprocessing_profile,
)
from .pipeline import preprocess_pose
from .serialization import read_preprocessing_result, write_preprocessing_result
from .diagnostics import summarize_preprocessing

__all__ = [
    "InterpolationSource", "NormalizationTransform", "PreprocessingProfile",
    "PreprocessingResult", "ProcessedJoint", "ProcessedPoseFrame", "ResolvedProfile",
    "SmoothingRun", "load_preprocessing_profile", "preprocess_pose",
    "read_preprocessing_result", "write_preprocessing_result", "summarize_preprocessing",
]
