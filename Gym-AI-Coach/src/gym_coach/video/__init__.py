"""Video metadata and streaming ingestion contracts."""

from .models import TimestampPolicy, VideoFrame, VideoMetadata
from .reader import OpenCVVideoReader, inspect_video

__all__ = [
    "OpenCVVideoReader",
    "TimestampPolicy",
    "VideoFrame",
    "VideoMetadata",
    "inspect_video",
]
