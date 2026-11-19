"""Auditable video-to-pose run; ingestion stays streaming and no later stages run."""

import hashlib
import json
from pathlib import Path

from .._version import __version__
from ..contracts import ErrorCategory, Exercise
from ..flags import FlagCode, GymError, Severity
from ..video import OpenCVVideoReader
from .coverage import PoseCoverageAccumulator
from .extractor import MediaPipePoseExtractor, pose_flag
from .mapping import JointName
from .models import PoseExtractionConfig, PoseSequenceSummary
from .serialization import write_pose_frames


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_video(video_path: str | Path, exercise: Exercise | str,
                  output_dir: str | Path, *, config: PoseExtractionConfig | None = None) -> PoseSequenceSummary:
    reader = OpenCVVideoReader(video_path)
    destination = Path(output_dir)
    with MediaPipePoseExtractor(reader.metadata.video_id, exercise, config=config) as extractor:
        accumulator = PoseCoverageAccumulator(
            extractor.video_id, extractor.exercise, extractor.identity, extractor.config,
        )
        issues = {joint.value: {"low_visibility_frames": [], "missing_frames": []}
                  for joint in JointName}
        no_pose_frames = []

        def frames():
            for source in reader.frames():
                pose = extractor.extract(source)
                accumulator.add(pose)
                if not pose.has_pose:
                    no_pose_frames.append(pose.frame_index)
                for observation in pose.joints:
                    if observation.status.value in ("low_visibility", "missing"):
                        key = observation.status.value + "_frames"
                        issues[observation.joint.value][key].append(pose.frame_index)
                yield pose

        try:
            # Caller chooses a new run directory; preserve earlier artifacts.
            destination.mkdir(parents=True, exist_ok=False)
            written = write_pose_frames(destination / "frames.jsonl", frames())
            summary = accumulator.summary()
            with (destination / "summary.json").open("x", encoding="utf-8") as stream:
                stream.write(summary.model_dump_json(indent=2) + "\n")
            manifest = {
                "package_version": __version__, "input_path": Path(video_path).as_posix(),
                "input_sha256": file_sha256(reader.metadata.source_path),
                "metadata": reader.metadata.model_dump(mode="json"),
                "frames_written": written, "pose_frames_sha256": file_sha256(destination / "frames.jsonl"),
                "completed": True,
                "limitations": list(extractor.identity.limitations),
                "no_pose_frames": no_pose_frames, "joint_issues": issues,
            }
            with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except OSError as exc:
            raise GymError(pose_flag(
                FlagCode.POSE_SERIALIZATION_ERROR, "Unable to create complete run artifacts",
                extractor.video_id, category=ErrorCategory.OPERATIONAL_ERROR,
                severity=Severity.ERROR, output_dir=str(destination), detail=str(exc),
            )) from exc
    return summary
