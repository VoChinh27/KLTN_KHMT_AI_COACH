"""Run the full Stage 1 -> 4 audit without enabling movement assessment."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..contracts import Exercise, Viewpoint
from ..flags import GymError
from ..pose import MediaPipePoseExtractor
from ..quality import evaluate_quality
from ..video import OpenCVVideoReader
from .diagnostics import summarize_preprocessing
from .models import PreprocessingProfile, load_preprocessing_profile
from .pipeline import preprocess_pose
from .serialization import write_preprocessing_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit video -> pose -> quality -> preprocessing")
    parser.add_argument("video")
    parser.add_argument("--exercise", required=True, choices=[e.value for e in Exercise])
    parser.add_argument("--viewpoint", default="unknown", choices=[v.value for v in Viewpoint])
    parser.add_argument("--profile", help="Versioned preprocessing TOML; default is S3 v1")
    parser.add_argument("--output", required=True, help="New self-contained JSON file; parent must exist")
    args = parser.parse_args(argv)
    try:
        profile = load_preprocessing_profile(args.profile) if args.profile else PreprocessingProfile()
        reader = OpenCVVideoReader(args.video)
        with MediaPipePoseExtractor(video_id=reader.metadata.video_id,
                                   exercise=Exercise(args.exercise),
                                   viewpoint=Viewpoint(args.viewpoint)) as extractor:
            raw = tuple(extractor.extract_frames(reader.frames()))
        result = preprocess_pose(raw, evaluate_quality(raw), reader.metadata, profile=profile)
        write_preprocessing_result(result, args.output)
    except GymError as exc:
        print(json.dumps({"ok": False, "flag": exc.flag.model_dump(mode="json")}))
        return 2
    print(json.dumps({"ok": True, "output": str(Path(args.output)),
                      "summary": summarize_preprocessing(result)}))
    return 0
