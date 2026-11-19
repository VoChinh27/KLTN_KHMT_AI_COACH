"""CLI for an explicit video/exercise and a new pose output directory."""

import argparse
import json
from collections.abc import Sequence

from ..contracts import Exercise
from ..flags import GymError
from .run import extract_video


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract native MediaPipe 2D pose with provenance")
    parser.add_argument("video")
    parser.add_argument("--exercise", required=True, choices=[exercise.value for exercise in Exercise])
    parser.add_argument("--output-dir", required=True, help="New directory; existing runs are not overwritten")
    args = parser.parse_args(argv)
    try:
        summary = extract_video(args.video, args.exercise, args.output_dir)
    except GymError as exc:
        print(json.dumps({"ok": False, "error": exc.flag.model_dump(mode="json")}))
        return 1
    print(json.dumps({"ok": True, "summary": summary.model_dump(mode="json")}))
    return 0
