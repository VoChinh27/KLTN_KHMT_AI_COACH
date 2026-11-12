"""Command-line metadata inspection with machine-readable output."""

import argparse
import json
from collections.abc import Sequence

from ..flags import GymError
from .reader import inspect_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect video metadata without pose analysis")
    parser.add_argument("video", help="Path to one video file")
    parser.add_argument("--min-frames", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = inspect_video(args.video, min_frames=args.min_frames)
    except GymError as exc:
        print(json.dumps({"ok": False, "flag": exc.flag.model_dump(mode="json")}, ensure_ascii=False))
        return 2
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "metadata": metadata.model_dump(mode="json")}, ensure_ascii=False))
    return 0
