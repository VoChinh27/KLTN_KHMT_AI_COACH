"""Audit saved pose JSONL with the S3 quality profile."""

import argparse
import json
from collections.abc import Sequence

from ..flags import GymError
from ..pose import read_pose_frames
from .gate import evaluate_quality
from .models import QualityProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit pose coverage and metric eligibility")
    parser.add_argument("pose_jsonl", help="PoseFrame JSONL produced by gym-pose-extract")
    parser.add_argument("--profile-version", default="quality-s3-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frames = read_pose_frames(args.pose_jsonl)
        report = evaluate_quality(frames, profile=QualityProfile(profile_version=args.profile_version))
    except GymError as exc:
        print(json.dumps({"ok": False, "flag": exc.flag.model_dump(mode="json")}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "quality_report": report.model_dump(mode="json")},
                     ensure_ascii=False))
    return 0
