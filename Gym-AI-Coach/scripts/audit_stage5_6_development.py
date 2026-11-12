"""Read-only inventory of current squat files and recorded quality evidence.

No decoding, pose inference, viewpoint classification or feature measurement.
The existing heuristic camera_view field is not a documented camera protocol.
Unreviewed additional metadata is listed and keeps eligibility blocked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from gym_coach.features import SUPPORTED_KNEE_FEATURE_IDS
from gym_coach.preprocessing.models import content_hash
from gym_coach.quality import QualityReport

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
LEGACY_CSV = "Data/UCF-101_GYM/audit/UCF101_GYM_VIDEO_AUDIT.csv"
REVIEWED_METADATA = {
    "Data/UCF-101_GYM/DATASET_STATUS.md": "Action classes/statistics only; no per-video camera annotation; historical heuristic claims unverified",
    "Data/UCF-101_GYM/audit/UCF101_GYM_AUDIT_CONFIG.md": "Explicitly states camera_view UNKNOWN and automated heuristics, no pose/annotation protocol",
    "Data/UCF-101_GYM/audit/UCF101_GYM_AUDIT_SUMMARY.md": "Historical blur/lighting/readability statistics, no documented per-video viewpoint",
    LEGACY_CSV: "Per-file camera_view values are heuristic/unverified metadata, never evidence of documented side view",
    "Data/UCF-101_GYM/audit/UCF101_GYM_PROBLEM_VIDEOS.csv": "Heuristic problem subset; no camera protocol or annotator confirmation",
    "Data/UCF-101_GYM/audit/UCF101_GYM_CLASS_SUMMARY.csv": "Per-class aggregate statistics, no per-video viewpoint",
    "Data/UCF-101_GYM/audit/summary_data.json": "Aggregate heuristic totals, no per-video viewpoint",
}


def byte_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    paths = sorted(p for p in (root/"Data").rglob("*") if p.is_file()
                   and p.suffix.lower() in VIDEO_SUFFIXES
                   and ("BodyWeightSquats" in p.parts or p.name.startswith("v_BodyWeightSquats_")))
    metadata_files = sorted(p for p in (root/"Data").rglob("*")
                            if p.is_file() and p.suffix.lower() not in VIDEO_SUFFIXES)
    reviewed = [{"relative_path": p.relative_to(root).as_posix(), "sha256": byte_hash(p),
                 "review": REVIEWED_METADATA.get(p.relative_to(root).as_posix(), "NOT_REVIEWED: needs manual metadata/protocol verification"),
                 "documented_viewpoint_evidence": False} for p in metadata_files]
    with (root/LEGACY_CSV).open(encoding="utf-8-sig", newline="") as stream:
        legacy_rows = {row["file_name"]: row for row in csv.DictReader(stream)
                       if row["class_name"] == "BodyWeightSquats"}
    reports = {}
    report_sources = []
    # Actual saved Stage4 outputs, never synthetic reports embedded in Stage5
    # tests/contexts. Strictly revalidate the existing report and its recorded hash.
    for path in sorted((root/"outputs/stage4").rglob("BodyWeightSquats.processed.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = QualityReport.model_validate_json(json.dumps(payload["quality"], allow_nan=False), strict=True)
        if report.exercise.value != "BodyWeightSquats" or report.video_id != payload["metadata"]["video_id"]:
            raise ValueError(f"Saved report identity mismatch: {path}")
        if content_hash(report) != payload["quality_hash"]:
            raise ValueError(f"Saved report hash mismatch: {path}")
        source = Path(payload["metadata"]["source_path"])
        if not source.is_absolute():
            source = root/source
        try:
            relative_source = source.resolve().relative_to(root).as_posix()
        except ValueError:
            relative_source = None
        item = {"relative_path": path.relative_to(root).as_posix(), "sha256": byte_hash(path),
                "source_relative_path": relative_source, "video_id": report.video_id,
                "quality_schema_version": report.quality_schema_version, "profile_version": report.profile.profile_version,
                "profile_hash": report.profile_hash, "quality_hash": payload["quality_hash"],
                "viewpoint": report.viewpoint.value, "sequence": report.eligibility.model_dump(mode="json"),
                "knee_metrics": {key: report.metric_eligibility[key].model_dump(mode="json")
                                  if key in report.metric_eligibility else None for key in SUPPORTED_KNEE_FEATURE_IDS},
                "report_origin": "Existing Stage4 artifact; strict report/hash checks only, no new eligibility evaluation"}
        report_sources.append(item)
        if relative_source is not None:
            reports.setdefault(relative_source, []).append(item)
    candidates = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        row = legacy_rows.get(path.name)
        digest = byte_hash(path)
        saved = reports.get(relative, [])
        reasons = ["documented_side_view_evidence_missing"]
        if saved:
            if any(r["sequence"]["action"] != "allow" for r in saved):
                reasons.append("recorded_sequence_not_allow")
            if any(r["viewpoint"] != "side" for r in saved):
                reasons.append("recorded_viewpoint_not_side")
            if any(any(m is None for m in r["knee_metrics"].values()) for r in saved):
                reasons.append("recorded_knee_metric_eligibility_missing")
        else:
            reasons.append("pose_quality_report_not_recorded_for_this_path")
        group = re.fullmatch(r"v_BodyWeightSquats_g(\d+)_c(\d+)\.avi", path.name)
        candidates.append({"relative_path": relative, "video_id": path.name, "legacy_video_id": row["video_id"] if row else None,
            "group_id": group[1] if group else None, "clip_id": group[2] if group else None,
            "file_size_bytes": path.stat().st_size, "sha256": digest,
            "viewpoint": "unknown", "viewpoint_source": LEGACY_CSV if row else None,
            "viewpoint_source_scope": "Subset audit row; same filename in original dataset is not an independent annotation" if row else "No metadata",
            "source_camera_view": row.get("camera_view") if row else None,
            "evidence_type": "heuristic_metadata_unverified" if row else "none",
            "documented": False, "visual_review": "not_performed", "visual_assessment_confirmed": False,
            "historical_image_quality": row.get("overall_quality") if row else None,
            "historical_image_quality_is_pose_eligibility": False,
            "existing_quality_reports": saved,
            "decision": "ineligible" if "recorded_sequence_not_allow" in reasons else "blocked",
            "reasons": reasons})
    duplicates = {}
    for c in candidates:
        duplicates.setdefault(c["sha256"], []).append(c["relative_path"])
    return {"audit_version": "stage5_6-development-inventory-v1", "exercise": "BodyWeightSquats",
        "scope": "Current Data squat videos, all Data nonvideo metadata, existing Stage4 squat reports; no archive/video inference/visual classification",
        "status": "BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT", "candidate_file_count": len(candidates),
        "unique_content_count": len(duplicates), "documented_side_view_count": 0,
        "eligible_count": 0, "ineligible_count": sum(c["decision"] == "ineligible" for c in candidates),
        "blocked_count": sum(c["decision"] == "blocked" for c in candidates),
        "files_by_dataset": {name: sum(c["relative_path"].startswith(f"Data/{name}/") for c in candidates)
                             for name in sorted({Path(c["relative_path"]).parts[1] for c in candidates})},
        "dataset_metadata_review": reviewed, "saved_report_sources": report_sources,
        "duplicate_content_groups": [{"sha256": h, "relative_paths": ps} for h, ps in sorted(duplicates.items())],
        "candidates": candidates,
        "limitations": ["Legacy metadata is not a documented viewpoint protocol; no human visual review performed",
            "Hash equivalence identifies copies, not participant independence or camera validation",
            "QualityReport has no independent raw-sequence/video hash; identity/hash checks do not authenticate producer",
            "No new inference or numeric real-video result; unrecorded quality remains unknown"],
        "needed_input": ["Video path/ID and content SHA256, explicit BodyWeightSquats label and development-use/group/participant context",
            "Per-video annotation with viewpoint=side, annotated frame interval, annotator/date, protocol ID/version/source and explicit confirmation of camera orientation",
            "Camera fixed with full shoulders/hips/knees/ankles visible, recorded FPS/resolution/timestamp policy; occlusion/in-frame limitations noted",
            "Fresh QualityGate report from unchanged video pose with canonical left/right knee requirements/viewpoints; sequence and both metrics must allow",
            "If any gate fails, keep blocked/unavailable; no form/correctness labels required or inferred"]}


def write_inventory(data, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/"inventory.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    fields = ("relative_path", "video_id", "sha256", "viewpoint", "viewpoint_source", "source_camera_view",
              "evidence_type", "documented", "visual_review", "decision", "reasons", "quality_report_count")
    with (directory/"candidates.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in data["candidates"]:
            row = {f: item[f] for f in fields if f in item}
            row["reasons"] = ";".join(item["reasons"])
            row["quality_report_count"] = len(item["existing_quality_reports"])
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory, never an existing artifact")
    args = parser.parse_args()
    result = inventory(args.root)
    write_inventory(result, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "candidate_file_count", "unique_content_count",
          "documented_side_view_count", "eligible_count", "ineligible_count", "blocked_count", "files_by_dataset")}, indent=2))
