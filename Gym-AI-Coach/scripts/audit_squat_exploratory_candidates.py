"""Fresh, scope-limited bilateral audit. Saved visual labels are assumptions only.

prepare freezes selection/ranking before run starts any decoder or pose backend.
No preprocessing, geometry, temporal detector, inference cache, or policy edits.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import time
import warnings

from gym_coach.contracts import Exercise, JointProvenance, Viewpoint
from gym_coach.features.geometry_registry import GEOMETRY_REGISTRY
from gym_coach.pose.models import PoseExtractionConfig, PoseFrame
from gym_coach.quality import QualityGate, QualityProfile

DATA_ROOT = Path("Data/UCF-101_GYM/BodyWeightSquats")
KEYS = ("squat_knee_angle_left_deg", "squat_knee_angle_right_deg")
PLAN_VERSION = "squat-real-bilateral-selection-v1"
CRITERIA = {
    "candidate_pool": "All latest-audit likely_side contents in the exact allowed directory, unique by byte SHA-256",
    "analysis_interval": "Entire decoded clip; no trimming, windows or side switching",
    "demo_requires": "sequence allow AND exact left/right metric allow AND at least one bilateral allowed frame",
    "frame_allowed": "Full-clip sequence+both metrics allow AND six original joints observed, finite and in image bounds",
    "raw_joint_fraction": "Diagnostic only; cannot promote skip/degrade to allow",
    "rank": ["bilateral_allowed_frame_ratio descending", "bilateral_allowed_frames descending", "relative_path ascending"],
    "tie_policy": "No extra confidence threshold or threshold tuning",
    "no_match": "Stop after audit; no real-data demo or numeric/temporal calculation",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def local(root, relative, scope):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected scoped workspace-relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to((root / scope).resolve()):
        raise ValueError("Path outside permitted scope")
    # Reject redirected files/directories, including junctions on Windows.
    candidate = root / relative
    for item in (candidate, *candidate.parents):
        if item == root:
            break
        if item.exists() and (item.is_symlink() or getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Symlink/junction is not a permitted input/output")
    return path


def requirements():
    reqs = GEOMETRY_REGISTRY.quality_requirements(Exercise.BODY_WEIGHT_SQUATS)
    views = GEOMETRY_REGISTRY.quality_viewpoints(Exercise.BODY_WEIGHT_SQUATS)
    expected = (("left_hip", "left_knee", "left_ankle"),
                ("right_hip", "right_knee", "right_ankle"))
    for key, joints in zip(KEYS, expected):
        if tuple(j.value for j in reqs[key]) != joints or views[key] != (Viewpoint.SIDE,):
            raise ValueError("Canonical knee requirements/viewpoint policy mismatch")
    return {key: reqs[key] for key in KEYS}, {key: views[key] for key in KEYS}


def identities():
    reqs, views = requirements()
    model = Path(importlib.util.find_spec("mediapipe").origin).parent / "modules/pose_landmark/pose_landmark_full.tflite"
    return dict(quality_profile=QualityProfile().model_dump(mode="json"),
                quality_profile_hash=QualityProfile().content_hash(),
                pose_config=PoseExtractionConfig().model_dump(mode="json"),
                pose_config_hash=PoseExtractionConfig().content_hash(),
                registry_version=GEOMETRY_REGISTRY.registry_version,
                registry_hash=GEOMETRY_REGISTRY.content_hash(),
                mediapipe_version=importlib.metadata.version("mediapipe"),
                model_asset_sha256=sha(model),
                metric_requirements={k: [j.value for j in v] for k, v in reqs.items()},
                metric_viewpoints={k: [v.value for v in values] for k, values in views.items()})


def prepare(root, audit_relative, output_relative):
    root = Path(root).resolve()
    audit = local(root, audit_relative, Path("outputs"))
    output = local(root, output_relative, Path("outputs"))
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(audit.read_text(encoding="utf-8"))
    candidates, excluded, duplicates = [], [], []
    seen = set()
    for row in sorted(saved["videos"], key=lambda v: v["relative_path"]):
        source = local(root, row["relative_path"], DATA_ROOT)
        if source.parent != (root / DATA_ROOT).resolve() or source.name != row["file_name"]:
            raise ValueError("Audit file/path mismatch")
        if row["visual_viewpoint"] != "likely_side":
            excluded.append(dict(path=row["relative_path"], reason="not_latest_audit_likely_side",
                                 visual_viewpoint=row["visual_viewpoint"]))
            continue
        if (row["visual_review_status"] != "visually_assessed_unverified" or row["documented"] is not False
                or row["original_camera_view"] != "UNKNOWN" or sha(source) != row["video_sha256"]):
            raise ValueError("Visual audit/video hash or evidence status mismatch")
        if row["video_sha256"] in seen:
            duplicates.append(dict(path=row["relative_path"], sha256=row["video_sha256"]))
            continue
        seen.add(row["video_sha256"])
        samples = row["samples"]
        if len(samples) < 2 or len({s["frame_index"] for s in samples}) != len(samples):
            raise ValueError("Visual review needs distinct sampled times")
        for sample in samples:
            image = local(root, sample["image_path"], audit.parent.relative_to(root))
            if sample["agent_viewed"] is not True or sha(image) != sample["image_sha256"]:
                raise ValueError("Saved reviewed image hash mismatch")
        candidates.append(dict(path=row["relative_path"], video_id=source.name, sha256=row["video_sha256"],
            review_id=row["review_id"], camera_view="UNKNOWN", exploratory_viewpoint="side",
            evidence_status="visually_assessed_unverified", documented=False,
            observation=row["observation"], confidence=row["confidence"],
            reviewer=row["reviewer"], review_date=row["review_date"], samples=samples))
    if not candidates:
        raise ValueError("No reusable likely_side audit candidates")
    plan = dict(plan_version=PLAN_VERSION, inference_started=False, criteria=CRITERIA,
                audit_path=audit.relative_to(root).as_posix(), audit_sha256=sha(audit),
                candidate_count=len(candidates), candidates=candidates,
                excluded_before_inference=excluded, duplicate_paths_not_reinferred=duplicates,
                source_camera_view="UNKNOWN", assumption_status="visually_assessed_unverified",
                **identities())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection-plan.json", plan)
    return plan


def side_projection(raw):
    """Explicit assumption-only metadata projection; raw unknown frames unchanged."""
    if any(frame.viewpoint is not Viewpoint.UNKNOWN for frame in raw):
        raise ValueError("Raw source viewpoint must remain UNKNOWN")
    result = []
    for frame in raw:
        payload = frame.model_dump(mode="json")
        payload["viewpoint"] = "side"
        result.append(PoseFrame.model_validate_json(json.dumps(payload, allow_nan=False), strict=True))
    return tuple(result)


def valid_joint(joint):
    return (joint.status is JointProvenance.OBSERVED and joint.x is not None and joint.y is not None
            and math.isfinite(joint.x) and math.isfinite(joint.y)
            and 0 <= joint.x <= 1 and 0 <= joint.y <= 1)


def decisions(raw, quality):
    reqs, _ = requirements()
    if (not raw or quality.total_frames != len(raw) or quality.video_id != raw[0].video_id
            or quality.viewpoint is not Viewpoint.SIDE
            or quality.profile != QualityProfile()):
        raise ValueError("Quality/input identity or frame count mismatch")
    for key in KEYS:
        if key not in quality.metric_eligibility or quality.metric_eligibility[key].required_joints != reqs[key]:
            raise ValueError("Missing/wrong exact canonical metric requirements")
    gate_allow = quality.eligibility.action == "allow" and all(
        quality.metric_eligibility[key].action == "allow" for key in KEYS)
    frame_records = []
    required = set(reqs[KEYS[0]] + reqs[KEYS[1]])
    for frame in raw:
        invalid = {j.joint.value: (j.status.value if j.status is not JointProvenance.OBSERVED else "out_of_frame")
                   for j in frame.joints if j.joint in required and not valid_joint(j)}
        frame_records.append(dict(frame_index=frame.frame_index, timestamp_seconds=frame.timestamp_seconds,
                                  raw_bilateral_observed_in_frame=not invalid,
                                  bilateral_allowed=gate_allow and not invalid, invalid_joints=invalid))
    count = sum(f["bilateral_allowed"] for f in frame_records)
    raw_count = sum(f["raw_bilateral_observed_in_frame"] for f in frame_records)
    return dict(total_frames=len(raw), sequence_status=quality.sequence_status,
        sequence=quality.eligibility.model_dump(mode="json"),
        knees={key: quality.metric_eligibility[key].model_dump(mode="json") for key in KEYS},
        raw_bilateral_valid_frames=raw_count, raw_bilateral_valid_ratio=raw_count / len(raw),
        bilateral_allowed_frames=count, bilateral_allowed_frame_ratio=count / len(raw),
        exploratory_demo_eligible=gate_allow and count > 0, frames=frame_records)


def choose(results):
    eligible = [r for r in results if r.get("exploratory_demo_eligible") is True]
    return sorted(eligible, key=lambda r: (-r["bilateral_allowed_frame_ratio"],
                  -r["bilateral_allowed_frames"], r["path"]))[0] if eligible else None


def run(root, selection_relative, expected_sha, output_relative):
    root = Path(root).resolve()
    selection = local(root, selection_relative, Path("outputs"))
    destination = local(root, output_relative, Path("outputs"))
    if destination.exists():
        raise FileExistsError(destination)
    if sha(selection) != expected_sha:
        raise ValueError("Frozen selection byte hash mismatch")
    plan = json.loads(selection.read_text(encoding="utf-8"))
    if plan["plan_version"] != PLAN_VERSION or plan["inference_started"] is not False or plan["criteria"] != CRITERIA:
        raise ValueError("Expected frozen pre-inference selection/ranking")
    if any(plan[k] != v for k, v in identities().items()):
        raise ValueError("Runtime config/profile/registry changed after selection freeze")
    audit = local(root, plan["audit_path"], Path("outputs"))
    if sha(audit) != plan["audit_sha256"]:
        raise ValueError("Saved visual audit changed after selection freeze")
    audit_rows = {v["relative_path"]: v for v in json.loads(audit.read_text(encoding="utf-8"))["videos"]}
    seen = set()
    for candidate in plan["candidates"]:
        source = local(root, candidate["path"], DATA_ROOT)
        row = audit_rows[candidate["path"]]
        if (source.parent != (root / DATA_ROOT).resolve() or source.name != candidate["video_id"]
                or sha(source) != candidate["sha256"] or candidate["sha256"] in seen
                or row["video_sha256"] != candidate["sha256"] or row["visual_viewpoint"] != "likely_side"
                or candidate["camera_view"] != "UNKNOWN" or candidate["documented"] is not False
                or candidate["evidence_status"] != "visually_assessed_unverified"):
            raise ValueError("Selected source hash/path/status is not the frozen likely_side evidence")
        seen.add(candidate["sha256"])
    # Only after every candidate and policy identity has been checked.
    from gym_coach.flags import GymError
    from gym_coach.pose import MediaPipePoseExtractor
    from gym_coach.pose.serialization import write_pose_frames
    from gym_coach.video import OpenCVVideoReader
    reqs, views = requirements()
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for i, candidate in enumerate(plan["candidates"]):
        folder = destination / candidate["review_id"]
        folder.mkdir(exist_ok=False)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reader = OpenCVVideoReader(root / candidate["path"])
                with MediaPipePoseExtractor(reader.metadata.video_id, Exercise.BODY_WEIGHT_SQUATS,
                                           viewpoint=Viewpoint.UNKNOWN) as extractor:
                    raw = tuple(extractor.extract(frame) for frame in reader.frames())
                    backend = extractor.identity
                if (backend.model_asset_sha256 != plan["model_asset_sha256"]
                        or backend.mediapipe_version != plan["mediapipe_version"]):
                    raise ValueError("Backend/model identity differs from frozen plan")
                if len(raw) != reader.metadata.frame_count:
                    raise ValueError("Decoder frame count mismatch")
                write_pose_frames(folder / "raw-pose-unknown.jsonl", raw)
                projected = side_projection(raw)
                write_pose_frames(folder / "pose-assumed-side.jsonl", projected)
                gate = QualityGate()
                original_quality = gate.evaluate(raw, metric_requirements=reqs, metric_viewpoints=views)
                quality = gate.evaluate(projected, metric_requirements=reqs, metric_viewpoints=views)
                write_json(folder / "quality-unknown.json", original_quality.model_dump(mode="json"))
                write_json(folder / "quality-assumed-side.json", quality.model_dump(mode="json"))
                decision = decisions(raw, quality)
                write_json(folder / "frame-eligibility.json", decision.pop("frames"))
                write_json(folder / "source.json", dict(candidate=candidate,
                    decoder_metadata=reader.metadata.model_dump(mode="json"), camera_view="UNKNOWN",
                    assumption_status="visually_assessed_unverified", documented=False,
                    backend=backend.model_dump(mode="json"), pose_config=plan["pose_config"],
                    pose_config_hash=plan["pose_config_hash"],
                    projection="Same fresh raw observations; only viewpoint metadata differs in explicit exploratory copy",
                    frame_count=len(raw), frames_with_pose=sum(f.has_pose for f in raw),
                    warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught]))
                if sha(root / candidate["path"]) != candidate["sha256"]:
                    raise ValueError("Video bytes changed during run")
                record = dict(path=candidate["path"], video_id=candidate["video_id"], sha256=candidate["sha256"],
                    review_id=candidate["review_id"], pipeline_status="completed", **decision,
                    elapsed_seconds=time.perf_counter() - started, warning_count=len(caught))
        except GymError as exc:
            record = dict(path=candidate["path"], video_id=candidate["video_id"], sha256=candidate["sha256"],
                review_id=candidate["review_id"], pipeline_status="operational_or_input_failure",
                exploratory_demo_eligible=False, error=exc.flag.model_dump(mode="json"))
        write_json(folder / "result.json", record)
        hashes = {p.name: dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()}
        write_json(folder / "artifact-manifest.json", hashes)
        results.append(record)
        print(json.dumps(dict(progress=f"{i + 1}/{len(plan['candidates'])}", **record)), flush=True)
    selected = choose(results)
    summary = dict(selection_path=selection.relative_to(root).as_posix(), selection_sha256=expected_sha,
        criteria=CRITERIA, identities=identities(), attempted=len(results),
        completed=sum(r["pipeline_status"] == "completed" for r in results),
        results=results, selected_video_id=selected["video_id"] if selected else None,
        selected_review_id=selected["review_id"] if selected else None,
        status="EXPLORATORY_CANDIDATE_FOUND" if selected else "NO_BILATERAL_CANDIDATE",
        camera_view="UNKNOWN", assumption_status="visually_assessed_unverified", documented=False,
        validation_status="NOT_VALIDATED", official_eligibility=False, assessment=False,
        g57="BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT", g6="IN_PROGRESS",
        preprocessing_run=False, interpolation_run=False, numeric_adapters_run=False,
        temporal_detectors_run=False, browser_qa="NOT_RUN")
    write_json(destination / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.audit:
            parser.error("prepare requires --audit")
        plan = prepare(Path.cwd(), args.audit, args.output_dir)
        print(json.dumps(dict(candidates=plan["candidate_count"], selection_sha256=sha(args.output_dir / "selection-plan.json"))))
    else:
        if not args.selection or not args.selection_sha256:
            parser.error("run requires --selection and --selection-sha256")
        result = run(Path.cwd(), args.selection, args.selection_sha256, args.output_dir)
        print(json.dumps(dict(status=result["status"], selected=result["selected_video_id"])), flush=True)


if __name__ == "__main__":
    main()
