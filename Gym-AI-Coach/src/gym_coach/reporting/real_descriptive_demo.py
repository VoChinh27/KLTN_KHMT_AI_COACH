"""One fixed-side, real-video descriptive demo using saved audited poses.

This additive artifact workflow never runs pose inference or temporal analysis.
Domain registries, profiles, contracts, and legacy gates are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ..contracts import JointProvenance, Viewpoint
from ..features import (GEOMETRY_REGISTRY, FeatureInputContext, GeometryFeatureObservation,
                        build_feature_input_context, observe_geometry_feature)
from ..features.models import content_hash
from ..pose.models import PoseFrame
from ..preprocessing import load_preprocessing_profile, preprocess_pose, write_preprocessing_result
from ..preprocessing.models import content_hash as preprocessing_hash
from ..preprocessing.pipeline import validate_quality, validate_sequence
from ..quality import QualityProfile, QualityReport
from ..video.models import VideoMetadata

VERSION = "squat-real-descriptive-v1"
DATA_SCOPE = Path("Data/UCF-101_GYM/BodyWeightSquats")
KEYS = ("squat_knee_angle_left_deg", "squat_knee_angle_right_deg")
STATUS = dict(validation_status="NOT_VALIDATED", official_eligibility=False, assessment=False,
              g57="BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT", g6="IN_PROGRESS")
RULE = "Exact knee metric allow; rank observed/finite/in-frame hip-knee-ankle fraction descending; ties path ascending then left/right. Full clip, fixed side."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def scoped(root, relative, scope):
    name = Path(relative)
    path = (root / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to((root / scope).resolve()):
        raise ValueError("Path outside permitted scope")
    for part in (root / name, *(root / name).parents):
        if part == root:
            break
        if part.exists() and (part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Redirected input/output paths are not permitted")
    return path


def observed_in_frame(joint):
    return (joint.status is JointProvenance.OBSERVED and joint.x is not None and joint.y is not None
            and 0 <= joint.x <= 1 and 0 <= joint.y <= 1)


def rank_pair(path, side, raw, quality):
    key = KEYS[("left", "right").index(side)]
    definition = GEOMETRY_REGISTRY.get(key)
    metric = quality.metric_eligibility.get(key)
    if metric is None or metric.required_joints != definition.required_joints:
        raise ValueError("Missing/wrong exact knee metric requirements")
    count = sum(all(observed_in_frame(j) for j in frame.joints if j.joint in definition.required_joints)
                for frame in raw)
    return dict(path=path, side=side, feature_id=key, total_frames=len(raw), observed_valid_frames=count,
                observed_valid_ratio=count / len(raw) if raw else 0.,
                metric=metric.model_dump(mode="json"), sequence=quality.eligibility.model_dump(mode="json"),
                candidate=metric.action == "allow" and count > 0)


def choose_pair(rows):
    candidates = [row for row in rows if row["candidate"]]
    if not candidates:
        raise ValueError("No clip-side pair has exact metric allow and usable raw inputs")
    return sorted(candidates, key=lambda r: (-r["observed_valid_ratio"], r["path"],
                                            ("left", "right").index(r["side"])))[0]


def load_pool(root, audit_relative):
    """Verify only the supplied prior run and scoped videos; never reload a model."""
    root = Path(root).resolve()
    audit = scoped(root, audit_relative, Path("outputs"))
    inventory = json.loads((audit / "artifact-inventory.json").read_bytes())["run_files"]
    def verified(name):
        path = scoped(root, (audit / name).relative_to(root), audit.relative_to(root))
        entry = inventory[name]
        if path.stat().st_size != entry["size_bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Saved artifact hash/size mismatch: {name}")
        return path.read_bytes()
    summary = json.loads(verified("audit/summary.json"))
    plan = json.loads(verified("selection/selection-plan.json"))
    if (plan["registry_hash"] != GEOMETRY_REGISTRY.content_hash()
            or plan["quality_profile_hash"] != QualityProfile().content_hash()):
        raise ValueError("Saved registry/quality identity differs from current policies")
    if sha(audit / "selection/selection-plan.json") != summary["selection_sha256"]:
        raise ValueError("Selection linkage mismatch")
    original_audit = scoped(root, plan["audit_path"], Path("outputs"))
    if sha(original_audit) != plan["audit_sha256"]:
        raise ValueError("Original visual audit linkage mismatch")
    if any(summary[k] != v for k, v in STATUS.items()) or summary["camera_view"] != "UNKNOWN":
        raise ValueError("Prior run must retain unvalidated/disabled official status")
    if summary["assumption_status"] != "visually_assessed_unverified" or summary["documented"] is not False:
        raise ValueError("Saved side evidence must remain unverified")
    visual_rows = {r["relative_path"]: r for r in json.loads(original_audit.read_bytes())["videos"]}
    pairs, loaded = [], {}
    seen = set()
    for candidate in plan["candidates"]:
        path = candidate["path"]
        source = scoped(root, path, DATA_SCOPE)
        visual = visual_rows[path]
        if (source.parent != (root / DATA_SCOPE).resolve() or source.name != candidate["video_id"]
                or sha(source) != candidate["sha256"] or candidate["sha256"] in seen
                or candidate["documented"] is not False or candidate["camera_view"] != "UNKNOWN"
                or candidate["evidence_status"] != "visually_assessed_unverified"
                or visual["video_sha256"] != candidate["sha256"] or visual["visual_viewpoint"] != "likely_side"):
            raise ValueError("Candidate video/visual evidence linkage mismatch")
        seen.add(candidate["sha256"])
        prefix = "audit/" + candidate["review_id"] + "/"
        folder_manifest = json.loads(verified(prefix + "artifact-manifest.json"))
        # Verify all seven payloads even though not all are needed for selection.
        payloads = {name: verified(prefix + name) for name in folder_manifest}
        for name, entry in folder_manifest.items():
            if hashlib.sha256(payloads[name]).hexdigest() != entry["sha256"]:
                raise ValueError("Per-clip manifest linkage mismatch")
        raw = tuple(PoseFrame.model_validate_json(line, strict=True)
                    for line in payloads["raw-pose-unknown.jsonl"].splitlines())
        assumed = tuple(PoseFrame.model_validate_json(line, strict=True)
                        for line in payloads["pose-assumed-side.jsonl"].splitlines())
        info = json.loads(payloads["source.json"])
        metadata = VideoMetadata.model_validate_json(json.dumps(info["decoder_metadata"]), strict=True)
        quality = QualityReport.model_validate_json(payloads["quality-assumed-side.json"], strict=True)
        original_quality = QualityReport.model_validate_json(payloads["quality-unknown.json"], strict=True)
        if (quality.profile != QualityProfile() or metadata.source_path.resolve() != source
                or info["candidate"] != candidate or info["camera_view"] != "UNKNOWN"
                or info["assumption_status"] != "visually_assessed_unverified"):
            raise ValueError("Saved config/metadata/candidate mismatch")
        validate_sequence(raw, metadata)
        validate_sequence(assumed, metadata)
        validate_quality(assumed, quality)
        validate_quality(raw, original_quality)
        if len(raw) != len(assumed):
            raise ValueError("Raw/assumed timeline mismatch")
        for original, derived in zip(raw, assumed):
            a, b = original.model_dump(mode="json"), derived.model_dump(mode="json")
            if a.pop("viewpoint") != "unknown" or b.pop("viewpoint") != "side" or a != b:
                raise ValueError("Assumed-side copy must differ only in viewpoint")
        for side in ("left", "right"):
            pairs.append(rank_pair(path, side, raw, quality))
        loaded[path] = dict(raw=raw, assumed=assumed, quality=quality, original_quality=original_quality,
                            metadata=metadata, candidate=candidate, prefix=prefix, payloads=payloads)
    return audit, pairs, loaded


def measure_frames(processed, feature_id, link_sink=None):
    """Only the existing context factory/adapter computes numerical observations."""
    before = preprocessing_hash(processed)
    observations, links = [], []
    for frame in processed.frames:
        linked = build_feature_input_context(processed, frame_index=frame.frame_index)
        # The full source linkage is audited once by the factory; the adapter
        # receives the same frame/report snapshot as in Stage5's audit workflow.
        context = FeatureInputContext(frame=linked.frame, quality=linked.quality)
        observation = observe_geometry_feature(feature_id, context)
        if observation.availability.input_hash != linked.measurement_input_hash:
            raise ValueError("Measurement/factory input hash mismatch")
        observations.append(observation)
        link = linked.model_dump(mode="json", exclude={"frame", "quality", "preprocessing_result"})
        link.update(frame_index=frame.frame_index, timestamp_seconds=frame.timestamp_seconds,
                    measurement_input_hash=linked.measurement_input_hash, observation_hash=content_hash(observation))
        links.append(link)
        if link_sink:
            link_sink(frame.frame_index, len(processed.frames))
    if preprocessing_hash(processed) != before:
        raise ValueError("Measurement must preserve preprocessing/quality snapshots")
    return observations, links


def make_result(receipt, raw, processed, observations, links):
    records = []
    for frame, obs, link in zip(raw, observations, links):
        records.append(dict(video_id=frame.video_id, frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds, source_video_sha256=receipt["sha256"],
            raw_pose_hash=preprocessing_hash(frame), source_linkage=link,
            raw_joints=[j.model_dump(mode="json") for j in frame.joints],
            observation=obs.model_dump(mode="json"),
            quality_action=obs.availability.input.quality.metric_eligibility[receipt["feature_id"]].action,
            viewpoint_evidence="visually_assessed_unverified"))
    available = sum(o.status == "available" for o in observations)
    result = dict(demo_version=VERSION, **STATUS, camera_view="UNKNOWN", assumed_viewpoint="side",
        viewpoint_evidence="visually_assessed_unverified", documented=False,
        selection=receipt, metadata=processed.metadata.model_dump(mode="json"),
        preprocessing_profile=processed.profile.model_dump(mode="json"),
        preprocessing_result_hash=preprocessing_hash(processed), frames=records,
        coverage=dict(total_frames=len(records), available_frames=available,
                      unavailable_frames=len(records) - available,
                      available_ratio=available / len(records) if records else None),
        temporal=dict(status="unavailable", reason="not_run_no_independent_temporal_evidence",
                      phase=None, repetitions=None, point_turn=None, pause=None),
        coordinate_note="Displayed angle: Stage4 processed coordinates; skeleton: original raw pose",
        inference_run=False, synthetic=False)
    result["result_hash"] = artifact_hash(result)
    validate_result(result)
    return result


def validate_result(value):
    """Read-only rendering boundary; validates saved numeric contracts, no math."""
    data = dict(value)
    digest = data.pop("result_hash")
    if digest != artifact_hash(data) or data["demo_version"] != VERSION:
        raise ValueError("Demo artifact version/hash mismatch")
    if any(data[k] != v for k, v in STATUS.items()) or data["synthetic"] is not False:
        raise ValueError("Real demo must retain disabled official status")
    if data["camera_view"] != "UNKNOWN" or data["viewpoint_evidence"] != "visually_assessed_unverified" or data["documented"] is not False:
        raise ValueError("Viewpoint evidence cannot become documented")
    if data["temporal"] != dict(status="unavailable", reason="not_run_no_independent_temporal_evidence",
                                phase=None, repetitions=None, point_turn=None, pause=None):
        raise ValueError("Temporal capabilities remain unavailable")
    selection = data["selection"]
    if selection["side"] not in ("left", "right") or selection["feature_id"] != KEYS[("left", "right").index(selection["side"])]:
        raise ValueError("Fixed side/feature ID mismatch")
    meta = VideoMetadata.model_validate_json(json.dumps(data["metadata"]), strict=True)
    if len(data["frames"]) != meta.frame_count:
        raise ValueError("Coverage must include every source frame")
    available = 0
    for i, row in enumerate(data["frames"]):
        obs = GeometryFeatureObservation.model_validate_json(json.dumps(row["observation"]), strict=True)
        if ((row["video_id"], row["frame_index"], row["timestamp_seconds"]) != (meta.video_id, i, i / meta.fps)
                or (obs.video_id, obs.frame_index, obs.timestamp_seconds) != (meta.video_id, i, i / meta.fps)
                or obs.feature_id != selection["feature_id"] or obs.temporal_analysis_eligible is not False
                or row["quality_action"] != obs.availability.input.quality.metric_eligibility[obs.feature_id].action
                or row["source_video_sha256"] != selection["sha256"]
                or row["source_linkage"]["observation_hash"] != content_hash(obs)
                or row["source_linkage"]["measurement_input_hash"] != obs.availability.input_hash):
            raise ValueError("Frame/observation/source linkage mismatch")
        if obs.availability.input.quality.eligibility.model_dump(mode="json") != selection["sequence"]:
            raise ValueError("Sequence decision cannot be promoted")
        available += obs.status == "available"
    expected = dict(total_frames=meta.frame_count, available_frames=available,
                    unavailable_frames=meta.frame_count - available,
                    available_ratio=available / meta.frame_count if meta.frame_count else None)
    if data["coverage"] != expected:
        raise ValueError("Coverage numerator/denominator mismatch")
    return value


def build_demo(audit_relative, output_relative, *, workspace=None):
    root = Path(workspace or Path.cwd()).resolve()
    output = scoped(root, output_relative, Path("outputs"))
    if output.exists():
        raise FileExistsError(output)
    audit, pairs, loaded = load_pool(root, audit_relative)
    chosen = choose_pair(pairs)
    source = loaded[chosen["path"]]
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(selection_version="squat-real-fixed-side-selection-v1", rule=RULE, **chosen,
        video_id=source["metadata"].video_id, sha256=source["candidate"]["sha256"],
        candidate_pairs=len(pairs), eligible_pairs=sum(p["candidate"] for p in pairs),
        audit_run=audit.relative_to(root).as_posix(),
        audit_inventory_sha256=sha(audit / "artifact-inventory.json"),
        reused_artifacts={name: dict(path=(audit / source["prefix"] / name).relative_to(root).as_posix(),
                                   sha256=hashlib.sha256(raw).hexdigest()) for name, raw in source["payloads"].items()},
        geometry_started=False, camera_view="UNKNOWN", viewpoint_evidence="visually_assessed_unverified",
        documented=False, **STATUS)
    write_json(output / "selection.json", receipt)  # Before preprocessing or any angle computation.
    write_json(output / "ranking.json", dict(rule=RULE, pairs=pairs, selected=chosen))
    for name in ("raw-pose-unknown.jsonl", "pose-assumed-side.jsonl", "quality-assumed-side.json", "quality-unknown.json", "source.json"):
        with (output / name).open("xb") as stream:
            stream.write(source["payloads"][name])
    with (output / "original.avi").open("xb") as stream:
        stream.write((root / chosen["path"]).read_bytes())
    profile = load_preprocessing_profile(root / "configs/preprocessing_s3_v1.toml")
    processed = preprocess_pose(source["assumed"], source["quality"], source["metadata"], profile=profile)
    write_preprocessing_result(processed, output / "preprocessing.json")
    def progress(index, count):
        if index % 25 == 0 or index == count - 1:
            print(f"geometry {index + 1}/{count}", flush=True)
    observations, links = measure_frames(processed, chosen["feature_id"], progress)
    result = make_result(receipt, source["raw"], processed, observations, links)
    write_json(output / "result.json", result)
    write_json(output / "coverage.json", result["coverage"])
    from .real_descriptive_render import render_video_bundle
    render_video_bundle(root, output)
    files = {p.name: dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    write_json(output / "manifest.json", dict(demo_version=VERSION, **STATUS, files=files,
        selected_source_sha256=receipt["sha256"], result_hash=result["result_hash"], inference_run=False,
        source_camera_view="UNKNOWN", viewpoint_evidence="visually_assessed_unverified", browser_qa="NOT_RUN"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_demo(args.audit_run, args.output_dir)
    print(json.dumps(dict(selected=result["selection"]["video_id"], side=result["selection"]["side"],
                          coverage=result["coverage"], official_eligibility=False)), flush=True)


if __name__ == "__main__":
    main()
