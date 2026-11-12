"""Coverage is measurement availability, never a form score."""

from collections import Counter

from ..pose.mapping import JointName
from .models import PreprocessingResult


def summarize_preprocessing(result: PreprocessingResult) -> dict:
    n = len(result.frames)
    by_joint = {}
    before, after = Counter(), Counter()
    for name in JointName:
        raw = [next(j for j in f.joints if j.joint == name) for f in result.raw_frames]
        processed = [next(j for j in f.joints if j.joint == name) for f in result.frames]
        b = Counter(j.status.value for j in raw)
        a = Counter(j.status.value for j in processed)
        before.update(b)
        after.update(a)
        usable_before = result.quality.by_joint[name].observed_frames - result.quality.by_joint[name].out_of_frame_observed_frames
        usable_after = sum(j.usable for j in processed)
        by_joint[name.value] = {
            "raw_provenance": {s: b[s] for s in ("observed", "low_visibility", "missing")},
            "processed_provenance": {s: a[s] for s in ("observed", "low_visibility", "imputed", "missing")},
            "raw_observed_coverage": b["observed"] / n if n else None,
            "usable_before": usable_before, "usable_after": usable_after,
            "usable_coverage_after": usable_after / n if n else None,
        }
    total = n * len(JointName)
    return {
        "video_id": result.metadata.video_id, "total_frames": n, "fps": result.metadata.fps,
        "profile_hash": result.profile_hash,
        "resolved_profile": result.resolved_profile.model_dump(mode="json"),
        "raw_provenance": {s: before[s] for s in ("observed", "low_visibility", "missing")},
        "processed_provenance": {s: after[s] for s in ("observed", "low_visibility", "imputed", "missing")},
        "raw_observed_coverage": before["observed"] / total if total else None,
        "usable_coverage_before": sum(v["usable_before"] for v in by_joint.values()) / total if total else None,
        "usable_coverage_after": sum(v["usable_after"] for v in by_joint.values()) / total if total else None,
        "imputed_points": after["imputed"],
        "fallback_origin_frames": sum(f.normalization.translation_status == "fallback_origin" for f in result.frames),
        "fallback_scale_frames": sum(f.normalization.scale_status == "fallback_unit" for f in result.frames),
        "finite_runs": len(result.smoothing_runs),
        "smoothed_runs": sum(r.status == "smoothed" for r in result.smoothing_runs),
        "short_runs": sum(r.status == "skipped_short_run" for r in result.smoothing_runs),
        "quality_sequence_status": result.quality.sequence_status,
        "by_joint": by_joint,
    }
