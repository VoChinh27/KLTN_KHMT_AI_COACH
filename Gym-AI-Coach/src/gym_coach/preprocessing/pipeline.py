"""Conservative, FPS-aware preprocessing; quality decisions are never promoted."""

import math
from collections.abc import Iterable

from ..contracts import ErrorCategory, Exercise, JointProvenance, SourceTier, Stage, Viewpoint
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..pose.mapping import JointName
from ..pose.models import PoseFrame
from ..quality.gate import _joint_quality
from ..quality.models import QualityReport
from ..video.models import VideoMetadata
from .models import (
    InterpolationSource, NormalizationTransform, PreprocessingProfile,
    PreprocessingResult, ProcessedJoint, ProcessedPoseFrame, SmoothingRun,
    content_hash, sequence_hash,
)


def failure(code: FlagCode, message: str) -> GymError:
    category = (ErrorCategory.CONFIGURATION_ERROR if code == FlagCode.INVALID_CONFIG else
                ErrorCategory.OPERATIONAL_ERROR if code in (
                    FlagCode.PREPROCESSING_ERROR, FlagCode.PREPROCESSING_SERIALIZATION_ERROR
                ) else ErrorCategory.INPUT_ERROR)
    return GymError(StructuredFlag(
        code=code, category=category, severity=Severity.ERROR,
        stage=Stage.PREPROCESSING, source_tier=SourceTier.S3, message=message,
    ))


def _flag(code: FlagCode, video_id: str, start: int, end: int, message: str,
          joint: JointName | None = None) -> StructuredFlag:
    return StructuredFlag(
        code=code, category=ErrorCategory.MEASUREMENT_ERROR,
        severity=Severity.WARNING, stage=Stage.PREPROCESSING, source_tier=SourceTier.S3,
        message=message, location=FlagLocation(
            video_id=video_id, frame_start=start, frame_end=end,
            joint=joint.value if joint else None,
        ),
    )


def validate_sequence(frames: tuple[PoseFrame, ...], metadata: VideoMetadata) -> None:
    if len(frames) != metadata.frame_count:
        raise ValueError("preprocessing requires the complete video sequence")
    for position, frame in enumerate(frames):
        if frame.video_id != metadata.video_id or frame.frame_index != position:
            raise ValueError("video ID/order mismatch; expected contiguous frame indices starting at zero")
        if not math.isclose(frame.timestamp_seconds, position / metadata.fps, rel_tol=0, abs_tol=1e-9):
            raise ValueError("timestamps must match the Stage 1 CFR/FPS contract")
        if (frame.exercise, frame.viewpoint, frame.backend, frame.config) != (
                frames[0].exercise, frames[0].viewpoint, frames[0].backend, frames[0].config):
            raise ValueError("mixed exercise/viewpoint/backend/config sequence")


def validate_quality(frames: tuple[PoseFrame, ...], quality: QualityReport) -> None:
    # Custom metric decisions and flags remain intact. Only pose-derived core
    # measurements are audited; do not build a new report or run eligibility.
    # validate_sequence already enforces the complete, ordered CFR input.
    first = frames[0] if frames else None
    with_pose = sum(frame.has_pose for frame in frames)
    expected = {
        "video_id": first.video_id if first else "unknown",
        "exercise": first.exercise if first else Exercise.BODY_WEIGHT_SQUATS,
        "viewpoint": first.viewpoint if first else Viewpoint.UNKNOWN,
        "total_frames": len(frames), "frames_with_pose": with_pose,
        "frames_without_pose": len(frames) - with_pose,
        "by_joint": {joint: _joint_quality(frames, joint) for joint in JointName},
    }
    for name in ("video_id", "exercise", "viewpoint", "total_frames", "frames_with_pose",
                 "frames_without_pose", "by_joint"):
        if expected[name] != getattr(quality, name):
            raise ValueError(f"QualityReport does not match raw input: {name}")


def preprocess_pose(frames: Iterable[PoseFrame], quality: QualityReport,
                    metadata: VideoMetadata, *,
                    profile: PreprocessingProfile | None = None) -> PreprocessingResult:
    """Process the full CFR sequence for audit, even if quality says skip.

    Coordinates are usable only for in-frame observed endpoints or explicit
    imputations. Low-visibility points are excluded by default. Raw snapshots
    retain all coordinates/confidence/flags; no backend is called here.
    """
    try:
        # Revalidation catches unsafe model_construct/model_copy inputs and
        # detaches mutable nested dictionaries in the caller's quality report.
        raw = tuple(PoseFrame.model_validate_json(f.model_dump_json()) for f in frames)
        meta = VideoMetadata.model_validate_json(metadata.model_dump_json())
        selected = profile if profile is not None else PreprocessingProfile()
        selected = PreprocessingProfile.model_validate_json(selected.model_dump_json())
        report = QualityReport.model_validate_json(quality.model_dump_json())
        validate_sequence(raw, meta)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise failure(FlagCode.INVALID_PREPROCESSING_INPUT, str(exc)) from exc
    try:
        validate_quality(raw, report)
    except (ValueError, GymError) as exc:
        raise failure(FlagCode.QUALITY_INPUT_MISMATCH, str(exc)) from exc
    try:
        return _process(raw, report, meta, selected)
    except GymError:
        raise
    except Exception as exc:
        raise failure(FlagCode.PREPROCESSING_ERROR,
                      f"Preprocessing failed ({type(exc).__name__}): {exc}") from exc


def _torso_scale(hips: tuple[float, float], shoulders: tuple[float, float],
                 minimum: float) -> float | None:
    scale = math.hypot(shoulders[0] - hips[0], shoulders[1] - hips[1])
    return scale if math.isfinite(scale) and scale > minimum else None


def _process(raw: tuple[PoseFrame, ...], quality: QualityReport,
             meta: VideoMetadata, profile: PreprocessingProfile) -> PreprocessingResult:
    resolved = profile.resolve(meta.fps)
    n = len(raw)
    source = [{j.joint: j for j in frame.joints} for frame in raw]
    points = {joint: [None] * n for joint in JointName}
    interpolation = {joint: [None] * n for joint in JointName}
    unavailable = {joint: [None] * n for joint in JointName}
    aspect = meta.height / meta.width
    for i, frame in enumerate(raw):
        for joint in frame.joints:
            if joint.status == JointProvenance.MISSING:
                unavailable[joint.joint][i] = joint.missing_reason
            elif joint.status == JointProvenance.LOW_VISIBILITY:
                unavailable[joint.joint][i] = FlagCode.LOW_POSE_VISIBILITY
            elif not (0 <= joint.x <= 1 and 0 <= joint.y <= 1):
                unavailable[joint.joint][i] = FlagCode.OUT_OF_FRAME
            else:
                # Image-width units make x and y isotropic for torso distance.
                # No crop, rotation or calibration is inferred.
                points[joint.joint][i] = (joint.x, joint.y * aspect)

    for joint in JointName:
        i = 0
        while i < n:
            candidate = source[i][joint].status == JointProvenance.MISSING or (
                profile.interpolate_low_visibility and
                source[i][joint].status == JointProvenance.LOW_VISIBILITY)
            if not candidate:
                i += 1
                continue
            start = i
            while i < n and (source[i][joint].status == JointProvenance.MISSING or (
                    profile.interpolate_low_visibility and
                    source[i][joint].status == JointProvenance.LOW_VISIBILITY)):
                i += 1
            end = i
            left, right = start - 1, end
            if (left < 0 or right >= n or end - start > resolved.max_gap_frames or
                    source[left][joint].status != JointProvenance.OBSERVED or
                    source[right][joint].status != JointProvenance.OBSERVED or
                    points[joint][left] is None or points[joint][right] is None):
                continue
            refs = InterpolationSource(
                left_frame_index=raw[left].frame_index, right_frame_index=raw[right].frame_index,
                left_timestamp_seconds=raw[left].timestamp_seconds,
                right_timestamp_seconds=raw[right].timestamp_seconds,
            )
            a, b = points[joint][left], points[joint][right]
            for k in range(start, end):
                weight = (raw[k].timestamp_seconds - raw[left].timestamp_seconds) / (
                    raw[right].timestamp_seconds - raw[left].timestamp_seconds)
                points[joint][k] = (a[0] + weight * (b[0] - a[0]),
                                    a[1] + weight * (b[1] - a[1]))
                interpolation[joint][k] = refs
                unavailable[joint][k] = None

    transforms = []
    flags: list[list[StructuredFlag]] = [[] for _ in raw]
    normalized = {joint: [None] * n for joint in JointName}
    for i, frame in enumerate(raw):
        lh, rh = points[JointName.LEFT_HIP][i], points[JointName.RIGHT_HIP][i]
        ls, rs = points[JointName.LEFT_SHOULDER][i], points[JointName.RIGHT_SHOULDER][i]
        origin = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2) if lh and rh else None
        shoulders = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if ls and rs else None
        scale = _torso_scale(origin, shoulders, profile.min_torso_scale) if origin and shoulders else None
        if origin is None:
            flags[i].append(_flag(FlagCode.NORMALIZATION_ORIGIN_FALLBACK, meta.video_id,
                                 i, i, "Mid-hip unavailable; translation omitted, origin is null"))
        if scale is None:
            flags[i].append(_flag(FlagCode.NORMALIZATION_SCALE_FALLBACK, meta.video_id,
                                 i, i, "Torso unavailable/degenerate; division omitted, scale is null"))
        transforms.append(NormalizationTransform(
            translation_status="mid_hip" if origin else "fallback_origin",
            scale_status="torso" if scale else "fallback_unit",
            origin_x=origin[0] if origin else None, origin_y=origin[1] if origin else None,
            torso_scale=scale, distance_ratio_eligible=scale is not None,
        ))
        for joint in JointName:
            p = points[joint][i]
            if p is not None:
                x, y = p
                if origin is not None:
                    x, y = x - origin[0], y - origin[1]
                if scale is not None:
                    x, y = x / scale, y / scale
                normalized[joint][i] = (x, y)

    from scipy.signal import savgol_filter
    smoothed = {joint: list(normalized[joint]) for joint in JointName}
    statuses = {joint: ["unavailable"] * n for joint in JointName}
    runs = []
    for joint in JointName:
        i = 0
        while i < n:
            if normalized[joint][i] is None:
                i += 1
                continue
            start = i
            transform_key = (transforms[i].translation_status, transforms[i].scale_status)
            while (i < n and normalized[joint][i] is not None and
                   (transforms[i].translation_status, transforms[i].scale_status) == transform_key):
                i += 1
            end = i
            status = "smoothed" if end - start >= resolved.smoothing_window_frames else "skipped_short_run"
            runs.append(SmoothingRun(
                joint=joint, start_frame=start, end_frame=end - 1, status=status,
                window_frames=resolved.smoothing_window_frames,
                translation_status=transform_key[0], scale_status=transform_key[1],
            ))
            if status == "smoothed":
                values = savgol_filter(normalized[joint][start:end],
                                       resolved.smoothing_window_frames,
                                       profile.smoothing_polyorder, axis=0, mode=profile.smoothing_mode)
                if not all(math.isfinite(float(v)) for pair in values for v in pair):
                    raise failure(FlagCode.PREPROCESSING_ERROR, "Smoothing produced non-finite coordinates")
                smoothed[joint][start:end] = [(float(x), float(y)) for x, y in values]
            else:
                flags[start].append(_flag(
                    FlagCode.SMOOTHING_SHORT_RUN, meta.video_id, start, end - 1,
                    "Finite run shorter than window; unsmoothed coordinates retained", joint,
                ))
            statuses[joint][start:end] = [status] * (end - start)

    processed = []
    for i, frame in enumerate(raw):
        joints = []
        for joint in JointName:
            p, u, s = points[joint][i], normalized[joint][i], smoothed[joint][i]
            refs = interpolation[joint][i]
            joints.append(ProcessedJoint(
                joint=joint, source_status=source[i][joint].status,
                status=JointProvenance.IMPUTED if refs else source[i][joint].status,
                usable=p is not None,
                pre_normalization_x=p[0] if p else None, pre_normalization_y=p[1] if p else None,
                unsmoothed_x=u[0] if u else None, unsmoothed_y=u[1] if u else None,
                x=s[0] if s else None, y=s[1] if s else None,
                interpolation=refs, unavailable_reason=unavailable[joint][i],
                smoothing_status=statuses[joint][i],
            ))
        processed.append(ProcessedPoseFrame(
            video_id=frame.video_id, frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds, raw_frame_hash=content_hash(frame),
            normalization=transforms[i], joints=tuple(joints), flags=tuple(flags[i]),
        ))
    return PreprocessingResult(
        profile=profile, profile_hash=content_hash(profile), resolved_profile=resolved,
        metadata=meta, quality=quality, quality_hash=content_hash(quality),
        raw_sequence_hash=sequence_hash(raw), raw_frames=raw, frames=tuple(processed),
        smoothing_runs=tuple(runs),
    )
