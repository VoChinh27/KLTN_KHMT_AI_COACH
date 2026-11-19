"""S3 pose-quality diagnostics and conservative metric eligibility."""

from collections.abc import Iterable, Mapping
from typing import Sequence

from ..contracts import (
    ErrorCategory,
    Exercise,
    JointProvenance,
    SourceTier,
    Stage,
    Viewpoint,
)
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..pose.mapping import REQUIRED_JOINTS, JointName
from ..pose.models import PoseFrame
from .models import JointGapSpan, JointQuality, MetricEligibility, QualityProfile, QualityReport


def _quality_flag(
    code: FlagCode,
    message: str,
    video_id: str,
    *,
    category: ErrorCategory = ErrorCategory.INSUFFICIENT_DATA,
    severity: Severity = Severity.WARNING,
    frame_start: int | None = None,
    frame_end: int | None = None,
    joint: JointName | None = None,
    feature: str | None = None,
    **evidence: str | int | float | bool | None,
) -> StructuredFlag:
    return StructuredFlag(
        code=code,
        category=category,
        severity=severity,
        stage=Stage.QUALITY,
        message=message,
        source_tier=SourceTier.S3,
        location=FlagLocation(
            video_id=video_id,
            frame_start=frame_start,
            frame_end=frame_end,
            joint=joint.value if joint is not None else None,
            feature=feature,
        ),
        evidence=evidence,
    )


def _invalid_spans(frames: Sequence[PoseFrame], joint: JointName) -> tuple[JointGapSpan, ...]:
    spans: list[JointGapSpan] = []
    start: int | None = None
    previous: int | None = None
    for frame in frames:
        observation = next(item for item in frame.joints if item.joint == joint)
        is_out_of_frame = (
            observation.x is not None and observation.y is not None
            and not (0 <= observation.x <= 1 and 0 <= observation.y <= 1)
        )
        is_invalid = observation.status is not JointProvenance.OBSERVED or is_out_of_frame
        contiguous = previous is not None and frame.frame_index == previous + 1
        if is_invalid:
            if start is None or not contiguous:
                if start is not None and previous is not None:
                    spans.append(JointGapSpan(
                        start_frame=start, end_frame=previous,
                        length=previous - start + 1,
                    ))
                start = frame.frame_index
        elif start is not None and previous is not None:
            spans.append(JointGapSpan(
                start_frame=start, end_frame=previous, length=previous - start + 1,
            ))
            start = None
        previous = frame.frame_index
    if start is not None and previous is not None:
        spans.append(JointGapSpan(
            start_frame=start, end_frame=previous, length=previous - start + 1,
        ))
    return tuple(spans)


def _joint_quality(frames: Sequence[PoseFrame], joint: JointName) -> JointQuality:
    observations = [next(item for item in frame.joints if item.joint == joint) for frame in frames]
    statuses = [item.status for item in observations]
    total = len(statuses)
    observed = sum(status is JointProvenance.OBSERVED for status in statuses)
    low = sum(status is JointProvenance.LOW_VISIBILITY for status in statuses)
    missing = sum(status is JointProvenance.MISSING for status in statuses)
    out_of_frame = sum(
        item.x is not None and item.y is not None
        and not (0 <= item.x <= 1 and 0 <= item.y <= 1)
        for item in observations
    )
    out_of_frame_observed = sum(
        item.status is JointProvenance.OBSERVED and item.x is not None and item.y is not None
        and not (0 <= item.x <= 1 and 0 <= item.y <= 1)
        for item in observations
    )
    spans = _invalid_spans(frames, joint)
    return JointQuality(
        total_frames=total,
        observed_frames=observed,
        low_visibility_frames=low,
        missing_frames=missing,
        out_of_frame_frames=out_of_frame,
        out_of_frame_observed_frames=out_of_frame_observed,
        observed_ratio=observed / total if total else None,
        invalid_ratio=(low + missing + out_of_frame_observed) / total if total else None,
        consecutive_invalid_spans=spans,
        max_consecutive_invalid_frames=max((span.length for span in spans), default=0),
    )


def _eligibility(
    frames: Sequence[PoseFrame],
    joints: tuple[JointName, ...],
    profile: QualityProfile,
    *,
    allowed_viewpoints: tuple[Viewpoint, ...] | None = None,
) -> MetricEligibility:
    reasons: list[str] = []
    if not frames:
        reasons.append(FlagCode.INSUFFICIENT_SEQUENCE_LENGTH.value)
        return MetricEligibility(eligible=False, action="skip", required_joints=joints,
                                 reasons=tuple(dict.fromkeys(reasons)))

    selected = [_joint_quality(frames, joint) for joint in joints]
    # A bilateral metric needs every declared joint, so a weak side must not
    # be diluted by averaging it with otherwise clean joints.
    missing_ratio = max(item.missing_frames / len(frames) for item in selected)
    low_ratio = max(item.low_visibility_frames / len(frames) for item in selected)
    out_of_frame_ratio = max(item.out_of_frame_frames / len(frames) for item in selected)
    max_gap = max((item.max_consecutive_invalid_frames for item in selected), default=0)
    if len(frames) < profile.min_sequence_frames:
        reasons.append(FlagCode.INSUFFICIENT_SEQUENCE_LENGTH.value)
    if missing_ratio > profile.max_missing_ratio:
        reasons.append(FlagCode.EXCESSIVE_MISSINGNESS.value)
    if low_ratio > profile.max_low_visibility_ratio:
        reasons.append(FlagCode.LOW_POSE_VISIBILITY.value)
    if out_of_frame_ratio > profile.max_missing_ratio:
        reasons.append(FlagCode.OUT_OF_FRAME.value)
    if (max_gap > profile.max_consecutive_invalid_frames and not any(
        reason in reasons for reason in (
            FlagCode.EXCESSIVE_MISSINGNESS.value,
            FlagCode.LOW_POSE_VISIBILITY.value,
            FlagCode.OUT_OF_FRAME.value,
        )
    )):
        reasons.append(FlagCode.EXCESSIVE_MISSINGNESS.value)
    if allowed_viewpoints is not None:
        if frames[0].viewpoint is Viewpoint.UNKNOWN:
            reasons.append(FlagCode.VIEWPOINT_UNKNOWN.value)
        elif frames[0].viewpoint not in allowed_viewpoints:
            reasons.append(FlagCode.VIEWPOINT_MISMATCH.value)

    reasons = list(dict.fromkeys(reasons))
    if any(reason in {
        FlagCode.INSUFFICIENT_SEQUENCE_LENGTH.value,
        FlagCode.EXCESSIVE_MISSINGNESS.value,
    } for reason in reasons):
        action = "skip"
    elif reasons:
        action = "skip"
    elif low_ratio > 0 or missing_ratio > 0 or out_of_frame_ratio > 0:
        reasons.append(FlagCode.PARTIAL_JOINT_COVERAGE.value)
        action = "degrade"
    else:
        action = "allow"
    return MetricEligibility(
        eligible=action == "allow",
        action=action,
        required_joints=joints,
        reasons=tuple(reasons),
    )


class QualityGate:
    """Evaluate a pose sequence; profile thresholds are engineering-only S3."""

    def __init__(self, profile: QualityProfile | None = None):
        self.profile = profile or QualityProfile()

    def evaluate(
        self,
        frames: Iterable[PoseFrame],
        *,
        metric_requirements: Mapping[str, Sequence[JointName | str]] | None = None,
        metric_viewpoints: Mapping[str, Sequence[Viewpoint | str]] | None = None,
    ) -> QualityReport:
        sequence = tuple(frames)
        if sequence:
            first = sequence[0]
            expected_identity = (first.video_id, first.exercise, first.viewpoint)
            previous_index = previous_timestamp = None
            for frame in sequence:
                if (frame.video_id, frame.exercise, frame.viewpoint) != expected_identity:
                    raise GymError(_quality_flag(
                        FlagCode.INVALID_POSE_SEQUENCE,
                        "Quality input mixes video, exercise, or viewpoint identity",
                        first.video_id,
                        category=ErrorCategory.INPUT_ERROR,
                        severity=Severity.ERROR,
                    ))
                if previous_index is not None and (
                    frame.frame_index != previous_index + 1 or
                    frame.timestamp_seconds <= previous_timestamp
                ):
                    raise GymError(_quality_flag(
                        FlagCode.INVALID_POSE_SEQUENCE,
                        "Quality input must contain contiguous frame indices and increasing timestamps",
                        first.video_id,
                        category=ErrorCategory.INPUT_ERROR,
                        severity=Severity.ERROR,
                        frame_start=frame.frame_index,
                        frame_end=frame.frame_index,
                    ))
                previous_index, previous_timestamp = frame.frame_index, frame.timestamp_seconds
        else:
            # Empty input has no identity; caller supplies a stable sentinel for the report.
            first = None

        video_id = first.video_id if first else "unknown"
        exercise = first.exercise if first else Exercise.BODY_WEIGHT_SQUATS
        viewpoint = first.viewpoint if first else Viewpoint.UNKNOWN
        by_joint = {joint: _joint_quality(sequence, joint) for joint in JointName}
        required = REQUIRED_JOINTS[exercise]
        requirements = metric_requirements or {"movement_analysis": required}
        viewpoints = metric_viewpoints or {}
        eligibility: dict[str, MetricEligibility] = {}
        for metric_name, raw_joints in requirements.items():
            joints = tuple(JointName(joint) for joint in raw_joints)
            if not joints or len(set(joints)) != len(joints):
                raise ValueError(f"Metric {metric_name!r} requires unique, non-empty joints")
            allowed = viewpoints.get(metric_name)
            allowed_viewpoints = (
                tuple(Viewpoint(view) for view in allowed) if allowed is not None else None
            )
            eligibility[metric_name] = _eligibility(
                sequence, joints, self.profile, allowed_viewpoints=allowed_viewpoints,
            )

        if "movement_analysis" not in eligibility:
            eligibility["movement_analysis"] = _eligibility(
                sequence, required, self.profile,
            )
        overall = eligibility["movement_analysis"]
        flags: list[StructuredFlag] = []
        if not sequence:
            flags.append(_quality_flag(
                FlagCode.INSUFFICIENT_SEQUENCE_LENGTH,
                "Pose sequence is empty; eligibility cannot be established",
                video_id,
            ))
        elif len(sequence) < self.profile.min_sequence_frames:
            flags.append(_quality_flag(
                FlagCode.INSUFFICIENT_SEQUENCE_LENGTH,
                "Pose sequence is shorter than the configured engineering minimum",
                video_id,
                frame_start=sequence[0].frame_index,
                frame_end=sequence[-1].frame_index,
                frame_count=len(sequence),
                minimum_frames=self.profile.min_sequence_frames,
            ))
        for joint in JointName:
            quality = by_joint[joint]
            out_of_frame_indices = [
                frame.frame_index for frame in sequence
                if (observation := next(item for item in frame.joints if item.joint == joint)).x is not None
                and observation.y is not None
                and not (0 <= observation.x <= 1 and 0 <= observation.y <= 1)
            ]
            out_of_frame_spans: list[tuple[int, int]] = []
            for frame_index in out_of_frame_indices:
                if out_of_frame_spans and frame_index == out_of_frame_spans[-1][1] + 1:
                    out_of_frame_spans[-1] = (out_of_frame_spans[-1][0], frame_index)
                else:
                    out_of_frame_spans.append((frame_index, frame_index))
            for start, end in out_of_frame_spans:
                flags.append(_quality_flag(
                    FlagCode.OUT_OF_FRAME,
                    "Pose coordinate is outside normalized image bounds; affected metrics are gated",
                    video_id,
                    category=ErrorCategory.MEASUREMENT_ERROR,
                    severity=Severity.WARNING,
                    frame_start=start,
                    frame_end=end,
                    joint=joint,
                    span_frames=end - start + 1,
                ))
            for span in quality.consecutive_invalid_spans:
                if span.length > self.profile.max_consecutive_invalid_frames:
                    observations_in_span = [
                        next(item for item in frame.joints if item.joint == joint)
                        for frame in sequence
                        if span.start_frame <= frame.frame_index <= span.end_frame
                    ]
                    has_out_of_frame = any(
                        item.x is not None and item.y is not None
                        and not (0 <= item.x <= 1 and 0 <= item.y <= 1)
                        for item in observations_in_span
                    )
                    has_low_visibility = any(
                        item.status is JointProvenance.LOW_VISIBILITY
                        for item in observations_in_span
                    )
                    gap_code = (
                        FlagCode.OUT_OF_FRAME if has_out_of_frame else
                        FlagCode.LOW_POSE_VISIBILITY if has_low_visibility else
                        FlagCode.EXCESSIVE_MISSINGNESS
                    )
                    flags.append(_quality_flag(
                        gap_code,
                        "Joint has a consecutive unusable span above the S3 limit",
                        video_id,
                        category=(ErrorCategory.MEASUREMENT_ERROR
                                  if gap_code is FlagCode.OUT_OF_FRAME
                                  else ErrorCategory.INSUFFICIENT_DATA),
                        frame_start=span.start_frame,
                        frame_end=span.end_frame,
                        joint=joint,
                        gap_length=span.length,
                        maximum=self.profile.max_consecutive_invalid_frames,
                    ))
            if quality.low_visibility_frames and quality.invalid_ratio is not None:
                low_ratio = quality.low_visibility_frames / len(sequence) if sequence else 0.0
                if low_ratio > self.profile.max_low_visibility_ratio:
                    flags.append(_quality_flag(
                        FlagCode.LOW_POSE_VISIBILITY,
                        "Joint low-visibility ratio exceeds the S3 metric gate",
                        video_id,
                        joint=joint,
                        low_visibility_ratio=low_ratio,
                        maximum=self.profile.max_low_visibility_ratio,
                    ))
            if quality.missing_frames and quality.invalid_ratio is not None:
                missing_ratio = quality.missing_frames / len(sequence) if sequence else 0.0
                if missing_ratio > self.profile.max_missing_ratio:
                    flags.append(_quality_flag(
                        FlagCode.EXCESSIVE_MISSINGNESS,
                        "Joint missing ratio exceeds the S3 metric gate",
                        video_id,
                        joint=joint,
                        missing_ratio=missing_ratio,
                        maximum=self.profile.max_missing_ratio,
                    ))
            if quality.out_of_frame_frames and quality.invalid_ratio is not None:
                out_ratio = quality.out_of_frame_frames / len(sequence) if sequence else 0.0
                if out_ratio > self.profile.max_missing_ratio:
                    flags.append(_quality_flag(
                        FlagCode.OUT_OF_FRAME,
                        "Joint out-of-frame ratio exceeds the S3 metric gate",
                        video_id,
                        category=ErrorCategory.MEASUREMENT_ERROR,
                        joint=joint,
                        out_of_frame_ratio=out_ratio,
                        maximum=self.profile.max_missing_ratio,
                    ))
        if viewpoint is Viewpoint.UNKNOWN:
            flags.append(_quality_flag(
                FlagCode.VIEWPOINT_UNKNOWN,
                "Viewpoint metadata is unknown; viewpoint-dependent metrics require explicit gating",
                video_id,
                category=ErrorCategory.MEASUREMENT_ERROR,
                severity=Severity.INFO,
            ))
        for metric_name, item in eligibility.items():
            for reason in item.reasons:
                if reason in {flag.code.value for flag in flags}:
                    continue
                try:
                    code = FlagCode(reason)
                except ValueError:
                    continue
                flags.append(_quality_flag(
                    code,
                    f"Metric {metric_name} is {item.action}: {reason}",
                    video_id,
                    feature=metric_name,
                    category=(ErrorCategory.MEASUREMENT_ERROR
                              if code in {FlagCode.VIEWPOINT_UNKNOWN, FlagCode.VIEWPOINT_MISMATCH}
                              else ErrorCategory.INSUFFICIENT_DATA),
                ))

        sequence_status = (
            "eligible" if overall.action == "allow" else
            "degraded" if overall.action == "degrade" else "skipped"
        )
        total = len(sequence)
        with_pose = sum(frame.has_pose for frame in sequence)
        return QualityReport(
            video_id=video_id,
            exercise=exercise,
            viewpoint=viewpoint,
            profile=self.profile,
            profile_hash=self.profile.content_hash(),
            total_frames=total,
            frames_with_pose=with_pose,
            frames_without_pose=total - with_pose,
            sequence_status=sequence_status,
            eligibility=overall,
            by_joint=by_joint,
            metric_eligibility=eligibility,
            flags=tuple(flags),
        )


def evaluate_quality(
    frames: Iterable[PoseFrame],
    *,
    profile: QualityProfile | None = None,
    metric_requirements: Mapping[str, Sequence[JointName | str]] | None = None,
    metric_viewpoints: Mapping[str, Sequence[Viewpoint | str]] | None = None,
) -> QualityReport:
    return QualityGate(profile).evaluate(
        frames,
        metric_requirements=metric_requirements,
        metric_viewpoints=metric_viewpoints,
    )
