"""Small, offline synthetic demo and explicitly blocked dataset preview.

Heavy dependencies are imported only inside the operation that needs them.
There is deliberately no real-video pose/feature/temporal pipeline here.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

from ..contracts import ErrorCategory, Exercise, JointProvenance, SourceTier, Stage, Viewpoint
from ..features import GEOMETRY_REGISTRY
from ..features.models import content_hash
from ..flags import FlagCode, FlagLocation, GymError, Severity, StructuredFlag
from ..segmentation import SquatSegmentationProfile
from .squat_builder import build_squat_feedback, blocked_segmentation, interval, phase_at
from .squat_models import (FeedbackSource, FeedbackFrame, FeedbackMetric, FeedbackMessage,
                           SquatFeedbackResult, SQUAT_IDS)


def byte_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def synthetic_preprocessing(*, fps: float = 30., duration: float = 6., missing_frames=()):
    """Generated 2D fixture, never a recording or human pose accuracy evidence."""
    from ..pose.mapping import JointName as J
    from ..pose.models import JointObservation, PoseBackendIdentity, PoseExtractionConfig, PoseFrame
    from ..preprocessing import preprocess_pose
    from ..quality import evaluate_quality
    from ..video.models import VideoMetadata

    if (isinstance(fps, bool) or isinstance(duration, bool) or not math.isfinite(fps)
            or not math.isfinite(duration) or fps <= 0 or duration < 0
            or duration * fps > 10000):
        raise ValueError('Finite positive FPS and a bounded synthetic duration are required')
    count = round(fps * duration) + 1
    video_id = 'synthetic-squat-feedback-gap-v1' if missing_frames else 'synthetic-squat-feedback-v1'
    config = PoseExtractionConfig()
    backend = PoseBackendIdentity(mediapipe_version='NOT_RUN_SYNTHETIC_FIXTURE', limitations=(
        'Generated coordinates; MediaPipe was not executed; no accuracy or form evidence',))
    raw = []
    for index in range(count):
        t = index / fps
        theta = 140 + 30 * math.cos(math.pi * (t - 1))
        bend = (170 - theta) / 60
        phi = -math.pi / 2 - .35 * bend
        knee = (.43 - .10 * bend, .62 + .05 * bend)
        hip = (knee[0] + .18 * math.cos(phi), knee[1] + .18 * math.sin(phi))
        ankle = (knee[0] + .18 * math.cos(phi + math.radians(theta)),
                 knee[1] + .18 * math.sin(phi + math.radians(theta)))
        shoulder = (hip[0] + .04 * bend, hip[1] - .22)
        points = {J.NOSE: (shoulder[0], shoulder[1] - .09)}
        for side, offset in (('left', 0.), ('right', .035)):
            for name, point in (('hip', hip), ('knee', knee), ('ankle', ankle), ('shoulder', shoulder),
                                ('elbow', (shoulder[0] + .08, shoulder[1] + .06)),
                                ('wrist', (shoulder[0] + .14, shoulder[1] + .07))):
                points[J(f'{side}_{name}')] = (point[0] + offset, point[1])
        joints = []
        for joint in J:
            if index in missing_frames and joint == J.LEFT_KNEE:
                joints.append(JointObservation(joint=joint, status=JointProvenance.MISSING,
                                               missing_reason=FlagCode.MISSING_JOINT))
            else:
                x, y = points[joint]
                joints.append(JointObservation(joint=joint, x=x, y=y, visibility=.99,
                                               status=JointProvenance.OBSERVED))
        raw.append(PoseFrame(video_id=video_id, exercise=Exercise.BODY_WEIGHT_SQUATS,
            viewpoint=Viewpoint.SIDE, frame_index=index, timestamp_seconds=t, has_pose=True,
            backend=backend, config=config, joints=tuple(joints)))
    metadata = VideoMetadata(video_id=video_id, source_path=Path(video_id), fps=fps,
        frame_count=count, width=320, height=320, decoder='synthetic_fixture_no_decoder')
    quality = evaluate_quality(raw,
        metric_requirements=GEOMETRY_REGISTRY.quality_requirements(Exercise.BODY_WEIGHT_SQUATS),
        metric_viewpoints=GEOMETRY_REGISTRY.quality_viewpoints(Exercise.BODY_WEIGHT_SQUATS))
    return preprocess_pose(raw, quality, metadata)


def _preview_error(reason: str) -> GymError:
    return GymError(StructuredFlag(code=FlagCode.INVALID_FEEDBACK_INPUT,
        category=ErrorCategory.INPUT_ERROR, severity=Severity.ERROR, stage=Stage.REPORTING,
        source_tier=SourceTier.S3, message=reason, evidence={'reason': 'invalid_dataset_preview'}))


def build_blocked_video_feedback(video: Path, workspace: Path, output: Path) -> SquatFeedbackResult:
    """Read permitted video pixels + original CSV only; never infer pose/viewpoint.

    The original automated audit's UNKNOWN fields are preserved in a sidecar.
    Hashes identify bytes, not viewpoint ground truth. Output must be new.
    """
    root = workspace.resolve()
    source = (root / video).resolve()
    allowed = (root / 'Data/UCF-101_GYM/BodyWeightSquats').resolve()
    try:
        relative = source.relative_to(root).as_posix()
        source.relative_to(allowed)
    except ValueError as exc:
        raise _preview_error('Only BodyWeightSquats videos within Data/UCF-101_GYM are admitted') from exc
    if source.parent != allowed or not source.is_file() or source.suffix.lower() != '.avi':
        raise _preview_error('Expected an existing direct BodyWeightSquats AVI input')
    audit = root / 'Data/UCF-101_GYM/audit/UCF101_GYM_VIDEO_AUDIT.csv'
    with audit.open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['class_name'] == 'BodyWeightSquats' and row['file_name'] == source.name]
    if len(rows) != 1 or rows[0]['camera_view'] != 'UNKNOWN':
        raise _preview_error('This preview requires one original UNKNOWN-viewpoint audit row; no automatic annotation admission')
    from ..video import OpenCVVideoReader
    import cv2

    before = byte_hash(source)
    reader = OpenCVVideoReader(source)
    meta = reader.metadata
    selected = {0, meta.frame_count // 2, meta.frame_count - 1}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    flag = StructuredFlag(code=FlagCode.DOCUMENTED_INPUT_MISSING,
        category=ErrorCategory.INSUFFICIENT_DATA, severity=Severity.WARNING,
        stage=Stage.REPORTING, source_tier=SourceTier.S3,
        location=FlagLocation(video_id=source.name),
        message='Official Stage 5.7 blocked: dataset viewpoint UNKNOWN; no linked confirmed annotation',
        evidence={'camera_view': rows[0]['camera_view'], 'person_visibility': rows[0]['person_visibility'],
                  'occlusion': rows[0]['occlusion'], 'source': 'original_automated_dataset_audit'})
    profile = SquatSegmentationProfile()
    segmentation = blocked_segmentation((flag,), profile)
    frames = []
    decoded = 0
    for frame in reader.frames():
        decoded += 1
        if frame.frame_index not in selected:
            continue
        name = f'frames/frame-{frame.frame_index}.jpg'
        ok, encoded = cv2.imencode('.jpg', frame.image)
        if not ok:
            raise _preview_error('Could not encode a selected preview frame')
        with (output / name).open('xb') as stream:
            stream.write(encoded.tobytes())
        metrics = tuple(FeedbackMetric(feature_id=key, definition_hash=GEOMETRY_REGISTRY.get(key).content_hash(),
            status='unavailable', value=None, evidence=interval(frame.frame_index, frame.frame_index, meta.fps),
            quality_action='not_run', observation_hash=None, input_hash=None, joint_inputs=(),
            reasons=(flag.code,), flags=(flag,)) for key in SQUAT_IDS)
        phase, evidence, reasons = phase_at(segmentation, frame.frame_index, meta.fps)
        frames.append(FeedbackFrame(frame_index=frame.frame_index, timestamp_seconds=frame.timestamp_seconds,
            phase=phase, phase_evidence=evidence, phase_reasons=reasons, source_frame_hash=None,
            overlay_source='not_run', overlay_joints=(), metrics=metrics,
            preview_image=name, preview_image_sha256=byte_hash(output / name)))
    if decoded != meta.frame_count or before != byte_hash(source):
        raise _preview_error('Video decode count/byte identity changed during preview')
    source_audit = FeedbackSource(kind='real_preview', evidence_status='dataset_viewpoint_unknown',
        video_id=source.name, relative_path=relative, video_byte_sha256=before,
        metadata_byte_sha256=byte_hash(audit), fps=meta.fps, frame_count=meta.frame_count,
        width=meta.width, height=meta.height, preview_sampling='three_preview_frames')
    report = SquatFeedbackResult(status='blocked', registry_hash=GEOMETRY_REGISTRY.content_hash(),
        source=source_audit, source_hash=content_hash(source_audit), quality=None,
        segmentation=segmentation, segmentation_hash=segmentation.content_hash(), frames=tuple(frames),
        messages=(FeedbackMessage(kind='data_status', text='Preview video gốc: chưa chạy pose, QualityGate hoặc feature. Camera metadata UNKNOWN.'),
                  FeedbackMessage(kind='assessment_blocked', text='Thiếu bằng chứng viewpoint được xác nhận và sequence/bilateral knee allow; giữ blocked.')),
        flags=(flag,))
    # Preserve the audit row as received; never rewrite its UNKNOWN fields.
    write_json(output / 'dataset_row.json', rows[0])
    write_json(output / 'decoder_metadata.json', json.loads(meta.model_dump_json()))
    return report


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def write_case(output: Path, feedback: SquatFeedbackResult, preprocessing=None):
    from .renderer import render_feedback
    from ..features import FeatureInputContext, observe_geometry_feature
    feedback = SquatFeedbackResult.model_validate_json(feedback.model_dump_json(), strict=True)
    write_json(output / 'feedback.json', json.loads(feedback.model_dump_json()))
    if preprocessing is not None:
        write_json(output / 'preprocessing.json', json.loads(preprocessing.model_dump_json()))
        # Full observations persist alongside projected hashes, including provenance.
        with (output / 'observations.jsonl').open('x', encoding='utf-8') as stream:
            for frame in preprocessing.frames:
                context = FeatureInputContext(frame=frame, quality=preprocessing.quality)
                for key in SQUAT_IDS:
                    stream.write(observe_geometry_feature(key, context).model_dump_json() + '\n')
    with (output / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(render_feedback(feedback))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline Gym Result Feedback, synthetic + blocked dataset preview')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing outputs are refused')
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--video', type=Path, help='Optional AVI under Data/UCF-101_GYM/BodyWeightSquats')
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, missing in (('synthetic', ()), ('synthetic-gap', tuple(range(58, 63)))):
            case = args.output / name
            case.mkdir()
            source = synthetic_preprocessing(missing_frames=missing)
            write_case(case, build_squat_feedback(source), source)
        if args.video is not None:
            case = args.output / 'real-blocked'
            write_case(case, build_blocked_video_feedback(args.video, args.workspace, case))
        write_json(args.output / 'feedback.schema.json', SquatFeedbackResult.model_json_schema())
        write_json(args.output / 'feature-registry.json', GEOMETRY_REGISTRY.model_dump(mode='json'))
        write_json(args.output / 'temporal-profile.json', SquatSegmentationProfile().model_dump(mode='json'))
        write_json(args.output / 'inventory.json', {
            'artifact_version': 'gym-feedback-demo-artifacts-v1',
            'command': [sys.executable, '-m', 'gym_coach.reporting', *(argv if argv is not None else sys.argv[1:])],
            'assessment_enabled': False, 'official_gate_status': 'BLOCKED_PENDING_STAGE5_7',
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                **{name: importlib.metadata.version(name) for name in ('ai-coach-gym','pydantic','numpy','scipy','mediapipe','opencv-contrib-python')}},
            'implementation_sha256': {p.relative_to(Path(__file__).parents[1]).as_posix(): byte_hash(p)
                                     for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
            'files': {p.relative_to(args.output).as_posix(): byte_hash(p)
                      for p in sorted(args.output.rglob('*')) if p.is_file()},
        })
        print(json.dumps({'output': str(args.output.resolve()), 'official_gate': 'BLOCKED_PENDING_STAGE5_7',
                          'synthetic': 'synthetic/index.html', 'real_preview': args.video is not None}))
        return 0
    except (OSError, ValueError, TypeError, GymError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
