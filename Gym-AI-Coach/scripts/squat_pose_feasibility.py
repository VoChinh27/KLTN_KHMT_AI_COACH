"""Resume-safe orchestration of existing pose/quality APIs; no counting or training."""
import argparse
from collections import Counter
from importlib.metadata import distribution, version
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from gym_coach.contracts import Exercise
from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.readiness import file_info, write_json
from gym_coach.features import GEOMETRY_REGISTRY
from gym_coach.pose.models import PoseExtractionConfig, PoseSequenceSummary
from gym_coach.pose.serialization import read_pose_frames
from gym_coach.pose.coverage import summarize_pose_sequence
from gym_coach.quality import QualityGate, QualityProfile, QualityReport

KEYS = ('squat_knee_angle_left_deg', 'squat_knee_angle_right_deg')
VERSION = 'squat-pose-feasibility-audit-v1'


def configuration():
    asset = distribution('mediapipe').locate_file('mediapipe/modules/pose_landmark/pose_landmark_full.tflite')
    if version('mediapipe') != '0.10.14' or not asset.is_file():
        raise ValueError('Existing tested MediaPipe/full asset required; no download/install')
    requirements = GEOMETRY_REGISTRY.quality_requirements(Exercise.BODY_WEIGHT_SQUATS)
    viewpoints = GEOMETRY_REGISTRY.quality_viewpoints(Exercise.BODY_WEIGHT_SQUATS)
    return dict(pose=PoseExtractionConfig().model_dump(mode='json'),
                pose_hash=PoseExtractionConfig().content_hash(),
                quality=QualityProfile().model_dump(mode='json'),
                quality_hash=QualityProfile().content_hash(),
                mediapipe_version=version('mediapipe'), model_asset=file_info(asset),
                registry_version=GEOMETRY_REGISTRY.registry_version,
                registry_hash=GEOMETRY_REGISTRY.content_hash(),
                metric_requirements={k: [v.value for v in requirements[k]] for k in KEYS},
                metric_viewpoints={k: [v.value for v in viewpoints[k]] for k in KEYS},
                camera_view='UNKNOWN', preprocessing=False, temporal=False, training=False)


def read_plan(run):
    run = Path(run).resolve()
    if not run.is_relative_to(Path.cwd().resolve()/'outputs'):
        raise ValueError('Run must be under workspace outputs')
    plan = json.loads((run/'plan.json').read_bytes())
    if (file_info(run/'plan.json')['sha256'] != (run/'plan.sha256').read_text().strip()
            or plan['version'] != VERSION or plan['configuration'] != configuration()):
        raise ValueError('Frozen plan/configuration mismatch')
    for name, identity in plan['code_identity'].items():
        if file_info(name) != identity:
            raise ValueError('Code changed after freeze: '+name)
    return plan


def source_check(source):
    path = Path(source['path'])
    if (not path.resolve().is_relative_to(Path.cwd().resolve()/'Data/UCF-101_GYM/BodyWeightSquats')
            or path.name != source['video_id'] or file_info(path)['sha256'] != source['sha256']):
        raise ValueError('Source path/ID/hash mismatch')


def validate_pose(source, folder, config):
    """Read all pose records and decoder identity, without running inference."""
    source_check(source)
    manifest = json.loads((folder/'manifest.json').read_bytes())
    if (manifest.get('completed') is not True or manifest['input_sha256'] != source['sha256']
            or Path(manifest['input_path']).resolve() != Path(source['path']).resolve()
            or manifest['pose_frames_sha256'] != file_info(folder/'frames.jsonl')['sha256']):
        raise ValueError('Pose completion/source/hash mismatch')
    metadata = manifest['metadata']
    if any(metadata[k] != source[k] for k in ('video_id', 'fps', 'frame_count', 'width', 'height', 'timestamp_policy')):
        raise ValueError('Pose metadata differs from frozen decoder identity')
    frames = tuple(read_pose_frames(folder/'frames.jsonl'))
    if len(frames) != source['frame_count'] or manifest['frames_written'] != len(frames):
        raise ValueError('Incomplete pose timeline')
    for i, f in enumerate(frames):
        if (f.video_id != source['video_id'] or f.exercise != Exercise.BODY_WEIGHT_SQUATS
                or f.viewpoint.value != 'unknown' or f.frame_index != i
                or f.timestamp_seconds != i/source['fps'] or f.config.model_dump(mode='json') != config['pose']
                or f.backend.mediapipe_version != config['mediapipe_version']
                or f.backend.model_asset_sha256 != config['model_asset']['sha256']):
            raise ValueError('Pose source/timeline/config/viewpoint mismatch')
    summary = PoseSequenceSummary.model_validate_json((folder/'summary.json').read_bytes(), strict=True)
    if summary != summarize_pose_sequence(frames, video_id=source['video_id'], exercise=Exercise.BODY_WEIGHT_SQUATS,
                                         backend=frames[0].backend, config=frames[0].config):
        raise ValueError('Saved coverage differs from actual records')
    return frames, summary


def quality(frames, config):
    return QualityGate(QualityProfile.model_validate(config['quality'])).evaluate(
        frames, metric_requirements=config['metric_requirements'], metric_viewpoints=config['metric_viewpoints'])


def verify_completed(plan, row, attempt):
    """Receipt alone is insufficient: verify hashes, full timeline and recomputed gate."""
    result = json.loads((attempt/'completed.json').read_bytes())
    if (result['plan_hash'] != canonical_hash(plan) or result['source'] != row['source']
            or result['status'] != 'completed' or result['version'] != VERSION):
        raise ValueError('Completion identity mismatch')
    for name, identity in result['files'].items():
        path = (attempt/name).resolve()
        if not path.is_relative_to(attempt.resolve()) or file_info(path) != identity:
            raise ValueError('Completion artifact hash mismatch')
    frames, summary = validate_pose(row['source'], attempt/'pose', plan['configuration'])
    saved = QualityReport.model_validate_json((attempt/'quality.json').read_bytes(), strict=True)
    if saved != quality(frames, plan['configuration']):
        raise ValueError('Saved QualityReport differs from canonical gate')
    return result


def find_completed(plan, row, run):
    issues = []
    for attempt in sorted((run/'clips'/row['source']['video_id']).glob('attempt-*')):
        if not (attempt/'completed.json').is_file():
            issues.append(dict(attempt=str(attempt), reason='incomplete_or_failed_attempt'))
            continue
        try:
            return verify_completed(plan, row, attempt), attempt, issues
        except Exception as exc:
            issues.append(dict(attempt=str(attempt), reason='invalid_completed_artifact', detail=str(exc)))
    return None, None, issues


def worker(run, video_id, attempt):
    # Backend import occurs only after frozen plan and source checks.
    plan = read_plan(run)
    row = next(r for r in plan['videos'] if r['source']['video_id'] == video_id and r['action'] == 'infer')
    attempt = Path(attempt).resolve()
    if attempt.parent != Path(run).resolve()/'clips'/video_id:
        raise ValueError('Unexpected attempt destination')
    source_check(row['source'])
    started = time.perf_counter()
    try:
        from gym_coach.pose.run import extract_video
        extract_video(row['source']['path'], Exercise.BODY_WEIGHT_SQUATS, attempt/'pose',
                      config=PoseExtractionConfig.model_validate(plan['configuration']['pose']))
        frames, summary = validate_pose(row['source'], attempt/'pose', plan['configuration'])
        report = quality(frames, plan['configuration'])
        write_json(attempt/'quality.json', report.model_dump(mode='json'))
        result = dict(version=VERSION, **STATUS, status='completed', source=row['source'],
            plan_hash=canonical_hash(plan), seconds=time.perf_counter()-started,
            decode='complete', pose='complete', frames=len(frames),
            frames_with_pose=summary.frames_with_pose, frames_without_pose=summary.frames_without_pose,
            pose_observed_coverage=summary.pose_coverage,
            sequence=report.eligibility.model_dump(mode='json'),
            metrics={k: report.metric_eligibility[k].model_dump(mode='json') for k in KEYS},
            files={p.relative_to(attempt).as_posix():file_info(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')})
        write_json(attempt/'completed.json', result)
        print(json.dumps(dict(video_id=video_id, status='completed', seconds=result['seconds'])))
        return 0
    except Exception as exc:
        from gym_coach.flags import GymError
        write_json(attempt/'failure.json',dict(status='failed', source=row['source'], plan_hash=canonical_hash(plan),
            seconds=time.perf_counter()-started, reason='pose_quality_operational_failure',
            error_type=type(exc).__name__, detail=str(exc),
            flag=exc.flag.model_dump(mode='json') if isinstance(exc,GymError) else None))
        return 1


def execute(run, limit=None):
    run=Path(run).resolve();plan=read_plan(run);records=[];started=time.perf_counter();attempted=0
    for row in plan['videos']:
        if row['action'] != 'infer':continue
        previous, folder, issues=find_completed(plan,row,run)
        if previous:
            records.append(dict(video_id=row['source']['video_id'],status='verified_resume',attempt=str(folder),issues=issues))
            continue
        if limit is not None and attempted>=limit:continue
        source_check(row['source'])
        attempt=run/'clips'/row['source']['video_id']/('attempt-'+uuid.uuid4().hex)
        attempt.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--run',str(run),'--worker',row['source']['video_id'],'--attempt',str(attempt)]
        write_json(attempt/'command.json',dict(command=cmd,source=row['source'],plan_hash=canonical_hash(plan)))
        with (attempt/'stdout.txt').open('xb') as out,(attempt/'stderr.txt').open('xb') as err:
            process=subprocess.run(cmd,stdout=out,stderr=err)
        checked=False;error=None
        try:verify_completed(plan,row,attempt);checked=process.returncode==0
        except Exception as exc:error=str(exc)
        record=dict(video_id=row['source']['video_id'],status='completed' if checked else 'failed',
                    exit_code=process.returncode,attempt=str(attempt),verification_error=error,issues=issues)
        write_json(attempt/'process.json',record);records.append(record);attempted+=1
        print(json.dumps(dict(attempted=attempted,**record)),flush=True)
    receipt=dict(version=VERSION,plan_hash=canonical_hash(plan),seconds=time.perf_counter()-started,
                 attempted=attempted,states=dict(Counter(r['status'] for r in records)),records=records)
    write_json(run/('execution-'+uuid.uuid4().hex+'.json'),receipt)
    return receipt


def aggregate(run):
    run=Path(run).resolve();plan=read_plan(run);rows=[]
    for item in plan['videos']:
        source=item['source'];source_check(source)
        q=None;attempt=None;issues=[]
        if item['action']=='reuse':
            saved=item['snapshot'];folder=Path(saved['directory'])
            for name,identity in saved['files'].items():
                if file_info(folder/name)!=identity:raise ValueError('Reused snapshot changed: '+str(folder/name))
            q=QualityReport.model_validate(json.loads((folder/'preprocessing.json').read_bytes())['quality'])
            status='reused';cohort='reused_exploratory_side' if q.viewpoint.value=='side' else 'reused_strict_unknown'
        else:
            result,attempt,issues=find_completed(plan,item,run)
            cohort='fresh_strict_unknown'
            status='completed' if result else ('failed_or_incomplete' if issues else 'not_processed')
            if result:q=QualityReport.model_validate_json((attempt/'quality.json').read_bytes(),strict=True)
        seq=q.eligibility.model_dump(mode='json') if q else None
        metrics={k:q.metric_eligibility[k].model_dump(mode='json') for k in KEYS} if q else None
        canonical=bool(q) and all(tuple(q.metric_eligibility[k].required_joints)==GEOMETRY_REGISTRY.get(k).required_joints for k in KEYS)
        sq=seq['action']=='allow' if seq else None
        left=metrics[KEYS[0]]['action']=='allow' if metrics else None
        right=metrics[KEYS[1]]['action']=='allow' if metrics else None
        rows.append(dict(source=source,cohort=cohort,status=status,decode_prior='complete',
            decode_current='complete' if status=='completed' else 'not_rerun' if status=='reused' else 'UNKNOWN',
            pose_status='complete' if q else status,sequence=seq,metrics=metrics,
            frames_with_pose=q.frames_with_pose if q else None,frames_without_pose=q.frames_without_pose if q else None,
            raw_joint_quality={k.value:v.model_dump(mode='json') for k,v in q.by_joint.items()} if q else None,
            source_camera_view='UNKNOWN',quality_viewpoint=q.viewpoint.value if q else None,
            quality_evidence=str(attempt/'quality.json') if attempt else item.get('snapshot',{}).get('directory'),
            conditions_A=dict(documented_side='FAIL',sequence_allow=sq,bilateral_allow=bool(left and right) if q else None,
                              canonical_joint_keys=canonical,checked_quality_subset_pass=bool(sq and left and right),overall='BLOCKED'),
            conditions_B=dict(sequence_allow=sq,left_metric_allow=left,canonical_joint_keys=canonical,
                              checked_quality_subset_pass=bool(sq and left),source_hash='PASS',timeline='PASS' if q else 'UNKNOWN',
                              raw_provenance_contract='PASS' if q else 'UNKNOWN',processed_count_input_gap_policy='UNKNOWN',
                              fixed_left_temporal_adapter='NOT_IMPLEMENTED',method_approval='PENDING',overall='BLOCKED'),
            temporal_eligibility=False,issues=issues))
    cohorts={}
    for name in sorted({r['cohort'] for r in rows}):
        subset=[r for r in rows if r['cohort']==name]
        cohorts[name]=dict(clips=len(subset),frames=sum(r['source']['frame_count'] for r in subset),
            states=dict(Counter(r['status'] for r in subset)),
            sequence=dict(Counter(r['sequence']['action'] if r['sequence'] else 'UNKNOWN' for r in subset)),
            left=dict(Counter(r['metrics'][KEYS[0]]['action'] if r['metrics'] else 'UNKNOWN' for r in subset)),
            right=dict(Counter(r['metrics'][KEYS[1]]['action'] if r['metrics'] else 'UNKNOWN' for r in subset)),
            frames_with_pose=sum(r['frames_with_pose'] or 0 for r in subset),
            frames_without_pose=sum(r['frames_without_pose'] or 0 for r in subset),
            processed_clips=sum(r['sequence'] is not None for r in subset))
    return dict(version=VERSION,**STATUS,plan_hash=canonical_hash(plan),video_denominator=len(rows),
                cohorts=cohorts,rows=rows,conditions_A_pass=sum(r['conditions_A']['overall']=='PASS' for r in rows),
                conditions_B_quality_subset_pass=sum(r['conditions_B']['checked_quality_subset_pass'] for r in rows),
                accuracy=None,agreement=None,training_readiness='NOT_READY',training_runs=0,real_count_runs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--worker');parser.add_argument('--attempt',type=Path)
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args.run,args.worker,args.attempt)
    if args.summarize:
        result=aggregate(args.run)
        destination=args.run/('summary-'+uuid.uuid4().hex+'.json')
        write_json(destination,result);print(str(destination));return 0
    execute(args.run,args.limit);return 0


if __name__=='__main__':sys.exit(main())
