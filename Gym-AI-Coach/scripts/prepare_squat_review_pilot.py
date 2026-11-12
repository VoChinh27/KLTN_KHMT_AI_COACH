"""Select pending review work from frozen evidence; never annotate or run a model."""
import argparse
import json
from pathlib import Path

from gym_coach.contracts import Viewpoint
from gym_coach.counting.evaluation import checked
from gym_coach.counting.models import VideoIdentity, STATUS
from gym_coach.counting.readiness import file_info, write_json

LEFT = 'squat_knee_angle_left_deg'
LEFT_JOINTS = ['left_hip', 'left_knee', 'left_ankle']


def source_index(manifest):
    if manifest['version'] != 'squat-count-dataset-v1':
        raise ValueError('Unknown dataset version')
    sources = [checked(VideoIdentity, r['source']).model_dump(mode='json') for r in manifest['videos']]
    index = {s['video_id']: s for s in sources}
    if len(index) != len(sources) or any(s['fps'] is None for s in sources):
        raise ValueError('Duplicate source or missing timeline')
    return index


def whole_clip(source):
    last = source['frame_count'] - 1
    return dict(scope='whole_clip', inclusive_start_frame=0, inclusive_end_frame=last,
                first_sample_seconds=0.0, last_sample_seconds=last/source['fps'],
                container_duration_seconds=source['frame_count']/source['fps'],
                timestamp_policy=source['timestamp_policy'])


def select_viewpoint(summary, manifest):
    sources = source_index(manifest)
    if summary['version'] != 'squat-pose-feasibility-audit-v1':
        raise ValueError('Unknown feasibility version')
    rows = summary['rows']
    if (summary['video_denominator'] != len(sources) or len(rows) != len(sources)
            or {r['source']['video_id'] for r in rows} != set(sources)):
        raise ValueError('Full cohort required; no silent missing/duplicate source')
    selected = []
    for row in rows:
        source = sources[row['source']['video_id']]
        if row['source'] != source:
            raise ValueError('Summary source/hash/timeline mismatch')
        metric = (row.get('metrics') or {}).get(LEFT)
        sequence = row.get('sequence') or {}
        if (row['cohort'] not in ('fresh_strict_unknown', 'reused_strict_unknown')
                or row['status'] not in ('completed', 'reused')
                or row['source_camera_view'] != 'UNKNOWN' or row['quality_viewpoint'] != 'unknown'
                or sequence.get('eligible') is not True or sequence.get('action') != 'allow'
                or metric is None or metric['eligible'] is not False or metric['action'] != 'skip'
                or metric['reasons'] != ['viewpoint_unknown']):
            continue
        if metric['required_joints'] != LEFT_JOINTS:
            raise ValueError('Canonical left-knee joints mismatch')
        selected.append(dict(source=source, exercise='BodyWeightSquats', review_scope=whole_clip(source),
            selection_reason='Strict UNKNOWN; sequence allow; canonical left metric skip only viewpoint_unknown',
            status='VIEWPOINT_REVIEW_PENDING', source_camera_view='UNKNOWN', documented=False,
            evidence_status='pending', reviewer=None, reviewed_at=None, protocol_reference=None,
            declared_whole_clip_viewpoint=None, camera_changes=None, reviewed_intervals=[],
            camera_direction_notes=None, framing_notes=None, occlusion_notes=None,
            in_frame_notes=None, uncertainty_notes=None, ai_visual_assessment=None))
    return sorted(selected, key=lambda row: row['source']['path'])


def select_pilot(manifest, split, manifest_sha256, size=5):
    """This function intentionally accepts no quality, viewpoint or prediction data."""
    sources = source_index(manifest)
    if (split['version'] != 'squat-count-split-proposal-v1'
            or split['dataset_manifest_sha256'] != manifest_sha256):
        raise ValueError('Frozen split linkage/version mismatch')
    assignments = split['assignments']
    if (len(assignments) != len(sources)
            or {r['video_id'] for r in assignments} != set(sources)):
        raise ValueError('Every source needs exactly one frozen split assignment')
    group_roles = {}; hash_roles = {}; train = []
    for row in assignments:
        source = sources[row['video_id']]
        if any(row[k] != source[k] for k in ('video_id', 'path', 'sha256', 'group_id')):
            raise ValueError('Split source linkage mismatch')
        if row['role'] not in ('train', 'validation', 'development_test'):
            raise ValueError('Unknown split role')
        for field, roles in (('group_id', group_roles), ('sha256', hash_roles)):
            if roles.setdefault(row[field], row['role']) != row['role']:
                raise ValueError('Group/duplicate leakage in frozen split')
        if row['role'] == 'train':
            train.append(source)
    first_per_group = {}
    for source in sorted(train, key=lambda s: (s['group_id'], s['path'])):
        first_per_group.setdefault(source['group_id'], source)
    if len(first_per_group) < size:
        raise ValueError('Insufficient distinct train groups; no fallback/reselection')
    return list(first_per_group.values())[:size]


def verify_inventory(folder, filename):
    folder = Path(folder).resolve()
    inventory = json.loads((folder/filename).read_bytes())
    for name, identity in inventory['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or file_info(path) != identity:
            raise ValueError('Inventory linkage/hash mismatch: '+name)
    return len(inventory['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    plan = json.loads(args.plan.read_bytes())
    if plan['version'] != 'squat-review-pilot-selection-v1':
        raise ValueError('Unknown selection plan')
    for name, identity in plan['inputs'].items():
        if file_info(root/name) != identity:
            raise ValueError('Frozen input changed: '+name)
    for path in (args.dataset/'dataset-manifest.json', args.dataset/'split-proposal.json', args.summary):
        if path.resolve().relative_to(root).as_posix() not in plan['inputs']:
            raise ValueError('Input not bound by plan')
    verified = dict(
        feasibility_files=verify_inventory(args.summary.parent, 'artifact-inventory.json'),
        dataset_files=verify_inventory(args.dataset, 'artifact-inventory.json'))
    manifest = json.loads((args.dataset/'dataset-manifest.json').read_bytes())
    split = json.loads((args.dataset/'split-proposal.json').read_bytes())
    summary = json.loads(args.summary.read_bytes())
    manifest_hash = file_info(args.dataset/'dataset-manifest.json')['sha256']
    candidates = select_viewpoint(summary, manifest)
    pilot = select_pilot(manifest, split, manifest_hash)
    sources = source_index(manifest)
    if len(sources) != 112 or len(candidates) != 33:
        raise ValueError('Measured cohort differs from expected 112/33; review without reselection')
    for source in sources.values():
        if file_info(root/source['path'])['sha256'] != source['sha256']:
            raise ValueError('Source bytes changed')
    out = args.output_dir.resolve()
    if not out.is_relative_to(root/'outputs'):
        raise ValueError('New output must be under workspace outputs')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'viewpoint-review.pending.json', dict(version='squat-viewpoint-worklist-v1',
        artifact_role='pending worksheet, not a runtime annotation schema or gate authorization',
        **STATUS, status='VIEWPOINT_REVIEW_PENDING', documented_evidence_rule_approval='PENDING',
        supported_viewpoints=[v.value for v in Viewpoint], plan_sha256=file_info(args.plan)['sha256'],
        evidence=file_info(args.summary), cohort_denominator=len(sources), records=candidates))
    write_json(out/'pilot-coordinator.json', dict(version='squat-temporal-pilot-selection-v1', **STATUS,
        status='TEMPORAL_LABELS_PENDING', purpose='Protocol usability pilot, not accuracy evaluation',
        plan_sha256=file_info(args.plan)['sha256'], cohort_denominator=len(sources),
        split=file_info(args.dataset/'split-proposal.json'), role='train', sources=pilot,
        template_records_per_pass=len(sources), template_modification=False, copied_videos=0))
    write_json(out/'receipt.json', dict(inputs={str(p):file_info(p) for p in (args.plan,args.summary)},
        verified=verified, source_hashes=112, viewpoint_pending=33, pilot_clips=5,
        human_review_claims=0, inference_runs=0, training_runs=0,
        outputs={p.name:file_info(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(viewpoint_pending=len(candidates), pilot_ids=[s['video_id'] for s in pilot])))


if __name__ == '__main__':
    main()
