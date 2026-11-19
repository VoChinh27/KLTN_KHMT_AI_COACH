"""Saved raw pose -> separate research tensors. Never invokes a pose backend."""
from collections import Counter
import json
import math
from pathlib import Path

from ..models import VideoIdentity, canonical_hash, STATUS
from ..readiness import file_info, write_json
from .contract import CHANNELS, Config, INPUT_SIZE, JOINTS, ROLES, VERSION, flag


def read(path):
    return json.loads(Path(path).read_bytes())


def local(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path outside project: ' + str(name))
    return path


def identity(root, path):
    path = local(root, path)
    return dict(path=path.relative_to(Path(root).resolve()).as_posix(), **file_info(path))


def check_identity(root, item):
    path = local(root, item['path'])
    if file_info(path) != {k: item[k] for k in ('sha256', 'size_bytes')}:
        raise ValueError('Artifact changed: ' + item['path'])
    return path


def verify_inventory(root, inventory_path):
    path = local(root, inventory_path)
    inventory = read(path)
    for name, expected in inventory['files'].items():
        candidate = (path.parent/name).resolve()
        if not candidate.is_relative_to(path.parent) or file_info(candidate) != expected:
            raise ValueError('Historical inventory mismatch: ' + str(candidate))
    return len(inventory['files'])


def verify_viewpoints(root,path,sources):
    from ..ai_annotations import AIViewpoint, verify_evidence
    views=read(path)
    ids=[]
    for envelope in views['records']:
        if canonical_hash(envelope['payload'])!=envelope['sha256']:
            raise ValueError('AI viewpoint envelope hash mismatch')
        view=AIViewpoint.model_validate_json(json.dumps(envelope['payload']))
        verify_evidence(view,root)
        if view.source.model_dump(mode='json')!=sources[view.source.video_id]:
            raise ValueError('Viewpoint/source linkage mismatch')
        ids.append(view.source.video_id)
    if len(ids)!=33 or len(set(ids))!=33:
        raise ValueError('Viewpoint review scope is exactly 33, not full cohort')
    return ids


def split_index(manifest, split, dataset_hash):
    if (manifest['version'] != 'squat-count-dataset-v1'
            or split['version'] != 'squat-count-split-proposal-v1'
            or split['dataset_manifest_sha256'] != dataset_hash):
        raise ValueError('Dataset/split version or linkage mismatch')
    sources = [VideoIdentity.model_validate_json(json.dumps(r['source'])) for r in manifest['videos']]
    ids = [s.video_id for s in sources]
    assigned = [r['video_id'] for r in split['assignments']]
    if len(set(ids)) != len(ids) or len(set(assigned)) != len(assigned) or set(ids) != set(assigned):
        raise ValueError('Duplicate/incomplete cohort')
    source_map = {s.video_id: s.model_dump(mode='json') for s in sources}
    roles = {key: {} for key in ('group_id', 'sha256', 'component_id')}
    for row in split['assignments']:
        source = source_map[row['video_id']]
        if row['role'] not in ROLES or any(row[k] != source[k] for k in ('video_id', 'path', 'sha256', 'group_id')):
            raise ValueError('Split source/role mismatch')
        for key, mapping in roles.items():
            if mapping.setdefault(row[key], row['role']) != row['role']:
                raise ValueError('Group/hash/component split leakage')
    return source_map, {r['video_id']: r for r in split['assignments']}


def encode(frames, source, config=Config(), evidence_flags=()):
    """No targets accepted. Invalid coordinates have a masked tensor placeholder.

    Complete-frame policy is stricter than marginal joint coverage. Hidden state
    will reset at every incomplete sample. No samples are dropped or interpolated.
    """
    n = len(frames)
    if source['fps'] is None or n != source['frame_count']:
        raise ValueError('Incomplete source/timeline')
    encoded, complete, counts = [], [], Counter()
    all_flags = list(evidence_flags)
    for i, frame in enumerate(frames):
        if (frame.video_id != source['video_id'] or frame.frame_index != i
                or frame.timestamp_seconds != i/source['fps']
                or frame.exercise.value != 'BodyWeightSquats' or frame.viewpoint.value != 'unknown'
                or frame.config.visibility_threshold != config.visibility_threshold):
            raise ValueError('Raw frame/timeline/provenance identity mismatch')
        joints = {j.joint.value: j for j in frame.joints}
        vector, usability = [], []
        for name in JOINTS:
            j = joints[name]
            status = j.status.value
            usable = (status == 'observed' and j.x is not None and j.y is not None
                      and 0 <= j.x <= 1 and 0 <= j.y <= 1)
            x = (j.x-0.5)*source['width']/source['height'] if usable else 0.0
            y = j.y-0.5 if usable else 0.0
            vector.extend([x, y, j.visibility if j.visibility is not None else 0.0,
                           float(usable), float(status == 'observed'),
                           float(status == 'low_visibility'), float(status == 'missing')])
            usability.append(usable)
            counts[status] += 1
            counts['usable'] += int(usable)
        vector.extend([frame.timestamp_seconds, 0.0 if i == 0 else 1/source['fps']])
        if len(vector) != INPUT_SIZE or not all(math.isfinite(x) for x in vector):
            raise ValueError('Non-finite/incorrect research vector')
        encoded.append(vector)
        complete.append(all(usability))
        all_flags.extend(f.model_dump(mode='json') for f in frame.flags)
    longest = current = 0
    for valid in complete:
        current = 0 if valid else current + 1
        longest = max(longest, current)
    reasons = []
    if not config.min_frames <= n <= config.max_frames:
        reasons.append(flag('research_sequence_length', f'{n} frames; permitted {config.min_frames}..{config.max_frames}'))
    fraction = sum(complete)/n if n else 0.0
    if fraction < config.min_complete_frame_fraction:
        reasons.append(flag('research_incomplete_support_ratio', f'Complete eight-joint support {sum(complete)}/{n}'))
    if longest > config.max_incomplete_span_frames:
        reasons.append(flag('research_long_gap', f'Longest incomplete support {longest} frames'))
    for code in sorted({f['code'] for f in all_flags} &
                       {'multi_person', 'side_switch_discontinuity', 'unstable_signal'}):
        reasons.append(flag('research_subject_or_signal_ambiguity', 'Original evidence flag: ' + code))
    return dict(version=VERSION, config_hash=config.content_hash(), source=source,
                status='rejected' if reasons else 'accepted', reasons=reasons,
                values=encoded, complete=complete, duration_seconds=n/source['fps'],
                stats=dict(frames=n, complete_frames=sum(complete), complete_fraction=fraction,
                           longest_incomplete_span=longest, joint_samples=n*len(JOINTS),
                           provenance=dict(counts)),
                subject_identity='UNVERIFIED_BACKEND_SINGLE_PROMINENT_SUBJECT',
                preprocessing='image_center_height_units; no imputation/smoothing/statistical fit')


def summarize(rows):
    output = {}
    for role in (*ROLES, 'all'):
        subset = [r for r in rows if role == 'all' or r['role'] == role]
        eligible = [r for r in subset if r['supervised_eligible']]
        output[role] = dict(clips=len(subset), groups=len({r['source']['group_id'] for r in subset}),
            frames=sum(r['source']['frame_count'] for r in subset),
            labels_available=sum(r['target'] is not None for r in subset),
            labels_unresolved=sum(r['target'] is None for r in subset),
            input_status=dict(Counter(r['input_status'] for r in subset)),
            supervised_clips=len(eligible), supervised_groups=len({r['source']['group_id'] for r in eligible}),
            exclusion_reasons=dict(Counter(f['code'] for r in subset for f in r['exclusion_reasons'])))
    return output


def saved_input(root, source, planned, summary, configuration):
    """Support the two already verified storage layouts; never consult angle values."""
    from ...pose.models import PoseFrame
    from ...quality.models import QualityReport
    links = []
    if planned['action'] == 'reuse':
        snapshot = planned['snapshot']
        folder = local(root, snapshot['directory'])
        raw_name = 'raw-pose-unknown.jsonl' if 'raw-pose-unknown.jsonl' in snapshot['files'] else 'raw-pose.jsonl'
        for name in (raw_name, 'preprocessing.json', 'source.json'):
            expected = snapshot['files'][name]
            item = identity(root, folder/name)
            if {k:item[k] for k in expected} != expected:
                raise ValueError('Saved snapshot hash mismatch: ' + name)
            links.append(item)
        raw_path = folder/raw_name
        pre = read(folder/'preprocessing.json')
        q = pre['quality']
        metadata = pre['metadata']
        original = read(folder/'source.json')
        if 'candidate' in original:
            video_hash = original['candidate']['sha256']
            original_path = original['candidate']['path']
        else:
            video_hash = original['identity']['video']['sha256']
            original_path = original['identity']['metadata']['source_path']
        if video_hash != source['sha256'] or local(root, original_path) != local(root, source['path']):
            raise ValueError('Snapshot source linkage mismatch')
    elif planned['action'] == 'infer' and summary['status'] == 'completed':
        # This describes a PAST run. No inference API is imported or called.
        quality_path = local(root, summary['quality_evidence'])
        folder = quality_path.parent
        completed = read(folder/'completed.json')
        if completed['source'] != source or completed['status'] != 'completed':
            raise ValueError('Saved completion source mismatch')
        for name, expected in completed['files'].items():
            item = identity(root, folder/name)
            if {k:item[k] for k in expected} != expected:
                raise ValueError('Saved completion hash mismatch: ' + name)
            links.append(item)
        links.append(identity(root, folder/'completed.json'))
        manifest = read(folder/'pose/manifest.json')
        raw_path = folder/'pose/frames.jsonl'
        if (manifest['completed'] is not True or manifest['input_sha256'] != source['sha256']
                or local(root, manifest['input_path']) != local(root, source['path'])
                or manifest['pose_frames_sha256'] != file_info(raw_path)['sha256']):
            raise ValueError('Pose manifest linkage mismatch')
        metadata, q = manifest['metadata'], read(quality_path)
    else:
        raise ValueError('No compatible saved pose; no automatic inference fallback')
    if any(metadata[k] != source[k] for k in ('video_id','fps','frame_count','width','height','timestamp_policy')):
        raise ValueError('Decoder/source timeline mismatch')
    frames = [PoseFrame.model_validate_json(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
    report = QualityReport.model_validate_json(json.dumps(q))
    if (report.video_id != source['video_id'] or report.total_frames != len(frames)
            or report.profile_hash != configuration['quality_hash']
            or report.eligibility.model_dump(mode='json') != summary['sequence']
            or report.viewpoint.value != summary['quality_viewpoint']
            or {k:report.metric_eligibility[k].model_dump(mode='json') for k in summary['metrics']} != summary['metrics']):
        raise ValueError('Quality/summary/profile mismatch')
    for f in frames:
        if (f.config.model_dump(mode='json') != configuration['pose']
                or f.backend.mediapipe_version != configuration['mediapipe_version']
                or f.backend.model_asset_sha256 != configuration['model_asset']['sha256']):
            raise ValueError('Pose backend/model/config mismatch')
    for name, quality in report.by_joint.items():
        js = [next(j for j in f.joints if j.joint == name) for f in frames]
        c = Counter(j.status.value for j in js)
        if (quality.observed_frames != c['observed'] or quality.low_visibility_frames != c['low_visibility']
                or quality.missing_frames != c['missing']):
            raise ValueError('Quality/raw provenance count mismatch')
    return frames, report.model_dump(mode='json'), links


def prepare(root, plan_path, out):
    from ..ai_annotations import load_pass, verify_evidence, conservative_pair
    root, out = Path(root).resolve(), Path(out).resolve()
    plan = read(plan_path)
    config = Config.model_validate_json(json.dumps(plan['config']))
    if plan['config_hash'] != config.content_hash():
        raise ValueError('Frozen configuration mismatch')
    for name, expected in plan['inputs'].items():
        if file_info(local(root, name)) != expected:
            raise ValueError('Frozen input changed: ' + name)
    paths = list(plan['inputs'])
    find = lambda ending: local(root, next(p for p in paths if p.endswith(ending)))
    dataset_path, split_path = find('/dataset-manifest.json'), find('/split-proposal.json')
    sources, assignments = split_index(read(dataset_path), read(split_path), file_info(dataset_path)['sha256'])
    if len(sources) != 112:
        raise ValueError('Expected fixed 112 cohort')
    verified = {p:verify_inventory(root,p) for p in paths if p.endswith('/artifact-inventory.json')}
    pose_plan_path = next(local(root,p) for p in paths if p.endswith('/plan.json'))
    pose_plan = read(pose_plan_path)
    summary_path = next(local(root,p) for p in paths if '/summary-' in p)
    summaries = {r['source']['video_id']:r for r in read(summary_path)['rows']}
    planned = {r['source']['video_id']:r for r in pose_plan['videos']}
    if set(sources) != set(planned) or set(sources) != set(summaries):
        raise ValueError('Pose cohort mismatch')
    out.mkdir(parents=True, exist_ok=False)
    (out/'inputs').mkdir()
    rows = []
    # ALL input decisions are made before opening count targets.
    for video_id, source in sorted(sources.items()):
        row = dict(source=source, role=assignments[video_id]['role'],
                   component_id=assignments[video_id]['component_id'], source_camera_view='UNKNOWN',
                   input_status='unchecked', input_artifact=None, input_reasons=[], source_links=[])
        try:
            if source != planned[video_id]['source'] or source != summaries[video_id]['source']:
                raise ValueError('Pose source differs from frozen dataset')
            source_link = identity(root, source['path'])
            if source_link['sha256'] != source['sha256']:
                raise ValueError('Source byte mismatch')
            frames, q, links = saved_input(root, source, planned[video_id], summaries[video_id], pose_plan['configuration'])
            payload = encode(frames, source, config, q['flags'])
            payload['original_quality_report'] = q  # verbatim semantic copy; never promoted
            payload['backend'] = frames[0].backend.model_dump(mode='json')
            payload['pose_config'] = frames[0].config.model_dump(mode='json')
            path = out/'inputs'/(video_id+'.json')
            write_json(path, payload)
            row.update(input_status=payload['status'], input_artifact=identity(root,path),
                       input_reasons=payload['reasons'], input_stats=payload['stats'],
                       source_links=[source_link,*links])
        except (ValueError, OSError, KeyError) as exc:
            row.update(input_status='error', input_reasons=[flag('research_integrity_error',str(exc),'input_error')])
        rows.append(row)
    write_json(out/'input-admission-before-targets.json', rows)
    labels_path = find('/results-v1/ai-reference.json')
    labels = read(labels_path)
    reference = {r['video_id']:r for r in labels['records']}
    if len(reference) != 112 or len(labels['records']) != 112 or set(reference) != set(sources):
        raise ValueError('AI reference cohort mismatch')
    for row in rows:
        ref = reference[row['source']['video_id']]
        if ref['source'] != row['source']:
            raise ValueError('AI reference source mismatch')
        passes = [load_pass(check_identity(root,ref['pass_links'][p])) for p in ('A','B')]
        for p in passes:
            verify_evidence(p,root)
            if p.source.model_dump(mode='json') != row['source'] or p.protocol_sha256 != file_info(find('docs/SQUAT_AI_ANNOTATION_PROTOCOL.md'))['sha256']:
                raise ValueError('AI protocol/source mismatch')
        if any(ref[k] != v for k,v in conservative_pair(*passes).items()):
            raise ValueError('AI reference differs from conserved original passes')
        target = ref['whole_clip_count']
        row.update(target=target, target_kind='AI_weak_video_complete_count',
                   label_links=list(ref['pass_links'].values()), label_record_hash=canonical_hash(ref),
                   supervised_eligible=row['input_status']=='accepted' and target is not None,
                   exclusion_reasons=row['input_reasons'] + ([] if target is not None else
                       [flag('ai_count_unresolved',ref['count_reason'])]))
    # No extrapolation to the other 79 clips; viewpoint is NOT admission input.
    view_ids = verify_viewpoints(root,find('/results-v1/ai-viewpoint.json'),sources)
    result = dict(version=VERSION, **STATUS, target_kind='AI_weak_video_complete_count',
        config=config.model_dump(mode='json'), config_hash=config.content_hash(),
        plan_link=identity(root,plan_path), frozen_inputs=plan['inputs'], rows=rows,
        summary=summarize(rows), historical_inventories_verified=verified,
        viewpoint_review_scope='33 only; unverified; not used for input selection',
        viewpoint_review_ids=view_ids,
        exact_event_labels=0, training_runs=0, baseline_fits=0, inference_runs=0)
    write_json(out/'dataset.json', dict(payload=result, sha256=canonical_hash(result)))
    return result


def load_dataset(root, path):
    envelope = read(path)
    data = envelope['payload']
    if canonical_hash(data) != envelope['sha256'] or data['version'] != VERSION:
        raise ValueError('Research dataset hash/version mismatch')
    config = Config.model_validate_json(json.dumps(data['config']))
    if data['config_hash'] != config.content_hash() or data['summary'] != summarize(data['rows']):
        raise ValueError('Config/summary mismatch')
    check_identity(root,data['plan_link'])
    for name, expected in data['frozen_inputs'].items():
        if file_info(local(root,name)) != expected:
            raise ValueError('Frozen dataset input changed: '+name)
    paths=list(data['frozen_inputs'])
    source_path=local(root,next(p for p in paths if p.endswith('/dataset-manifest.json')))
    split_path=local(root,next(p for p in paths if p.endswith('/split-proposal.json')))
    sources,assignments=split_index(read(source_path),read(split_path),file_info(source_path)['sha256'])
    verify_viewpoints(root,local(root,next(p for p in paths if p.endswith('/results-v1/ai-viewpoint.json'))),sources)
    rows=data['rows']
    if (len(rows)!=112 or len({r['source']['video_id'] for r in rows})!=112
            or set(sources)!={r['source']['video_id'] for r in rows}):
        raise ValueError('Full fixed cohort required')
    labels_path=local(root,next(p for p in paths if p.endswith('/results-v1/ai-reference.json')))
    labels={r['video_id']:r for r in read(labels_path)['records']}
    for row in data['rows']:
        video_id=row['source']['video_id']
        ref=labels[video_id]
        if (row['source']!=sources[video_id] or row['role']!=assignments[video_id]['role']
                or row['component_id']!=assignments[video_id]['component_id']
                or row['target']!=ref['whole_clip_count'] or row['label_record_hash']!=canonical_hash(ref)
                or row['label_links']!=list(ref['pass_links'].values())
                or row['supervised_eligible']!=(row['input_status']=='accepted' and row['target'] is not None)):
            raise ValueError('Research row source/split/AI target linkage mismatch')
        if file_info(local(root,row['source']['path']))['sha256']!=row['source']['sha256']:
            raise ValueError('Source byte mismatch')
        for link in row['source_links'] + row['label_links']:
            check_identity(root,link)
        if row['input_artifact']:
            value = read(check_identity(root,row['input_artifact']))
            if (value['source'] != row['source'] or value['config_hash'] != config.content_hash()
                    or value['status'] != row['input_status'] or value['reasons']!=row['input_reasons']
                    or len(value['values'])!=row['source']['frame_count']
                    or len(value['complete'])!=row['source']['frame_count']):
                raise ValueError('Cached input linkage mismatch')
    return data


class CountDataset:
    """Torch-free cached input loader; explicit eligible subset of full manifest."""
    def __init__(self, root, data, role, supervised=True):
        if role not in ROLES:
            raise ValueError('Unknown split')
        self.root = Path(root)
        self.rows = [r for r in data['rows'] if r['role']==role and r['input_status']=='accepted'
                     and (not supervised or r['supervised_eligible'])]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        value = read(check_identity(self.root,row['input_artifact']))
        return dict(video_id=row['source']['video_id'], values=value['values'],
                    complete=value['complete'], duration_seconds=value['duration_seconds'],
                    target=row['target'])
