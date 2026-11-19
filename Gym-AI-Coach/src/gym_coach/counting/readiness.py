"""All-squat data/label preparation. Decode only; never pose inference or training."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import re
import shutil
import time

from .models import (AnnotationBundle, AnnotationPass, EvaluationPolicy, Prediction,
                     STATUS, VideoIdentity, canonical_hash, reason)
from .split import propose_split, SEED

SCOPE = Path('Data/UCF-101_GYM/BodyWeightSquats')
VISUAL = Path('outputs/stage5_7/subset-visual-20261015-077a183104c94d28a301b4d538b69de8/visual_audit.json')
REPLAY_PLAN = Path('outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20/plan.json')
AUDIT = Path('Data/UCF-101_GYM/audit/UCF101_GYM_VIDEO_AUDIT.csv')
PROTOCOL_FILE = Path('docs/SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md')


def file_info(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def safe_video_paths(root):
    scope = root/SCOPE
    if scope.is_symlink() or scope.resolve() != scope.absolute():
        raise ValueError('Redirected dataset scope')
    entries = sorted(scope.iterdir())
    videos, other = [], []
    for path in entries:
        if path.is_symlink() or path.resolve().parent != scope.resolve():
            raise ValueError('Redirected video or nested source')
        if path.is_file() and re.fullmatch(r'v_BodyWeightSquats_g\d{2}_c\d{2}\.avi', path.name):
            videos.append(path)
        else:
            other.append(path.relative_to(root).as_posix())
    return videos, other


def decode_video(path):
    from ..video import OpenCVVideoReader
    from ..flags import GymError
    count = 0
    metadata = None
    try:
        reader = OpenCVVideoReader(path)
        metadata = reader.metadata
        for frame in reader.frames():
            if frame.frame_index != count or frame.timestamp_seconds != count/metadata.fps:
                raise ValueError('Decoder timeline mismatch')
            if tuple(frame.image.shape[:2]) != (metadata.height, metadata.width):
                raise ValueError('Resolution changed during decode')
            count += 1
        return dict(status='complete', frames_decoded=count, metadata=metadata.model_dump(mode='json'), reason=None)
    except (GymError, ValueError, OSError) as exc:
        return dict(status='failed', frames_decoded=count, metadata=metadata.model_dump(mode='json') if metadata else None,
                    reason=exc.flag.model_dump(mode='json') if isinstance(exc, GymError) else
                    reason('decode_error', str(exc), 'input_error').model_dump(mode='json'))


def saved_compatibility(root, sources):
    """Reuse existing strict Stage4/source validator. No processing or measurements."""
    from ..reporting import context_performance as perf
    from ..pose.models import PoseExtractionConfig
    from ..quality.models import QualityProfile
    results, errors = {}, []
    try:
        plan = json.loads((root/REPLAY_PLAN).read_bytes())
        jobs, _, profile, _ = perf.verified_jobs(root, plan)
    except (ValueError, OSError, KeyError) as exc:
        return results, [dict(stage='batch_inventory', error=str(exc))]
    model_path = distribution('mediapipe').locate_file('mediapipe/modules/pose_landmark/pose_landmark_full.tflite')
    model_hash = file_info(model_path)['sha256']
    by_path = {s.path: s for s in sources}
    for job in jobs:
        path = job['path']
        try:
            source = by_path[path]
            if source.sha256 != job['source_video_sha256']:
                raise ValueError('Saved video bytes mismatch')
            processed = perf.load_source(job, profile)
            m = processed.metadata
            if (source.fps, source.frame_count, source.width, source.height) != (m.fps, m.frame_count, m.width, m.height):
                raise ValueError('Saved metadata differs from fresh decoder')
            if processed.quality.profile != QualityProfile():
                raise ValueError('Quality profile differs from current')
            if any(f.config != PoseExtractionConfig() or f.backend.mediapipe_version != version('mediapipe') or
                   f.backend.model_asset_sha256 != model_hash for f in processed.raw_frames):
                raise ValueError('Model bytes/version/config differ from current')
            results[path] = dict(status='verified_compatible_saved_snapshots',
                directory=Path(job['directory']).relative_to(root).as_posix(), files=job['files'],
                frame_count=len(processed.frames), backend=processed.raw_frames[0].backend.model_dump(mode='json'),
                pose_config_sha256=processed.raw_frames[0].config.content_hash(),
                quality_profile_sha256=processed.quality.profile_hash,
                preprocessing_profile_sha256=canonical_hash(processed.profile),
                sequence=processed.quality.eligibility.model_dump(mode='json'),
                metrics={k: processed.quality.metric_eligibility[k].model_dump(mode='json') for k in perf.single.KEYS},
                viewpoint='visually_assessed_unverified', source_camera_view='UNKNOWN',
                real_temporal_admission=False,
                compatibility_scope='Saved typed/hash/source/timeline/current model-config/profile; no fresh inference or accuracy assertion')
        except (ValueError, OSError, KeyError) as exc:
            results[path] = dict(status='incompatible', reason=str(exc), real_temporal_admission=False)
            errors.append(dict(path=path, stage='saved_snapshot', error=str(exc)))
    return results, errors


def catalog_evidence(root, output, hashes):
    """Metadata index only. Matching bytes in a manifest is a lead, not compatibility."""
    catalog = defaultdict(list); annotations = []; searched = 0
    excluded_run = output.parent if output.parent.name.startswith('run-') else output
    candidate_names = {'source.json', 'manifest.json'}
    for directory, folders, files in os.walk(root/'outputs', followlinks=False):
        folder = Path(directory)
        folders[:] = [n for n in folders if not (n.startswith(('tests', 'pytest', '.')) or
                       'snapshot' in n or 'restore' in n or (folder/n).is_symlink() or
                       (folder/n).resolve().is_relative_to(excluded_run))]
        for name in files:
            path = folder/name
            is_annotation = 'annotat' in name.lower() and name.endswith('.json')
            is_candidate = name in candidate_names and any(n in files for n in (
                'raw-pose-unknown.jsonl', 'frames.jsonl', 'pose.jsonl', 'preprocessing.json'))
            if not (is_annotation or is_candidate) or path.is_symlink() or path.stat().st_size > 8*1024*1024:
                continue
            data = path.read_bytes(); searched += 1
            try:
                value = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            ref = dict(path=path.relative_to(root).as_posix(), **file_info(path))
            if is_annotation:
                annotations.append(dict(**ref, kind='annotation_named_document_not_automatically_temporal_ground_truth',
                    declared_version=value.get('version', value.get('protocol_version')) if isinstance(value, dict) else None))
            if is_candidate:
                present = set(re.findall(r'[a-f0-9]{64}', data.decode('utf-8')))&hashes
                for sha in present:
                    catalog[sha].append(dict(**ref, compatibility='catalogued_not_validated',
                        available_files=[n for n in files if n in ('raw-pose-unknown.jsonl', 'frames.jsonl', 'pose.jsonl',
                                                                  'preprocessing.json', 'quality.json', 'quality-assumed-side.json')]))
    return dict(catalog), dict(documents=annotations, metadata_documents_read=searched,
        search_scope='Non-test outputs metadata and annotation-named JSON only; no archive/Data videos outside exact squat class',
        qualification='No automatic acceptance by filename. Current known independent pixel templates are not temporal labels; real temporal bundle import requires strict schema/source/protocol/adjudication.')


def verify_additional_snapshot(root, source, entry):
    """Check a discovered single-video snapshot without rendering or inference."""
    from ..job_controller import local_path
    from ..reporting import single_video as pipeline
    from ..preprocessing import PreprocessingResult, load_preprocessing_profile
    from ..quality.models import QualityProfile
    from ..pose.models import PoseExtractionConfig
    folder = local_path(root/Path(entry['path']).parent)
    if not folder.is_relative_to(root/'outputs'): raise ValueError('Snapshot outside outputs')
    manifest = json.loads((folder/'manifest.json').read_bytes())
    if manifest.get('version') != pipeline.VERSION or manifest.get('status') != 'completed':
        raise ValueError('Not a completed single-video snapshot')
    if any(manifest[k] != v for k,v in STATUS.items()): raise ValueError('Disabled gate mismatch')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or ':' in name or '/' in name or '\\' in name:
            raise ValueError('Unsafe snapshot path')
        path=local_path(folder/name)
        if path.parent != folder or file_info(path) != expected: raise ValueError('Snapshot artifact hash mismatch')
    result = pipeline.load_saved(folder)
    processed = PreprocessingResult.model_validate_json((folder/'preprocessing.json').read_bytes(),strict=True)
    identity = json.loads((folder/'source.json').read_bytes())['identity']
    if (identity['video']['sha256'] != source.sha256 or manifest['source_video_sha256'] != source.sha256 or
            Path(identity['metadata']['source_path']).resolve() != (root/source.path).resolve() or
            result['source']['video_sha256'] != source.sha256 or
            processed.quality.profile != QualityProfile() or
            processed.profile != load_preprocessing_profile(root/'configs/preprocessing_s3_v1.toml')):
        raise ValueError('Source/profile linkage mismatch')
    m=processed.metadata
    if (m.fps,m.frame_count,m.width,m.height)!=(source.fps,source.frame_count,source.width,source.height):
        raise ValueError('Saved decoder timeline mismatch')
    backend_hash=file_info(distribution('mediapipe').locate_file('mediapipe/modules/pose_landmark/pose_landmark_full.tflite'))['sha256']
    if any(f.config != PoseExtractionConfig() or f.backend.mediapipe_version != version('mediapipe') or
           f.backend.model_asset_sha256 != backend_hash for f in processed.raw_frames):
        raise ValueError('Saved model/config incompatible')
    keys=('squat_knee_angle_left_deg','squat_knee_angle_right_deg')
    from ..features import GEOMETRY_REGISTRY
    for key in keys:
        if processed.quality.metric_eligibility[key].required_joints != GEOMETRY_REGISTRY.get(key).required_joints:
            raise ValueError('Metric joints mismatch')
    return dict(status='verified_compatible_saved_snapshots', directory=folder.relative_to(root).as_posix(),
        files=manifest['files'], frame_count=m.frame_count, sequence=processed.quality.eligibility.model_dump(mode='json'),
        metrics={k:processed.quality.metric_eligibility[k].model_dump(mode='json') for k in keys},
        backend=processed.raw_frames[0].backend.model_dump(mode='json'),
        pose_config_sha256=processed.raw_frames[0].config.content_hash(),
        quality_profile_sha256=processed.quality.profile_hash, preprocessing_profile_sha256=canonical_hash(processed.profile),
        viewpoint=result['mode'], source_camera_view='UNKNOWN', real_temporal_admission=False,
        compatibility_scope='Typed saved raw/processed/result and byte/source/timeline/model/profile compatibility, not cache authorization or accuracy')


def load_decode_reuse(root, folder):
    folder=(root/folder).resolve()
    if not folder.is_relative_to(root/'outputs'): raise ValueError('Decode evidence must be under outputs')
    inv=json.loads((folder/'artifact-inventory.json').read_bytes())
    if file_info(folder/'dataset-manifest.json') != inv['files']['dataset-manifest.json']:
        raise ValueError('Prior decoder manifest hash mismatch')
    manifest=json.loads((folder/'dataset-manifest.json').read_bytes())
    if manifest['version']!='squat-count-dataset-v1': raise ValueError('Unknown decoder evidence version')
    rows={}
    for row in manifest['videos']:
        source=VideoIdentity.model_validate_json(json.dumps(row['source']),strict=True)
        if source.path in rows: raise ValueError('Duplicate decoder source')
        decoded=row['decode']
        if decoded['status'] not in ('complete','failed') or (decoded['status']=='complete' and decoded['frames_decoded']!=source.frame_count):
            raise ValueError('Invalid prior full-decode accounting')
        rows[source.path]=row
    return rows


def prepare(output, *, workspace=None, reuse_decode_from=None):
    root = Path(workspace or Path.cwd()).resolve(); out = (root/output).resolve()
    if not out.is_relative_to(root/'outputs') or out == root/'outputs':
        raise ValueError('New run child under outputs required')
    if out.exists(): raise FileExistsError(out)
    videos, unclassified = safe_video_paths(root)
    if not videos: raise ValueError('No in-scope squat videos')
    out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    policy = EvaluationPolicy(); protocol_hash = file_info(root/PROTOCOL_FILE)['sha256']
    write_json(out/'plan.json', dict(version='squat-count-readiness-plan-v1', seed=SEED,
        scope=SCOPE.as_posix(), paths=[p.relative_to(root).as_posix() for p in videos],
        evaluation_policy=policy.model_dump(mode='json'), protocol_sha256=protocol_hash,
        inference_authorized=False, training_authorized=False, **STATUS))
    visual_doc = json.loads((root/VISUAL).read_bytes())
    visual = {row['relative_path']: row for row in visual_doc['videos']}
    with (root/AUDIT).open(encoding='utf-8-sig', newline='') as stream:
        audit = {row['file_name']: row for row in csv.DictReader(stream) if row['class_name'] == 'BodyWeightSquats'}
    reused = load_decode_reuse(root,reuse_decode_from) if reuse_decode_from else {}
    records = []; problems = []; exposed = []; actual_decode_frames=0; reused_count=0
    for index, path in enumerate(videos):
        info = file_info(path); relative = path.relative_to(root).as_posix()
        if relative in reused:
            prior_row=reused[relative]
            if prior_row['source']['sha256'] != info['sha256'] or prior_row['size_bytes'] != info['size_bytes']:
                raise ValueError('Video bytes no longer match full-decode evidence')
            decoded=prior_row['decode'];reused_count+=1
        else:
            decoded=decode_video(path);actual_decode_frames+=decoded['frames_decoded']
        metadata = decoded['metadata']
        source = VideoIdentity(video_id=path.name, path=relative, sha256=info['sha256'],
            group_id=re.search(r'_(g\d{2})_', path.name)[1],
            **({k: metadata[k] for k in ('fps', 'frame_count', 'width', 'height')} if metadata else {}))
        if file_info(path) != info: raise ValueError('Source changed during decode')
        prior = visual.get(relative); visual_ok = prior is not None and prior['video_sha256'] == source.sha256
        if prior and not visual_ok: problems.append(dict(path=relative, reason='visual_audit_hash_mismatch'))
        if visual_ok: exposed.append(source.video_id)
        if decoded['status'] != 'complete': problems.append(dict(path=relative, reason=decoded['reason']))
        old_metadata = prior.get('decoder_metadata') if visual_ok else None
        metadata_match = old_metadata is not None and metadata is not None and all(
            old_metadata.get(k) == metadata[k] for k in ('fps', 'frame_count', 'width', 'height'))
        records.append(dict(source=source.model_dump(mode='json'), size_bytes=info['size_bytes'], decode=decoded,
            duration_seconds=source.frame_count/source.fps if source.fps else None,
            last_sample_seconds=(source.frame_count-1)/source.fps if source.fps else None,
            existing_metadata_match=metadata_match,
            labels=dict(action=dict(value='BodyWeightSquats', source='dataset class folder, declared action, not recognition/form truth'),
                        count=dict(status='pending', value=None), temporal_boundaries=dict(status='pending'),
                        viewpoint=dict(official='UNKNOWN', evidence='visually_assessed_unverified' if visual_ok else 'not_assessed',
                            visual_value=prior.get('visual_viewpoint') if visual_ok else None, source=VISUAL.as_posix() if visual_ok else None)),
            audit_fields={k: audit.get(path.name, {}).get(k) for k in ('camera_view', 'person_visibility', 'occlusion', 'decode_status')},
            prior_exposure=dict(known_visual_review=visual_ok, evidence=VISUAL.as_posix() if visual_ok else None,
                                actual_annotation=False, unseen_holdout=False)))
        if index % 20 == 0: print(json.dumps(dict(stage='decode_inventory', completed=index+1, total=len(videos))), flush=True)
    sources = [VideoIdentity.model_validate_json(json.dumps(r['source']), strict=True) for r in records]
    saved, saved_errors = saved_compatibility(root, sources); problems += saved_errors
    catalog, label_search = catalog_evidence(root, out, {s.sha256 for s in sources})
    for source in sources:
        if source.path in saved: continue
        candidates=[e for e in catalog.get(source.sha256,[]) if Path(e['path']).name=='manifest.json']
        for candidate in candidates:
            try:
                saved[source.path]=verify_additional_snapshot(root,source,candidate)
                break
            except (ValueError,OSError,KeyError) as exc:
                problems.append(dict(path=source.path,stage='additional_snapshot',evidence=candidate['path'],error=str(exc)))
    duplicates = defaultdict(list)
    for s in sources: duplicates[s.sha256].append(s.path)
    for row in records:
        s = row['source']; row['duplicate_paths'] = duplicates[s['sha256']]
        row['saved_artifacts'] = saved.get(s['path'], dict(status='no_verified_reusable_snapshot', real_temporal_admission=False))
        row['other_artifact_catalog'] = catalog.get(s['sha256'], [])
        row['prior_exposure']['saved_pipeline_use'] = s['path'] in saved or bool(row['other_artifact_catalog'])
    split = propose_split(sources, exposed_ids=exposed)
    manifest = dict(version='squat-count-dataset-v1', **STATUS, data_kind='real_development',
        protocol_sha256=protocol_hash, evaluation_policy=policy.model_dump(mode='json'),
        video_scope=SCOPE.as_posix(), videos=records, unclassified_scope_entries=unclassified,
        problems=problems, metadata_sources={p.as_posix(): file_info(root/p) for p in (VISUAL, AUDIT, REPLAY_PLAN)},
        duplicate_clusters={sha: paths for sha, paths in duplicates.items() if len(paths)>1},
        related_source_policy='gXX grouping and byte hashes only; no verified shared capture/subject IDs, no guessed near-duplicates',
        label_search=label_search, elapsed_seconds=time.perf_counter()-started,
        inference_runs=0, training_runs=0, fresh_decoding=reused_count==0,
        actual_decoded_frames_this_run=actual_decode_frames, reused_decode_clips=reused_count,
        reused_decode_evidence=dict(path=str(reuse_decode_from),manifest=file_info(root/reuse_decode_from/'dataset-manifest.json')) if reuse_decode_from else None,
        decoder_reason='Reuse prior metadata/viewpoint/pose snapshots; full decode refresh checks all112, not only the selected40. No new visual labeling.')
    write_json(out/'dataset-manifest.json', manifest)
    manifest_sha = file_info(out/'dataset-manifest.json')['sha256']
    write_json(out/'split-proposal.json', dict(**split, dataset_manifest_sha256=manifest_sha))
    private = out/'private'; private.mkdir()
    write_json(private/'artifact-catalog.json', catalog)
    write_json(private/'label-search.json', label_search)
    write_json(private/'exposure.json', dict(clips=[dict(video_id=r['source']['video_id'], **r['prior_exposure']) for r in records],
        classification='Development exposure; does not claim every viewed clip was used to tune thresholds. Profile unchanged.'))
    blind = out/'annotation_package'; blind.mkdir(); (blind/'videos').mkdir()
    shutil.copyfile(root/PROTOCOL_FILE, blind/'PROTOCOL.md')
    passes = [AnnotationPass(source=s, protocol_sha256=protocol_hash) for s in sources]
    blind_sources = []
    for row in records:
        source = row['source']; video = root/source['path']
        target = blind/'videos'/source['video_id']
        shutil.copyfile(video, target)
        if file_info(target) != file_info(video): raise ValueError('Annotation video copy mismatch')
        blind_sources.append(dict(source=source, local_video='videos/'+source['video_id'], decode_status=row['decode']['status']))
    write_json(blind/'manifest.json', dict(version='squat-count-blind-package-v1', protocol_sha256=protocol_hash,
        dataset_manifest_sha256=manifest_sha, videos=blind_sources, no_model_predictions=True))
    for name in ('pass_a.blank.json', 'pass_b.blank.json'):
        write_json(blind/name, dict(version='squat-count-pass-collection-v1', dataset_manifest_sha256=manifest_sha,
                                   annotations=[p.model_dump(mode='json') for p in passes]))
    bundles = [AnnotationBundle(data_kind='real_development', pass_a=p, pass_b=p) for p in passes]
    write_json(out/'adjudication.blank.json', dict(version='squat-count-adjudication-collection-v1',
        dataset_manifest_sha256=manifest_sha, bundles=[b.model_dump(mode='json') for b in bundles]))
    from ..segmentation.models import SquatSegmentationProfile
    predictions = [Prediction(source=s, status='not_run', algorithm_version='squat-pvp-hysteresis-v1',
        config_sha256=SquatSegmentationProfile().content_hash(), evidence_ref='docs/SQUAT_COUNT_RESEARCH_CONTRACT_PROPOSAL.md',
        reasons=(reason('real_temporal_admission_blocked', 'Existing producer is synthetic-only; unchanged G5.7/sequence/bilateral admission.'),)) for s in sources]
    write_json(out/'predictions.not_run.json', dict(version='squat-count-prediction-collection-v1',
        dataset_manifest_sha256=manifest_sha, predictions=[p.model_dump(mode='json') for p in predictions]))
    for model in (AnnotationPass, AnnotationBundle, Prediction, EvaluationPolicy):
        write_json(out/(model.__name__+'.schema.json'), model.model_json_schema())
    summary = dict(video_paths=len(records), unique_hashes=len(duplicates), groups=len({s.group_id for s in sources}),
        total_frames=sum(r['decode']['frames_decoded'] for r in records),
        total_duration_seconds=sum(r['duration_seconds'] or 0 for r in records),
        decode_status=dict(Counter(r['decode']['status'] for r in records)),
        prior_metadata_matches=sum(r['existing_metadata_match'] for r in records),
        saved_status=dict(Counter(r['saved_artifacts']['status'] for r in records)),
        sequence_actions=dict(Counter(r['saved_artifacts']['sequence']['action'] for r in records if 'sequence' in r['saved_artifacts'])),
        previously_visually_reviewed=len(exposed), independent_temporal_labels=0,
        untouched_holdout_established=False, split_roles=dict(Counter(a['role'] for a in split['assignments'])),
        issues=len(problems), unclassified_scope_entries=unclassified, **STATUS)
    write_json(out/'summary.json', summary)
    (blind/'HUONG_DAN.txt').write_text('Chi giao video, manifest, PROTOCOL.md va mot template rieng cho moi rater.\nKhong giao private/, predictions, split hoac ket qua nguoi kia. Khong tu dien count=0.\nLuu annotation da dien o file moi; giu template goc. Hai nguoi doc lap, mot nguoi phan xu khac.\n', encoding='utf-8')
    write_json(out/'artifact-inventory.json', dict(version='squat-count-preparation-inventory-v1',
        files={p.relative_to(out).as_posix(): file_info(p) for p in out.rglob('*') if p.is_file()}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-decode-from',type=Path,help='Verified prior readiness bundle; source byte identities must still match')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output_dir,reuse_decode_from=args.reuse_decode_from), indent=2))


if __name__ == '__main__': main()
