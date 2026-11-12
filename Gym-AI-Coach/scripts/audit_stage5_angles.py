"""Fresh development audit from a byte-bound, precommitted visual selection.

Run from the workspace with an explicit new output directory. This audit never
creates annotation, promotes QualityReport or chooses clips from pose outcomes.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata as packages
import json
from pathlib import Path
import platform
import sys
import time
import warnings

from gym_coach.contracts import Exercise, Viewpoint
from gym_coach.features import GEOMETRY_REGISTRY
from gym_coach.features.registry import input_error
from gym_coach.flags import FlagCode, GymError


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def workspace_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Audit inputs must use workspace-relative paths')
    return path


def load_selection(root, selection_path, expected_hash):
    """Verify path/ID/hash/visual evidence before any decoder or backend starts."""
    try:
        if sha(selection_path) != expected_hash:
            raise ValueError('Precommitted selection byte hash mismatch')
        selection = json.loads(Path(selection_path).read_text('utf-8'))
        if selection['selection_version'] != 'development-angle-audit-v1' or selection['inference_started'] is not False:
            raise ValueError('Expected frozen selection made before inference')
        hashes = set()
        for candidate in selection['candidates']:
            path = workspace_path(root, candidate['path'])
            if path.name != candidate['video_id'] or sha(path) != candidate['sha256']:
                raise ValueError('Selected video path/ID/byte hash mismatch')
            if candidate['sha256'] in hashes:
                raise ValueError('Selection must deduplicate byte-identical videos')
            hashes.add(candidate['sha256'])
            Exercise(candidate['exercise']); Viewpoint(candidate['exploratory_viewpoint'])
            if candidate['evidence_status'] != 'visually_assessed_unverified' or candidate['documented'] is not False:
                raise ValueError('Agent visual audit cannot establish documented eligibility')
            evidence_path = workspace_path(root,candidate['visual_evidence_path'])
            if sha(evidence_path) != candidate['visual_evidence_sha256']:
                raise ValueError('Visual evidence byte hash mismatch')
            evidence = json.loads(evidence_path.read_text('utf-8'))
            row = next(v for v in evidence.get('videos',evidence.get('candidates',[])) if v['review_id']==candidate['review_id'])
            if (row.get('representative_path',row.get('path')) != candidate['path']
                    or row.get('input_byte_sha256',row.get('sha256')) != candidate['sha256']
                    or row['video_id'] != candidate['video_id']):
                raise ValueError('Visual evidence does not identify selected video')
            visual = row.get('visual_assessment',row.get('assessment'))
            expected_view = {'likely_side':'side','front':'front','oblique':'oblique','uncertain':'unknown'}[visual['viewpoint']]
            if candidate['exploratory_viewpoint'] != expected_view:
                raise ValueError('Exploratory assumption differs from recorded visual assessment')
            samples=row['samples']
            if len(samples)<2 or len({s['frame_index'] for s in samples})!=len(samples):
                raise ValueError('Visual evidence requires distinct multiple sampled times')
            for sample in samples:
                sample_path=workspace_path(root,sample.get('image_relative_path',sample.get('path')))
                if (sample['agent_viewed'] is not True
                        or sha(sample_path)!=sample.get('image_byte_sha256',sample.get('sha256'))):
                    raise ValueError('Visual sample evidence is missing, unviewed or byte-mismatched')
        if not hashes:
            raise ValueError('Empty selection')
        return selection
    except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:
        flag = input_error(FlagCode.FEATURE_CONTEXT_MISMATCH, str(exc) or 'Selected review ID is absent from visual evidence').flag
        raise GymError(flag.model_copy(update={'evidence':{'component':'development_angle_audit','reason':'selection_linkage_invalid'}})) from exc


def run_branch(root, candidate, viewpoint, destination):
    # Heavy backends are lazy and used only for fresh inference/preprocessing.
    from gym_coach.features import (FeatureInputContext, GeometryFeatureObservation,
        build_feature_input_context, observe_geometry_feature)
    from gym_coach.features.models import content_hash
    from gym_coach.pose import MediaPipePoseExtractor, PoseCoverageAccumulator
    from gym_coach.pose.serialization import write_pose_frames
    from gym_coach.preprocessing import load_preprocessing_profile, preprocess_pose, write_preprocessing_result
    from gym_coach.preprocessing.models import content_hash as source_hash
    from gym_coach.quality import QualityGate
    from gym_coach.video import OpenCVVideoReader

    destination.mkdir(parents=True,exist_ok=False)
    source_path = workspace_path(root,candidate['path'])
    start = time.perf_counter()
    reader = OpenCVVideoReader(source_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with MediaPipePoseExtractor(reader.metadata.video_id,candidate['exercise'],viewpoint=viewpoint) as extractor:
            raw = tuple(extractor.extract(frame) for frame in reader.frames())
            accumulator = PoseCoverageAccumulator(extractor.video_id,extractor.exercise,extractor.identity,
                                                  extractor.config,viewpoint=viewpoint)
            for frame in raw: accumulator.add(frame)
            pose_summary = accumulator.summary()
        assert len(raw)==reader.metadata.frame_count
        write_pose_frames(destination/'raw_pose.jsonl',raw)
        write_json(destination/'metadata.json',reader.metadata.model_dump(mode='json'))
        write_json(destination/'pose_summary.json',pose_summary.model_dump(mode='json'))
        gate = QualityGate()
        quality = gate.evaluate(raw,metric_requirements=GEOMETRY_REGISTRY.quality_requirements(candidate['exercise']),
                               metric_viewpoints=GEOMETRY_REGISTRY.quality_viewpoints(candidate['exercise']))
        write_json(destination/'quality.json',quality.model_dump(mode='json'))
        profile = load_preprocessing_profile(root/'configs/preprocessing_s3_v1.toml')
        processed = preprocess_pose(raw,quality,reader.metadata,profile=profile)
        write_preprocessing_result(processed,destination/'preprocessing.json')
        before = source_hash(processed),content_hash(quality)
        features = GEOMETRY_REGISTRY.for_exercise(candidate['exercise'])
        stats = {f.feature_id:dict(available=0,unavailable=0,failed_measurement=0,reasons=Counter(),
                                  input_provenance=Counter(),available_min_deg=None,available_max_deg=None) for f in features}
        with (destination/'features.jsonl').open('x',encoding='utf-8') as stream, (destination/'frame_links.jsonl').open('x',encoding='utf-8') as links:
            for frame in processed.frames:
                linked = build_feature_input_context(processed,frame_index=frame.frame_index)
                audit = linked.model_dump(mode='json',exclude={'frame','quality','preprocessing_result'})
                audit.update(frame_index=frame.frame_index,timestamp_seconds=frame.timestamp_seconds,
                             measurement_input_hash=linked.measurement_input_hash,linked_context_hash=linked.content_hash())
                links.write(json.dumps(audit,allow_nan=False)+'\n')
                if frame.frame_index in (0,len(processed.frames)-1):
                    write_json(destination/f'linked_context_{frame.frame_index}.json',linked.model_dump(mode='json'))
                # Factory already validates linkage for this frame. Individual
                # observations carry the exact detached frame/quality hash;
                # frame_links ties each snapshot to the persisted full source.
                context = FeatureInputContext(frame=linked.frame,quality=linked.quality)
                for definition in features:
                    obs = observe_geometry_feature(definition,context)
                    line = obs.model_dump_json()
                    assert GeometryFeatureObservation.model_validate_json(line,strict=True)==obs
                    assert obs.availability.input_hash==linked.measurement_input_hash
                    stream.write(line+'\n')
                    item = stats[definition.feature_id];item[obs.status]+=1
                    item['failed_measurement']+=obs.computation_status=='failed_measurement'
                    item['reasons'].update(reason.value for reason in obs.reasons)
                    item['input_provenance'].update(j.status.value for j in obs.joint_inputs)
                    if obs.value is not None:
                        item['available_min_deg']=obs.value if item['available_min_deg'] is None else min(item['available_min_deg'],obs.value)
                        item['available_max_deg']=obs.value if item['available_max_deg'] is None else max(item['available_max_deg'],obs.value)
        assert before==(source_hash(processed),content_hash(quality))
    summary = dict(completed=True,video_id=reader.metadata.video_id,input_path=candidate['path'],input_byte_sha256=sha(source_path),
        viewpoint=viewpoint.value,viewpoint_evidence=candidate['evidence_status'],documented=False,official_eligibility=False,
        source_metadata_viewpoint='unknown',branch='original_unknown' if viewpoint==Viewpoint.UNKNOWN else 'exploratory_visual_assumption',
        exploratory_assumption=None if viewpoint==Viewpoint.UNKNOWN else viewpoint.value,
        decoded_frames=len(raw),timestamp_policy=reader.metadata.timestamp_policy,
        backend=raw[0].backend.model_dump(mode='json'),pose_config=raw[0].config.model_dump(mode='json'),
        pose_config_hash=raw[0].config.content_hash(),quality_profile_hash=quality.profile_hash,
        quality_profile=quality.profile.model_dump(mode='json'),preprocessing_profile_hash=processed.profile_hash,
        preprocessing_profile=processed.profile.model_dump(mode='json'),preprocessing_result_hash=source_hash(processed),
        raw_sequence_hash=processed.raw_sequence_hash,quality_report_hash=content_hash(quality),
        sequence_status=quality.sequence_status,sequence_decision=quality.eligibility.model_dump(mode='json'),
        metric_decisions={f.feature_id:quality.metric_eligibility[f.feature_id].model_dump(mode='json') for f in features},
        feature_outcomes=stats,python_warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
        elapsed_seconds=time.perf_counter()-start,temporal_analysis_eligible=False,assessment_status='blocked_pending_expert_validation')
    summary['artifact_byte_hashes']={p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination/'summary.json',summary)
    return summary


def run_audit(root, selection_path, expected_hash, destination):
    selection=load_selection(root,selection_path,expected_hash)
    destination.mkdir(parents=True,exist_ok=False)
    manifest=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                  selection_path=str(Path(selection_path).relative_to(root)),selection_sha256=expected_hash,
                  registry=GEOMETRY_REGISTRY.model_dump(mode='json'),registry_hash=GEOMETRY_REGISTRY.content_hash(),
                  audit_script_sha256=sha(Path(__file__)),
                  environment=dict(python=sys.version,platform=platform.platform(),dependencies={
                      p:packages.version(p) for p in ('ai-coach-gym','pydantic','mediapipe','numpy','scipy','opencv-contrib-python','pytest')}))
    write_json(destination/'manifest.json',manifest)
    results=[]
    for candidate in selection['candidates']:
        for branch,view in (('original_unknown',Viewpoint.UNKNOWN),('exploratory',Viewpoint(candidate['exploratory_viewpoint']))):
            output=destination/(candidate['review_id']+'_'+branch)
            print(f"START {candidate['review_id']} {branch}",flush=True)
            try:
                summary=run_branch(root,candidate,view,output)
            except GymError as exc:
                summary=dict(completed=False,video_id=candidate['video_id'],branch=branch,structured_error=exc.flag.model_dump(mode='json'))
                if not output.exists(): output.mkdir()
                write_json(output/'failure.json',summary)
            results.append(dict(review_id=candidate['review_id'],output=output.relative_to(root).as_posix(),**summary))
            print(f"DONE {candidate['review_id']} {branch} completed={summary['completed']}",flush=True)
    write_json(destination/'results.json',dict(selection_sha256=expected_hash,results=results,documented_input_status='BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT'))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=run_audit(args.root.resolve(),args.selection.resolve(),args.selection_sha256,args.output.resolve())
    return 0 if all(r['completed'] for r in results) else 1


if __name__=='__main__':
    raise SystemExit(main())
