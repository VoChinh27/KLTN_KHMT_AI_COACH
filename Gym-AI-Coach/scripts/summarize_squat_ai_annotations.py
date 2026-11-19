"""Verify saved AI observations and export an explicitly non-independent reference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from gym_coach.counting.ai_annotations import AIAnnotation, AIViewpoint, load_pass, verify_evidence, summarize_pairs
from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.readiness import file_info, write_json


def summarize(run, observations, output):
    root=Path.cwd().resolve(); run=Path(run); output=Path(output)
    if output.exists() or not output.resolve().is_relative_to(root/'outputs'):
        raise ValueError('A new output directory under outputs is required')
    decision=json.loads((run/'decision.json').read_bytes())
    for path,identity in decision['inputs'].items():
        if file_info(path)!=identity:raise ValueError('Frozen input changed: '+path)
    dataset_path=next(p for p in decision['inputs'] if p.endswith('/dataset-manifest.json'))
    viewpoint_path=next(p for p in decision['inputs'] if p.endswith('/viewpoint-review.pending.json'))
    semantics_path=next(p for p in decision['inputs'] if p.endswith('/PROTOCOL.md'))
    dataset=json.loads(Path(dataset_path).read_bytes())
    sources={r['source']['video_id']:r['source'] for r in dataset['videos']}
    if len(sources)!=112 or len(dataset['videos'])!=112:raise ValueError('Expected fixed 112 cohort')
    ids=sorted(sources,key=lambda i:sources[i]['path'])
    protocol=Path('docs/SQUAT_AI_ANNOTATION_PROTOCOL.md'); protocol_hash=file_info(protocol)['sha256']
    pairs=[];links={}; evidence_inventory={}
    for label in ('A','B'):
        actual={p.name for p in (run/('pass-'+label)).glob('*.json')}
        if actual!={i+'.json' for i in ids}:raise ValueError('Missing/extra checkpoint in '+label)
    for video in ids:
        passes=[];refs={}
        for label in ('A','B'):
            path=run/('pass-'+label)/(video+'.json');p=load_pass(path)
            if (p.source.model_dump(mode='json')!=sources[video] or p.protocol_sha256!=protocol_hash
                    or p.semantics_protocol_sha256!=decision['inputs'][semantics_path]['sha256']):
                raise ValueError('Frozen source/protocol mismatch: '+video)
            verify_evidence(p,root);passes.append(p)
            refs[label]=dict(path=path.as_posix(),**file_info(path))
            for e in p.evidence:
                evidence_inventory[e.path]=file_info(e.path)
                m=Path(e.path).parent/'manifest.json';evidence_inventory[m.as_posix()]=file_info(m)
        pairs.append(tuple(passes));links[video]=refs
    result=summarize_pairs(pairs,ids)
    for row in result['records']:row['pass_links']=links[row['video_id']]
    originals=json.loads(Path(viewpoint_path).read_bytes())['records']
    manual=json.loads(Path(observations).read_bytes())['records']
    if (len(originals)!=33 or len(manual)!=33 or len({r['video_id'] for r in manual})!=33
            or {r['video_id'] for r in manual}!={r['source']['video_id'] for r in originals}):
        raise ValueError('Expected exact frozen 33 viewpoint IDs')
    annotations={a.source.video_id:a for a,b in pairs};viewpoints=[]
    rubric=Path(viewpoint_path).parent/'VIEWPOINT_REVIEW.md'
    original_sources={r['source']['video_id']:r['source'] for r in originals}
    for row in sorted(manual,key=lambda r:r['video_id']):
        p=annotations[row['video_id']]; source=p.source
        if source.model_dump(mode='json')!=original_sources[source.video_id]:raise ValueError('Viewpoint source mismatch')
        intervals=row.get('intervals',[dict(start=0,end=source.frame_count-1,viewpoint=row['viewpoint'],reason=row['camera_direction_notes'])])
        support=[dict(interval=dict(start=dict(frame_index=i['start'],timestamp_seconds=i['start']/source.fps),end=dict(frame_index=i['end'],timestamp_seconds=i['end']/source.fps)),viewpoint=i['viewpoint'],reason=i['reason']) for i in intervals]
        payload=dict(protocol_sha256=protocol_hash,rubric_sha256=file_info(rubric)['sha256'],source=source.model_dump(mode='json'),
            recorded_at=datetime.now(timezone.utc).isoformat(),evidence=[e.model_dump(mode='json') for e in p.evidence],
            reviewed_intervals=support,declared_whole_clip_viewpoint=row['viewpoint'],subject_description=p.subject_description,
            camera_changes=row['camera_changes'],camera_direction_notes=row['camera_direction_notes'],framing_notes=row['framing_notes'],occlusion_notes=row['occlusion_notes'],
            uncertainty_notes='AI contact-sheet judgment, no calibration or human validation; small images limit detail. Recording time is not a human review time.')
        view=AIViewpoint.model_validate_json(json.dumps(payload));verify_evidence(view,root)
        viewpoints.append(dict(payload=view.model_dump(mode='json'),sha256=canonical_hash(view)))
    result.update(version='squat-ai-reconciliation-v1',**STATUS)
    result['summary']['viewpoint_distribution']=dict(Counter(v['payload']['declared_whole_clip_viewpoint'] for v in viewpoints))
    result['summary'].update(viewpoint_reviewed=len(viewpoints),documented_viewpoint=0,training_readiness='NOT_READY',real_counting_run=False,training_run=False)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'ai-reference.json',result)
    write_json(output/'ai-viewpoint.json',dict(version='squat-ai-viewpoint-collection-v1',records=viewpoints,**STATUS))
    write_json(output/'AIAnnotation.schema.json',AIAnnotation.model_json_schema())
    write_json(output/'AIViewpoint.schema.json',AIViewpoint.model_json_schema())
    write_json(output/'evidence-integrity.json',dict(source_count=112,passes_verified=224,evidence_files=evidence_inventory,
        inputs=decision['inputs'],observations=dict(path=Path(observations).as_posix(),**file_info(observations)),
        protocol=dict(path=protocol.as_posix(),**file_info(protocol)),rubric=dict(path=rubric.as_posix(),**file_info(rubric)),
        source_hashes={i:sources[i]['sha256'] for i in ids},errors=[],scope='Hash/schema/timeline linkage, not label correctness'))
    write_json(output/'artifact-inventory.json',dict(version='squat-ai-summary-inventory-v1',files={p.name:file_info(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return result['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--viewpoint-observations',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.run,args.viewpoint_observations,args.output_dir),ensure_ascii=False,indent=2))
