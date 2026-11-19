"""Intake explicit AI visual observations; never infer labels from extracted frames."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from gym_coach.counting.ai_annotations import AIAnnotation, save_pass, load_pass, verify_evidence
from gym_coach.counting.readiness import file_info


def intake(run, observations, pass_id):
    run=Path(run); observations=Path(observations)
    decision=json.loads((run/'decision.json').read_bytes())
    protocol=Path('docs/SQUAT_AI_ANNOTATION_PROTOCOL.md')
    semantics=next(p for p in decision['inputs'] if p.endswith('/PROTOCOL.md'))
    if file_info(semantics)!=decision['inputs'][semantics]:raise ValueError('Frozen semantics changed')
    dataset=next(p for p in decision['inputs'] if p.endswith('/dataset-manifest.json'))
    if file_info(dataset)!=decision['inputs'][dataset]:raise ValueError('Frozen dataset changed')
    sources={r['source']['video_id']:r['source'] for r in json.loads(Path(dataset).read_bytes())['videos']}
    rows=json.loads(observations.read_bytes())
    target=run/('pass-'+pass_id);target.mkdir(exist_ok=True)
    results=[]
    for row in rows:
        video=row['video_id'];sheets_dir=run/'source-sheets'/video
        sheets=json.loads((sheets_dir/'manifest.json').read_bytes())
        source=sheets['source'];fps=source['fps']
        if sources.get(video)!=source:raise ValueError('Sheet source differs from frozen cohort')
        def span(pair):
            return {k:dict(frame_index=i,timestamp_seconds=i/fps) for k,i in zip(('start','end'),pair)}
        viewed=row['viewed_sheets']
        evidence=[dict(path=(sheets_dir/s['file']).as_posix(),sha256=s['identity']['sha256'],frame_indices=s['frame_indices']) for s in sheets['sheets'] if s['file'] in viewed]
        if len(evidence)!=len(viewed):raise ValueError('Unknown/repeated viewed sheet')
        hints=[dict(start_range=span(h[0]),bottom_range=span(h[1]),end_range=span(h[2]),reason='AI approximate boundary envelope; not exact event or pause') for h in row.get('hints',[])]
        partials=[dict(kind=p[0],interval=span(p[1]),reason=p[2]) for p in row.get('partials',[])]
        uncertain=[dict(interval=span(u[0]),affects_count=u[1],reason=u[2]) for u in row.get('uncertain',[])]
        payload=dict(protocol_sha256=file_info(protocol)['sha256'],semantics_protocol_sha256=file_info(semantics)['sha256'],
            source=source,pass_id=pass_id,recorded_at=datetime.now(timezone.utc).isoformat(),evidence=evidence,
            subject_description=row['subject'],whole_clip_count=row['count'],observed_complete_count=row['observed'],
            count_reason=row.get('reason'),boundaries_reason='Exact event samples not established by contact-sheet observation; hints are uncertainty ranges only',
            cycle_hints=hints,partials=partials,uncertain=uncertain,observations=row['notes'])
        annotation=AIAnnotation.model_validate_json(json.dumps(payload))
        path=target/(video+'.json')
        if path.exists():
            previous=load_pass(path)
            verify_evidence(previous,Path.cwd())
            new=annotation.model_dump(mode='json');old=previous.model_dump(mode='json')
            for d in (new,old):d.pop('recorded_at')
            if new!=old:raise ValueError('Resume would change existing annotation')
            results.append(dict(video=video,status='verified_resume'));continue
        save_pass(path,annotation,Path.cwd())
        results.append(dict(video=video,status='created'))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--observations',required=True,type=Path)
    p.add_argument('--pass-id',choices=['A','B'],required=True)
    a=p.parse_args();print(json.dumps(intake(a.run,a.observations,a.pass_id),indent=2))


if __name__=='__main__':main()
