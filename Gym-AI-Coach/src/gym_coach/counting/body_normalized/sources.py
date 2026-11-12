"""Reuse existing saved-source validators; no inference, fitting or source mutation."""
import importlib.util
from pathlib import Path
from ..inner_cv import manifest as cv
from ..models import canonical_hash
from ..research.data import read,identity,local
from ..mse.manifest import diagnostic_tool
from .contract import SEEDS

SENSITIVITY='outputs/squat_frame_order_sensitivity/run-9f0b2c4b3383435985f0f40cf330047d'
REVIEW='outputs/squat_frame_order_sensitivity/review-e836aefa2afe409da5d539f55f564931'


def load_verifier(root):
    path=local(root,'scripts/diagnose_squat_frame_order.py')
    spec=importlib.util.spec_from_file_location('body_normalization_source_check',path)
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def verify(root):
    tool=load_verifier(root)
    ledger=tool.Ledger(root)
    ledger.take('scripts/diagnose_squat_frame_order.py')
    conditions={seed:tool.load_condition(root,seed,ledger) for seed in SEEDS}
    a,b=[conditions[s]['manifest'] for s in SEEDS]
    for key in ('rows','folds','dataset','core_identity','cv_code'):
        if a[key]!=b[key]:
            raise ValueError('Raw controls differ in '+key)
    if dict(a['config'],seed=SEEDS[1])!=b['config'] or b['parent_manifest_sha256']!=canonical_hash(a):
        raise ValueError('Raw config/seed lineage mismatch')
    baseline_link=a['controls']['verified_lineage']['baseline']
    if baseline_link!=b['sources']['baseline']:
        raise ValueError('Raw baselines differ')
    baseline=cv.unseal(read(ledger.link(baseline_link)))
    diagnostic_link=a['controls']['diagnostic_tool']
    ledger.link(diagnostic_link)
    stats=diagnostic_tool(root,diagnostic_link)
    records={};control_links={};metrics={}
    for seed,condition in conditions.items():
        man=condition['manifest']
        result=cv.unseal(read(ledger.link(condition['comparison'])))
        endpoint=result['endpoints']['499']
        method='cv500' if seed==SEEDS[0] else 'seed20261027'
        controls=[]
        for name in (method,'duration','constant'):
            train={};held={}
            for f in range(3):
                train[f]=stats.validate_records(man,f,'train',[r for r in endpoint['records'] if r['method']==name and r['fold']==f and r['side']=='train'])
                held[f]=stats.validate_records(man,f,'evaluation',[r for r in endpoint['records'] if r['method']==name and r['fold']==f and r['side']=='held_out'])
                if name==method:
                    for r in held[f]:
                        if r['prediction']!=condition['oof'][r['video_id']]['checkpoint_prediction']:
                            raise ValueError('Raw OOF/checkpoint prediction mismatch')
                for side,items in (('train',train[f]),('held_out',held[f])):
                    controls.extend(dict(r,method='gru' if name==method else name,side=side) for r in items)
            aggregate=stats.aggregate_unique_videos(man,train,held)
            if aggregate!=endpoint['results'][name]['aggregate']:
                raise ValueError('Saved raw metric mismatch')
            if name in ('duration','constant'):
                left=sorted((r for rs in held.values() for r in rs),key=lambda r:r['video_id'])
                right=sorted(baseline['predictions'][name],key=lambda r:r['video_id'])
                if len(left)!=len(right) or any(any(x[k]!=y[k] for k in ('video_id','fold','source_sha256','target','prediction'))
                                               for x,y in zip(left,right)):
                    raise ValueError('Saved baseline linkage mismatch')
            metrics.setdefault(str(seed),{})['raw58' if name==method else name]=aggregate
        records[str(seed)]=controls
        initials={}
        for f,link in condition['links'].items():
            run=read(ledger.take(local(root,link['path']).parent/'run.json'))
            initials[str(f)]=run['initial_fresh_state_sha256']
        control_links[str(seed)]=dict(manifest=condition['manifest_link'],comparison=condition['comparison'],
            checkpoint_sources={str(f):link for f,link in condition['links'].items()},
            initializations=initials,raw_code=man['code'],raw_config=man['config'])
    # Review is a saved rerun. Compare scientific outputs and verify every referenced source.
    for name in ('manifest.json','permutations.json','original-replay.json','draws.jsonl','per-video.json','summary.json'):
        paths=[ledger.take(Path(folder)/name) for folder in (SENSITIVITY,REVIEW)]
        if paths[0].read_bytes()!=paths[1].read_bytes():
            raise ValueError('Sensitivity review differs: '+name)
        if name.endswith('.json'):
            cv.unseal(read(paths[0]))
    for folder in (SENSITIVITY,REVIEW):
        receipt=read(ledger.take(Path(folder)/'receipt.json'))
        if (receipt['status']!='COMPLETE_DEVELOPMENT_SENSITIVITY_ONLY' or receipt['original_forwards']!=66
                or receipt['identity_forwards']!=66 or receipt['shuffled_forwards']!=1320
                or any(w['before']!=w['after'] for w in receipt['weights'])):
            raise ValueError('Sensitivity receipt mismatch')
        for item in receipt['sources']:
            ledger.link(item)
    for row in a['rows']:
        for link in [row['input_artifact'],*row['source_links'],*row['label_links']]:
            ledger.link(link)
    evidence=dict(controls=control_links,metrics=metrics,baseline=baseline_link,diagnostic_tool=diagnostic_link,
        sensitivity=identity(root,local(root,SENSITIVITY)/'summary.json'),
        review=identity(root,local(root,REVIEW)/'summary.json'),
        source_verifier=identity(root,local(root,'scripts/diagnose_squat_frame_order.py')),
        sources=ledger.preserved(),new_model_forwards=0,training_steps=0,baseline_fits=0)
    return a,conditions[SEEDS[0]]['data'],records,evidence
