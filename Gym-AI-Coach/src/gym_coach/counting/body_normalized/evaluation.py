"""Common MAE metrics at fixed499; saved controls and equal-video two-seed aggregation."""
import statistics
from ..inner_cv_500.evaluation import compare_endpoint as common_metrics
from ..inner_cv.manifest import unseal
from ..mse.manifest import diagnostic_tool
from ..models import STATUS,canonical_hash
from ..research.data import read,identity,check_identity
from ..readiness import file_info
from .contract import SEEDS,VERSION
from . import runtime

METHODS={'cv500':'body_normalized','gru':'raw58','duration':'duration','constant':'constant'}


def compare(manifest,seed,train,held,controls,tool):
    result=common_metrics(manifest,train,held,controls,tool,499)
    result['results']={METHODS[k]:v for k,v in result['results'].items()}
    result['records']=[dict(r,method=METHODS[r['method']],seed=seed) for r in result['records']]
    lookup={(r['method'],r['side'],r['fold'],r['video_id']):r for r in result['records']}
    paired=[]
    for r in result['records']:
        if r['method']!='body_normalized':
            continue
        peers={m:lookup[m,r['side'],r['fold'],r['video_id']] for m in METHODS.values()}
        paired.append(dict(video_id=r['video_id'],source_sha256=r['source_sha256'],label_record_hash=r['label_record_hash'],
            row_sha256=r['row_sha256'],group_id=r['group_id'],target=r['target'],seed=seed,fold=r['fold'],side=r['side'],
            predictions={m:p['prediction'] for m,p in peers.items()},
            signed_errors={m:p['signed_error'] for m,p in peers.items()},
            absolute_errors={m:p['absolute_error'] for m,p in peers.items()},
            delta_AE={m:r['absolute_error']-p['absolute_error'] for m,p in peers.items() if m!='body_normalized'}))
    result['paired_records']=paired
    result['fold0_held_out']=[p for p in paired if p['fold']==0 and p['side']=='held_out']
    result['paired_equal_video']={}
    for side,field in (('train','train_mean_AE'),('held_out','held_out_AE')):
        normalized={r['video_id']:r for r in result['results']['body_normalized']['aggregate']['records']}
        comparison={}
        for method in ('raw58','duration','constant'):
            raw={r['video_id']:r for r in result['results'][method]['aggregate']['records']}
            pairs=[dict(video_id=v,group_id=r['group_id'],target=r['target'],body_AE=r[field],control_AE=raw[v][field],
                        delta_AE=r[field]-raw[v][field]) for v,r in normalized.items()]
            comparison[method]=dict(records=pairs,**difference_summary(pairs))
        result['paired_equal_video'][side]=comparison
    result['prediction_sha256']=canonical_hash(result['records'])
    return result


def difference_summary(rows):
    return dict(unique_videos=len(rows),mean_delta_AE=statistics.mean(r['delta_AE'] for r in rows),
        improved=sum(r['delta_AE']<0 for r in rows),worsened=sum(r['delta_AE']>0 for r in rows),
        equal=sum(r['delta_AE']==0 for r in rows))


def combine(seeds):
    if set(seeds)!={str(s) for s in SEEDS}:
        raise ValueError('Both seeds required')
    methods={}
    for method in METHODS.values():
        if any(len(q['results'][method]['aggregate']['records'])!=33 for q in seeds.values()):
            raise ValueError('Two-seed aggregation requires exactly33 records per method/seed')
        values={seed:{r['video_id']:r for r in q['results'][method]['aggregate']['records']} for seed,q in seeds.items()}
        a,b=[values[str(s)] for s in SEEDS]
        if len(a)!=33 or set(a)!=set(b):
            raise ValueError('Two-seed pairing requires same33videos')
        rows=[]
        for video,x in a.items():
            y=b[video]
            if any(x[k]!=y[k] for k in ('target','group_id','held_out_fold','train_folds')):
                raise ValueError('Two-seed group/target/fold mismatch')
            rows.append(dict(video_id=video,group_id=x['group_id'],target=x['target'],
                train_mean_AE=(x['train_mean_AE']+y['train_mean_AE'])/2,
                held_out_mean_AE=(x['held_out_AE']+y['held_out_AE'])/2))
        group_errors={g:statistics.mean(r['held_out_mean_AE'] for r in rows if r['group_id']==g)
                      for g in sorted({r['group_id'] for r in rows})}
        if len(group_errors)!=12:
            raise ValueError('Require locked12 groups')
        methods[method]=dict(records=rows,unique_videos=33,seeds_per_video=2,
            train_MAE=statistics.mean(r['train_mean_AE'] for r in rows),
            OOF_MAE=statistics.mean(r['held_out_mean_AE'] for r in rows),
            macro_group_MAE=statistics.mean(group_errors.values()),group_MAE=group_errors)
    deltas={seed:q['paired_equal_video']['held_out']['raw58']['mean_delta_AE'] for seed,q in seeds.items()}
    directions=[-1 if d<0 else 1 if d>0 else 0 for d in deltas.values()]
    interpretation=('inconsistent_across_seeds' if -1 in directions and 1 in directions else
                    'same_observed_direction' if len(set(directions))==1 else 'mixed_with_exact_tie')
    return dict(methods=methods,per_seed_delta_OOF_MAE=deltas,descriptive_result=interpretation,
        unit='mean absolute errors within video across seeds, then equal mean33; not66independent videos',
        criterion='lower MAE is better; no significance/equivalence threshold or seed selection')


def evaluate(root,data,manifest,paths,controls):
    import torch
    if len(paths)!=6:
        raise ValueError('Require six index499 checkpoints')
    loaded={};links=[]
    for path in paths:
        if file_info(path)!=read(str(path)+'.sha256.json'):
            raise ValueError('Checkpoint byte hash mismatch')
        header=torch.load(path,map_location='cpu',weights_only=True)
        seed=header['identity']['seed'];fold=header['identity']['fold']
        engine,selected,_=runtime.engine(manifest,seed)
        saved=engine.load_checkpoint(path,selected,fold)
        if saved['epoch']!=499 or (seed,fold) in loaded:
            raise ValueError('Duplicate or wrong endpoint checkpoint')
        loaded[seed,fold]=saved
        links.append(dict(**identity(root,path),seed=seed,fold=fold))
    if set(loaded)!={(s,f) for s in SEEDS for f in range(3)}:
        raise ValueError('Incomplete seed/fold grid')
    tool=diagnostic_tool(root,manifest['sources']['diagnostic_tool'])
    results={};receipts=[]
    for seed in SEEDS:
        train={};held={}
        for f in range(3):
            saved=loaded[seed,f]
            train[f],receipt=runtime.train_predictions(root,data,manifest,seed,f,saved)
            receipts.append(receipt)
            held[f]=[dict(r,fold=f) for r in unseal(saved['epoch_reports'][499])['records']]
        results[str(seed)]=compare(manifest,seed,train,held,controls[str(seed)],tool)
    for link in links:
        check_identity(root,link)
    return dict(version=VERSION,**STATUS,status='COMPLETE_DEVELOPMENT_ONLY',manifest_sha256=canonical_hash(manifest),
        endpoint_index=499,seeds=results,combined=combine(results),checkpoints=links,inference_receipts=receipts,
        train_forward_calls=132,held_forward_calls=0,training_steps=0,baseline_fits=0,
        limitation=manifest['limitation'])


def report(result):
    lines=['# Body-normalized GRU500 · index 499','',result['limitation'],'',
        '|Seed|Method|Train MAE33|OOF MAE33|Macro MAE12groups|','|---|---|---:|---:|---:|']
    for seed,q in result['seeds'].items():
        for method,s in q['results'].items():
            a=s['aggregate']
            lines.append(f"|{seed}|{method}|{a['train_video_equal_MAE']:.6f}|{a['held_out_pooled_MAE']:.6f}|{s['macro_group_MAE']:.6f}|")
        change=q['paired_equal_video']['held_out']['raw58']
        lines.extend(['',f"{seed} ΔOOF MAE body−raw {change['mean_delta_AE']:+.6f}. Improved {change['improved']}/33, worsened {change['worsened']}/33, equal {change['equal']}/33.",''])
    lines.extend(['','## Cùng video, trung bình hai seed','',result['combined']['unit'],'',
        result['combined']['descriptive_result'],'','|Method|Train MAE|OOF MAE|Macro MAE12groups|','|---|---:|---:|---:|'])
    for method,q in result['combined']['methods'].items():
        lines.append(f"|{method}|{q['train_MAE']:.6f}|{q['OOF_MAE']:.6f}|{q['macro_group_MAE']:.6f}|")
    lines.extend(['','## Fold, group và count','',
        '|Seed|Method|Fold|Subset|n/groups|MAE|Signed mean|','|---|---|---|---|---|---:|---:|'])
    for seed,q in result['seeds'].items():
        for method,s in q['results'].items():
            for f,fold in s['folds'].items():
                for side in ('train','held_out'):
                    metric=fold[side]
                    lines.append(f"|{seed}|{method}|{f}|{side}|{metric['clips']}/{metric['groups']}|{metric['MAE']:.6f}|{metric['signed_error_mean']:.6f}|")
    lines.extend(['','## Đủ13 video held-out fold0, gồm g11 và g20','',
        '|Seed|Video|Group|AI count|Body|Raw|ΔAE body−raw|','|---|---|---|---:|---:|---:|---:|'])
    for seed,q in result['seeds'].items():
        for p in q['fold0_held_out']:
            lines.append(f"|{seed}|{p['video_id']}|{p['group_id']}|{p['target']}|{p['predictions']['body_normalized']:.6f}|{p['predictions']['raw58']:.6f}|{p['delta_AE']['raw58']:+.6f}|")
        for group in ('g11','g20'):
            lines.extend(['',f"{seed} {group}: "+str({m:q['results'][m]['held_out']['by_group'][group] for m in METHODS.values()}),''])
    lines.extend(['','Full paired train/held errors, by-group/by-count, prediction distributions and source hashes are in result.json and paired-errors.json.',
        'AI weak reference, adaptive development. Lower development MAE does not validate squat events. '
        'NOT_VALIDATED, official_eligibility=false, assessment=false. No checkpoint/seed winner or final refit.',''])
    return '\n'.join(lines)
