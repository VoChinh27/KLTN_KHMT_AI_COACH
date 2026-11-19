"""Fixed endpoint comparison, exact held replay and equal-video paired metrics."""
import statistics
from ..body_normalized.evaluation import difference_summary
from ..inner_cv_500.evaluation import compare_endpoint
from ..inner_cv.manifest import unseal
from ..mse.manifest import diagnostic_tool
from ..models import STATUS,canonical_hash
from ..research.data import read,identity,check_identity
from ..readiness import file_info
from .contract import SEEDS,VERSION
from . import runtime

METHODS=('fixed_origin','body_normalized','raw58','duration','constant')


def compare(manifest,seed,train,held,controls,tool):
    q=compare_endpoint(manifest,train,held,controls['raw'],tool,499)
    names={'cv500':'fixed_origin','gru':'raw58','duration':'duration','constant':'constant'}
    q['results']={names[k]:v for k,v in q['results'].items()}
    q['records']=[dict(r,method=names[r['method']],seed=seed) for r in q['records']]
    body_controls=[dict(r,method='gru') for r in controls['body']]+[r for r in controls['raw'] if r['method']!='gru']
    b=compare_endpoint(manifest,train,held,body_controls,tool,499)
    q['results']['body_normalized']=b['results']['gru']
    q['records'].extend(dict(r,method='body_normalized',seed=seed) for r in b['records'] if r['method']=='gru')
    lookup={(r['method'],r['side'],r['fold'],r['video_id']):r for r in q['records']};paired=[]
    for r in q['records']:
        if r['method']!='fixed_origin':continue
        peers={m:lookup[m,r['side'],r['fold'],r['video_id']] for m in METHODS}
        paired.append(dict(video_id=r['video_id'],source_sha256=r['source_sha256'],label_record_hash=r['label_record_hash'],
            row_sha256=r['row_sha256'],group_id=r['group_id'],target=r['target'],seed=seed,fold=r['fold'],side=r['side'],
            predictions={m:p['prediction'] for m,p in peers.items()},signed_errors={m:p['signed_error'] for m,p in peers.items()},
            absolute_errors={m:p['absolute_error'] for m,p in peers.items()},
            delta_AE={m:r['absolute_error']-p['absolute_error'] for m,p in peers.items() if m!='fixed_origin'}))
    q['paired_records']=paired
    for f in (0,1):q[f'fold{f}_held_out']=[p for p in paired if p['fold']==f and p['side']=='held_out']
    q['paired_equal_video']={}
    for side,field in (('train','train_mean_AE'),('held_out','held_out_AE')):
        new=q['results']['fixed_origin']['aggregate']['records'];comparisons={}
        for m in METHODS[1:]:
            old={r['video_id']:r for r in q['results'][m]['aggregate']['records']}
            pairs=[dict(video_id=r['video_id'],group_id=r['group_id'],target=r['target'],
                        fixed_origin_AE=r[field],control_AE=old[r['video_id']][field],delta_AE=r[field]-old[r['video_id']][field]) for r in new]
            comparisons[m]=dict(records=pairs,**difference_summary(pairs))
        q['paired_equal_video'][side]=comparisons
    q['prediction_sha256']=canonical_hash(q['records'])
    return q


def directions(deltas):
    signs=[-1 if v<0 else 1 if v>0 else 0 for v in deltas.values()]
    return 'inconsistent_across_seeds' if -1 in signs and 1 in signs else 'same_observed_direction' if len(set(signs))==1 else 'mixed_with_exact_tie'


def combine(seeds):
    if set(seeds)!={str(s) for s in SEEDS}:raise ValueError('Both locked seeds required')
    methods={}
    for m in METHODS:
        lists=[seeds[str(s)]['results'][m]['aggregate']['records'] for s in SEEDS]
        if any(len(rs)!=33 or len({r['video_id'] for r in rs})!=33 for rs in lists):raise ValueError('Require33 unique videos per seed/method')
        a,b=[{r['video_id']:r for r in rs} for rs in lists]
        if set(a)!=set(b):raise ValueError('Cross-seed video mismatch')
        rows=[]
        for v,x in a.items():
            y=b[v]
            if any(x[k]!=y[k] for k in ('target','group_id','held_out_fold','train_folds')):raise ValueError('Cross-seed identity mismatch')
            rows.append(dict(video_id=v,group_id=x['group_id'],target=x['target'],
                train_mean_AE=(x['train_mean_AE']+y['train_mean_AE'])/2,held_out_mean_AE=(x['held_out_AE']+y['held_out_AE'])/2))
        groups={g:statistics.mean(r['held_out_mean_AE'] for r in rows if r['group_id']==g) for g in sorted({r['group_id'] for r in rows})}
        if len(groups)!=12:raise ValueError('Require12groups')
        methods[m]=dict(records=rows,unique_videos=33,seeds_per_video=2,train_MAE=statistics.mean(r['train_mean_AE'] for r in rows),
            OOF_MAE=statistics.mean(r['held_out_mean_AE'] for r in rows),macro_group_MAE=statistics.mean(groups.values()),group_MAE=groups)
    oof={s:q['paired_equal_video']['held_out']['body_normalized']['mean_delta_AE'] for s,q in seeds.items()}
    f1={s:q['results']['fixed_origin']['folds']['1']['held_out']['MAE']-q['results']['body_normalized']['folds']['1']['held_out']['MAE'] for s,q in seeds.items()}
    return dict(methods=methods,per_seed_delta_OOF_MAE=oof,per_seed_delta_fold1_MAE=f1,
        descriptive_result=directions(oof),fold1_direction=directions(f1),
        unit='Mean AE within same video across two seeds then equal mean33; not AE of mean prediction or66independent videos')


def evaluate(root,data,manifest,paths,controls):
    import torch
    if len(paths)!=6:raise ValueError('Require six index499 checkpoints')
    loaded={};links=[]
    for path in paths:
        if file_info(path)!=read(str(path)+'.sha256.json'):raise ValueError('Checkpoint hash mismatch')
        header=torch.load(path,map_location='cpu',weights_only=True)
        seed=header['identity']['seed'];fold=header['identity']['fold']
        engine,selected,_=runtime.engine(manifest,seed);saved=engine.load_checkpoint(path,selected,fold)
        if saved['epoch']!=499 or (seed,fold) in loaded:raise ValueError('Wrong/duplicate endpoint')
        loaded[seed,fold]=saved;links.append(dict(**identity(root,path),seed=seed,fold=fold))
    if set(loaded)!={(s,f) for s in SEEDS for f in range(3)}:raise ValueError('Incomplete checkpoint grid')
    tool=diagnostic_tool(root,manifest['sources']['diagnostic_tool']);results={};receipts=[]
    for seed in SEEDS:
        train={};held={}
        for f in range(3):
            cp=loaded[seed,f]
            train[f],receipt=runtime.train_predictions(root,data,manifest,seed,f,cp);receipts.append(receipt)
            held[f],receipt=runtime.train_predictions(root,data,manifest,seed,f,cp,side='evaluation')
            expected=[dict(r,fold=f) for r in unseal(cp['epoch_reports'][499])['records']]
            if held[f]!=expected:raise ValueError('Exact held-out checkpoint replay mismatch; stop comparison')
            receipts.append(dict(receipt,checkpoint_prediction_replay_exact=True))
        results[str(seed)]=compare(manifest,seed,train,held,controls[str(seed)],tool)
    for link in links:check_identity(root,link)
    return dict(version=VERSION,**STATUS,status='COMPLETE_DEVELOPMENT_ONLY',manifest_sha256=canonical_hash(manifest),
        endpoint_index=499,seeds=results,combined=combine(results),checkpoints=links,inference_receipts=receipts,
        train_forward_calls=132,held_forward_calls=66,training_steps=0,baseline_fits=0,limitation=manifest['limitation'])


def report(result):
    lines=['# Fixed video hip origin — index499','',result['limitation'],
           'NOT_VALIDATED; official_eligibility=false; assessment=false. No seed/checkpoint selection.','',
           '|Seed|Method|Train MAE33|OOF MAE33|Macro12groups|','|---|---|---:|---:|---:|']
    for s,q in result['seeds'].items():
        for m,r in q['results'].items():
            a=r['aggregate'];lines.append(f"|{s}|{m}|{a['train_video_equal_MAE']:.9f}|{a['held_out_pooled_MAE']:.9f}|{r['macro_group_MAE']:.9f}|")
        d=q['paired_equal_video']['held_out']['body_normalized']
        lines+=['',f"{s}: ΔOOF new−body={d['mean_delta_AE']:+.9f}; improved/worsened/equal {d['improved']}/{d['worsened']}/{d['equal']} of33; Δfold1={result['combined']['per_seed_delta_fold1_MAE'][s]:+.9f}."]
    lines+=['','OOF direction: '+result['combined']['descriptive_result'],'Fold1 direction: '+result['combined']['fold1_direction'],
            '',result['combined']['unit'],'|Method|Train|OOF|Macro12|','|---|---:|---:|---:|']
    for m,q in result['combined']['methods'].items():lines.append(f"|{m}|{q['train_MAE']:.9f}|{q['OOF_MAE']:.9f}|{q['macro_group_MAE']:.9f}|")
    lines+=['','|Seed|Method|Fold|Subset|n/groups|MAE|Signed mean|','|---|---|---|---|---|---:|---:|']
    for s,q in result['seeds'].items():
        for m,r in q['results'].items():
            for f,rs in r['folds'].items():
                for side in ('train','held_out'):
                    x=rs[side];lines.append(f"|{s}|{m}|{f}|{side}|{x['clips']}/{x['groups']}|{x['MAE']:.9f}|{x['signed_error_mean']:.9f}|")
    for f in (1,0):
        lines+=['',f'## All13 fold{f} held videos; g17/g19 or g11/g20 retained',
                '|Seed|Video|Group|AI count|New|Body|Raw|Duration|ΔAE new−body|','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for s,q in result['seeds'].items():
            for r in q[f'fold{f}_held_out']:
                p=r['predictions'];lines.append(f"|{s}|{r['video_id']}|{r['group_id']}|{r['target']}|{p['fixed_origin']:.6f}|{p['body_normalized']:.6f}|{p['raw58']:.6f}|{p['duration']:.6f}|{r['delta_AE']['body_normalized']:+.6f}|")
    lines+=['','Full signed errors,12groups/count breakdowns, paired predictions and hashes: result.json / paired-errors.json.',
            'Development evidence only; not a unique cause or validation of squat events. No significance/equivalence threshold.']
    return '\n'.join(lines)+'\n'
