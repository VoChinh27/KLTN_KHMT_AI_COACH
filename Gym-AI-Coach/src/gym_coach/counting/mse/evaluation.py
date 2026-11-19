"""Common metrics at index49; train appearances are not independent clips."""
import math
import statistics
from ..models import STATUS, canonical_hash
from ..inner_cv import runtime as original
from ..inner_cv.manifest import unseal
from ..research.data import identity, check_identity
from .manifest import ENDPOINT, VERSION, fold_identity, diagnostic_tool
from .loss import Config

METHODS=('mse','gru','duration','constant','zero_xy','velocity')


def train_predictions(root,data,manifest,fold,saved,tool):
    import torch
    from ..research.network import setup,CountGRU
    if saved['identity']!=fold_identity(manifest,fold) or saved['epoch']!=49:
        raise ValueError('Train evaluation requires exact MSE/fold/config/final checkpoint')
    config=Config.model_validate_json(__import__('json').dumps(manifest['config']))
    setup(config);model=CountGRU(config);model.load_state_dict(saved['model']);model.eval()
    before=original.state_hash(model.state_dict())
    if before!=saved['model_sha256']:raise ValueError('Loaded weights mismatch')
    ids=manifest['folds'][fold]['train_ids'];dataset=original.subset(root,data,ids)
    if [r['source']['video_id'] for r in dataset.rows]!=ids:raise ValueError('Incorrect train subset')
    records=original.predict(model,dataset)  # frozen eval + inference_mode; no optimizer
    records=[dict(r,fold=fold) for r in records]
    after=original.state_hash(model.state_dict())
    if after!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Weights/gradients changed')
    return tool.validate_records(manifest,fold,'train',records),dict(fold=fold,train_ids=ids,
        model_before_sha256=before,model_after_sha256=after,train_forwards=len(ids),
        model_eval=True,inference_mode=True,optimizer_restored=False)


def common_metrics(tool,records):
    result=tool.breakdown(records)
    result['fixed_checkpoint_MSE']=statistics.mean((r['prediction']-r['target'])**2 for r in records)
    return result


def compare(manifest,train,held,controls,tool):
    if set(train)!=set(range(3)) or set(held)!=set(range(3)):raise ValueError('Require three folds')
    if set(r['method'] for r in controls)!=set(METHODS)-{'mse'} or len(controls)!=495:
        raise ValueError('Require all saved diagnostic controls with full coverage')
    all_records=[];results={}
    for method in METHODS:
        tr={};ev={};folds={}
        for f in range(3):
            tr[f]=tool.validate_records(manifest,f,'train',train[f] if method=='mse' else
                [r for r in controls if r['method']==method and r['fold']==f and r['side']=='train'])
            ev[f]=tool.validate_records(manifest,f,'evaluation',held[f] if method=='mse' else
                [r for r in controls if r['method']==method and r['fold']==f and r['side']=='held_out'])
            folds[str(f)]={s:common_metrics(tool,rs) for s,rs in (('train',tr[f]),('held_out',ev[f]))}
            folds[str(f)]['gap_held_minus_train']=folds[str(f)]['held_out']['MAE']-folds[str(f)]['train']['MAE']
            for side,rs in (('train',tr[f]),('held_out',ev[f])):
                all_records.extend(dict(r,method=method,side=side,signed_error=r['prediction']-r['target'],
                    absolute_error=abs(r['prediction']-r['target'])) for r in rs)
        results[method]=dict(folds=folds,aggregate=tool.aggregate_unique_videos(manifest,tr,ev))
        pooled=[r for rs in ev.values() for r in rs]
        results[method]['held_out']=common_metrics(tool,pooled)
        results[method]['macro_group_MAE']=statistics.mean(s['MAE'] for s in results[method]['held_out']['by_group'].values())
    lookup={(r['method'],r['fold'],r['side'],r['video_id']):r for r in all_records}
    paired=[]
    for r in all_records:
        if r['method']!='mse':continue
        peers={m:lookup[(m,r['fold'],r['side'],r['video_id'])] for m in METHODS}
        paired.append(dict(video_id=r['video_id'],fold=r['fold'],side=r['side'],target=r['target'],
            source_sha256=r['source_sha256'],label_record_hash=r['label_record_hash'],row_sha256=r['row_sha256'],
            predictions={m:q['prediction'] for m,q in peers.items()},
            absolute_errors={m:q['absolute_error'] for m,q in peers.items()},
            delta_AE_mse_minus_gru=r['absolute_error']-peers['gru']['absolute_error'],
            delta_AE_mse_minus_duration=r['absolute_error']-peers['duration']['absolute_error']))
    return dict(version=VERSION,**STATUS,status='COMPLETE',endpoint=ENDPOINT,
        manifest_sha256=canonical_hash(manifest),results=results,records=all_records,paired_records=paired,
        prediction_sha256=canonical_hash(all_records),train_unit='mean two AE per video, then equal mean33; not66independent samples',
        delta_definition='AE_MSE - AE_control; positive means control has smaller absolute error',
        new_baseline_fits=0,outer_validation_used=False,development_test_used=False,final_refit=False,
        limitation='AI weak reference; one seed; OOF used in design; adaptive development, not independent final evaluation. '
                   'Online MSE and SmoothL1 losses are not directly comparable performance metrics.')


def evaluate(root,data,manifest,paths,controls):
    from .runtime import endpoint_reports,load_checkpoint
    reports,sources=endpoint_reports(manifest,paths)
    tool=diagnostic_tool(root,manifest['controls']['diagnostic_tool'])
    train={};held={};checks=[]
    for report in reports:
        f=report['fold'];held[f]=[dict(r,fold=f) for r in report['records']]
    for source in sources:
        f=source['fold'];source.update(identity(root,source['path']))
        saved=load_checkpoint(check_identity(root,source),manifest,f)
        train[f],receipt=train_predictions(root,data,manifest,f,saved,tool);checks.append(receipt)
        check_identity(root,source)
    result=compare(manifest,train,held,controls,tool)
    result.update(checkpoint_sources=sources,inference_receipts=checks,controls=manifest['controls'],
        train_forward_calls=66,held_out_forward_calls=0,new_training_steps=0)
    return result


def report_markdown(result):
    lines=['# Loss-only MSE versus original SmoothL1 — fixed epoch50/index49','',result['limitation'],'',
        'NOT_VALIDATED; official_eligibility=false; assessment=false; G5.7 blocked; G6 IN_PROGRESS.','',
        result['train_unit'],'',result['delta_definition'],'',
        '|Method|Train MAE,33videos|OOF MAE,33videos|OOF macro12groups|Held − train|',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        b=result['results'][m];a=b['aggregate']
        lines.append(f"|{m}|{a['train_video_equal_MAE']:.9f}|{a['held_out_pooled_MAE']:.9f}|{b['macro_group_MAE']:.9f}|{a['gap_held_minus_train']:.9f}|")
    lines+=['','## Fold metrics','',
        '|Method|Fold|Subset|n/g|MAE|Signed mean|Under/over/exact|Pred min/max/mean/popSD|Fixed MSE|Fixed SmoothL1|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for m in METHODS:
        for f,b in result['results'][m]['folds'].items():
            for side in ('train','held_out'):
                s=b[side];p=s['predictions'];d='/'.join(f'{p[k]:.6f}' for k in ('minimum','maximum','mean','population_sd'))
                lines.append(f"|{m}|{f}|{side}|{s['clips']}/{s['groups']}|{s['MAE']:.6f}|{s['signed_error_mean']:.6f}|{s['under']}/{s['over']}/{s['exact']}|{d}|{s['fixed_checkpoint_MSE']:.6f}|{s['fixed_checkpoint_smooth_l1']:.6f}|")
    lines+=['','## Errors by count (including3–5), each fold/subset','',
        '|Method|Fold|Subset|AI count|n|MAE|Signed mean|Under/over|','|---|---|---|---|---|---|---|---|']
    for m in METHODS:
        for f,b in result['results'][m]['folds'].items():
            for side in ('train','held_out'):
                for c,s in b[side]['by_count'].items():
                    lines.append(f"|{m}|{f}|{side}|{c}|{s['clips']}|{s['MAE']:.6f}|{s['signed_error_mean']:.6f}|{s['under']}/{s['over']}|")
    lines+=['','## Paired predictions','',
        '|Video|Fold|Subset|Target|MSE|GRU|Duration|ΔAE vs GRU|ΔAE vs duration|','|---|---|---|---|---|---|---|---|---|']
    for r in result['paired_records']:
        p=r['predictions'];lines.append(f"|{r['video_id']}|{r['fold']}|{r['side']}|{r['target']}|{p['mse']:.6f}|{p['gru']:.6f}|{p['duration']:.6f}|{r['delta_AE_mse_minus_gru']:.6f}|{r['delta_AE_mse_minus_duration']:.6f}|")
    return '\n'.join(lines)+'\n'
