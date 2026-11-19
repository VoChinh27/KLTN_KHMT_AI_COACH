"""Two fixed endpoints, equal-video train MAE, saved controls and exact reproduction."""
import json
import statistics
from ..models import STATUS,canonical_hash
from ..inner_cv import manifest as cv,runtime as original
from ..research.data import identity,check_identity
from .contract import VERSION,Config,ENDPOINTS
from .manifest import fold_identity,diagnostic_tool
from . import runtime,reproduction

METHODS=('cv500','gru','duration','constant')


def train_predictions(root,data,manifest,fold,saved,tool):
    import torch
    from ..research.network import CountGRU,setup
    if saved['identity']!=fold_identity(manifest,fold) or saved['epoch'] not in ENDPOINTS:
        raise ValueError('Train evaluation requires correct fixed endpoint/fold/condition')
    config=Config.model_validate_json(json.dumps(manifest['config']));setup(config);model=CountGRU(config)
    model.load_state_dict(saved['model']);model.eval();before=original.state_hash(model.state_dict())
    if before!=saved['model_sha256']:raise ValueError('Loaded model differs')
    ids=manifest['folds'][fold]['train_ids'];dataset=original.subset(root,data,ids)
    if [r['source']['video_id'] for r in dataset.rows]!=ids:raise ValueError('Incorrect train subset/order')
    records=[dict(r,fold=fold) for r in original.predict(model,dataset)]
    after=original.state_hash(model.state_dict())
    if after!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Evaluation changed weights/gradients')
    return tool.validate_records(manifest,fold,'train',records),dict(fold=fold,epoch=saved['epoch'],train_ids=ids,
        model_before_sha256=before,model_after_sha256=after,train_forwards=len(ids),model_eval=True,
        inference_mode=True,optimizer_instantiated=False,optimizer_restored=False)


def unique_error_groups(rows,key,field):
    return {str(value):dict(unique_videos=len(rs),MAE=statistics.mean(r[field] for r in rs))
        for value in sorted({r[key] for r in rows})
        for rs in [[r for r in rows if r[key]==value]]}


def compare_endpoint(manifest,train,held,controls,tool,epoch):
    if type(epoch) is not int or epoch not in ENDPOINTS or set(train)!={0,1,2} or set(held)!={0,1,2}:
        raise ValueError('Require fixed endpoint and all three folds')
    if len(controls)!=297 or {r['method'] for r in controls}!=set(METHODS)-{'cv500'}:
        raise ValueError('Require all297 saved control appearances')
    records=[];results={}
    for method in METHODS:
        tr={};ev={};folds={}
        for f in range(3):
            tr[f]=tool.validate_records(manifest,f,'train',train[f] if method=='cv500' else
                [r for r in controls if r['method']==method and r['fold']==f and r['side']=='train'])
            ev[f]=tool.validate_records(manifest,f,'evaluation',held[f] if method=='cv500' else
                [r for r in controls if r['method']==method and r['fold']==f and r['side']=='held_out'])
            folds[str(f)]={side:tool.breakdown(rs) for side,rs in (('train',tr[f]),('held_out',ev[f]))}
            folds[str(f)]['gap_held_minus_train']=folds[str(f)]['held_out']['MAE']-folds[str(f)]['train']['MAE']
            for side,rs in (('train',tr[f]),('held_out',ev[f])):
                records.extend(dict(r,method=method,side=side,signed_error=r['prediction']-r['target'],absolute_error=abs(r['prediction']-r['target'])) for r in rs)
        aggregate=tool.aggregate_unique_videos(manifest,tr,ev)
        pooled=tool.breakdown([r for rs in ev.values() for r in rs]);unique=aggregate['records']
        train_groups=unique_error_groups(unique,'group_id','train_mean_AE')
        results[method]=dict(folds=folds,aggregate=aggregate,held_out=pooled,
            train_appearance_descriptions=tool.breakdown([r for rs in tr.values() for r in rs]),
            train_appearance_unit='66 repeated predictions from33videos; descriptive only, not66independent samples',
            macro_group_MAE=statistics.mean(s['MAE'] for s in pooled['by_group'].values()),
            train_macro_group_MAE=statistics.mean(s['MAE'] for s in train_groups.values()),
            train_unique_video_by_group=train_groups,train_unique_video_by_count=unique_error_groups(unique,'target','train_mean_AE'))
    lookup={(r['method'],r['fold'],r['side'],r['video_id']):r for r in records};paired=[]
    for r in records:
        if r['method']!='cv500':continue
        peers={method:lookup[(method,r['fold'],r['side'],r['video_id'])] for method in METHODS}
        paired.append(dict(video_id=r['video_id'],fold=r['fold'],side=r['side'],target=r['target'],
            source_sha256=r['source_sha256'],label_record_hash=r['label_record_hash'],row_sha256=r['row_sha256'],
            predictions={k:v['prediction'] for k,v in peers.items()},absolute_errors={k:v['absolute_error'] for k,v in peers.items()},
            delta_AE_cv500_minus_gru=r['absolute_error']-peers['gru']['absolute_error'],
            delta_AE_cv500_minus_duration=r['absolute_error']-peers['duration']['absolute_error']))
    return dict(epoch_index=epoch,results=results,records=records,paired_records=paired,prediction_sha256=canonical_hash(records))


def trend(a,b):
    dt=b['train_video_equal_MAE']-a['train_video_equal_MAE'];do=b['held_out_pooled_MAE']-a['held_out_pooled_MAE']
    label='train_and_oof_MAE_decreased' if dt<0 and do<0 else 'train_MAE_decreased_OOF_not_decreased' if dt<0 else 'other_pattern'
    return dict(delta_train_MAE=dt,delta_OOF_MAE=do,descriptive_pattern=label,
        limitation='literal changes in common metrics, not significance/equivalence/causality')


def summarize_change(endpoints):
    a=endpoints['49']['results']['cv500'];b=endpoints['499']['results']['cv500']
    folds={}
    for f in ('0','1','2'):
        folds[f]=trend(dict(train_video_equal_MAE=a['folds'][f]['train']['MAE'],held_out_pooled_MAE=a['folds'][f]['held_out']['MAE']),
            dict(train_video_equal_MAE=b['folds'][f]['train']['MAE'],held_out_pooled_MAE=b['folds'][f]['held_out']['MAE']))
    return dict(aggregate=trend(a['aggregate'],b['aggregate']),folds=folds,
        fold_patterns_differ=len({v['descriptive_pattern'] for v in folds.values()})>1)


def evaluate(root,data,manifest,paths50,paths500,controls):
    tool=diagnostic_tool(root,manifest['controls']['diagnostic_tool'])
    _,s50=runtime.endpoint_reports(manifest,paths50,49);_,s500=runtime.endpoint_reports(manifest,paths500,499)
    old_manifest,_=cv.load(root,check_identity(root,manifest['controls']['original_manifest']))
    loaded={};checks=[];reproductions=[];sources=[]
    for epoch,items in ((49,s50),(499,s500)):
        loaded[epoch]={}
        for item in items:
            f=item['fold'];path=item['path'];saved=runtime.load_checkpoint(path,manifest,f)
            loaded[epoch][f]=saved;sources.append(dict(**identity(root,path),fold=f,epoch=epoch))
    for f in range(3):
        old_link=manifest['controls']['verified_lineage']['checkpoint_links']['gru'][str(f)]
        old=original.load_checkpoint(check_identity(root,old_link),cv.fold_identity(old_manifest,f))
        reproductions.append(reproduction.compare(old,loaded[49][f],loaded[499][f]))
    endpoints={}
    for epoch in ENDPOINTS:
        train={};held={}
        for f in range(3):
            saved=loaded[epoch][f]
            train[f],receipt=train_predictions(root,data,manifest,f,saved,tool);checks.append(receipt)
            held[f]=[dict(r,fold=f) for r in cv.unseal(saved['epoch_reports'][epoch])['records']]
        endpoints[str(epoch)]=compare_endpoint(manifest,train,held,controls,tool,epoch)
    for source in sources:check_identity(root,source)
    exact=all(r['identical'] for r in reproductions)
    return dict(version=VERSION,**STATUS,status='COMPLETE_NOT_VALIDATION' if exact else 'REPRODUCTION_MISMATCH_REQUIRES_INVESTIGATION',
        manifest_sha256=canonical_hash(manifest),endpoints=endpoints,reproducibility=dict(identical=exact,folds=reproductions),
        change=summarize_change(endpoints),checkpoint_sources=sources,inference_receipts=checks,controls=manifest['controls'],
        train_forward_calls=132,held_out_forward_calls=0,new_training_steps=0,new_baseline_fits=0,
        train_unit='mean two absolute errors per video then equal mean33; not66independent samples',
        endpoint_selection='fixed49/499; no minimum/fold-best/early-stop/refit',
        limitation='AI weak reference, one seed, OOF used in design; adaptive development, not independent final evaluation/generalization/form validation.')


def report_markdown(result):
    lines=['# GRU500epoch inner-CV — fixed endpoints','',result['limitation'],'',
        'NOT_VALIDATED; official_eligibility=false; assessment=false; G5.7blocked; G6 IN_PROGRESS.',
        result['train_unit'],'',
        f"Reproduction first50epochs per fold: {'EXACT_MATCH' if result['reproducibility']['identical'] else 'MISMATCH_REQUIRES_INVESTIGATION'}.",
        'Original history did not log realized batch order, gradients/update deltas or train predictions. Do not claim direct checks of those fields.',
        'Mismatch requires investigation before attributing differences to budget; do not alter tolerance or retrain automatically.','',
        '|Endpoint|Method|Train MAE33|OOF MAE33|Train macro12groups|OOF macro12groups|',
        '|---|---|---:|---:|---:|---:|']
    for e in ('49','499'):
        for method,q in result['endpoints'][e]['results'].items():
            a=q['aggregate'];lines.append(f"|{int(e)+1}({e})|{method}|{a['train_video_equal_MAE']:.9f}|{a['held_out_pooled_MAE']:.9f}|{q['train_macro_group_MAE']:.9f}|{q['macro_group_MAE']:.9f}|")
    lines+=['','## Descriptive changes500 minus50','',json.dumps(result['change'],ensure_ascii=False,indent=2),'',
        'Not a statistical significance/equivalence or causal conclusion. A lower train MAE need not imply lower OOF MAE.','',
        '|Endpoint|Method|Fold|Subset|n/groups|MAE|Signed mean|Under/over/exact|Pred min/max/mean/popSD|',
        '|---|---|---|---|---|---|---|---|---|']
    for e in ('49','499'):
        for method,q in result['endpoints'][e]['results'].items():
            for f,fold in q['folds'].items():
                for side in ('train','held_out'):
                    s=fold[side];p=s['predictions'];dist='/'.join(f'{p[k]:.6f}' for k in ('minimum','maximum','mean','population_sd'))
                    lines.append(f"|{e}|{method}|{f}|{side}|{s['clips']}/{s['groups']}|{s['MAE']:.6f}|{s['signed_error_mean']:.6f}|{s['under']}/{s['over']}/{s['exact']}|{dist}|")
    lines+=['','## Errors by count, including0 and3–5','',
        '|Endpoint|Method|Fold|Subset|AI count|n|MAE|Signed mean|','|---|---|---|---|---|---|---|---|']
    for e in ('49','499'):
        for method,q in result['endpoints'][e]['results'].items():
            for f,fold in q['folds'].items():
                for side in ('train','held_out'):
                    for count,s in fold[side]['by_count'].items():
                        lines.append(f"|{e}|{method}|{f}|{side}|{count}|{s['clips']}|{s['MAE']:.6f}|{s['signed_error_mean']:.6f}|")
    lines+=['','See per-video.json and paired-errors.json for full paired source/target/hash-linked values and ΔAE versus original GRU/duration.',
        'Positive ΔAE means the control has smaller absolute error. Controls are fixed saved50epoch GRU and saved fold-fitted baselines; no baseline refit.',
        'No automatic model promotion, final refit or demo integration.','']
    return '\n'.join(lines)
