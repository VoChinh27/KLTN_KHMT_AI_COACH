"""Fixed endpoint comparisons; reuse saved controls, no baseline fitting."""
import json
import statistics
from ..inner_cv import manifest as cv
from ..inner_cv_500.evaluation import compare_endpoint as parent_compare
from ..mse.manifest import diagnostic_tool
from ..models import STATUS,canonical_hash
from ..research.data import identity,check_identity
from .contract import VERSION,ENDPOINTS,OBSERVATIONS,LIMITATION
from . import runtime

NEW='seed20261027'
OLD='seed20261026'
METHODS=(NEW,OLD,'duration','constant')
ALIASES=dict(cv500=NEW,gru=OLD,duration='duration',constant='constant')


def observations(result,records,method):
    fold=result['folds']['0']
    selected=[r for r in records if r['method']==method and r['fold']==0 and r['side']=='held_out']
    if len(selected)!=13:raise ValueError('Require all13 fold0 held-out videos')
    groups={g:[r for r in selected if r['group_id']==g] for g in ('g11','g20')}
    if {g:len(rs) for g,rs in groups.items()}!={'g11':3,'g20':4}:
        raise ValueError('Require locked g11/g20 membership')
    gap=fold['held_out']['MAE']-fold['train']['MAE']
    g11=statistics.mean(r['signed_error'] for r in groups['g11'])
    g20=statistics.mean(r['signed_error'] for r in groups['g20'])
    return {OBSERVATIONS[0]:dict(value=gap,condition_holds=gap>0,train_n=20,held_n=13,
                train_MAE=fold['train']['MAE'],held_MAE=fold['held_out']['MAE']),
        OBSERVATIONS[1]:dict(value=g11,condition_holds=g11>0,n=3,videos=groups['g11']),
        OBSERVATIONS[2]:dict(value=g20,condition_holds=g20<0,n=4,videos=groups['g20'])}


def paired_summary(rows):
    return dict(videos=len(rows),mean_delta_AE=statistics.mean(r['delta_AE'] for r in rows),
        new_smaller_AE=sum(r['delta_AE']<0 for r in rows),new_larger_AE=sum(r['delta_AE']>0 for r in rows),
        equal_AE=sum(r['delta_AE']==0 for r in rows))


def compare_endpoint(manifest,train,held,saved_endpoint,tool,epoch):
    if saved_endpoint['epoch_index'] != epoch:raise ValueError('Saved control endpoint mismatch')
    controls=[]
    for r in saved_endpoint['records']:
        if r['method'] in ('cv500','duration','constant'):
            controls.append(dict(r,method='gru' if r['method']=='cv500' else r['method']))
    # The pure metric engine requires exactly297 controls and all99 new appearances.
    q=parent_compare(manifest,train,held,controls,tool,epoch)
    results={ALIASES[k]:v for k,v in q['results'].items()}
    records=[dict(r,method=ALIASES[r['method']]) for r in q['records']]
    lookup={(r['method'],r['fold'],r['side'],r['video_id']):r for r in records}
    pairs=[]
    for r in records:
        if r['method']!=NEW:continue
        peers={m:lookup[(m,r['fold'],r['side'],r['video_id'])] for m in METHODS}
        pairs.append(dict(video_id=r['video_id'],group_id=r['group_id'],fold=r['fold'],side=r['side'],target=r['target'],
            source_sha256=r['source_sha256'],row_sha256=r['row_sha256'],label_record_hash=r['label_record_hash'],
            predictions={m:p['prediction'] for m,p in peers.items()},
            signed_errors={m:p['signed_error'] for m,p in peers.items()},
            absolute_errors={m:p['absolute_error'] for m,p in peers.items()},
            delta_AE={m:r['absolute_error']-peers[m]['absolute_error'] for m in METHODS if m!=NEW}))
    unique={m:{r['video_id']:r for r in s['aggregate']['records']} for m,s in results.items()}
    equal_video={}
    for side,field in (('train','train_mean_AE'),('held_out','held_AE')):
        # aggregate_unique_videos names its held error explicitly; no mean-prediction shortcut.
        if side=='held_out':field='held_out_AE'
        equal_video[side]={}
        for control in METHODS[1:]:
            rows=[]
            for video,a in unique[NEW].items():
                b=unique[control][video]
                if side=='held_out':
                    pa=next(p for p in pairs if p['video_id']==video and p['side']=='held_out')
                    ae_new=pa['absolute_errors'][NEW];ae_control=pa['absolute_errors'][control]
                else:ae_new=a[field];ae_control=b[field]
                rows.append(dict(video_id=video,group_id=a['group_id'],target=a['target'],
                    new_AE=ae_new,control_AE=ae_control,delta_AE=ae_new-ae_control))
            equal_video[side][control]=dict(summary=paired_summary(rows),records=rows)
    obs={m:observations(results[m],records,m) for m in (NEW,OLD)}
    return dict(epoch_index=epoch,role='primary' if epoch==499 else 'descriptive_only',results=results,
        records=records,paired_records=pairs,equal_video_comparisons=equal_video,
        fold0_held_out=[p for p in pairs if p['fold']==0 and p['side']=='held_out'],
        separate_directional_observations=obs,prediction_sha256=canonical_hash(records),
        delta_definition='AE_new_seed - AE_control; positive means control has smaller error')


def evaluate(root,data,manifest,paths50,paths500,controls):
    tool=diagnostic_tool(root,manifest['sources']['diagnostic_tool'])
    loaded={};sources=[];receipts=[]
    for epoch,paths in ((49,paths50),(499,paths500)):
        _,items=runtime.endpoint_reports(manifest,paths,epoch)
        loaded[epoch]={}
        for item in items:
            f=item['fold'];path=item['path']
            loaded[epoch][f]=runtime.load_checkpoint(path,manifest,f)
            sources.append(dict(**identity(root,path),fold=f,epoch=epoch))
    for f in range(3):
        a=loaded[49][f];b=loaded[499][f]
        if a['epoch_reports'] != b['epoch_reports'][:50] or a['model_sha256']!=cv.unseal(b['epoch_reports'][49])['model_sha256']:
            raise ValueError('index49 checkpoint is not on the index499 history lineage')
    endpoints={}
    for epoch in ENDPOINTS:
        train={};held={}
        for f in range(3):
            saved=loaded[epoch][f]
            train[f],receipt=runtime.train_predictions(root,data,manifest,f,saved,tool);receipts.append(receipt)
            held[f]=[dict(r,fold=f) for r in cv.unseal(saved['epoch_reports'][epoch])['records']]
        endpoints[str(epoch)]=compare_endpoint(manifest,train,held,controls['endpoints'][str(epoch)],tool,epoch)
    for link in sources:check_identity(root,link)
    return dict(version=VERSION,**STATUS,status='COMPLETE_NOT_VALIDATION',manifest_sha256=canonical_hash(manifest),
        endpoints=endpoints,checkpoint_sources=sources,inference_receipts=receipts,controls=manifest['sources'],
        train_forward_calls=132,held_out_forward_calls=0,new_training_steps=0,new_baseline_fits=0,
        train_unit='mean two absolute errors per video, then equal mean33; 66 appearances are not independent',
        primary_endpoint=499,descriptive_endpoint=49,seeds=[20261026,20261027],limitation=LIMITATION)


def report_markdown(result):
    lines=['# GRU500 seed repeat — fixed primary index499','',result['limitation'],'',
        'NOT_VALIDATED; official_eligibility=false; assessment=false; G5.7 blocked; G6 IN_PROGRESS.',
        result['train_unit'],'Index49 is descriptive only; no best epoch or seed selection.','',
        '|Index|Method|Train MAE33|OOF MAE33|OOF macro12groups|','|---|---|---:|---:|---:|']
    for e in ('499','49'):
        for m,q in result['endpoints'][e]['results'].items():
            a=q['aggregate'];lines.append(f"|{e}|{m}|{a['train_video_equal_MAE']:.9f}|{a['held_out_pooled_MAE']:.9f}|{q['macro_group_MAE']:.9f}|")
    q=result['endpoints']['499']
    lines+=['','## Three separate directional observations (no combined stability label)','']
    for m,obs in q['separate_directional_observations'].items():
        for key,v in obs.items():lines.append(f"- {m}: {key}: {v['condition_holds']}; magnitude={v['value']:.9f}.")
    lines+=['','## All13 fold0 held-out videos, primary endpoint','',
        '|Video|Group|AI count|New seed|Old seed|Duration|New signed|ΔAE new−old|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for p in q['fold0_held_out']:
        lines.append(f"|{p['video_id']}|{p['group_id']}|{p['target']}|{p['predictions'][NEW]:.6f}|{p['predictions'][OLD]:.6f}|{p['predictions']['duration']:.6f}|{p['signed_errors'][NEW]:.6f}|{p['delta_AE'][OLD]:.6f}|")
    lines+=['','## Paired equal-video comparisons (positive ΔAE favors control)','']
    for side,peers in q['equal_video_comparisons'].items():
        for control,s in peers.items():lines.append(f"- {side} vs {control}: {json.dumps(s['summary'],sort_keys=True)}")
    lines+=['','## Fold/count/group details and distributions','',
        '|Index|Method|Fold|Subset|n/groups|MAE|Signed mean|Under/over/exact|Prediction min/max/mean/popSD|',
        '|---|---|---|---|---|---|---|---|---|']
    for e in ('499','49'):
        for m,res in result['endpoints'][e]['results'].items():
            for f,fold in res['folds'].items():
                for side in ('train','held_out'):
                    s=fold[side];p=s['predictions'];dist='/'.join(f'{p[k]:.6f}' for k in ('minimum','maximum','mean','population_sd'))
                    lines.append(f"|{e}|{m}|{f}|{side}|{s['clips']}/{s['groups']}|{s['MAE']:.6f}|{s['signed_error_mean']:.6f}|{s['under']}/{s['over']}/{s['exact']}|{dist}|")
    lines+=['','Full by-count/by-group metrics, unique-video train errors, all paired predictions and source hashes are in result.json/per-video.json/paired-errors.json.',
        'Training losses are not endpoint MAE. Repeated error directions do not establish a cause; changed directions do not prove seed explains all errors.','']
    return '\n'.join(lines)
