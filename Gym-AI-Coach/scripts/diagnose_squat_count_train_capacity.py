"""Final-checkpoint train-only diagnosis. Never train, restore an optimizer, or refit."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from gym_coach.counting import cv_baseline_comparison as comparison
from gym_coach.counting.inner_cv import manifest as cv, runtime as original_rt
from gym_coach.counting.inner_cv.oof import pair_identity, valid_prediction
from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.readiness import file_info, write_json
from gym_coach.counting.research import baselines
from gym_coach.counting.research.contract import Config
from gym_coach.counting.research.data import read, local, identity, check_identity
from gym_coach.counting.xy_ablation import manifest as xy_m, runtime as xy_rt
from gym_coach.counting.velocity import manifest as v_m, runtime as v_rt, evaluation as v_eval
from gym_coach.counting.velocity.safety import no_learning as existing_guard

NEURAL = ('gru', 'zero_xy', 'velocity')
METHODS = (*NEURAL, 'constant', 'duration')
RUNTIMES = dict(gru=original_rt, zero_xy=xy_rt, velocity=v_rt)
MANIFESTS = dict(gru=cv, zero_xy=xy_m, velocity=v_m)


@contextmanager
def inference_only_guard():
    import torch
    def blocked(*args, **kwargs):
        raise AssertionError('Diagnosis forbids training and optimizer creation/restoration')
    with existing_guard(), patch.object(v_rt, 'train', blocked), \
            patch.object(torch.optim.AdamW, '__init__', blocked), \
            patch.object(torch.optim.Optimizer, 'load_state_dict', blocked):
        yield


def manifest_from_receipt(root, folder):
    receipt = read(folder/'receipt.json'); command = receipt['command']
    if command.count('--manifest') != 1:
        raise ValueError('Manifest receipt ambiguous or missing')
    return local(root, command[command.index('--manifest')+1])


def verify_inventory(folder):
    inventory = read(folder/'inventory.json')
    for name, expected in inventory['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_info(path) != expected:
            raise ValueError('Inventory mismatch: '+name)


def verify_sources(root, runs, gru_oof_dir, baseline_dir):
    """Frozen loaders validate original controls; replay velocity without model forward."""
    folders = {k: local(root, v) for k, v in runs.items()}
    oof_dir, baseline_dir = local(root, gru_oof_dir), local(root, baseline_dir)
    vel_dir = folders['velocity']/'comparison'; zero_dir = folders['zero_xy']/'comparison'
    paths = dict(velocity=manifest_from_receipt(root, vel_dir),
                 zero_xy=manifest_from_receipt(root, zero_dir), gru=manifest_from_receipt(root, oof_dir))
    velocity, data, controls = v_m.load(root, paths['velocity'])
    links = velocity['controls']
    if (identity(root, paths['gru']) != links['original_manifest']
            or identity(root, paths['zero_xy']) != links['zero_xy']['manifest']
            or identity(root, oof_dir/'oof.json') != links['gru_oof']
            or baseline_dir != local(root, links['comparison_dir'])
            or zero_dir != local(root, links['zero_xy']['directory'])):
        raise ValueError('Requested runs differ from verified experiment lineage')
    manifests = dict(velocity=velocity, zero_xy=cv.unseal(read(paths['zero_xy'])), gru=cv.unseal(read(paths['gru'])))
    base = manifests['gru']
    for m in manifests.values():
        if (m['rows'] != base['rows'] or m['folds'] != base['folds'] or m['config'] != base['config']):
            raise ValueError('Cohort/config differ across conditions')
    verify_inventory(vel_dir)
    saved_velocity = cv.unseal(read(vel_dir/'comparison.json'))
    velocity_paths = [check_identity(root, s) for s in saved_velocity['checkpoint_sources']]
    endpoint, sources = v_rt.endpoint_reports(velocity, velocity_paths)
    for s in sources: s.update(identity(root, s['path']))
    replay = v_eval.compare_reports(velocity, endpoint, controls)
    replay.update(checkpoint_sources=sources, controls=velocity['controls'])
    if (canonical_hash(replay) != canonical_hash(saved_velocity)
            or read(vel_dir/'receipt.json')['result'] != saved_velocity
            or cv.unseal(read(vel_dir/'per-video-errors.json')) != replay['records']
            or (vel_dir/'REPORT.md').read_text(encoding='utf-8') != v_eval.report_markdown(replay)):
        raise ValueError('Velocity comparison/receipt/report differs from checkpoint replay')
    oof = cv.unseal(read(oof_dir/'oof.json'))
    saved_sources = dict(gru=oof['checkpoint_sources'], zero_xy=links['zero_xy']['checkpoint_sources'],
                         velocity=sources)
    held = dict(controls, velocity=[dict(r, fold=e['fold']) for e in endpoint for r in e['records']])
    held = {k: comparison.paired_records(base, held[k]) for k in METHODS}
    baseline = cv.unseal(read(baseline_dir/'baselines.json'))
    checkpoints, histories, checkpoint_links = {}, {}, {}
    for method in NEURAL:
        checkpoints[method] = {}; histories[method] = {}; checkpoint_links[method] = {}
        if len(saved_sources[method]) != 3 or sorted(s['fold'] for s in saved_sources[method]) != [0,1,2]:
            raise ValueError('Checkpoint coverage differs from three folds')
        for source in saved_sources[method]:
            fold = source['fold']; path = check_identity(root, source)
            if not path.is_relative_to(folders[method]) or source['epoch'] != 49:
                raise ValueError('Wrong requested run/endpoint')
            man = manifests[method]
            saved = (original_rt.load_checkpoint(path, cv.fold_identity(man, fold)) if method == 'gru'
                     else RUNTIMES[method].load_checkpoint(path, man, fold))
            if saved['epoch'] != 49:
                raise ValueError('Diagnosis permits only final index49')
            run_record = read(path.parent/'run.json')
            complete_record = read(path.parent/'complete.json')
            expected = MANIFESTS[method].fold_identity(man, fold)
            if run_record['identity'] != expected or complete_record['identity'] != expected:
                raise ValueError('Run/complete checkpoint identity mismatch')
            if complete_record['checkpoint'] != file_info(path):
                raise ValueError('Complete record checkpoint bytes mismatch')
            history = []
            for epoch, envelope in enumerate(saved['epoch_reports']):
                report = cv.unseal(envelope)
                value = report.get('training_loss')
                if (report['epoch'] != epoch or type(value) not in (int,float)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError('Invalid saved online training loss/history')
                log_path = path.parent/f'epoch-{epoch:03d}.json'
                pt_path = path.parent/f'epoch-{epoch:03d}.pt'
                if read(log_path) != envelope or file_info(pt_path) != read(str(pt_path)+'.sha256.json'):
                    raise ValueError('Saved per-epoch report/checkpoint byte mismatch')
                history.append(dict(epoch=epoch, training_loss=value, training_steps=report['training_steps'],
                    report=identity(root, log_path), checkpoint=identity(root, pt_path)))
            if len(history) != 50:
                raise ValueError('Require all50saved losses; never silently shorten history')
            checkpoint_records = [dict(r, fold=fold) for r in cv.unseal(saved['epoch_reports'][-1])['records']]
            stored = [r for r in held[method] if r['fold'] == fold]
            if canonical_hash(sorted(checkpoint_records, key=lambda r:r['video_id'])) != canonical_hash(sorted(stored, key=lambda r:r['video_id'])):
                raise ValueError('Held-out predictions differ from final checkpoint')
            checkpoints[method][fold] = saved
            histories[method][str(fold)] = history
            checkpoint_links[method][str(fold)] = identity(root, path)
    evidence = dict(manifests={k:identity(root,p) for k,p in paths.items()},
        manifest_payload_hashes={k:canonical_hash(v) for k,v in manifests.items()},
        checkpoint_links=checkpoint_links, dataset=base['dataset'],
        velocity_comparison=identity(root, vel_dir/'comparison.json'),
        original_oof=identity(root, oof_dir/'oof.json'), baseline=identity(root, baseline_dir/'baselines.json'),
        frozen_control_lineage=links, checkpoint_reports_verified=450,
        losses_per_model_fold=50, held_out_inference_calls=0, endpoint_index=49,
        config=base['config'], conditions={k:manifests[k].get('transform', {'kind':'frozen_original_xy'}) for k in NEURAL})
    return manifests, data, checkpoints, held, baseline, histories, evidence


def validate_records(manifest, fold, side, records):
    if type(fold) is not int or fold not in (0,1,2) or side not in ('train','evaluation'):
        raise ValueError('Invalid fold or subset')
    expected_ids = manifest['folds'][fold][side+'_ids']
    ids = [r.get('video_id') for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Missing/duplicate/unexpected predictions; denominator fixed')
    rows = {r['source']['video_id']:r for r in manifest['rows']}
    for record in records:
        expected = pair_identity(rows[record['video_id']])
        if (canonical_hash({k:record.get(k) for k in expected}) != canonical_hash(expected)
                or type(record.get('fold')) is not int or record['fold'] != fold):
            raise ValueError('Prediction source/target/group/fold linkage mismatch')
        if record.get('status') != 'estimated' or record.get('reason') is not None or not valid_prediction(record.get('prediction')):
            raise ValueError('Invalid/unavailable prediction; no dropping')
    return sorted(records, key=lambda r:r['video_id'])


def infer_train(root, data, manifest, method, fold, checkpoint):
    import torch
    from gym_coach.counting.research.network import CountGRU, setup, collate
    if (method not in NEURAL or manifest.get('version') != MANIFESTS[method].VERSION
            or checkpoint['identity'] != MANIFESTS[method].fold_identity(manifest, fold)
            or checkpoint['epoch'] != 49):
        raise ValueError('Model/transform/fold/endpoint identity mismatch')
    config = Config.model_validate_json(json.dumps(manifest['config']))
    setup(config); model = CountGRU(config)
    model.load_state_dict(checkpoint['model'])  # no optimizer object or optimizer restore
    model.eval()
    before = original_rt.state_hash(model.state_dict())
    if before != checkpoint['model_sha256']:
        raise ValueError('Loaded final weights mismatch')
    ids = manifest['folds'][fold]['train_ids']
    dataset = RUNTIMES[method].subset(root, data, ids)
    if [r['source']['video_id'] for r in dataset.rows] != ids:
        raise ValueError('Inference attempted outside exact ordered train subset')
    records, inputs = [], []
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            sample = dataset[index]; batch = collate([sample]); output = model(batch)
            if model.training or output.requires_grad or not torch.isfinite(output).all():
                raise ValueError('Inference mode/finite output invariant failed')
            records.append(dict(**pair_identity(row), fold=fold, prediction=float(output[0]), status='estimated', reason=None))
            inputs.append(dict(video_id=sample['video_id'], transformed_sample_sha256=canonical_hash(sample),
                               source_cache=row['input_artifact']))
    after = original_rt.state_hash(model.state_dict())
    if after != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Weights or gradients changed in inference-only diagnosis')
    return validate_records(manifest, fold, 'train', records), dict(method=method, fold=fold,
        train_ids=ids, input_samples=inputs, model_before_sha256=before, model_after_sha256=after,
        parameters_unchanged=True, optimizer_instantiated=False, optimizer_restored=False,
        grad_enabled_during_forward=False, inference_mode=True, model_eval=True, model_forward_calls=len(ids))


def describe(values):
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('Descriptive metrics require finite nonempty values')
    return dict(n=len(values), minimum=min(values), maximum=max(values), mean=statistics.mean(values),
                population_sd=statistics.pstdev(values))


def summarize(records):
    if not records:
        raise ValueError('Empty diagnostic subset')
    target=[r['target'] for r in records]; prediction=[r['prediction'] for r in records]
    describe(target); describe(prediction)
    errors=[p-t for p,t in zip(prediction,target)]; absolute=[abs(v) for v in errors]
    return dict(clips=len(records), groups=len({r['group_id'] for r in records}),
        MAE=math.fsum(absolute)/len(records), signed_error_mean=statistics.mean(errors),
        under=sum(v<0 for v in errors), over=sum(v>0 for v in errors), exact=sum(v==0 for v in errors),
        targets=describe(target), target_histogram=dict(sorted(Counter(str(v) for v in target).items())),
        predictions=describe(prediction),
        fixed_checkpoint_smooth_l1=statistics.mean(.5*v*v if v<1 else v-.5 for v in absolute))


def breakdown(records):
    result=summarize(records)
    result['by_count']={str(v):summarize([r for r in records if r['target']==v]) for v in sorted({r['target'] for r in records})}
    result['by_group']={g:summarize([r for r in records if r['group_id']==g]) for g in sorted({r['group_id'] for r in records})}
    result['top5_errors']=[dict(video_id=r['video_id'],target=r['target'],prediction=r['prediction'],
        signed_error=r['prediction']-r['target'],absolute_error=abs(r['prediction']-r['target']))
        for r in sorted(records,key=lambda r:(-abs(r['prediction']-r['target']),r['video_id']))[:5]]
    return result


def aggregate_unique_videos(manifest, train_by_fold, held_by_fold):
    train=[];held=[]
    if set(train_by_fold)!=set(range(3)) or set(held_by_fold)!=set(range(3)):
        raise ValueError('Missing/extra fold for aggregation')
    for f in range(3):
        train.extend(validate_records(manifest,f,'train',train_by_fold[f]))
        held.extend(validate_records(manifest,f,'evaluation',held_by_fold[f]))
    rows=[]
    for source in manifest['rows']:
        v=source['source']['video_id'];tr=[r for r in train if r['video_id']==v];ev=[r for r in held if r['video_id']==v]
        if len(tr)!=2 or len(ev)!=1:
            raise ValueError('Each clip requires2train appearances and1held-out prediction')
        rows.append(dict(video_id=v, group_id=source['source']['group_id'], target=source['target'],
            train_folds=[r['fold'] for r in tr], train_predictions=[r['prediction'] for r in tr],
            train_mean_AE=statistics.mean(abs(r['prediction']-r['target']) for r in tr),
            held_out_fold=ev[0]['fold'],held_out_prediction=ev[0]['prediction'],
            held_out_AE=abs(ev[0]['prediction']-ev[0]['target'])))
    train_mae=statistics.mean(r['train_mean_AE'] for r in rows);held_mae=statistics.mean(r['held_out_AE'] for r in rows)
    return dict(unique_videos=33, train_appearances=66, held_out_predictions=33,
        train_video_equal_MAE=train_mae, held_out_pooled_MAE=held_mae,
        gap_held_minus_train=held_mae-train_mae,records=rows,
        unit='mean of2absolute_errors per video, then equal mean over33videos; not66independent observations, notAE(mean prediction)')


def analyze(root, manifests, data, checkpoints, held, baseline, histories):
    base=manifests['gru'];train={};forward_receipts=[]
    for method in NEURAL:
        train[method]={}
        for fold in range(3):
            train[method][fold],receipt=infer_train(root,data,manifests[method],method,fold,checkpoints[method][fold])
            forward_receipts.append(receipt)
    for method in ('constant','duration'):
        train[method]={}
        for fold in base['folds']:
            f=fold['fold'];origin=baseline['folds'][f];parameters=origin['parameters']
            if origin['train_ids']!=fold['train_ids'] or parameters['training_ids']!=sorted(fold['train_ids']):
                raise ValueError('Wrong fold saved baseline parameters')
            lookup={r['source']['video_id']:r for r in base['rows']};rows=[lookup[v] for v in fold['train_ids']]
            values=baselines.predict(parameters,rows,'train',method)  # saved coefficients only
            records=[dict(**pair_identity(r),fold=f,prediction=values[r['source']['video_id']],status='estimated',reason=None) for r in rows]
            train[method][f]=validate_records(base,f,'train',records)
    results={};all_records=[]
    for method in METHODS:
        held_by_fold={f:[r for r in held[method] if r['fold']==f] for f in range(3)}
        folds={}
        for f in range(3):
            tr=validate_records(base,f,'train',train[method][f]);ev=validate_records(base,f,'evaluation',held_by_fold[f])
            folds[str(f)]=dict(train=breakdown(tr),held_out=breakdown(ev),gap_held_minus_train=summarize(ev)['MAE']-summarize(tr)['MAE'])
            for side,values in (('train',tr),('held_out',ev)):
                all_records.extend(dict(r,method=method,side=side,signed_error=r['prediction']-r['target'],
                    absolute_error=abs(r['prediction']-r['target'])) for r in values)
        results[method]=dict(folds=folds,aggregate=aggregate_unique_videos(base,train[method],held_by_fold))
    trends={}
    for method,folds in histories.items():
        trends[method]={}
        for f,values in folds.items():
            loss=[r['training_loss'] for r in values]
            trends[method][f]=dict(epochs=50,first=loss[0],last=loss[-1],first5_mean=statistics.mean(loss[:5]),
                last5_mean=statistics.mean(loss[-5:]),epoch40_to50_change=loss[-1]-loss[39],
                adjacent_increases=sum(b>a for a,b in zip(loss,loss[1:])),loss_definition='online sample-weighted SmoothL1 beta1 during updates; not final-checkpoint MAE')
    return dict(**STATUS,status='COMPLETE',epoch_index=49,results=results,records=all_records,
        loss_histories=histories,loss_trends=trends,inference_receipts=forward_receipts,
        train_model_forward_calls=sum(r['model_forward_calls'] for r in forward_receipts),held_out_model_forward_calls=0,
        new_training_steps=0,new_baseline_fits=0,optimizer_restorations=0,
        limitation='AI weak reference; one seed; adaptive development OOF used in design; in-sample train predictions are not evidence of generalization.')


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(gru='#176b96',zero_xy='#ca6b19',velocity='#248353')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True,layout='constrained')
    for f,ax in enumerate(axes):
        for method in NEURAL:
            h=result['loss_histories'][method][str(f)]
            ax.plot(range(1,51),[r['training_loss'] for r in h],label=method,color=colors[method])
        ax.set_title(f'Fold {f}: train {20 if f<2 else 26} clips / 8 groups')
        ax.set_xlabel('Epoch (1-based); fixed endpoint50');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Online training SmoothL1 beta1 (not MAE)')
    fig.suptitle('Saved training loss | no training or epoch selection in this diagnosis')
    fig.savefig(out/'training-loss.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,sharey=True,layout='constrained')
    for i,method in enumerate(NEURAL):
        for j,side in enumerate(('train','held_out')):
            ax=axes[i,j]
            for f in range(3):
                rows=[r for r in result['records'] if r['method']==method and r['side']==side and r['fold']==f]
                ax.scatter([r['target'] for r in rows],[r['prediction'] for r in rows],alpha=.65,s=30,label=f'fold{f}, n={len(rows)}')
            ax.plot([0,5],[0,5],'--',color='gray',linewidth=1);ax.set_title(method+' | '+side)
            ax.set_xlabel('AI weak count');ax.set_ylabel('Prediction');ax.set_xlim(-.2,5.3);ax.set_ylim(-.2,5.3);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Final index49 | Train: 66 appearances / 33 unique videos (dependent)\nHeld-out: 33 stored OOF predictions; no held-out inference')
    fig.savefig(out/'train-heldout-predictions.png',dpi=140);plt.close(fig)


def tables(result):
    lines=['# Final index49 — prediction/error records','',result['limitation'],'',
        'Training video appearances repeat across2folds, not independent observations. No sample omitted.','',
        '|Method|Side|Fold|Video|Group|AI target|Prediction|Signed error|Absolute error|',
        '|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in result['records']:
        lines.append(f"|{r['method']}|{r['side']}|{r['fold']}|{r['video_id']}|{r['group_id']}|{r['target']}|{r['prediction']:.9f}|{r['signed_error']:.9f}|{r['absolute_error']:.9f}|")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--gru-run',type=Path,required=True);p.add_argument('--gru-oof-dir',type=Path,required=True)
    p.add_argument('--zero-run',type=Path,required=True);p.add_argument('--velocity-run',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv);out=local(args.root,args.output_dir);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();command=[sys.executable,__file__,*(argv if argv is not None else sys.argv[1:])]
    try:
        with inference_only_guard():
            manifests,data,checkpoints,held,baseline,histories,evidence=verify_sources(args.root,
                dict(gru=args.gru_run,zero_xy=args.zero_run,velocity=args.velocity_run),args.gru_oof_dir,args.baseline_dir)
            result=analyze(args.root,manifests,data,checkpoints,held,baseline,histories)
            for methods in evidence['checkpoint_links'].values():
                for link in methods.values():check_identity(args.root,link)
            write_json(out/'verification.json',cv.sealed(evidence));write_json(out/'diagnosis.json',cv.sealed(result))
            write_json(out/'per-video-errors.json',cv.sealed(result['records']))
            (out/'PER_VIDEO_ERRORS.md').write_text(tables(result),encoding='utf-8')
            plot(result,out)
        from gym_coach.counting.research.cli import environment
        write_json(out/'receipt.json',dict(command=command,seconds=time.perf_counter()-start,environment=environment(),
            status='COMPLETE',train_inference_calls=result['train_model_forward_calls'],held_out_inference_calls=0,
            training_steps=0,baseline_fits=0,optimizer_restorations=0,tool=file_info(__file__)))
        write_json(out/'inventory.json',dict(files={f.name:file_info(f) for f in sorted(out.iterdir()) if f.is_file()}))
        print(json.dumps({m:result['results'][m]['aggregate']|{'records':'see diagnosis.json'} for m in METHODS}))
        return 0
    except (ValueError,KeyError,RuntimeError,OSError,TypeError) as exc:
        write_json(out/'failure.json',dict(status='INCOMPLETE_OR_INVALID',command=command,detail=type(exc).__name__+': '+str(exc)))
        raise


if __name__=='__main__':
    raise SystemExit(main())
