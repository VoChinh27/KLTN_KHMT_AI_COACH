"""Read-only first-experiment diagnosis. No fitting or development-test inference."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

from gym_coach.counting.models import canonical_hash, STATUS
from gym_coach.counting.readiness import file_info, write_json
from gym_coach.counting.research.contract import Config, JOINTS
from gym_coach.counting.research.data import read, load_dataset, CountDataset, encode, verify_inventory
from gym_coach.counting.research.evaluation import evaluate, rounded_count
from gym_coach.counting.research.experiment import experiment_identity, load_checkpoint

XY = [7*j+k for j in range(8) for k in (0,1)]
METHODS = ('freeze_xy', 'reverse_xy', 'shuffle_xy', 'elapsed_zero',
           'duration_one_second', 'time_scale_two', 'duplicate_reset_fixed_time_duration')


def describe(values):
    xs=[x for x in values if x is not None]
    return dict(n=len(xs),null=len(values)-len(xs),minimum=min(xs) if xs else None,
        maximum=max(xs) if xs else None,mean=statistics.mean(xs) if xs else None,
        median=statistics.median(xs) if xs else None,population_sd=statistics.pstdev(xs) if xs else None)


def distribution(rows):
    return dict(clips=len(rows),groups=dict(sorted(Counter(r['source']['group_id'] for r in rows).items())),
        counts=dict(sorted(Counter('null' if r['target'] is None else str(r['target']) for r in rows).items())),
        count_summary=describe([r['target'] for r in rows]),
        duration_seconds=describe([r['source']['frame_count']/r['source']['fps'] for r in rows]),
        reasons=dict(Counter(f['code'] for r in rows for f in r['exclusion_reasons'])))


def valid_runs(mask):
    runs=[];current=[]
    for i,valid in enumerate(mask):
        if valid:current.append(i)
        elif current:runs.append(current);current=[]
    if current:runs.append(current)
    return runs


def perturb(sample,method):
    """OOD diagnostic input only. Never overwrite a cache, source, or target."""
    out=copy.deepcopy(sample)
    if method in ('freeze_xy','reverse_xy','shuffle_xy'):
        first=next(i for i,v in enumerate(sample['complete']) if v)
        for run in valid_runs(sample['complete']):
            indices=list(run)
            if method=='freeze_xy':indices=[first]*len(run)
            elif method=='reverse_xy':indices.reverse()
            else:random.Random(20261027+run[0]).shuffle(indices)
            for dst,src in zip(run,indices):
                for channel in XY:out['values'][dst][channel]=sample['values'][src][channel]
    elif method=='elapsed_zero':
        for value in out['values']:value[-2]=0.0
    elif method=='duration_one_second':out['duration_seconds']=1.0
    elif method=='time_scale_two':
        for value in out['values']:value[-2]*=2;value[-1]*=2
        out['duration_seconds']*=2
    elif method=='duplicate_reset_fixed_time_duration':
        # Structural probe: replay exactly the same hidden-state run with a reset.
        # Time channels and duration intentionally inconsistent; not a real clip.
        out['values']=out['values']+[[0.0]*len(out['values'][0])]+copy.deepcopy(out['values'])
        out['complete']=out['complete']+[False]+out['complete']
    else:raise ValueError('Unknown diagnostic perturbation')
    return out


def verify_report(rows,report,role):
    predicted={r['video_id']:r['prediction'] for r in report['records'] if r['prediction'] is not None}
    result=evaluate(rows,predicted,role)
    if result!=report:raise ValueError('Saved report disagrees with evaluator/source/target linkage')
    return result


def infer(model,samples,batch_size=8):
    import torch
    from gym_coach.counting.research.network import collate
    result={}
    with torch.inference_mode():
        for start in range(0,len(samples),batch_size):
            part=samples[start:start+batch_size]
            values=model(collate(part))
            if values.requires_grad or not torch.isfinite(values).all():raise ValueError('Invalid forward-only result')
            result.update(zip([s['video_id'] for s in part],map(float,values)))
    return result


def parameter_equal(a,b):
    import torch
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def forbid_learning():
    import torch
    import gym_coach.counting.research.baselines as baseline
    import gym_coach.counting.research.experiment as experiment
    def blocked(*args,**kwargs):raise AssertionError('Diagnostic cannot learn parameters')
    torch.Tensor.backward=blocked
    torch.autograd.backward=blocked
    torch.optim.AdamW.step=blocked
    baseline.fit_baselines=blocked
    experiment.train=blocked
    experiment.run_baselines=blocked
    torch.set_grad_enabled(False)


def execute(root,prepared,training,baseline,baseline_test,model_test,out):
    import torch
    from gym_coach.counting.research.network import setup,CountGRU,collate,count_loss
    from gym_coach.pose.models import PoseFrame
    started=time.perf_counter();forbid_learning();setup()
    paths=[prepared,training,baseline,baseline_test,model_test]
    before={p.relative_to(root).as_posix():file_info(p) for folder in paths for p in sorted(folder.rglob('*')) if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',dict(version='squat-count-diagnosis-v1',**STATUS,
        source_files=before,diagnostic_script=file_info(__file__),
        forward_roles=['train','validation'],epochs='all 50 stored model states; no new checkpoint selection',
        perturbation_states=['saved best epoch','scheduled final epoch'],perturbations=METHODS,
        perturbation_scope='Supervised train/validation only; OOD, not causal evidence or accuracy',
        development_test='Saved JSON verification only; already exposed; never infer alternate checkpoints',
        training=False,baseline_fit=False,source_mutation=False))
    verify_inventory(root,prepared/'artifact-inventory.json')
    dataset_path=prepared/'dataset-v1/dataset.json'
    data=load_dataset(root,dataset_path);rows=data['rows'];identity=experiment_identity(dataset_path,Config())
    if read(training/'experiment.json')['identity']!=identity:raise ValueError('Training/code/data/environment identity mismatch')
    selection=read(training/'selection.json')
    if file_info(training/selection['checkpoint'])!=selection['checkpoint_identity']:raise ValueError('Selection/checkpoint mismatch')
    baseline_envelope=read(baseline/'baselines.json');baseline_payload=baseline_envelope['payload']
    if canonical_hash(baseline_payload)!=baseline_envelope['sha256'] or baseline_payload['identity']!=identity:
        raise ValueError('Baseline identity mismatch')
    expected_train=sorted((r['source']['video_id'],r['source']['sha256'],r['target']) for r in rows if r['role']=='train' and r['supervised_eligible'])
    if (baseline_payload['parameters']['training_subset_hash']!=canonical_hash(expected_train)
            or baseline_payload['parameters']['training_ids']!=[r[0] for r in expected_train]):
        raise ValueError('Baseline trained subset mismatch')
    summaries={role:{name:distribution([r for r in rows if r['role']==role and predicate(r)]) for name,predicate in (
        ('all',lambda r:True),('accepted',lambda r:r['input_status']=='accepted'),
        ('rejected',lambda r:r['input_status']=='rejected'),('supervised',lambda r:r['supervised_eligible']))}
        for role in ('train','validation')}
    by_group={}
    for group in sorted({r['source']['group_id'] for r in rows if r['role'] in ('train','validation')}):
        group_rows=[r for r in rows if r['source']['group_id']==group]
        by_group[group]={name:distribution([r for r in group_rows if predicate(r)]) for name,predicate in
            (('all',lambda r:True),('supervised',lambda r:r['supervised_eligible']),('rejected',lambda r:r['input_status']=='rejected'))}
    write_json(out/'distributions.json',dict(by_split=summaries,by_group=by_group,
        train_validation_records=[dict(video_id=r['source']['video_id'],group=r['source']['group_id'],role=r['role'],
            target=r['target'],duration=r['source']['frame_count']/r['source']['fps'],input_status=r['input_status'],
            supervised=r['supervised_eligible'],reasons=r['exclusion_reasons'],source_sha256=r['source']['sha256'])
            for r in rows if r['role'] in ('train','validation')]))
    saved_results={}
    for role in ('train','validation'):
        for method in ('constant','duration'):
            saved_results[f'{role}-{method}']=verify_report(rows,read(baseline/f'{role}-{method}.json'),role)
    for name,folder in (('constant',baseline_test),('duration',baseline_test),('model',model_test)):
        value=read(folder/(name+'.json'))
        if value['identity']!=identity:raise ValueError('Test result experiment mismatch')
        if name=='model':
            if value['checkpoint']!=selection['checkpoint_identity'] or value['best_epoch']!=selection['best_epoch']:
                raise ValueError('Test result selection mismatch')
        elif value['baseline']!=file_info(baseline/'baselines.json'):raise ValueError('Test baseline mismatch')
        saved_results['development_test-'+name]=verify_report(rows,value['result'],'development_test')
    write_json(out/'verified-saved-results.json',saved_results)
    samples={}
    raw_checks=[]
    for role in ('train','validation'):
        ds=CountDataset(root,data,role,supervised=False);samples[role]=[ds[i] for i in range(len(ds))]
        for row in ds.rows:
            cache=read(root/row['input_artifact']['path'])
            links=[link for link in row['source_links'] if Path(link['path']).name in ('raw-pose-unknown.jsonl','raw-pose.jsonl')]
            if not links:links=[link for link in row['source_links'] if link['path'].endswith('/pose/frames.jsonl')]
            if len(links)!=1:raise ValueError('Ambiguous raw-pose linkage')
            raw_path=root/links[0]['path']
            fs=[PoseFrame.model_validate_json(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
            rebuilt=encode(fs,row['source'],Config(),cache['original_quality_report']['flags'])
            if any(rebuilt[k]!=cache[k] for k in ('values','complete','duration_seconds','stats','status','reasons')):
                raise ValueError('Encoded raw input differs from cache')
            runs=valid_runs(cache['complete'])
            raw_checks.append(dict(video_id=row['source']['video_id'],role=role,supervised=row['supervised_eligible'],
                frames=len(fs),complete=sum(cache['complete']),runs=len(runs),longest_run=max(map(len,runs)),
                incomplete_frames=len(fs)-sum(cache['complete']),raw_link=links[0],recomputed_equal=True))
    write_json(out/'input-checks.json',raw_checks)
    curve=[];best_state=None;best_score=None;best_epoch=None;state_cache={};max_replay_error=0.
    for epoch in range(Config().epochs):
        checkpoint=load_checkpoint(training/f'epoch-{epoch:03d}.pt',identity)
        log=read(training/f'epoch-{epoch:03d}.json')
        if checkpoint['epoch']!=epoch or checkpoint['training_steps']!=(epoch+1)*5:
            raise ValueError('Epoch/step identity mismatch')
        score=verify_report(rows,log['validation'],'validation')['continuous_MAE']
        if best_score is None or score<best_score:best_score,best_epoch,best_state=score,epoch,checkpoint['model']
        if (checkpoint['best_epoch']!=best_epoch or checkpoint['best_score']!=best_score
                or log['best_epoch']!=best_epoch or not parameter_equal(checkpoint['best_model'],best_state)):
            raise ValueError('Checkpoint best state/history mismatch')
        model=CountGRU();model.load_state_dict(checkpoint['model']);model.eval()
        record=dict(epoch_index=epoch,epoch_number=epoch+1,logged_training_loss=log['training_loss'],
                    elapsed_training_seconds=log['elapsed_seconds'],best_epoch_so_far=best_epoch,roles={})
        for role in ('train','validation'):
            predictions=infer(model,samples[role]);report=evaluate(rows,predictions,role)
            known=[r for r in report['records'] if r['comparable']]
            with torch.inference_mode():
                part=[s for s in samples[role] if s['target'] is not None];batch=collate(part)
                loss=float(count_loss(model(batch),batch))
            record['roles'][role]=dict(result=report,prediction_distribution=describe([r['prediction'] for r in known]),
                rounded_distribution=dict(Counter(str(r['rounded_prediction']) for r in known)),
                inference_full_split_loss=loss)
            if role=='validation':
                old={r['video_id']:r['prediction'] for r in log['validation']['records'] if r['comparable']}
                difference=max(abs(predictions[k]-v) for k,v in old.items());max_replay_error=max(max_replay_error,difference)
                if difference>1e-5:raise ValueError('Validation checkpoint replay mismatch')
        if not parameter_equal(model.state_dict(),checkpoint['model']) or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Inference mutated model/created gradient')
        if epoch in (selection['best_epoch'],Config().epochs-1):state_cache[epoch]=checkpoint['model']
        curve.append(record)
        if epoch%10==0:print(json.dumps(dict(verified_epoch=epoch,roles=['train','validation'])),flush=True)
    if best_epoch!=selection['best_epoch'] or best_score!=selection['best_validation_MAE']:
        raise ValueError('Selection not earliest validation minimum')
    write_json(out/'epoch-diagnostics.json',curve)
    perturbations=[];implementation_checks=[]
    for epoch,state in state_cache.items():
        model=CountGRU();model.load_state_dict(state);model.eval()
        for role in ('train','validation'):
            cohort=[s for s in samples[role] if s['target'] is not None]
            original=infer(model,cohort)
            for method in METHODS:
                modified=[perturb(s,method) for s in cohort];predictions=infer(model,modified)
                deltas=[predictions[s['video_id']]-original[s['video_id']] for s in cohort]
                perturbations.append(dict(epoch_index=epoch,role=role,method=method,scope='diagnostic_OOD_not_accuracy_or_causality',
                    signed_delta=describe(deltas),absolute_delta=describe(list(map(abs,deltas))),
                    records=[dict(video_id=s['video_id'],original=original[s['video_id']],modified=predictions[s['video_id']],
                                  delta=predictions[s['video_id']]-original[s['video_id']]) for s in cohort]))
            single=infer(model,cohort,batch_size=1)
            batch_error=max(abs(single[k]-v) for k,v in original.items())
            batch=collate(cohort);heads=[];states=[]
            head_hook=model.head.register_forward_pre_hook(lambda module,args:heads.append(args[0].clone()))
            state_hook=model.cell.register_forward_pre_hook(lambda module,args:states.append(args[1].clone()))
            with torch.inference_mode():
                expected=model(batch)
                reset_error=max((float(states[t][~batch['complete'][:,t-1]].abs().max())
                                 for t in range(1,len(states)) if (~batch['complete'][:,t-1]).any()),default=0.)
                reset_cells=sum(int((~batch['complete'][:,t-1]).sum()) for t in range(1,len(states)))
                preactivation=model.head(heads[0]).squeeze(1)
                head_duration=model.head.weight[0,-1]*batch['duration']
                exact=torch.nn.functional.softplus(preactivation+model.head.weight[0,-1]*(1-batch['duration']))
                actual=model(collate([perturb(s,'duration_one_second') for s in cohort]))
                duration_error=float((actual-exact).abs().max())
            head_hook.remove();state_hook.remove()
            padded=copy.deepcopy(batch)
            for i,n in enumerate(batch['lengths']):padded['x'][i,int(n):]=12345.
            with torch.inference_mode():padding_error=float((model(padded)-expected).abs().max())
            if max(batch_error,duration_error,padding_error)>1e-5 or reset_error!=0:raise ValueError('Implementation probe failed')
            implementation_checks.append(dict(epoch_index=epoch,role=role,batch_single_max_error=batch_error,
                altered_padding_max_error=padding_error,reset_prestate_max_abs=reset_error,
                reset_cells_including_padding=reset_cells,duration_formula_max_error=duration_error,
                duration_head_weight=float(model.head.weight[0,-1].detach()),bias=float(model.head.bias[0].detach()),
                duration_logit=describe(head_duration.tolist()),hidden_logit=describe((preactivation-head_duration-model.head.bias[0]).tolist())))
        if not parameter_equal(model.state_dict(),state):raise ValueError('Probe mutated checkpoint weights')
    write_json(out/'perturbations.json',perturbations)
    write_json(out/'implementation-checks.json',implementation_checks)
    if any(file_info(root/p)!=info for p,info in before.items()):raise ValueError('Original artifacts changed')
    write_json(out/'receipt.json',dict(**STATUS,script=file_info(__file__),command=sys.argv,source_files_verified=len(before),
        epoch_checkpoints_verified=50,validation_replay_max_absolute_error=max_replay_error,
        raw_cache_reencoded=len(raw_checks),train_validation_inferred_epochs=50,
        perturbation_records=len(perturbations),development_test_inference=0,
        training_runs=0,backward_calls=0,optimizer_steps=0,baseline_fits=0,
        source_files_unchanged=True,checkpoint_selection_unchanged=True,seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();root=Path.cwd();base=root/'outputs/squat_count_experiment'
    execute(root,base/'run-6d45f415095746f9add7fa32044f72b5',
        base/'user-training-d34e179493d74e07ab3a7b13e2254aff',base/'user-baseline-d34e179493d74e07ab3a7b13e2254aff',
        base/'baseline-test-71b2e1ac019d46d1aa9a7bc4c41bea3d',base/'model-test-71b2e1ac019d46d1aa9a7bc4c41bea3d',
        args.output_dir.resolve())


if __name__=='__main__':main()
