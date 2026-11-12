"""Budget1000 condition; frozen500epoch update loop retained verbatim, separate identity."""
import json
import math
from pathlib import Path
import time
from ..inner_cv import runtime as original
from ..inner_cv.manifest import sealed,unseal
from ..models import STATUS,canonical_hash
from ..readiness import file_info,write_json
from ..research.data import read
from ..research.experiment import save_checkpoint,restore_state
from .contract import Config,BUDGET,CHECKPOINT_VERSION
from .manifest import experiment_identity
from .diagnostics import metrics,gradient_observation,parameter_change,step_log


def subset(root,data,manifest):
    dataset=original.subset(root,data,manifest['training_ids'])
    if len(dataset)!=6 or [r['source']['video_id'] for r in dataset.rows]!=manifest['training_ids']:
        raise ValueError('Tiny runtime may read only ordered six training IDs')
    if dataset.rows!=manifest['rows']:raise ValueError('Tiny dataset/manifest rows differ')
    return dataset


def forward_only(root,data,manifest):
    import torch
    from ..research.network import CountGRU,setup,collate,count_loss
    from ..research.contract import Config as OriginalConfig
    config=Config.model_validate_json(json.dumps(manifest['config']))
    setup(OriginalConfig());reference=CountGRU(OriginalConfig());reference_hash=original.state_hash(reference.state_dict())
    setup(config);model=CountGRU(config);model.eval();before=original.state_hash(model.state_dict())
    dataset=subset(root,data,manifest);batch=collate([dataset[i] for i in range(6)])
    if before!=reference_hash or len(batch['video_ids'])>config.batch_size or not batch['target_mask'].all():
        raise ValueError('Initial state/single batch/target mismatch')
    with torch.inference_mode():output=model(batch);loss=count_loss(output,batch)
    after=original.state_hash(model.state_dict())
    if (not torch.isfinite(output).all() or not torch.isfinite(loss) or output.requires_grad
            or before!=after or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Forward-only invariant failed')
    return dict(kind='UNTRAINED_FORWARD_ONLY_NOT_PERFORMANCE',ids=batch['video_ids'],
        input_shape=list(batch['x'].shape),output_shape=list(output.shape),finite=True,
        initial_model_sha256=before,after_model_sha256=after,original_initial_sha256=reference_hash,
        untrained_SmoothL1=float(loss),batch_size_setting=config.batch_size,actual_batch_size=6,
        batches_per_epoch=1,parameters_unchanged=True,training_steps=0,baseline_fits=0,
        gradient_update_logging='FIXTURE_ONLY_UNTIL_USER_TRAINING')


def checkpoint_payload(identity,model,optimizer,epoch,history,initial_hash,*,fixture=False):
    import torch
    return dict(version=CHECKPOINT_VERSION,identity=identity,kind='synthetic_fixture_no_learning' if fixture else 'user_trained',
        epoch=epoch,training_steps=0 if fixture else epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),initial_model_sha256=initial_hash,
        model_sha256=original.state_hash(model.state_dict()),epoch_reports=history,history_sha256=canonical_hash(history))


def validate_report(manifest,report,epoch,previous_hash,initial_hash,*,fixture=False):
    ident=experiment_identity(manifest)
    if (type(report.get('epoch')) is not int or report['epoch']!=epoch
            or report['identity_sha256']!=canonical_hash(ident) or report['initial_model_sha256']!=initial_hash
            or report['training_steps']!=(0 if fixture else epoch+1) or report['batch_size_actual']!=6):
        raise ValueError('Tiny report identity/epoch/steps mismatch')
    if metrics(manifest,report['records'])!=report['evaluation']:raise ValueError('Tiny metrics differ from six predictions')
    import torch
    ids=manifest['training_ids'];order=torch.randperm(6,generator=torch.Generator().manual_seed(Config().seed+epoch)).tolist()
    if report['batch_order_sha256']!=canonical_hash([ids[i] for i in order]):raise ValueError('Tiny sampler order mismatch')
    value=report['training_loss']
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid training loss')
    obs=report['step_observation'];grad=obs['gradient'];change=obs['update']
    if step_log(grad,change,obs['preclip_norm'])!=obs:raise ValueError('Invalid step observation')
    if (change['before_sha256']!=previous_hash or change['after_sha256']!=report['model_sha256']
            or change['parameters_finite'] is not True or change['parameter_tensors']!=6
            or type(change['changed_tensors']) is not int or not 0<=change['changed_tensors']<=6
            or grad['trainable_tensors']!=6 or grad['tensors_with_gradient']!=6 or grad['missing_gradient_names']):
        raise ValueError('Parameter/gradient logging or hash continuity mismatch')
    for key in ('delta_l2','delta_max_abs'):
        if type(change[key]) not in (int,float) or not math.isfinite(change[key]) or change[key]<0:
            raise ValueError('Invalid update size')


def load_checkpoint(path,manifest,*,allow_fixture=False):
    import torch
    if file_info(path)!=read(str(path)+'.sha256.json'):raise ValueError('Tiny checkpoint byte hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('version')!=CHECKPOINT_VERSION or saved.get('identity')!=experiment_identity(manifest):
        raise ValueError('Tiny checkpoint identity mismatch; no full-CV/MSE/other condition')
    fixture=saved.get('kind')=='synthetic_fixture_no_learning'
    if saved.get('kind') not in ('user_trained','synthetic_fixture_no_learning') or (fixture and not allow_fixture):
        raise ValueError('Require user-trained tiny checkpoint')
    epoch=saved['epoch']
    if type(epoch) is not int or epoch not in range(BUDGET):raise ValueError('Tiny budget/epoch invalid')
    if saved['training_steps']!=(0 if fixture else epoch+1):raise ValueError('Tiny checkpoint steps mismatch')
    history=saved['epoch_reports']
    if len(history)!=epoch+1 or canonical_hash(history)!=saved['history_sha256']:raise ValueError('Tiny checkpoint history mismatch')
    previous=saved['initial_model_sha256']
    for e,envelope in enumerate(history):
        r=unseal(envelope);validate_report(manifest,r,e,previous,saved['initial_model_sha256'],fixture=fixture);previous=r['model_sha256']
    if original.state_hash(saved['model'])!=saved['model_sha256'] or previous!=saved['model_sha256']:
        raise ValueError('Tiny final model hash mismatch')
    # Construction/inspection only, no optimizer update; preserve caller RNG.
    with torch.random.fork_rng(devices=[]):
        reference,optimizer=original.fresh_components(Config())
    if saved['initial_model_sha256']!=original.state_hash(reference.state_dict()):
        raise ValueError('Tiny initialization differs from fresh original GRU')
    reference.load_state_dict(saved['model'],strict=True)
    if saved['optimizer']['param_groups']!=optimizer.state_dict()['param_groups']:
        raise ValueError('Tiny optimizer configuration differs from original AdamW')
    if not all(bool(torch.isfinite(p).all()) for p in reference.parameters()):
        raise ValueError('Tiny checkpoint contains nonfinite weights')
    return saved


def resume_start(saved):
    epoch=saved['epoch']+1
    if epoch>=BUDGET:raise ValueError('Fixed1000epoch budget exhausted; report, do not continue')
    return epoch


def train(root,data,manifest,out,*,user_run=False,resume=None):
    if not user_run:raise PermissionError('Only user may train: --user-run required')
    import torch
    from ..research.network import collate,count_loss
    config=Config.model_validate_json(json.dumps(manifest['config']));ident=experiment_identity(manifest)
    dataset=subset(root,data,manifest);saved=load_checkpoint(resume,manifest) if resume else None
    first=resume_start(saved) if saved else 0
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model,optimizer=original.fresh_components(config);initial_hash=original.state_hash(model.state_dict());history=[];steps=0
    if saved:
        if initial_hash!=saved['initial_model_sha256']:raise ValueError('Fresh initialization differs from resume origin')
        first,steps=restore_state(saved,model,optimizer);history=saved['epoch_reports'][:]
    write_json(out/'run.json',dict(**STATUS,identity=ident,config=config.model_dump(mode='json'),
        initial_model_sha256=initial_hash,resume_checkpoint=str(resume) if resume else None,
        resume_identity=file_info(resume) if resume else None,first_epoch=first,input_audit=manifest['input_audit'],
        scope='same_six_train_only',held_out_used=False,outer_validation_used=False,development_test_used=False))
    started=time.perf_counter()
    for epoch in range(first,BUDGET):
        model.train();order=torch.randperm(6,generator=torch.Generator().manual_seed(config.seed+epoch)).tolist()
        batch=collate([dataset[i] for i in order])  # exactly one full batch; setting8, actual6
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True);loss=count_loss(model(batch),batch)
        has_path=loss.requires_grad and loss.grad_fn is not None
        if not torch.isfinite(loss) or not has_path:raise ValueError('Non-finite loss or missing gradient graph')
        loss.backward()
        gradient=gradient_observation(model.named_parameters(),has_path)
        if not gradient['gradients_finite'] or not gradient['loss_reaches_all_trainable_parameters']:
            raise ValueError('Missing/nonfinite parameter gradients; diagnostic stopped')
        preclip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
        optimizer.step();steps+=1
        change=parameter_change(before,model.state_dict());observation=step_log(gradient,change,preclip)
        records=original.predict(model,dataset)  # eval + inference_mode on same six only
        evaluation=metrics(manifest,records)
        report=dict(epoch=epoch,training_steps=steps,identity_sha256=canonical_hash(ident),
            initial_model_sha256=initial_hash,model_sha256=original.state_hash(model.state_dict()),
            batch_size_actual=6,batch_order_sha256=canonical_hash([manifest['training_ids'][i] for i in order]),
            training_loss=float(loss.detach()),records=records,evaluation=evaluation,step_observation=observation,
            elapsed_segment_seconds=time.perf_counter()-started)
        validate_report(manifest,report,epoch,change['before_sha256'],initial_hash)
        history.append(sealed(report));path=out/f'epoch-{epoch:03d}.pt'
        save_checkpoint(path,checkpoint_payload(ident,model,optimizer,epoch,history,initial_hash))
        write_json(out/f'epoch-{epoch:03d}.json',history[-1])
        print(json.dumps(dict(epoch=epoch,training_loss=report['training_loss'],MAE=evaluation['MAE'],
            RMSE=evaluation['RMSE'],max_absolute_error=evaluation['max_absolute_error'],preclip_norm=preclip)),flush=True)
    write_json(out/'complete.json',dict(**STATUS,identity=ident,checkpoint=file_info(path),final_checkpoint=path.name,
        epochs=1000,training_steps=steps,initial_model_sha256=initial_hash,final_model_sha256=report['model_sha256'],report_pending=True))
