"""Strict six-clip metrics and telemetry; helpers never update parameters."""
import math
from ..inner_cv.oof import pair_identity,valid_prediction
from ..inner_cv.runtime import state_hash
from ..models import canonical_hash
from .contract import CRITERION,BUDGET


def metrics(manifest,records):
    lookup={r['source']['video_id']:r for r in manifest['rows']};ids=[r['video_id'] for r in records]
    if len(ids)!=6 or len(set(ids))!=6 or set(ids)!=set(manifest['training_ids']):
        raise ValueError('Require exactly six selected training-video predictions')
    errors=[];paired=[]
    for r in records:
        expected=pair_identity(lookup[r['video_id']])
        if (canonical_hash({k:r.get(k) for k in expected})!=canonical_hash(expected)
                or r.get('status')!='estimated' or r.get('reason') is not None
                or not valid_prediction(r.get('prediction'))):raise ValueError('Tiny prediction linkage/target/value mismatch')
        if 'fold' in r and (type(r['fold']) is not int or r['fold']!=1):raise ValueError('Wrong parent fold')
        error=r['prediction']-r['target'];errors.append(error)
        paired.append(dict(r,signed_error=error,absolute_error=abs(error)))
    mae=math.fsum(abs(e) for e in errors)/6;rmse=math.sqrt(math.fsum(e*e for e in errors)/6);maximum=max(abs(e) for e in errors)
    if not all(math.isfinite(v) for v in (mae,rmse,maximum)):raise ValueError('Non-finite tiny metrics')
    return dict(clips=6,MAE=mae,RMSE=rmse,max_absolute_error=maximum,
        signed_error_mean=math.fsum(errors)/6,records=sorted(paired,key=lambda r:r['video_id']),
        scope='same_training_six_in_sample_not_generalization')


def final_criterion(history):
    if len(history)!=BUDGET or any(type(r.get('epoch')) is not int or r['epoch']!=i for i,r in enumerate(history)):
        raise ValueError('Require ordered complete500epochs; never select best/shorten denominator')
    checks=[]
    for r in history:
        q=r['evaluation']
        if q['clips']!=6 or any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) or q[k]<0 for k in ('MAE','RMSE','max_absolute_error')):
            raise ValueError('Invalid criterion metrics')
        checks.append(q['MAE']<=CRITERION['MAE_lte'] and q['max_absolute_error']<=CRITERION['max_absolute_error_lte'])
    last=checks[-20:];met=all(last)
    return dict(criterion=CRITERION,met=met,checked_epoch_indices=list(range(480,500)),
        passing_final_epochs=sum(last),required_final_consecutive_epochs=20,
        conclusion='MET_DIAGNOSTIC_FIT_CONVENTION' if met else 'NOT_FIT_IN_THIS_SETUP_AND_BUDGET',
        independent_evaluation=False)


def gradient_observation(named_parameters,loss_has_gradient_path):
    import torch
    params=[(n,p) for n,p in named_parameters if p.requires_grad]
    missing=[n for n,p in params if p.grad is None]
    present=[(n,p.grad.detach()) for n,p in params if p.grad is not None]
    finite=bool(present) and all(bool(torch.isfinite(g).all()) for _,g in present)
    norm=(float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g,2) for _,g in present]),2)) if finite else None)
    return dict(loss_requires_grad=bool(loss_has_gradient_path),trainable_tensors=len(params),
        tensors_with_gradient=len(present),missing_gradient_names=missing,gradients_finite=finite,
        loss_reaches_all_trainable_parameters=bool(loss_has_gradient_path and params and not missing),
        zero_gradient_tensors=sum(int(torch.count_nonzero(g))==0 for _,g in present),
        observed_preclip_norm=norm)


def parameter_change(before,after):
    import torch
    if set(before)!=set(after):raise ValueError('Parameter keys changed')
    squares=[];maximum=0.;changed=0
    for name,a in before.items():
        b=after[name].detach()
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Non-finite/mismatched parameter state')
        delta=b.double()-a.detach().double();squares.append(float(torch.sum(delta*delta)))
        maximum=max(maximum,float(torch.max(torch.abs(delta))));changed+=int(not torch.equal(a,b))
    return dict(before_sha256=state_hash(before),after_sha256=state_hash(after),
        delta_l2=math.sqrt(math.fsum(squares)),delta_max_abs=maximum,changed_tensors=changed,
        parameter_tensors=len(before),parameters_finite=True)


def step_log(gradient,change,preclip_norm):
    if (not gradient['loss_reaches_all_trainable_parameters'] or not gradient['gradients_finite']
            or type(preclip_norm) not in (int,float) or not math.isfinite(preclip_norm) or preclip_norm<0
            or not math.isclose(preclip_norm,gradient['observed_preclip_norm'],rel_tol=1e-5,abs_tol=1e-7)):
        raise ValueError('Gradient path/finite/norm logging invariant failed')
    return dict(gradient=gradient,update=change,preclip_norm=preclip_norm,clip_threshold=1.0,
        exceeded_clip_threshold=preclip_norm>1.0,steps=1)
