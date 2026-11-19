"""Compare only fields actually saved in the original50epoch control."""
from ..inner_cv.manifest import unseal
from ..models import canonical_hash
from .contract import REPRO_FIELDS


def compare(old,middle,final):
    if old['epoch']!=49 or middle['epoch']!=49 or final['epoch']!=499:
        raise ValueError('Require original49,new49,new499')
    f=old['identity']['fold']
    if middle['identity']['fold']!=f or final['identity']['fold']!=f:raise ValueError('Reproduction fold mismatch')
    if len(old['epoch_reports'])!=50 or len(middle['epoch_reports'])!=50 or len(final['epoch_reports'])!=500:
        raise ValueError('Missing reproduction history')
    if middle['epoch_reports']!=final['epoch_reports'][:50]:raise ValueError('New49 not exact prefix of new499; wrong lineage')
    differences=[];projections=[]
    for e,(a,b) in enumerate(zip(old['epoch_reports'],middle['epoch_reports'])):
        a=unseal(a);b=unseal(b)
        if any(k not in a or k not in b for k in REPRO_FIELDS):raise ValueError('Missing old/new shared history field')
        for k in REPRO_FIELDS:
            if canonical_hash(a[k])!=canonical_hash(b[k]):
                differences.append(dict(epoch_index=e,field=k,old_sha256=canonical_hash(a[k]),new_sha256=canonical_hash(b[k])))
        projections.append(dict(epoch_index=e,old_sha256=canonical_hash({k:a[k] for k in REPRO_FIELDS}),
            new_sha256=canonical_hash({k:b[k] for k in REPRO_FIELDS})))
    import torch
    if set(old['model'])!=set(middle['model']):raise ValueError('Reproduction model layout mismatch')
    tensors=[]
    for k in sorted(old['model']):
        a=old['model'][k];b=middle['model'][k]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Reproduction parameter shape/dtype mismatch')
        tensors.append(dict(name=k,equal=torch.equal(a,b),max_abs_difference=float((a.double()-b.double()).abs().max())))
    old_records=unseal(old['epoch_reports'][-1])['records'];new_records=unseal(middle['epoch_reports'][-1])['records']
    if [r['video_id'] for r in old_records]!=[r['video_id'] for r in new_records]:raise ValueError('Reproduction prediction IDs/order mismatch')
    predictions=[dict(video_id=a['video_id'],target=a['target'],old_prediction=a['prediction'],new_prediction=b['prediction']) for a,b in zip(old_records,new_records)]
    exact=not differences and all(r['equal'] for r in tensors)
    return dict(fold=f,status='EXACT_MATCH' if exact else 'MISMATCH_REQUIRES_INVESTIGATION',identical=exact,
        compared_epochs=50,tolerance='exact; no numeric tolerance',fields=list(REPRO_FIELDS),
        ignored_fields=['manifest_sha256','identity_sha256','elapsed_seconds'],
        not_directly_checked=['old realized batch order: not logged','old gradients/update deltas: not logged','train prediction history: not logged'],
        sampler_evidence='frozen code/config/seed+epoch order unchanged; new batch hashes validated; not a direct old order observation',
        differences=differences,history_projections=projections,tensors=tensors,predictions=predictions,
        old_weights_sha256=old['model_sha256'],new_epoch50_weights_sha256=middle['model_sha256'])
