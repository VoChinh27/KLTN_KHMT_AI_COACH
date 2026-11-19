"""Reuse the frozen CV500 loop with only an input wrapper and explicit condition identity."""
import copy
from types import SimpleNamespace
from ..inner_cv_500 import runtime as parent
from ..inner_cv import runtime as original
from ..inner_cv_500_seed.binding import bind
from ..models import canonical_hash
from .contract import CHECKPOINT_VERSION,ENDPOINT,SEEDS,config_type
from .manifest import select,fold_identity
from .transform import BodyDataset

# Capture before read-only guards patch public training entry points.
TEMPLATE=SimpleNamespace(**vars(parent))


def engine(manifest,seed):
    selected=select(manifest,seed)
    def identity_for_seed(value,fold):
        if value['seed']!=seed:
            raise ValueError('Checkpoint seed differs from selected engine')
        return fold_identity(value,fold)
    runtime,namespace=bind(TEMPLATE,dict(Config=config_type(seed),CHECKPOINT_VERSION=CHECKPOINT_VERSION,
        ENDPOINT=ENDPOINT,fold_identity=identity_for_seed))
    def subset(root,data,ids):
        return BodyDataset(original.subset(root,data,ids),manifest['input_audit'])
    # bind() clones module-local functions. Replace the one data seam explicitly afterwards.
    namespace['subset']=subset
    runtime.subset=subset
    return runtime,selected,namespace


def train(root,data,manifest,seed,fold,out,*,user_run=False,resume=None):
    if not user_run:
        raise PermissionError('User-run training requires --user-run')
    runtime,selected,_=engine(manifest,seed)
    return runtime.train(root,data,selected,fold,out,user_run=True,resume=resume)


def forward_only(root,data,manifest):
    import torch
    from ..research.network import CountGRU,setup,collate,count_loss
    from ..inner_cv_500.contract import Config as Raw26
    from ..inner_cv_500_seed.contract import Config as Raw27
    checks=[]
    for seed in SEEDS:
        config=config_type(seed)()
        runtime,selected,_=engine(manifest,seed)
        for fold in manifest['folds']:
            f=fold['fold']
            raw_config=(Raw26 if seed==SEEDS[0] else Raw27)()
            setup(raw_config)
            reference=CountGRU(raw_config)
            setup(config)
            model=CountGRU(config)
            model.eval()
            before=original.state_hash(model.state_dict())
            expected=manifest['sources']['controls'][str(seed)]['initializations'][str(f)]
            if before!=original.state_hash(reference.state_dict()) or before!=expected:
                raise ValueError('Body initialization differs from same-seed raw control')
            for side in ('train','evaluation'):
                dataset=runtime.subset(root,data,fold[side+'_ids'])
                losses=[];hashes=[]
                for i in range(len(dataset)):
                    sample=dataset[i]
                    batch=collate([sample])
                    with torch.inference_mode():
                        output=model(batch)
                        loss=count_loss(output,batch)
                    if not torch.isfinite(output).all() or not torch.isfinite(loss):
                        raise ValueError('Nonfinite preflight forward/loss')
                    losses.append(float(loss))
                    hashes.append(dict(video_id=sample['video_id'],sha256=canonical_hash(sample)))
                after=original.state_hash(model.state_dict())
                if after!=before or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Forward changed weights or gradients')
                checks.append(dict(seed=seed,fold=f,side=side,videos=len(dataset),input_hashes=hashes,
                    initial_model_sha256=before,model_after_sha256=after,raw_initialization_exact=True,
                    loss_min=min(losses),loss_max=max(losses),input_size=58,output_size=1))
    return dict(status='PREFLIGHT_ONLY_UNTRAINED',checks=checks,unique_videos=33,
        forward_calls=sum(c['videos'] for c in checks),training_steps=0,baseline_fits=0)


def train_predictions(root,data,manifest,seed,fold,checkpoint):
    from ..research.network import setup,CountGRU
    runtime,selected,_=engine(manifest,seed)
    if checkpoint['identity']!=fold_identity(selected,fold) or checkpoint['epoch']!=499:
        raise ValueError('Train inference requires matching condition/fold/index499')
    config=config_type(seed)()
    setup(config);model=CountGRU(config)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    before=original.state_hash(model.state_dict())
    if before!=checkpoint['model_sha256']:
        raise ValueError('Loaded model hash mismatch')
    dataset=runtime.subset(root,data,manifest['folds'][fold]['train_ids'])
    records=[dict(r,fold=fold) for r in original.predict(model,dataset)]
    after=original.state_hash(model.state_dict())
    if before!=after or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Inference changed weights/gradients')
    return records,dict(seed=seed,fold=fold,train_ids=manifest['folds'][fold]['train_ids'],
        before_model_sha256=before,after_model_sha256=after,optimizer_restored=False)
