"""Private binding of the unchanged CV500 training and checkpoint code."""
import json
from ..inner_cv_500 import runtime as parent
from ..inner_cv_500 import evaluation as parent_evaluation
from ..inner_cv import runtime as original
from ..models import canonical_hash
from .binding import bind
from .contract import Config, CHECKPOINT_VERSION, ENDPOINT
from .manifest import fold_identity

engine, namespace = bind(parent, dict(Config=Config, CHECKPOINT_VERSION=CHECKPOINT_VERSION,
    ENDPOINT=ENDPOINT, fold_identity=fold_identity))
train = engine.train
checkpoint_payload = engine.checkpoint_payload
load_checkpoint = engine.load_checkpoint
validate_report = engine.validate_report
endpoint_reports = engine.endpoint_reports
predict_engine, _ = bind(parent_evaluation, dict(Config=Config, fold_identity=fold_identity))
train_predictions = predict_engine.train_predictions


def forward_only(root,data,manifest):
    import torch
    from ..research.network import setup,CountGRU,collate,count_loss
    from ..inner_cv_500.contract import Config as OldConfig
    config=Config.model_validate_json(json.dumps(manifest['config']))
    checks=[];initializations=[]
    for fold in manifest['folds']:
        setup(OldConfig());old=CountGRU(OldConfig());old_hash=original.state_hash(old.state_dict())
        setup(config);a=CountGRU(config)
        setup(config);b=CountGRU(config);b.eval()
        before=original.state_hash(b.state_dict())
        if before != original.state_hash(a.state_dict()) or before == old_hash:
            raise ValueError('New seed initialization not deterministic/distinct')
        initializations.append(dict(fold=fold['fold'],seed=config.seed,initial_model_sha256=before,
            old_seed=OldConfig().seed,old_initial_model_sha256=old_hash))
        for side in ('train','evaluation'):
            ids=fold[side+'_ids'];dataset=original.subset(root,data,ids)
            hashes=[]
            for i in range(len(dataset)):
                sample=dataset[i]
                if sample['video_id'] != ids[i]:raise ValueError('Input order mismatch')
                hashes.append(dict(video_id=ids[i],sha256=canonical_hash(sample)))
            batch=collate([dataset[i] for i in range(min(2,len(dataset)))])
            with torch.inference_mode():
                prediction=b(batch);loss=count_loss(prediction,batch)
            after=original.state_hash(b.state_dict())
            if before != after or not torch.isfinite(loss) or prediction.requires_grad or any(p.grad is not None for p in b.parameters()):
                raise ValueError('Invalid forward-only result/changed weights')
            checks.append(dict(fold=fold['fold'],side=side,input_hashes=hashes,checked_clips=len(dataset),
                forward_ids=batch['video_ids'],input_shape=list(batch['x'].shape),output_shape=list(prediction.shape),
                untrained_SmoothL1=float(loss),finite=True,weights_before=before,weights_after=after))
    return dict(kind='untrained_forward_only_not_performance',initializations=initializations,checks=checks,
        unique_clips=33,checked_appearances=99,training_steps=0,baseline_fits=0,parameters_unchanged=True)
