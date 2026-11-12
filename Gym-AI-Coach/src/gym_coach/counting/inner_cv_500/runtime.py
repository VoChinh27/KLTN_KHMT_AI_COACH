"""Separate checkpoint family; learning is reachable only by explicit user-run CLI."""
import json
import math
from pathlib import Path
import time

from ..inner_cv import runtime as frozen_runtime
from ..inner_cv.manifest import sealed, unseal
from ..inner_cv.oof import pair_identity, valid_prediction, metric
from ..models import STATUS, canonical_hash
from ..readiness import file_info, write_json
from .contract import Config
from ..research.network import count_loss
from ..research.data import read
from ..research.experiment import save_checkpoint, restore_state
from .manifest import ENDPOINT, fold_identity

CHECKPOINT_VERSION = 'squat-count-inner-cv-500-checkpoint-v1'


def subset(root, data, ids):
    return frozen_runtime.subset(root, data, ids)


def forward_only(root, data, manifest):
    """Input/init parity and finite CV500 checks; no optimizer or parameter update."""
    import torch
    from ..research.network import CountGRU, setup, collate
    from ..research.contract import Config as OriginalConfig
    config=Config.model_validate_json(json.dumps(manifest['config']));checks=[]
    for fold in manifest['folds']:
        setup(OriginalConfig());reference=CountGRU(OriginalConfig())
        setup(config);model=CountGRU(config);model.eval()
        before=frozen_runtime.state_hash(model.state_dict())
        if before!=frozen_runtime.state_hash(reference.state_dict()):raise ValueError('CV500 initialization differs')
        recorded=manifest['controls'].get('original_initializations')
        if recorded is not None and before!=recorded[str(fold['fold'])]['model_sha256']:
            raise ValueError('CV500 initialization differs from saved original control')
        for side in ('train','evaluation'):
            dataset=subset(root,data,fold[side+'_ids']);raw=frozen_runtime.subset(root,data,fold[side+'_ids'])
            sample_hashes=[]
            for i in range(len(dataset)):
                sample=dataset[i]
                if sample!=raw[i]:raise ValueError('CV500 input differs from original')
                sample_hashes.append(dict(video_id=sample['video_id'],sha256=canonical_hash(sample)))
            batch=collate([dataset[i] for i in range(min(2,len(dataset)))])
            with torch.inference_mode():output=model(batch);loss=count_loss(output,batch)
            after=frozen_runtime.state_hash(model.state_dict())
            if before!=after or output.requires_grad or any(p.grad is not None for p in model.parameters()):
                raise ValueError('CV500 forward-only changed weights/gradients')
            checks.append(dict(fold=fold['fold'],side=side,checked_clips=len(dataset),input_hashes=sample_hashes,
                forward_ids=batch['video_ids'],input_shape=list(batch['x'].shape),output_shape=list(output.shape),
                untrained_SmoothL1=float(loss),finite=True,initial_model_sha256=before,after_model_sha256=after))
    return dict(kind='untrained_forward_only_not_performance',checks=checks,unique_clips=33,
        training_steps=0,baseline_fits=0,parameters_unchanged=True,checkpoint_created=False)


def checkpoint_payload(identity, model, optimizer, epoch, steps, history, *, fixture=False):
    import torch
    return dict(version=CHECKPOINT_VERSION, identity=identity,
        kind='synthetic_fixture_no_learning' if fixture else 'user_trained',
        model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        epoch=epoch, training_steps=steps, model_sha256=frozen_runtime.state_hash(model.state_dict()),
        epoch_reports=history, history_sha256=canonical_hash(history))


def validate_report(report, identity, manifest, epoch, *, fixture=False):
    if (type(report.get('epoch')) is not int or report['epoch'] != epoch
            or type(report.get('fold')) is not int or report['fold'] != identity['fold']
            or report['manifest_sha256'] != identity['manifest_sha256']
            or report['identity_sha256'] != canonical_hash(identity)):
        raise ValueError('CV500 checkpoint report identity/epoch mismatch')
    records = report['records']
    ids = [r['video_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(identity['partition']['evaluation_ids']):
        raise ValueError('CV500 fold prediction coverage mismatch')
    rows = {r['source']['video_id']: r for r in manifest['rows']}
    for r in records:
        if 'fold' in r and (type(r['fold']) is not int or r['fold']!=identity['fold']):
            raise ValueError('CV500 record fold mismatch')
        expected = pair_identity(rows[r['video_id']])
        if canonical_hash({k: r.get(k) for k in expected}) != canonical_hash(expected):
            raise ValueError('CV500 prediction source/target linkage mismatch')
        if r.get('status') == 'estimated':
            if r.get('reason') is not None or not valid_prediction(r.get('prediction')):
                raise ValueError('Invalid CV500 prediction')
        elif r.get('status') != 'failed' or r.get('prediction') is not None or not r.get('reason'):
            raise ValueError('Invalid CV500 failure record')
    expected_steps = (epoch+1)*math.ceil(len(identity['partition']['train_ids'])/Config().batch_size)
    if report['metric'] != metric(records) or report['training_steps'] != (0 if fixture else expected_steps):
        raise ValueError('CV500 report metric/step mismatch')
    if not fixture and (type(report['training_loss']) not in (int, float)
                        or not math.isfinite(report['training_loss']) or report['training_loss'] < 0):
        raise ValueError('Invalid saved training loss')
    if not fixture:
        import torch
        ids = identity['partition']['train_ids']
        generator = torch.Generator().manual_seed(Config().seed+epoch)
        order = torch.randperm(len(ids), generator=generator).tolist()
        if report.get('batch_order_sha256') != canonical_hash([ids[i] for i in order]):
            raise ValueError('CV500 saved batch order differs from frozen sampler')


def load_checkpoint(path, manifest, fold, *, allow_fixture=False):
    import torch
    identity = fold_identity(manifest, fold)
    if file_info(path) != read(str(path)+'.sha256.json'):
        raise ValueError('CV500 checkpoint byte hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('version') != CHECKPOINT_VERSION or saved.get('identity') != identity:
        raise ValueError('CV500 checkpoint identity mismatch: fold/transform/config/code/control')
    fixture = saved.get('kind') == 'synthetic_fixture_no_learning'
    if (fixture and not allow_fixture) or saved.get('kind') not in ('user_trained', 'synthetic_fixture_no_learning'):
        raise ValueError('Not a user-trained CV500 checkpoint')
    epoch = saved['epoch']
    if type(epoch) is not int or epoch not in range(500):
        raise ValueError('Invalid CV500 checkpoint epoch/budget')
    expected_steps = (epoch+1)*math.ceil(len(identity['partition']['train_ids'])/Config().batch_size)
    if saved['training_steps'] != (0 if fixture else expected_steps):
        raise ValueError('CV500 checkpoint step mismatch')
    history = saved['epoch_reports']
    if len(history) != epoch+1 or canonical_hash(history) != saved['history_sha256']:
        raise ValueError('CV500 checkpoint history mismatch')
    for e, envelope in enumerate(history):
        validate_report(unseal(envelope), identity, manifest, e, fixture=fixture)
    if (frozen_runtime.state_hash(saved['model']) != saved['model_sha256']
            or unseal(history[-1])['model_sha256'] != saved['model_sha256']):
        raise ValueError('CV500 model/report hash mismatch')
    with torch.random.fork_rng(devices=[]):
        reference,optimizer=frozen_runtime.fresh_components(Config())
    reference.load_state_dict(saved['model'],strict=True)
    if saved['optimizer']['param_groups']!=optimizer.state_dict()['param_groups']:
        raise ValueError('CV500 optimizer config mismatch')
    if not all(bool(torch.isfinite(p).all()) for p in reference.parameters()):
        raise ValueError('Nonfinite CV500 checkpoint weights')
    return saved



def train(root, data, manifest, fold, out, *, user_run=False, resume=None):
    if not user_run:
        raise PermissionError('Only the user may run learning; --user-run required')
    import torch
    from ..research.network import collate
    config = Config.model_validate_json(json.dumps(manifest['config']))
    identity = fold_identity(manifest, fold)
    train_set = subset(root, data, identity['partition']['train_ids'])
    evaluation = subset(root, data, identity['partition']['evaluation_ids'])
    previous = load_checkpoint(resume, manifest, fold) if resume else None
    first_epoch = previous['epoch']+1 if previous else 0
    if first_epoch >= config.epochs:
        raise ValueError('Frozen 500 epoch budget exhausted; aggregate final checkpoints')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, optimizer = frozen_runtime.fresh_components(config)
    initial_sha256 = frozen_runtime.state_hash(model.state_dict())
    steps, history = 0, []
    if previous:
        first_epoch, steps = restore_state(previous, model, optimizer)
        history = previous['epoch_reports'][:]
    write_json(out/'run.json', dict(**STATUS, identity=identity, config=config.model_dump(mode='json'),
        initial_fresh_state_sha256=initial_sha256, resume_checkpoint=str(resume) if resume else None,
        resume_identity=file_info(resume) if resume else None, first_epoch=first_epoch,
        endpoint=ENDPOINT, outer_validation_used=False, development_test_used=False, final_refit=False))
    started = time.perf_counter()
    for epoch in range(first_epoch, config.epochs):
        model.train()
        generator = torch.Generator().manual_seed(config.seed+epoch)
        order = torch.randperm(len(train_set), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), config.batch_size):
            samples = [train_set[i] for i in order[start:start+config.batch_size]]
            batch = collate(samples)
            optimizer.zero_grad(set_to_none=True)
            loss = count_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; no checkpoint committed')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append((float(loss.detach()), len(samples)))
        records = frozen_runtime.predict(model, evaluation)
        report = dict(epoch=epoch, fold=fold, manifest_sha256=canonical_hash(manifest),
            identity_sha256=canonical_hash(identity), model_sha256=frozen_runtime.state_hash(model.state_dict()),
            records=records, metric=metric(records), training_steps=steps,
            training_loss=sum(v*n for v, n in losses)/sum(n for _, n in losses),
            batch_order_sha256=canonical_hash([train_set.rows[i]['source']['video_id'] for i in order]),
            elapsed_seconds=time.perf_counter()-started)
        history.append(sealed(report))
        path = out/f'epoch-{epoch:03d}.pt'
        save_checkpoint(path, checkpoint_payload(identity, model, optimizer, epoch, steps, history))
        write_json(out/f'epoch-{epoch:03d}.json', history[-1])
        print(json.dumps(dict(fold=fold, epoch=epoch, steps=steps, evaluation=report['metric'])), flush=True)
    write_json(out/'complete.json', dict(**STATUS, final_checkpoint=path.name, checkpoint=file_info(path),
        identity=identity, endpoint=ENDPOINT, comparison_pending=True, final_refit=False))


def endpoint_reports(manifest, paths, epoch=499):
    """Fail closed; no minimum search and no fold-best substitution."""
    import torch
    if type(epoch) is not int or epoch not in (49,499):raise ValueError('Only fixed indices49 and499')
    if len(paths) != 3:
        raise ValueError('Exactly three final CV500 checkpoints required')
    reports, sources, seen = [], [], set()
    for path in paths:
        if file_info(path) != read(str(path)+'.sha256.json'):
            raise ValueError('CV500 checkpoint byte hash mismatch')
        header = torch.load(path, map_location='cpu', weights_only=True)
        fold = header['identity']['fold']
        saved = load_checkpoint(path, manifest, fold)
        if fold in seen or saved['epoch'] != epoch:
            raise ValueError('Duplicate fold or wrong endpoint: require requested fixed endpoint for each fold')
        seen.add(fold)
        reports.append(unseal(saved['epoch_reports'][epoch]))
        sources.append(dict(path=str(path), **file_info(path), fold=fold, epoch=epoch))
    if seen != {0, 1, 2}:
        raise ValueError('Missing CV500 fold')
    return reports, sorted(sources, key=lambda s: s['fold'])
