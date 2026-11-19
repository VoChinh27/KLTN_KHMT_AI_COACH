"""Separate CV checkpoint identity and opt-in user learning; old research code is unchanged."""
import json
import hashlib
import math
import pickle
from pathlib import Path
import time

from ..models import canonical_hash, STATUS
from ..readiness import file_info, write_json
from ..research.contract import Config
from ..research.data import CountDataset, read
from ..research.experiment import save_checkpoint, restore_state
from .manifest import fold_identity, sealed, unseal
from .oof import pair_identity, valid_prediction, metric, aggregate

CHECKPOINT_VERSION = 'squat-count-inner-cv-checkpoint-v1'


def fresh_components(config):
    """Same frozen seed, independent storage and empty optimizer history per fold."""
    import torch
    from ..research.network import CountGRU, setup
    setup(config)
    model = CountGRU(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return model, optimizer


def subset(root, data, ids):
    dataset = CountDataset(root, data, 'train')
    by_id = {r['source']['video_id']: r for r in dataset.rows}
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError('Subset is not a unique admitted outer-train subset')
    dataset.rows = [by_id[v] for v in ids]
    return dataset


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def predict(model, dataset):
    """Current epoch only; failed inputs/outputs retain their IDs and null values."""
    import torch
    from ..research.network import collate
    result = []
    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(dataset.rows):
            value = dict(**pair_identity(row), prediction=None, status='failed', reason='prediction_operational_error')
            try:
                estimate = float(model(collate([dataset[i]]))[0])
                if not valid_prediction(estimate):
                    raise ValueError('Non-finite/negative output')
                value.update(prediction=estimate, status='estimated', reason=None)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                value['detail'] = type(exc).__name__+': '+str(exc)
            result.append(value)
    return result


def checkpoint_payload(identity, model, optimizer, epoch, steps, history, *, fixture=False):
    import torch
    return dict(version=CHECKPOINT_VERSION, identity=identity,
        kind='synthetic_fixture_no_learning' if fixture else 'user_trained',
        model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        epoch=epoch, training_steps=steps, model_sha256=state_hash(model.state_dict()),
        epoch_reports=history, history_sha256=canonical_hash(history))


def load_checkpoint(path, identity, *, allow_fixture=False):
    import torch
    if file_info(path) != read(str(path)+'.sha256.json'):
        raise ValueError('CV checkpoint byte hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('version') != CHECKPOINT_VERSION or saved.get('identity') != identity:
        raise ValueError('CV checkpoint/fold identity mismatch; old or other-fold weights forbidden')
    fixture = saved.get('kind') == 'synthetic_fixture_no_learning'
    if (fixture and not allow_fixture) or saved.get('kind') not in ('user_trained', 'synthetic_fixture_no_learning'):
        raise ValueError('Not a user-trained CV checkpoint')
    epoch = saved['epoch']
    if type(epoch) is not int or epoch not in range(50):
        raise ValueError('Invalid checkpoint epoch')
    expected_steps = (epoch+1)*math.ceil(len(identity['partition']['train_ids'])/Config().batch_size)
    if saved['training_steps'] != (0 if fixture else expected_steps):
        raise ValueError('Checkpoint step/budget mismatch')
    history = saved['epoch_reports']
    if canonical_hash(history) != saved['history_sha256'] or len(history) != epoch+1:
        raise ValueError('Checkpoint history mismatch')
    for e, envelope in enumerate(history):
        report = unseal(envelope)
        if (report['epoch'] != e or report['fold'] != identity['fold']
                or report['manifest_sha256'] != identity['manifest_sha256']
                or report['identity_sha256'] != canonical_hash(identity)):
            raise ValueError('Checkpoint report pairing/identity mismatch')
    if (state_hash(saved['model']) != saved['model_sha256']
            or unseal(history[-1])['model_sha256'] != saved['model_sha256']):
        raise ValueError('Checkpoint current model/report mismatch')
    return saved


def forward_only(root, data, manifest):
    """No optimizer creation is needed here; no predictions are presented as performance."""
    import torch
    from ..research.network import CountGRU, setup, collate
    config = Config.model_validate_json(json.dumps(manifest['config']))
    receipts = []
    for fold in manifest['folds']:
        setup(config)
        model = CountGRU(config)
        before = state_hash(model.state_dict())
        for side in ('train', 'evaluation'):
            dataset = subset(root, data, fold[side+'_ids'])
            samples = [dataset[i] for i in range(min(2, len(dataset)))]
            batch = collate(samples)
            with torch.inference_mode():
                output = model(batch)
            if (not torch.isfinite(output).all() or output.requires_grad
                    or state_hash(model.state_dict()) != before or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Forward-only invariant failed')
            receipts.append(dict(fold=fold['fold'], side=side, ids=batch['video_ids'],
                                 input_shape=list(batch['x'].shape), output_shape=list(output.shape), finite=True))
    return dict(kind='untrained_forward_only_not_performance', checks=receipts,
                training_steps=0, baseline_fits=0, parameters_unchanged=True, checkpoint_created=False)


def train(root, data, manifest, fold, out, *, user_run=False, resume=None):
    if not user_run:
        raise PermissionError('Only the user may run learning; --user-run required')
    import torch
    from ..research.network import collate, count_loss
    config = Config.model_validate_json(json.dumps(manifest['config']))
    identity = fold_identity(manifest, fold)
    train_set = subset(root, data, identity['partition']['train_ids'])
    evaluation = subset(root, data, identity['partition']['evaluation_ids'])
    previous = load_checkpoint(resume, identity) if resume else None
    first_epoch = previous['epoch']+1 if previous else 0
    if first_epoch >= config.epochs:
        raise ValueError('Frozen 50 epoch budget exhausted; aggregate final checkpoints instead')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, optimizer = fresh_components(config)
    initial_sha256 = state_hash(model.state_dict())
    steps, history = 0, []
    if previous:
        first_epoch, steps = restore_state(previous, model, optimizer)
        history = previous['epoch_reports'][:]
    write_json(out/'run.json', dict(**STATUS, identity=identity, config=config.model_dump(mode='json'),
        initial_fresh_state_sha256=initial_sha256, resume_checkpoint=str(resume) if resume else None,
        resume_identity=file_info(resume) if resume else None, first_epoch=first_epoch,
        command_scope='user-run inner-CV only', outer_validation_used=False, development_test_used=False,
        selection='only pooled same-epoch OOF after all three folds; no fold-best weights substituted'))
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
        records = predict(model, evaluation)
        report = dict(epoch=epoch, fold=fold, manifest_sha256=canonical_hash(manifest),
            identity_sha256=canonical_hash(identity), model_sha256=state_hash(model.state_dict()),
            records=records, metric=metric(records), training_steps=steps,
            training_loss=sum(v*n for v, n in losses)/sum(n for _, n in losses),
            elapsed_seconds=time.perf_counter()-started)
        history.append(sealed(report))
        # Checkpoint contains the complete immutable OOF history for interruption/resume.
        # A partial file without its SHA sidecar is never considered committed.
        path = out/f'epoch-{epoch:03d}.pt'
        save_checkpoint(path, checkpoint_payload(identity, model, optimizer, epoch, steps, history))
        write_json(out/f'epoch-{epoch:03d}.json', history[-1])
        print(json.dumps(dict(fold=fold, epoch=epoch, steps=steps, evaluation=report['metric'])), flush=True)
    write_json(out/'complete.json', dict(**STATUS, final_checkpoint=path.name,
        checkpoint=file_info(path), identity=identity, selection_pending_other_folds=True, refit=False))


def aggregate_checkpoints(manifest, paths):
    reports, issues, sources = [], [], []
    for path in paths:
        try:
            # Read fold from a byte-verified checkpoint, then validate the full expected identity.
            import torch
            if file_info(path) != read(str(path)+'.sha256.json'):
                raise ValueError('Checkpoint byte hash mismatch')
            header = torch.load(path, map_location='cpu', weights_only=True)
            expected = fold_identity(manifest, header['identity']['fold'])
            saved = load_checkpoint(path, expected)
            reports.extend(unseal(e) for e in saved['epoch_reports'])
            sources.append(dict(path=str(path), **file_info(path), fold=expected['fold'], epoch=saved['epoch']))
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, TypeError) as exc:
            issues.append(dict(code='checkpoint_unavailable_or_invalid', path=str(path), detail=str(exc)))
    result = aggregate(manifest, reports)
    result['checkpoint_sources'] = sources
    result['global_issues'].extend(issues)
    if issues:
        result.update(status='INCOMPLETE_OR_INVALID', selection=None)
    return result
