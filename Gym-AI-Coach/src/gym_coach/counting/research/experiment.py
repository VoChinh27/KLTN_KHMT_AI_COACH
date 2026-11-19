"""User-run learning entry points. Preparation/verify/forward never call these."""
import copy
import json
from pathlib import Path
import time

from ..models import canonical_hash, STATUS
from ..readiness import file_info, write_json
from .contract import Config
from .data import CountDataset, read
from .evaluation import evaluate, better_checkpoint


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def experiment_identity(dataset_path, config):
    from importlib.metadata import version
    import sys
    return dict(dataset=file_info(dataset_path),config_hash=config.content_hash(),code=code_identity(),
                python=list(sys.version_info[:3]),torch=version('torch'),numpy=version('numpy'))


def save_checkpoint(path, payload):
    import torch
    with Path(path).open('xb') as stream:
        torch.save(payload,stream)
    write_json(Path(str(path)+'.sha256.json'),file_info(path))


def load_checkpoint(path, expected_identity):
    import torch
    if file_info(path) != read(str(path)+'.sha256.json'):
        raise ValueError('Checkpoint byte hash mismatch')
    result = torch.load(path,map_location='cpu',weights_only=True)
    if result['version']!='squat-count-checkpoint-v1' or result['identity']!=expected_identity:
        raise ValueError('Checkpoint experiment identity mismatch; no implicit migration')
    if result['training_steps']<=0 or result['best_model'] is None:
        raise ValueError('Checkpoint is not a user-trained experiment')
    return result


def restore_state(previous,model,optimizer):
    """Restore only; no learning. Epoch permutation is seed+epoch, no dropout."""
    import torch
    model.load_state_dict(previous['model'])
    optimizer.load_state_dict(previous['optimizer'])
    torch.set_rng_state(previous['torch_rng'])
    return previous['epoch']+1,previous['training_steps']


def require_ready(data):
    if any(r['input_status'] not in ('accepted','rejected') for r in data['rows']):
        raise ValueError('Unchecked or corrupt input remains in cohort; fix integrity before learning')
    if any(data['summary'][role]['supervised_clips']==0 for role in ('train','validation')):
        raise ValueError('Train and validation require admitted AI targets')


def predict_model(model,root,data,role):
    import torch
    from .network import collate
    dataset = CountDataset(root,data,role,supervised=False)
    values = {}
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            # Input integrity errors stop above; model failures remain in denominator.
            try:
                values[sample['video_id']] = float(model(collate([sample]))[0])
            except (ValueError,RuntimeError) as exc:
                values[sample['video_id']]=dict(status='failed',reason='prediction_operational_error',
                                                detail=type(exc).__name__+': '+str(exc))
    return evaluate(data['rows'],values,role)


def train(root,data,dataset_path,out,*,user_run=False,resume=None):
    if not user_run:
        raise PermissionError('Learning is user-run only; explicitly pass --user-run')
    import torch
    from .network import CountGRU, setup, collate, count_loss
    require_ready(data)
    config = Config.model_validate_json(json.dumps(data['config']))
    setup(config)
    identity = experiment_identity(dataset_path,config)
    train_set = CountDataset(root,data,'train')
    validation = CountDataset(root,data,'validation')
    if not len(train_set) or not len(validation):
        raise ValueError('Train and validation require admitted AI targets')
    model = CountGRU(config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    first_epoch, steps, best_score, best_epoch, best_model = 0,0,None,None,None
    if resume:
        previous = load_checkpoint(resume,identity)
        first_epoch,steps = restore_state(previous,model,optimizer)
        best_score,best_epoch,best_model = previous['best_score'],previous['best_epoch'],previous['best_model']
    if first_epoch>=config.epochs:
        raise ValueError('Frozen epoch budget already exhausted')
    out = Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'experiment.json',dict(**STATUS,identity=identity,config=config.model_dump(mode='json'),
        scope='AI reference video-count research',resume= str(resume) if resume else None,
        training_ids=[r['source']['video_id'] for r in train_set.rows],
        validation_ids=[r['source']['video_id'] for r in validation.rows],development_test_used=False))
    started = time.perf_counter()
    for epoch in range(first_epoch,config.epochs):
        model.train()
        generator = torch.Generator().manual_seed(config.seed+epoch)
        order = torch.randperm(len(train_set),generator=generator).tolist()
        losses = []
        for start in range(0,len(order),config.batch_size):
            samples = [train_set[i] for i in order[start:start+config.batch_size]]
            batch = collate(samples)
            optimizer.zero_grad(set_to_none=True)
            loss = count_loss(model(batch),batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; no checkpoint promotion')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append((float(loss.detach()),len(samples)))
        validation_result = predict_model(model,root,data,'validation')
        score = validation_result['continuous_MAE']
        expected = data['summary']['validation']['supervised_clips']
        if validation_result['comparable']!=expected or score is None:
            raise ValueError('Validation coverage changed; checkpoint selection blocked')
        if better_checkpoint(score,best_score):
            best_score,best_epoch,best_model = score,epoch,copy.deepcopy(model.state_dict())
        checkpoint = dict(version='squat-count-checkpoint-v1',identity=identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,training_steps=steps,
            best_score=best_score,best_epoch=best_epoch,best_model=best_model,torch_rng=torch.get_rng_state())
        path = out/f'epoch-{epoch:03d}.pt'
        save_checkpoint(path,checkpoint)
        write_json(out/f'epoch-{epoch:03d}.json',dict(epoch=epoch,training_steps=steps,
            training_loss=sum(x*n for x,n in losses)/sum(n for _,n in losses),validation=validation_result,
            best_epoch=best_epoch,best_validation_MAE=best_score,elapsed_seconds=time.perf_counter()-started))
    write_json(out/'selection.json',dict(checkpoint=path.name,checkpoint_identity=file_info(path),
        state_to_evaluate='best_model',best_epoch=best_epoch,best_validation_MAE=best_score,
        selection=config.selection,development_test_used=False,**STATUS))


def run_baselines(root,data,dataset_path,out,*,user_run=False):
    if not user_run:
        raise PermissionError('Baseline parameter estimation is user-run only')
    from .baselines import fit_baselines,predict
    require_ready(data)
    config = Config.model_validate_json(json.dumps(data['config']))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    parameters = fit_baselines(data['rows'])
    value = dict(parameters=parameters,identity=experiment_identity(dataset_path,config),**STATUS)
    write_json(out/'baselines.json',dict(payload=value,sha256=canonical_hash(value)))
    for role in ('train','validation'):
        for method in ('constant','duration'):
            write_json(out/f'{role}-{method}.json',evaluate(data['rows'],predict(parameters,data['rows'],role,method),role))


def evaluate_saved(root,data,dataset_path,out,role,*,checkpoint=None,baseline=None,final_evaluation=False):
    if role not in ('validation','development_test'):
        raise ValueError('Evaluation role must be validation or development_test')
    if role=='development_test' and not final_evaluation:
        raise PermissionError('Development-test is final evaluation only; pass --final-evaluation after freezing config')
    if bool(checkpoint)==bool(baseline):
        raise ValueError('Select exactly one checkpoint or baseline')
    config=Config.model_validate_json(json.dumps(data['config']))
    identity=experiment_identity(dataset_path,config)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if checkpoint:
        from .network import CountGRU, setup
        setup(config)
        saved=load_checkpoint(checkpoint,identity)
        model=CountGRU(config);model.load_state_dict(saved['best_model'])
        write_json(out/'model.json',dict(**STATUS,checkpoint=file_info(checkpoint),best_epoch=saved['best_epoch'],
            identity=identity,result=predict_model(model,root,data,role)))
    else:
        from .baselines import predict
        envelope=read(baseline);saved=envelope['payload']
        if canonical_hash(saved)!=envelope['sha256'] or saved['identity']!=identity:
            raise ValueError('Baseline hash/experiment mismatch')
        for method in ('constant','duration'):
            write_json(out/(method+'.json'),dict(**STATUS,baseline=file_info(baseline),identity=identity,
                result=evaluate(data['rows'],predict(saved['parameters'],data['rows'],role,method),role)))
