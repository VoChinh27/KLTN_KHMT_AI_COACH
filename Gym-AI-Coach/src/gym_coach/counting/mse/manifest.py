"""Bind MSE to original folds plus byte-verified saved diagnostic controls."""
import importlib.util
from pathlib import Path
from ..inner_cv import manifest as cv
from ..models import STATUS, canonical_hash
from ..readiness import file_info
from ..research.data import check_identity, identity, local, read
from ..xy_ablation.manifest import build as original_condition_build
from .loss import Config, descriptor

VERSION = 'squat-count-mse-v1'
ENDPOINT = dict(epoch_index=49, epoch_one_based=50, selection='fixed_endpoint_no_minimum_search')


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def diagnostic_tool(root, link):
    path=check_identity(root,link)
    if path != local(root,'scripts/diagnose_squat_count_train_capacity.py'):
        raise ValueError('Unexpected diagnostic implementation')
    spec=importlib.util.spec_from_file_location('mse_frozen_diagnostic',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def command_arg(command, name):
    if command.count(name)!=1 or command.index(name)+1>=len(command):
        raise ValueError('Missing/ambiguous source receipt option: '+name)
    return command[command.index(name)+1]


def verify_controls(root, original_manifest, gru_run, diagnosis_dir):
    root=Path(root).resolve();folder=local(root,diagnosis_dir)/'analysis-v1'
    receipt=read(folder/'receipt.json');command=receipt['command']
    tool_link=identity(root,command[1])
    if file_info(check_identity(root,tool_link))!=receipt['tool']:
        raise ValueError('Saved diagnostic tool hash changed')
    tool=diagnostic_tool(root,tool_link);tool.verify_inventory(folder)
    runs={k:command_arg(command,flag) for k,flag in
          (('gru','--gru-run'),('zero_xy','--zero-run'),('velocity','--velocity-run'))}
    if local(root,runs['gru']) != local(root,gru_run):
        raise ValueError('Requested original GRU run differs from diagnosis')
    manifests,data,checkpoints,held,baseline,histories,evidence=tool.verify_sources(root,runs,
        command_arg(command,'--gru-oof-dir'),command_arg(command,'--baseline-dir'))
    if evidence['manifests']['gru']!=identity(root,local(root,original_manifest)):
        raise ValueError('Requested original manifest differs from verified lineage')
    if cv.unseal(read(folder/'verification.json'))!=evidence:
        raise ValueError('Saved verification differs from checkpoint/control replay')
    saved=cv.unseal(read(folder/'diagnosis.json'));base=manifests['gru']
    if (cv.unseal(read(folder/'per-video-errors.json'))!=saved['records']
            or saved['epoch_index']!=49 or saved['loss_histories']!=histories
            or saved['train_model_forward_calls']!=198 or saved['held_out_model_forward_calls']!=0
            or saved['new_training_steps']!=0 or saved['new_baseline_fits']!=0):
        raise ValueError('Saved diagnostic records/history/endpoint mismatch')
    if len(saved['records'])!=495 or set(r['method'] for r in saved['records'])!=set(tool.METHODS):
        raise ValueError('Saved control coverage mismatch')
    for method in tool.METHODS:
        train_by_fold={};held_by_fold={}
        for f in range(3):
            for side,key in (('train','train'),('held_out','evaluation')):
                rs=[r for r in saved['records'] if r['method']==method and r['fold']==f and r['side']==side]
                valid=tool.validate_records(base,f,key,rs)
                if tool.breakdown(valid)!=saved['results'][method]['folds'][str(f)][side]:
                    raise ValueError('Saved control metrics differ from predictions')
                if side=='train':train_by_fold[f]=valid
                else:
                    held_by_fold[f]=valid
                    expected={r['video_id']:r['prediction'] for r in held[method] if r['fold']==f}
                    if {r['video_id']:r['prediction'] for r in valid}!=expected:
                        raise ValueError('Saved control OOF differs from checkpoint/baseline replay')
        if tool.aggregate_unique_videos(base,train_by_fold,held_by_fold)!=saved['results'][method]['aggregate']:
            raise ValueError('Saved train/control aggregate mismatch')
    receipts=saved['inference_receipts']
    if len(receipts)!=9 or {(r['method'],r['fold']) for r in receipts}!={(m,f) for m in tool.NEURAL for f in range(3)}:
        raise ValueError('Saved train checkpoint provenance incomplete')
    for r in receipts:
        f=r['fold'];ck=checkpoints[r['method']][f]
        if (r['train_ids']!=base['folds'][f]['train_ids'] or r['model_before_sha256']!=ck['model_sha256']
                or r['model_after_sha256']!=ck['model_sha256'] or not r['model_eval'] or not r['inference_mode']):
            raise ValueError('Saved train inference checkpoint linkage mismatch')
    initializations={}
    for f,source in evidence['checkpoint_links']['gru'].items():
        run_path=check_identity(root,source).parent/'run.json';record=read(run_path)
        if record['config']!=base['config'] or record['resume_checkpoint'] is not None or record['first_epoch']!=0:
            raise ValueError('Original control is not the expected fresh initialization run')
        initializations[f]=dict(run=identity(root,run_path),model_sha256=record['initial_fresh_state_sha256'])
    links=dict(original_manifest=identity(root,local(root,original_manifest)),
        original_gru_run=local(root,gru_run).relative_to(root).as_posix(),
        diagnosis_dir=local(root,diagnosis_dir).relative_to(root).as_posix(),diagnostic_tool=tool_link,
        diagnostic_files={n:identity(root,folder/n) for n in
            ('diagnosis.json','verification.json','per-video-errors.json','receipt.json','inventory.json')},
        verified_lineage=evidence,original_initializations=initializations)
    return base,data,links,saved['records']


def build(base,links):
    result=original_condition_build(base,links)
    original=dict(base['config']);changed=dict(original,loss='mse_video_mean')
    config=Config.model_validate_json(__import__('json').dumps(changed)).model_dump(mode='json')
    if {k for k in config if config[k]!=original[k]}!={'loss'}:
        raise ValueError('MSE may change exactly the loss field')
    result.update(version=VERSION,config=config,config_sha256=canonical_hash(config),code=code_identity(),
        transform=dict(version='identity-original-raw58-v1',operation='none'),loss=descriptor(),
        endpoint=ENDPOINT,scope='loss-only sensitivity; adaptive development; not independent final evaluation',
        retained_pose_information='all original raw58 channels, masks/time/duration unchanged',
        training_start='fresh initialization only; resume same MSE condition only',
        original_config=original,original_manifest_sha256=canonical_hash(base))
    return result


def load(root,path):
    payload=cv.unseal(read(local(root,path)));links=payload['controls']
    check_identity(root,links['original_manifest']);check_identity(root,links['diagnostic_tool'])
    for link in links['diagnostic_files'].values():check_identity(root,link)
    base,data,current,controls=verify_controls(root,links['original_manifest']['path'],
        links['original_gru_run'],links['diagnosis_dir'])
    if payload!=build(base,current):
        raise ValueError('MSE manifest/code/loss/config/control mismatch; no migration')
    return payload,data,controls


def fold_identity(manifest,fold):
    if type(fold) is not int or fold not in range(3):raise ValueError('Fold must be0,1,2')
    return dict(version=VERSION,manifest_sha256=canonical_hash(manifest),fold=fold,
        partition=manifest['folds'][fold],loss=manifest['loss'],config=manifest['config'],
        config_sha256=manifest['config_sha256'],transform=manifest['transform'],code=manifest['code'],
        core_identity=manifest['core_identity'],original_cv_code=manifest['original_cv_code'],endpoint=ENDPOINT)
