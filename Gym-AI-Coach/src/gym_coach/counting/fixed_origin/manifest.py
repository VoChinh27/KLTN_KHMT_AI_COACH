import copy
import json
import sys
from importlib.metadata import version
from pathlib import Path
from ..inner_cv import manifest as cv, runtime as original
from ..models import STATUS, canonical_hash
from ..readiness import file_info
from ..research.data import read, local, check_identity
from .contract import VERSION, SEEDS, ENDPOINT, config_type
from .transform import descriptor, normalize, independent_check
from . import sources


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def environment_identity():
    return dict(python=list(sys.version_info[:3]),torch=version('torch'),numpy=version('numpy'),pydantic=version('pydantic'))


def build(parent,evidence,audit,geometry):
    rows,_,folds=cv.partition(parent['rows'])
    if rows!=parent['rows'] or folds!=parent['folds'] or [len(f['train_ids']) for f in folds]!=[20,20,26]:
        raise ValueError('Locked cohort/partition mismatch')
    if set(audit)!=set(parent['input_audit']) or len(audit)!=33:raise ValueError('All33 clips required')
    configs={}
    for seed in SEEDS:
        cfg=config_type(seed)().model_dump(mode='json');old=parent['configs'][str(seed)]
        if {k for k in cfg if cfg[k]!=old[k]}!={'normalization'}:raise ValueError('Config changes exceed origin')
        configs[str(seed)]=cfg
    for v,q in audit.items():
        if q['scale']!=parent['input_audit'][v]['scale']:raise ValueError('Locked parent scale changed')
    return dict(version=VERSION,**STATUS,rows=copy.deepcopy(rows),folds=copy.deepcopy(folds),
        dataset=parent['dataset'],core_identity=parent['core_identity'],cv_code=parent['cv_code'],parent500_code=parent['parent500_code'],
        parent_code=parent['code'],parent_input_audit=parent['input_audit'],configs=configs,
        config_hashes={s:canonical_hash(c) for s,c in configs.items()},transform=descriptor(),transform_sha256=canonical_hash(descriptor()),
        code=code_identity(),launcher=dict(path='scripts/Run-SquatFixedOrigin.ps1',**file_info(Path(__file__).resolve().parents[4]/'scripts/Run-SquatFixedOrigin.ps1')),
        environment=environment_identity(),input_audit=audit,input_audit_sha256=canonical_hash(audit),
        independent_geometry=geometry,sources=evidence,seeds=list(SEEDS),epochs=500,expected_steps=[1500,1500,2000],endpoint=ENDPOINT,
        comparison=dict(primary='fixed-origin minus same-seed body-normalized OOF MAE',
            train='mean two AE per video then mean33',two_seeds='mean AE within same video across seeds then mean33, not AE of mean prediction',
            directions='report global OOF and fold1 separately; opposite seed directions means inconsistent',
            focus=['fold1','g17','g19','fold0','g11','g20'],threshold=None,checkpoint_selection=False,seed_selection=False),
        outer_validation_used=False,development_test_used=False,final_refit=False,
        limitation='AI weak reference; adaptive development, not independent evaluation or causal/event validation. Retains camera drift; whole clip, not streaming.')


def prepare(root):
    parent,data,controls,evidence=sources.verify(root)
    dataset=original.subset(root,data,[r['source']['video_id'] for r in parent['rows']]);audit={};checks={}
    for i,row in enumerate(dataset.rows):
        raw=dataset[i];v=raw['video_id'];src=row['source']
        if len(raw['values'])!=src['frame_count'] or raw['duration_seconds']!=src['frame_count']/src['fps']:
            raise ValueError('Source timeline mismatch: '+v)
        new,audit[v]=normalize(raw,parent['input_audit'][v])
        checks[v]=independent_check(raw,new,parent['input_audit'][v])
    return build(parent,evidence,audit,checks),data,controls


def load(root,path):
    saved=cv.unseal(read(local(root,path)))
    if saved['version']!=VERSION or saved['code']!=code_identity() or saved['environment']!=environment_identity():
        raise ValueError('Fixed origin code/version/environment mismatch')
    for link in saved['sources']['sources']:check_identity(root,link)
    rebuilt,data,controls=prepare(root)
    if rebuilt!=saved:raise ValueError('Fixed origin manifest/parent/scale/input identity mismatch')
    return saved,data,controls


def select(manifest,seed):
    config=config_type(seed).model_validate_json(json.dumps(manifest['configs'][str(seed)])).model_dump(mode='json')
    if canonical_hash(config)!=manifest['config_hashes'][str(seed)]:raise ValueError('Config hash mismatch')
    return dict(manifest,seed=seed,config=config,config_sha256=canonical_hash(config),bundle_manifest_sha256=canonical_hash(manifest))


def fold_identity(selected,fold):
    if type(fold) is not int or fold not in (0,1,2):raise ValueError('Invalid fold')
    config_type(selected['seed']).model_validate_json(json.dumps(selected['config']))
    return dict(version=VERSION,manifest_sha256=canonical_hash(selected),bundle_manifest_sha256=selected['bundle_manifest_sha256'],
        seed=selected['seed'],fold=fold,partition=selected['folds'][fold],config=selected['config'],config_sha256=selected['config_sha256'],
        transform=selected['transform'],transform_sha256=selected['transform_sha256'],input_audit_sha256=selected['input_audit_sha256'],
        code=selected['code'],environment=selected['environment'],parent_manifest=selected['sources']['parent_manifest'],
        core_identity=selected['core_identity'],parent500_code=selected['parent500_code'],original_cv_code=selected['cv_code'],endpoint=ENDPOINT)
