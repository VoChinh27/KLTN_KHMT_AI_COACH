import copy
import json
from pathlib import Path
from ..inner_cv import manifest as cv,runtime as original
from ..models import STATUS,canonical_hash
from ..readiness import file_info
from ..research.data import read,local,check_identity
from .contract import VERSION,SEEDS,ENDPOINT,config_type
from .transform import descriptor,normalize,NormalizationError
from . import sources


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def audit_inputs(root,data,base):
    ids=[r['source']['video_id'] for r in base['rows']]
    dataset=original.subset(root,data,ids)
    audit={};errors={}
    for i,row in enumerate(dataset.rows):
        v=ids[i]
        try:
            sample=dataset[i]
            source=row['source']
            if (sample['video_id']!=v or len(sample['values'])!=source['frame_count']
                    or sample['duration_seconds']!=source['frame_count']/source['fps']
                    or any(vector[-2:]!=[t/source['fps'],1/source['fps'] if t else 0.]
                           for t,vector in enumerate(sample['values']))):
                raise NormalizationError('source_timeline_mismatch',v)
            if any(complete!=all(vector[j*7+3] for j in range(8))
                   for complete,vector in zip(sample['complete'],sample['values'])):
                raise NormalizationError('complete_mask_mismatch',v)
            _,audit[v]=normalize(sample)
        except (ValueError,KeyError,OSError,TypeError) as error:
            errors[v]=dict(reason=getattr(error,'reason','input_integrity_error'),detail=str(error))
    if errors:
        error=ValueError('Normalization preflight blocked; retain all33 clips')
        error.diagnostics=dict(eligible_audits=audit,errors=errors,expected_clips=33)
        raise error
    if len(audit)!=33:
        raise ValueError('Require all33 input audits')
    return audit


def build(base,evidence,audit):
    rows,_,folds=cv.partition(base['rows'])
    if rows!=base['rows'] or folds!=base['folds'] or [len(f['train_ids']) for f in folds]!=[20,20,26]:
        raise ValueError('Locked cohort/fold drift')
    if set(audit)!={r['source']['video_id'] for r in rows}:
        raise ValueError('Audit must retain all33 clips')
    configs={}
    for seed in SEEDS:
        raw=evidence['controls'][str(seed)]['raw_config']
        cfg=config_type(seed)().model_dump(mode='json')
        if {k for k in cfg if cfg[k]!=raw[k]}!={'normalization'}:
            raise ValueError('Only representation metadata may change within each seed')
        configs[str(seed)]=cfg
    return dict(version=VERSION,**STATUS,rows=copy.deepcopy(rows),folds=copy.deepcopy(folds),
        dataset=base['dataset'],core_identity=base['core_identity'],cv_code=base['cv_code'],
        parent500_code=base['code'],configs=configs,config_hashes={s:canonical_hash(c) for s,c in configs.items()},
        transform=descriptor(),transform_sha256=canonical_hash(descriptor()),code=code_identity(),
        input_audit=audit,input_audit_sha256=canonical_hash(audit),sources=evidence,
        seeds=list(SEEDS),epochs=500,expected_steps=[1500,1500,2000],endpoint=ENDPOINT,
        comparison=dict(primary='per-seed body minus same-seed raw OOF MAE, lower is better',
            train='mean two absolute errors/video then equal mean33',
            two_seeds='mean absolute errors within video/seed then equal mean33, not66independent videos',
            opposite_directions='result inconsistent across the two seeds',
            threshold=None,checkpoint_selection=False,seed_selection=False),
        outer_validation_used=False,development_test_used=False,final_refit=False,
        limitation='AI weak reference; adaptive development on inspected OOF; not independent final evaluation or squat-event validation')


def prepare(root):
    base,data,controls,evidence=sources.verify(root)
    audit=audit_inputs(root,data,base)
    return build(base,evidence,audit),data,controls


def load(root,path):
    saved=cv.unseal(read(local(root,path)))
    if saved['version']!=VERSION or saved['code']!=code_identity():
        raise ValueError('Body condition version/code mismatch')
    for item in saved['sources']['sources']:
        check_identity(root,item)
    rebuilt,data,controls=prepare(root)
    if rebuilt!=saved:
        raise ValueError('Body manifest/config/scale/cache/source identity mismatch')
    return saved,data,controls


def select(manifest,seed):
    config=config_type(seed).model_validate_json(json.dumps(manifest['configs'][str(seed)])).model_dump(mode='json')
    if canonical_hash(config)!=manifest['config_hashes'][str(seed)]:
        raise ValueError('Body config hash mismatch')
    return dict(manifest,seed=seed,config=config,config_sha256=canonical_hash(config),
        bundle_manifest_sha256=canonical_hash(manifest))


def fold_identity(selected,fold):
    if type(fold) is not int or fold not in (0,1,2):
        raise ValueError('Invalid fold')
    config_type(selected['seed']).model_validate_json(json.dumps(selected['config']))
    return dict(version=VERSION,manifest_sha256=canonical_hash(selected),
        bundle_manifest_sha256=selected['bundle_manifest_sha256'],seed=selected['seed'],fold=fold,
        partition=selected['folds'][fold],config=selected['config'],config_sha256=selected['config_sha256'],
        transform=selected['transform'],transform_sha256=selected['transform_sha256'],
        input_audit_sha256=selected['input_audit_sha256'],code=selected['code'],core_identity=selected['core_identity'],
        parent500_code=selected['parent500_code'],original_cv_code=selected['cv_code'],endpoint=ENDPOINT)
