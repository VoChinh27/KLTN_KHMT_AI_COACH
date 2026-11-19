import copy
import json
from pathlib import Path
from ..inner_cv import manifest as cv
from ..models import canonical_hash
from ..readiness import file_info
from ..research.data import read, local, check_identity
from .contract import Config, VERSION, ENDPOINT, ENDPOINTS, OBSERVATIONS, LIMITATION
from . import sources


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def build(parent, links, evidence):
    rows, _, folds = cv.partition(parent['rows'])
    if (rows != parent['rows'] or folds != parent['folds']
            or [len(f['train_ids']) for f in folds] != [20,20,26]
            or parent['epochs'] != 500 or parent['config']['seed'] != 20261026):
        raise ValueError('Require locked original33/12/folds/GRU500 seed20261026')
    config = Config.model_validate_json(json.dumps(dict(parent['config'],seed=20261027))).model_dump(mode='json')
    if {k for k in config if config[k] != parent['config'][k]} != {'seed'}:
        raise ValueError('Only training seed may change')
    kept = ('rows','folds','dataset','core_identity','cv_code','group_order','assignment_rule','parent_cohort',
        'parent_roles','excluded_from_inner_cv','oof_clips','oof_groups','outer_validation_used',
        'development_test_used','development_test_previously_exposed','target_kind',
        'validation_status','official_eligibility','assessment','reference_status','g57','g6')
    result = {k:copy.deepcopy(parent[k]) for k in kept}
    result.update(version=VERSION, config=config, config_sha256=canonical_hash(config),
        parent_config=parent['config'], parent_manifest_sha256=canonical_hash(parent), parent500_code=parent['code'],
        code=code_identity(), sources=links, evidence=evidence, epochs=500, expected_steps=[1500,1500,2000],
        endpoint=ENDPOINT, endpoints=list(ENDPOINTS), observations=list(OBSERVATIONS),
        selection='primary499_descriptive49_no_minimum_or_seed_selection', fresh_initialization=True,
        seed_change='overall initialization plus seed+epoch batch order; effects not separated',
        input_transform='identity_raw58_no_mutation', final_refit=False, scope=LIMITATION,
        inherited_config_selection='unused research metadata; endpoint policy above governs; no best selection')
    return result


def prepare_inputs(root, parent_manifest, comparison, diagnosis_report, review):
    parent,data,saved,links,evidence = sources.verify(root,parent_manifest,comparison,diagnosis_report,review)
    return build(parent,links,evidence),data,saved


def load(root,path):
    saved = cv.unseal(read(local(root,path)))
    for link in saved['sources'].values():
        check_identity(root,link)
    links=saved['sources']
    rebuilt,data,controls=prepare_inputs(root,links['parent_manifest']['path'],
        Path(links['comparison']['path']).parent,links['diagnosis_report']['path'],links['review']['path'])
    if saved != rebuilt:
        raise ValueError('Seed condition manifest/config/code/input/control mismatch')
    return saved,data,controls


def fold_identity(manifest,fold):
    if type(fold) is not int or fold not in (0,1,2):
        raise ValueError('Fold must be0,1,2')
    config=Config.model_validate_json(json.dumps(manifest['config'])).model_dump(mode='json')
    if config != manifest['config'] or canonical_hash(config) != manifest['config_sha256']:
        raise ValueError('Seed condition config hash mismatch')
    return dict(version=VERSION,manifest_sha256=canonical_hash(manifest),fold=fold,partition=manifest['folds'][fold],
        config=config,config_sha256=manifest['config_sha256'],code=manifest['code'],
        parent500_code=manifest['parent500_code'],core_identity=manifest['core_identity'],
        original_cv_code=manifest['cv_code'],endpoint=ENDPOINT)
