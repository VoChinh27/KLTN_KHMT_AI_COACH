import copy
import json
from pathlib import Path
from ..inner_cv import manifest as cv
from ..models import canonical_hash
from ..readiness import file_info
from ..research.data import local,read,identity,check_identity
from ..mse.manifest import diagnostic_tool
from .contract import VERSION,Config,ENDPOINT,ENDPOINTS,REPRO_FIELDS
from . import sources


def code_identity():return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def build(base,controls,tiny_links,tiny_evidence):
    rows,_,folds=cv.partition(base['rows'])
    if rows!=base['rows'] or folds!=base['folds'] or [len(f['train_ids']) for f in folds]!=[20,20,26]:
        raise ValueError('Original33clips/12groups/locked folds must remain identical')
    if base['epochs']!=50 or base['config']['epochs']!=50:raise ValueError('Require original50epoch CV')
    config=Config.model_validate_json(json.dumps(dict(base['config'],epochs=500))).model_dump(mode='json')
    if {k for k in config if config[k]!=base['config'][k]}!={'epochs'}:raise ValueError('Only config budget may change')
    result=copy.deepcopy(base)
    result.update(version=VERSION,config=config,config_sha256=canonical_hash(config),code=code_identity(),
        original_config=base['config'],controls=controls,tiny1000_sources=tiny_links,tiny1000_evidence=tiny_evidence,
        epochs=500,selection='fixed_indices49_and499_no_best_selection',endpoint=ENDPOINT,endpoints=list(ENDPOINTS),
        expected_steps=[1500,1500,2000],steps_per_epoch=[3,3,4],original_manifest_sha256=canonical_hash(base),
        fresh_initialization=True,final_refit=False,scope='budget-only adaptive development; not independent evaluation',
        reproducibility=dict(epoch_index=49,history_indices=list(range(50)),fields=list(REPRO_FIELDS),
            comparison='exact canonical values and tensors; zero tolerance',
            ignored=['manifest_sha256','identity_sha256','elapsed_seconds'],
            unavailable_in_old_history=['batch_order_sha256','gradient_norm','parameter_delta','train_predictions'],
            sampler_check='same frozen seed+epoch randperm/update-loop source; old realized order not directly recorded'))
    return result


def prepare_inputs(root,original_manifest,gru_run,control_receipt,tiny_report,tiny_review):
    base,data,controls,records=sources.verify_controls(root,original_manifest,gru_run,control_receipt)
    links,evidence=sources.verify_tiny(root,tiny_report,tiny_review)
    return build(base,controls,links,evidence),data,records


def load(root,path):
    saved=cv.unseal(read(local(root,path)));c=saved['controls'];t=saved['tiny1000_sources']
    for link in (c['original_manifest'],c['discovery_receipt'],c['discovered_manifest'],t['review']):check_identity(root,link)
    rebuilt,data,records=prepare_inputs(root,c['original_manifest']['path'],c['original_gru_run'],c['discovery_receipt']['path'],
        Path(t['report']['result.json']['path']).parent,t['review']['path'])
    if saved!=rebuilt:raise ValueError('CV500 manifest/config/code/source/control mismatch')
    return saved,data,records


def fold_identity(manifest,fold):
    if type(fold) is not int or fold not in (0,1,2):raise ValueError('Fold must be0,1,2')
    return dict(version=VERSION,manifest_sha256=canonical_hash(manifest),fold=fold,partition=manifest['folds'][fold],
        config=manifest['config'],config_sha256=manifest['config_sha256'],code=manifest['code'],
        core_identity=manifest['core_identity'],original_cv_code=manifest['cv_code'],endpoint=ENDPOINT)
