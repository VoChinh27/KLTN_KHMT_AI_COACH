"""Copy verified500epoch cohort verbatim; freeze budget identity and old result links."""
import copy
import json
from pathlib import Path
from ..tiny_set import manifest as old,runtime as old_rt,evaluation as old_ev
from ..inner_cv.manifest import unseal
from ..models import canonical_hash
from ..readiness import file_info
from ..research.data import local,read,identity,check_identity
from .contract import VERSION,Config,BUDGET,CRITERION,ENDPOINTS,REPRO_FIELDS


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_old(root,old_manifest,report_dir,review_path):
    parent=local(root,old_manifest);report_dir=local(root,report_dir)
    base,data=old.load(root,parent)
    saved_result=unseal(read(report_dir/'result.json'))
    checkpoint=check_identity(root,saved_result['checkpoint'])
    replay=old_ev.evaluate(base,checkpoint);replay['checkpoint']=identity(root,checkpoint)
    if replay!=saved_result or read(report_dir/'receipt.json')['result']!=replay:
        raise ValueError('Old500 result/receipt differs from saved checkpoint history')
    if unseal(read(report_dir/'history.json'))!=replay['history']:
        raise ValueError('Old500 history differs')
    per_video=[dict(r,epoch=h['epoch']) for h in replay['history'] for r in h['evaluation']['records']]
    if unseal(read(report_dir/'per-video.json'))!=per_video:
        raise ValueError('Old500 video-epoch records differ')
    if (report_dir/'REPORT.md').read_text(encoding='utf-8')!=old_ev.report_markdown(replay):
        raise ValueError('Old500 Markdown result differs')
    for name,expected in read(report_dir/'inventory.json')['files'].items():
        if file_info(local(root,report_dir/name))!=expected:raise ValueError('Old500 report inventory mismatch')
    # All500 committed checkpoint bytes and external epoch reports, without inference.
    for e,h in enumerate(replay['history']):
        path=checkpoint.parent/f'epoch-{e:03d}.pt'
        if file_info(path)!=read(str(path)+'.sha256.json'):raise ValueError('Old500 epoch checkpoint bytes differ')
        if unseal(read(checkpoint.parent/f'epoch-{e:03d}.json'))!=h:raise ValueError('Old500 external history differs')
    review=read(local(root,review_path))
    if (review['status']!='VERIFIED' or review['epochs_verified']!=500 or review['video_epoch_records']!=3000
            or review['checkpoint']!=identity(root,checkpoint) or review['criterion']!=replay['criterion']
            or review['steps_exceeding_clip']!=replay['steps_exceeding_clip']
            or review['finite_gradients']!=replay['finite_gradients_all_steps']
            or review['all_parameters_have_gradients']!=replay['all_parameters_have_gradients_all_steps']
            or review['no_parameter_change_steps']!=replay['no_parameter_change_steps']):
        raise ValueError('Old review provenance/metrics mismatch')
    for i in ('49','499'):
        if review['endpoints'][i]!={k:replay['endpoints'][i][k] for k in ('MAE','RMSE','max_absolute_error')}:
            raise ValueError('Old review endpoint mismatch')
    links=dict(parent_manifest=identity(root,parent),checkpoint=identity(root,checkpoint),
        report={p.name:identity(root,p) for p in sorted(report_dir.iterdir()) if p.is_file()},
        review=identity(root,local(root,review_path)))
    evidence=dict(status='VERIFIED_SAVED_ONLY',endpoints=replay['endpoints'],criterion=replay['criterion'],
        initial_model_sha256=replay['initial_model_sha256'],final_model_sha256=replay['final_model_sha256'],
        finite_gradients=replay['finite_gradients_all_steps'],all_parameters_have_gradients=replay['all_parameters_have_gradients_all_steps'],
        steps_exceeding_clip=replay['steps_exceeding_clip'],no_parameter_change_steps=replay['no_parameter_change_steps'],
        verified_epochs=500,video_epoch_records=3000,new_training_steps=0,new_forward_calls=0,new_fits=0)
    return base,data,links,evidence


def build(base,links,evidence):
    if base['epochs']!=500 or base['config']['epochs']!=500 or base['report_epoch_indices']!=[49,499]:
        raise ValueError('Require original500epoch condition, no cohort reselection')
    if (len(base['rows'])!=6 or len(set(base['training_ids']))!=6
            or [r['target'] for r in base['rows']]!=list(range(6))
            or [r['source']['video_id'] for r in base['rows']]!=base['training_ids'] or base['evaluation_ids']!=base['training_ids']):
        raise ValueError('Original six ordered IDs must match train/evaluation')
    config=Config.model_validate_json(json.dumps(dict(base['config'],epochs=BUDGET))).model_dump(mode='json')
    if {k for k in config if config[k]!=base['config'][k]}!={'epochs'}:raise ValueError('Only config budget may change')
    result=copy.deepcopy(base)
    result.update(version=VERSION,code=code_identity(),config=config,config_sha256=canonical_hash(config),
        criterion=CRITERION,epochs=BUDGET,expected_total_steps=BUDGET,report_epoch_indices=list(ENDPOINTS),
        parent_tiny_manifest=links['parent_manifest'],old_tiny_code=base['code'],old500_sources=links,old500_evidence=evidence,
        reproducibility_policy=dict(epoch_index=499,history_indices=list(range(500)),fields=list(REPRO_FIELDS),
            comparison='exact canonical values and tensor/state hashes; no numeric tolerance',
            ignored_fields=['identity_sha256','elapsed_segment_seconds'],
            required_checkpoint_500=True,mismatch='report discrepancy; investigate; do not attribute to budget'),
        new_condition='fresh_original_raw58_SmoothL1_same_six_1000epochs',
        adaptive_development=True)
    return result


def prepare_inputs(root,old_manifest,report_dir,review_path):
    base,data,links,evidence=verify_old(root,old_manifest,report_dir,review_path)
    return build(base,links,evidence),data


def load(root,path):
    saved=unseal(read(local(root,path)));links=saved['old500_sources']
    for info in [links['parent_manifest'],links['checkpoint'],links['review'],*links['report'].values()]:check_identity(root,info)
    expected,data=prepare_inputs(root,links['parent_manifest']['path'],Path(links['report']['result.json']['path']).parent,links['review']['path'])
    if saved!=expected:raise ValueError('Tiny1000 manifest/code/config/cohort/control mismatch')
    return saved,data


def experiment_identity(manifest):
    return dict(version=VERSION,manifest_sha256=canonical_hash(manifest),config=manifest['config'],
        config_sha256=manifest['config_sha256'],code=manifest['code'],core_identity=manifest['core_identity'],
        training_ids=manifest['training_ids'],parent_fold=1,criterion=CRITERION,
        condition='fresh_original_raw58_SmoothL1_six_targets_1000epochs')
