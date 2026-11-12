"""Count-stratified source-hash selection, exclusively within original fold1 train."""
from collections import Counter
import json
from pathlib import Path
from ..inner_cv import manifest as cv,runtime as original
from ..models import STATUS,canonical_hash
from ..readiness import file_info
from ..research.data import local,read,identity,check_identity
from .contract import Config,VERSION,FOLD,COUNTS,CRITERION


def code_identity():
    return {p.name:file_info(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def select(base):
    _,_,folds=cv.partition(base['rows'])
    if folds!=base['folds']:raise ValueError('Parent folds differ from frozen original assignment')
    ids=base['folds'][FOLD]['train_ids'];lookup={r['source']['video_id']:r for r in base['rows']}
    if len(ids)!=20 or len(set(ids))!=20 or not set(ids)<=set(lookup):raise ValueError('Fold1 train must contain20unique clips')
    candidates=[lookup[v] for v in ids]
    if any(r['role']!='train' or r['input_status']!='accepted' or not r['supervised_eligible']
           or type(r['target']) is not int for r in candidates):raise ValueError('Invalid parent train admission/target')
    selected=[];ranking={}
    for count in COUNTS:
        group=sorted((r for r in candidates if r['target']==count),key=lambda r:(r['source']['sha256'],r['source']['video_id']))
        if not group:raise ValueError(f'Fold1 train has no count{count}; no fallback permitted')
        ranking[str(count)]=[dict(video_id=r['source']['video_id'],source_sha256=r['source']['sha256']) for r in group]
        selected.append(group[0])
    if len({r['source']['sha256'] for r in selected})!=6:raise ValueError('Duplicate source bytes among selected six clips')
    return selected,ranking


def input_audit(root,data,rows):
    dataset=original.subset(root,data,[r['source']['video_id'] for r in rows]);result=[]
    for i,row in enumerate(rows):
        sample=dataset[i];mask=sample['complete'];n=len(mask)
        if n==0 or len(sample['values'])!=n or n!=row['source']['frame_count'] or any(type(x) is not bool for x in mask):
            raise ValueError('Selected timeline/mask mismatch')
        longest=streak=0;starts=[]
        for t,valid in enumerate(mask):
            streak=0 if valid else streak+1;longest=max(streak,longest)
            if not valid and (t==0 or mask[t-1]):starts.append(t)
        if sum(mask)!=row['input_stats']['complete_frames'] or longest!=row['input_stats']['longest_incomplete_span']:
            raise ValueError('Selected support differs from locked admission statistics')
        result.append(dict(video_id=sample['video_id'],sample_sha256=canonical_hash(sample),
            cache=row['input_artifact'],frames=n,complete_frames=sum(mask),incomplete_frames=n-sum(mask),
            complete_fraction=sum(mask)/n,longest_incomplete_span=longest,gap_start_indices=starts,
            reset_frame_indices=[t for t,v in enumerate(mask) if not v],
            padding='separate masked padding to max length in batch; not missing source frames',
            hidden_start='zero for each clip and each forward',
            limitations=['AI weak count','source camera UNKNOWN','2D pose support is not accuracy',
                         'raw58; no imputation/smoothing/learned scaling; no independent subject tracking']))
    return result


def build(base,parent_link,audit):
    rows,ranking=select(base)
    config=Config.model_validate_json(json.dumps(dict(base['config'],epochs=500))).model_dump(mode='json')
    if {k for k in config if config[k]!=base['config'][k]}!={'epochs'}:raise ValueError('Only config epochs may differ')
    if len(rows)!=6 or config['batch_size']!=8:raise ValueError('Require six samples, original batch-size setting8')
    return dict(version=VERSION,**STATUS,parent_manifest=parent_link,dataset=base.get('dataset'),
        core_identity=base['core_identity'],original_cv_code=base['cv_code'],code=code_identity(),
        original_config=base['config'],config=config,config_sha256=canonical_hash(config),
        parent_fold=FOLD,parent_train_ids=base['folds'][FOLD]['train_ids'],
        selection_rule='one per count0..5 from fold1 train only; ascending source byte SHA256 then video_id; output order count0..5',
        candidate_rankings=ranking,parent_train_count_distribution=dict(sorted(Counter(str(r['target']) for r in base['rows'] if r['source']['video_id'] in base['folds'][FOLD]['train_ids']).items())),
        rows=rows,training_ids=[r['source']['video_id'] for r in rows],evaluation_ids=[r['source']['video_id'] for r in rows],
        evaluation_role='same_six_training_videos_in_sample_only',input_audit=audit,criterion=CRITERION,
        epochs=500,steps_per_epoch=1,expected_total_steps=500,report_epoch_indices=[49,499],
        initialization='fresh original GRU; no full-CV/MSE/other weights accepted',
        sampler='same seed+epoch randperm; one batch contains all six clips; batch_size setting8',
        legacy_config_selection='inherited descriptive field only; NOT executed; no validation/best checkpoint selection in tiny runtime',
        not_single_factor_comparison=True,outer_validation_used=False,development_test_used=False,held_out_fold_used=False,
        target_kind='AI_weak_reference_not_independent_ground_truth',final_refit=False,
        scope='tiny-set learning-capacity diagnostic; no generalization, model selection, accuracy or form validation')


def prepare_inputs(root,parent_path):
    path=local(root,parent_path);base,data=cv.load(root,path);rows,_=select(base)
    audit=input_audit(root,data,rows)
    return build(base,identity(root,path),audit),data


def load(root,path):
    saved=cv.unseal(read(local(root,path)));parent=check_identity(root,saved['parent_manifest'])
    expected,data=prepare_inputs(root,parent)
    if saved!=expected:raise ValueError('Tiny manifest/code/config/criterion/selection/source mismatch; no migration')
    return saved,data


def experiment_identity(manifest):
    return dict(version=VERSION,manifest_sha256=canonical_hash(manifest),config=manifest['config'],
        config_sha256=manifest['config_sha256'],code=manifest['code'],core_identity=manifest['core_identity'],
        training_ids=manifest['training_ids'],parent_fold=FOLD,criterion=CRITERION,
        condition='fresh_original_raw58_SmoothL1_six_targets_500epochs')
