"""Pure statistics, linkage and inference-only fixtures; no parameter fitting."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from gym_coach.counting.inner_cv.oof import pair_identity
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.research.contract import Config
from gym_coach.counting.xy_ablation.transform import AblatedDataset
from gym_coach.counting.velocity.transform import VelocityDataset
import test_count_velocity as fixtures

spec=importlib.util.spec_from_file_location('train_capacity',Path(__file__).parents[1]/'scripts/diagnose_squat_count_train_capacity.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


@pytest.fixture(autouse=True)
def guard():
    with d.inference_only_guard():yield


def manifest():
    return fixtures.fixtures.base_manifest()


def records(man,fold,side,error=1.):
    lookup={r['source']['video_id']:r for r in man['rows']}
    return [dict(**pair_identity(lookup[v]),fold=fold,prediction=lookup[v]['target']+error,
                 status='estimated',reason=None) for v in man['folds'][fold][side+'_ids']]


@pytest.mark.parametrize('side',['train','evaluation'])
@pytest.mark.parametrize('change',['missing','duplicate','source','target','group','fold','null','inf'])
def test_missing_duplicate_linkage_or_invalid_predictions_rejected(side,change):
    man=manifest();rs=records(man,0,side)
    if change=='missing':rs.pop()
    if change=='duplicate':rs[-1]=copy.deepcopy(rs[0])
    if change=='source':rs[0]['source_sha256']='bad'
    if change=='target':rs[0]['target']+=1
    if change=='group':rs[0]['group_id']='bad'
    if change=='fold':rs[0]['fold']=1
    if change=='null':rs[0].update(prediction=None,status='failed',reason='fixture failure')
    if change=='inf':rs[0]['prediction']=float('inf')
    with pytest.raises(ValueError):d.validate_records(man,0,side,rs)


def test_mae_signed_errors_sd_histogram_and_fixed_checkpoint_loss():
    rs=[dict(video_id='a',target=1,prediction=0.,group_id='g01'),dict(video_id='b',target=1,prediction=3.,group_id='g02')]
    result=d.summarize(rs)
    assert result['clips']==2 and result['groups']==2 and result['MAE']==1.5
    assert result['signed_error_mean']==.5 and result['under']==1 and result['over']==1 and result['exact']==0
    assert result['predictions']==dict(n=2,minimum=0.,maximum=3.,mean=1.5,population_sd=1.5)
    assert result['target_histogram']=={'1':2} and result['fixed_checkpoint_smooth_l1']==1.


def test_equal_video_aggregation_no_independent66_or_ae_of_mean_prediction():
    man=manifest();train={f:records(man,f,'train',f+1) for f in range(3)};held={f:records(man,f,'evaluation',.25) for f in range(3)}
    result=d.aggregate_unique_videos(man,train,held)
    assert result['unique_videos']==33 and result['train_appearances']==66 and result['held_out_predictions']==33
    expected=[]
    for row in man['rows']:
        v=row['source']['video_id'];values=[r for rs in train.values() for r in rs if r['video_id']==v]
        expected.append(sum(abs(r['prediction']-r['target']) for r in values)/2)
    assert result['train_video_equal_MAE']==pytest.approx(sum(expected)/33)
    assert result['held_out_pooled_MAE']==.25
    assert result['gap_held_minus_train']==pytest.approx(.25-sum(expected)/33)
    # Opposite signed errors must not cancel through averaging the prediction first.
    row=next(r for r in man['rows'] if r['target']>=1);v=row['source']['video_id'];visits=[r for rs in train.values() for r in rs if r['video_id']==v]
    visits[0]['prediction']=row['target']-1;visits[1]['prediction']=row['target']+1
    updated=d.aggregate_unique_videos(man,train,held)
    assert next(r for r in updated['records'] if r['video_id']==v)['train_mean_AE']==1.


def test_aggregation_missing_fold_or_appearance_rejected():
    man=manifest();tr={f:records(man,f,'train') for f in range(3)};ev={f:records(man,f,'evaluation') for f in range(3)}
    with pytest.raises(ValueError):d.aggregate_unique_videos(man,{0:tr[0],1:tr[1]},ev)
    tr[2].pop()
    with pytest.raises(ValueError):d.aggregate_unique_videos(man,tr,ev)


def inference_fixture(monkeypatch,method):
    from gym_coach.counting.research.network import CountGRU,setup
    base=manifest();base['version']=d.cv.VERSION
    for r in base['rows']:r['input_artifact']={'path':'synthetic-cache','sha256':'a'*64,'size_bytes':1}
    man=(base if method=='gru' else d.xy_m.build(base,{'synthetic':True}) if method=='zero_xy' else d.v_m.build(base,{'synthetic':True},{'synthetic':True}))
    ids=man['folds'][0]['train_ids'];lookup={r['source']['video_id']:r for r in man['rows']};rows=[lookup[v] for v in ids];samples=[]
    for row in rows:
        s=fixtures.sample(n=100);s['video_id']=row['source']['video_id'];s['target']=row['target'];samples.append(s)
    raw=fixtures.fixtures.MemoryDataset(rows,samples)
    dataset=raw if method=='gru' else AblatedDataset(raw) if method=='zero_xy' else VelocityDataset(raw)
    monkeypatch.setitem(d.RUNTIMES,method,SimpleNamespace(subset=lambda root,data,subset_ids:dataset))
    setup(Config());model=CountGRU();state=model.state_dict()
    saved=dict(identity=d.MANIFESTS[method].fold_identity(man,0),epoch=49,model=state,model_sha256=d.original_rt.state_hash(state))
    return man,saved,dataset


@pytest.mark.parametrize('method',d.NEURAL)
def test_final_train_inference_uses_exact_transform_fold_and_preserves_weights(monkeypatch,method):
    man,saved,dataset=inference_fixture(monkeypatch,method);before=d.original_rt.state_hash(saved['model'])
    rs,receipt=d.infer_train(None,None,man,method,0,saved)
    assert len(rs)==20 and receipt['model_forward_calls']==20 and receipt['model_eval'] and receipt['inference_mode']
    assert receipt['model_before_sha256']==receipt['model_after_sha256']==before
    assert receipt['optimizer_instantiated'] is False and receipt['optimizer_restored'] is False
    assert [r['transformed_sample_sha256'] for r in receipt['input_samples']]==[canonical_hash(dataset[i]) for i in range(20)]
    assert d.original_rt.state_hash(saved['model'])==before


@pytest.mark.parametrize('change',['method','fold','epoch','transform'])
def test_model_transform_fold_endpoint_identity_checked_before_forward(monkeypatch,change):
    man,saved,_=inference_fixture(monkeypatch,'velocity');fold=0;method='velocity'
    if change=='method':method='gru'
    if change=='fold':fold=1
    if change=='epoch':saved['epoch']=48
    if change=='transform':saved['identity']['transform']='wrong'
    with pytest.raises(ValueError,match='identity'):d.infer_train(None,None,man,method,fold,saved)


def test_weight_invariance_failure_is_detected_without_mutating_weights(monkeypatch):
    man,saved,_=inference_fixture(monkeypatch,'gru');actual=d.original_rt.state_hash;calls=[]
    def simulated_hash(state):
        calls.append(1)
        return actual(state) if len(calls)==1 else 'simulated-mutation'
    monkeypatch.setattr(d.original_rt,'state_hash',simulated_hash)
    with pytest.raises(ValueError,match='Weights'):d.infer_train(None,None,man,'gru',0,saved)
    assert actual(saved['model'])==saved['model_sha256']


def test_top_errors_count_group_breakdown_and_zero_targets_kept():
    man=manifest();rs=records(man,0,'train');result=d.breakdown(rs)
    assert sum(v['clips'] for v in result['by_count'].values())==20
    assert sum(v['clips'] for v in result['by_group'].values())==20
    assert len(result['top5_errors'])==5
    assert result['top5_errors'][0]['video_id']==min(r['video_id'] for r in rs)


def test_manifest_discovered_from_receipt_no_default_or_guess(tmp_path):
    d.write_json(tmp_path/'receipt.json',{'command':['python','aggregate','--manifest','actual.json']})
    assert d.manifest_from_receipt(tmp_path,tmp_path)==tmp_path/'actual.json'
    other=tmp_path/'ambiguous';other.mkdir();d.write_json(other/'receipt.json',{'command':['--manifest','a','--manifest','b']})
    with pytest.raises(ValueError):d.manifest_from_receipt(tmp_path,other)


def test_guard_forbids_optimizer_creation_and_new_training():
    import torch
    with pytest.raises(AssertionError):torch.optim.AdamW([])
    with pytest.raises(AssertionError):d.v_rt.train(None,None,None,None,None,user_run=True)


@pytest.mark.parametrize('change',['bytes','missing','escape'])
def test_inventory_rejects_missing_changed_or_outside_artifact(tmp_path,change):
    bundle=tmp_path/'bundle';bundle.mkdir();item=bundle/'result.json';item.write_text('{}')
    info=d.file_info(item);d.write_json(bundle/'inventory.json',{'files':{'result.json':info}})
    d.verify_inventory(bundle)
    invalid=tmp_path/'invalid';invalid.mkdir();entries={'result.json':info}
    if change=='bytes':(invalid/'result.json').write_text('{"changed":true}')
    if change=='escape':
        outside=tmp_path/'outside.json';outside.write_text('{}')
        entries={'../outside.json':d.file_info(outside)}
    d.write_json(invalid/'inventory.json',{'files':entries})
    with pytest.raises((ValueError,OSError)):d.verify_inventory(invalid)


@pytest.mark.parametrize('change',['order','held_out_id'])
def test_exact_train_subset_checked_before_forward(monkeypatch,change):
    man,saved,dataset=inference_fixture(monkeypatch,'gru')
    dataset.rows=copy.deepcopy(dataset.rows)
    if change=='order':dataset.rows.reverse()
    else:dataset.rows[0]['source']['video_id']=man['folds'][0]['evaluation_ids'][0]
    with pytest.raises(ValueError,match='exact ordered train subset'):
        d.infer_train(None,None,man,'gru',0,saved)


def test_loaded_weight_hash_mismatch_rejected(monkeypatch):
    man,saved,_=inference_fixture(monkeypatch,'gru');saved['model_sha256']='wrong'
    with pytest.raises(ValueError,match='Loaded final weights'):
        d.infer_train(None,None,man,'gru',0,saved)


def test_script_has_no_parameter_update_fit_or_held_out_prediction_dispatch():
    import ast
    tree=ast.parse(Path(d.__file__).read_text(encoding='utf-8'))
    forbidden={'fit_baselines','fit_inner_baselines','backward','step','train','restore_state'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr=='predict' and not (isinstance(n.func.value,ast.Name) and n.func.value.id=='baselines')
                   for n in ast.walk(tree))
