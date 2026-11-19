"""MSE loss-only condition: forward/restore/statistics only; no learned updates."""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from gym_coach.counting.mse import loss,manifest as m,runtime as rt,evaluation as ev,cli
from gym_coach.counting.mse.safety import no_learning
from gym_coach.counting.inner_cv import runtime as original,manifest as cv
from gym_coach.counting.inner_cv.oof import pair_identity,metric
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json
from gym_coach.counting.research.contract import Config as OriginalConfig
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.research.data import identity
import test_count_xy_ablation as fixtures
import test_count_train_capacity as diagnosis

TRAIN=rt.train


@pytest.fixture(autouse=True)
def forbid_learning():
    with no_learning():yield


def manifest():return m.build(fixtures.base_manifest(),{'synthetic':True})


def records(man,f,side,error=.25):
    rows={r['source']['video_id']:r for r in man['rows']}
    return [dict(**pair_identity(rows[v]),fold=f,prediction=rows[v]['target']+error,status='estimated',reason=None)
            for v in man['folds'][f][side+'_ids']]


def data(man):
    train={f:records(man,f,'train') for f in range(3)};held={f:records(man,f,'evaluation') for f in range(3)}
    controls=[dict(r,method=method,side=side) for method in ev.METHODS[1:] for f in range(3)
              for side,key in (('train','train'),('held_out','evaluation')) for r in records(man,f,key,.5)]
    return train,held,controls


def report(man,f,e):
    ident=m.fold_identity(man,f);rs=[{k:v for k,v in r.items() if k!='fold'} for r in records(man,f,'evaluation')]
    return dict(epoch=e,fold=f,manifest_sha256=canonical_hash(man),identity_sha256=canonical_hash(ident),
        records=rs,metric=metric(rs),training_steps=0,training_loss=None,model_sha256='fixture')


def checkpoint(tmp_path,man,f=0,epoch=1):
    model,optimizer=original.fresh_components(loss.Config())
    param=next(model.parameters())
    optimizer.state[param]=dict(step=torch.tensor(3.),exp_avg=torch.ones_like(param),exp_avg_sq=torch.ones_like(param)*2)
    history=[]
    for e in range(epoch+1):
        r=report(man,f,e);r['model_sha256']=original.state_hash(model.state_dict());history.append(cv.sealed(r))
    payload=rt.checkpoint_payload(m.fold_identity(man,f),model,optimizer,epoch,0,history,fixture=True)
    path=tmp_path/f'fixture-{f}-{epoch}.pt';save_checkpoint(path,payload)
    return path,payload


def test_mse_formula_mask_and_reduction_without_target_transform():
    p=torch.tensor([1.,4.,100.]);b=dict(target=torch.tensor([3.,1.,1000.]),target_mask=torch.tensor([True,True,False]))
    assert float(loss.count_loss(p,b))==6.5
    assert torch.equal(p,torch.tensor([1.,4.,100.])) and torch.equal(b['target'],torch.tensor([3.,1.,1000.]))
    # Unequal batch lengths must yield a sample-weighted log, as in frozen runtime.
    whole=loss.count_loss(torch.tensor([1.,4.,2.]),dict(target=torch.tensor([3.,1.,2.]),target_mask=torch.ones(3,dtype=torch.bool)))
    assert float(whole)==pytest.approx((6.5*2+0.*1)/3)


@pytest.mark.parametrize('case',['empty','nan_prediction','inf_target','overflow','shape','mask_shape','mask_type'])
def test_invalid_or_nonfinite_mse_rejected(case):
    p=torch.tensor([1.,2.]);b=dict(target=torch.tensor([3.,4.]),target_mask=torch.tensor([True,True]))
    if case=='empty':b['target_mask'][:]=False
    if case=='nan_prediction':p[0]=float('nan')
    if case=='inf_target':b['target'][0]=float('inf')
    if case=='overflow':p[0]=1e30
    if case=='shape':p=p[:,None]
    if case=='mask_shape':b['target_mask']=torch.tensor([True])
    if case=='mask_type':b['target_mask']=torch.tensor([1,1])
    with pytest.raises(ValueError):loss.count_loss(p,b)


def test_only_config_loss_changes_and_original_default_remains_smooth_l1():
    a=OriginalConfig().model_dump();b=loss.Config().model_dump()
    assert {k for k in b if b[k]!=a[k]}=={'loss'}
    assert a['loss']=='smooth_l1_beta1_video_mean' and b['loss']=='mse_video_mean'
    from gym_coach.counting.research.network import count_loss
    batch=dict(target=torch.tensor([0.,0.]),target_mask=torch.ones(2,dtype=torch.bool))
    assert float(count_loss(torch.tensor([2.,4.]),batch))==2.5
    assert float(loss.count_loss(torch.tensor([2.,4.]),batch))==10.
    for bad in ('smooth_l1_beta1_video_mean','mse_sum','mse_weighted'):
        with pytest.raises(ValueError):loss.Config(loss=bad)


def test_fresh_model_optimizer_input_parity_and_independence(monkeypatch):
    a,oa=original.fresh_components(OriginalConfig());b,ob=original.fresh_components(loss.Config())
    assert original.state_hash(a.state_dict())==original.state_hash(b.state_dict())
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    assert oa is not ob and oa.state==ob.state=={} and oa.state_dict()==ob.state_dict()
    raw=fixtures.MemoryDataset([],[]);monkeypatch.setattr(original,'subset',lambda *a:raw)
    assert rt.subset(None,None,[]) is raw


def test_training_update_loop_sampling_weighting_clipping_identical_to_original():
    def loop(source):
        tree=ast.parse(source);train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        epoch=next(n for n in train.body if isinstance(n,ast.For) and n.target.id=='epoch')
        stop=next(i for i,n in enumerate(epoch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='records' for t in n.targets))
        return [ast.dump(n,include_attributes=False) for n in epoch.body[:stop]]
    assert loop(Path(original.__file__).read_text())==loop(Path(rt.__file__).read_text())
    assert rt.count_loss is loss.count_loss
    source=Path(rt.__file__).read_text()
    assert 'frozen_runtime.fresh_components(config)' in source
    assert "training_loss=sum(v*n for v, n in losses)/sum(n for _, n in losses)" in source
    assert 'VelocityDataset' not in source and 'AblatedDataset' not in source


@pytest.mark.parametrize('key',['group_id','sha256','component_id'])
def test_frozen_fold_leakage_still_rejected(key):
    base=fixtures.base_manifest();lookup={r['source']['video_id']:r for r in base['rows']}
    a,b=[lookup[base['folds'][0][k+'_ids'][0]] for k in ('train','evaluation')]
    if key!='component_id':a,b=a['source'],b['source']
    b[key]=a[key]
    with pytest.raises(ValueError):m.build(base,{})


def test_resume_only_restores_fixture_state_optimizer_rng_without_update(tmp_path):
    man=manifest();path,payload=checkpoint(tmp_path,man);saved=rt.load_checkpoint(path,man,0,allow_fixture=True)
    model,optimizer=original.fresh_components(loss.Config())
    assert restore_state(saved,model,optimizer)==(2,0)
    assert original.state_hash(model.state_dict())==payload['model_sha256']
    assert torch.equal(torch.get_rng_state(),payload['torch_rng'])
    assert len(optimizer.state)==1 and float(next(iter(optimizer.state.values()))['step'])==3
    with pytest.raises(ValueError,match='user-trained'):rt.load_checkpoint(path,man,0)
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,man,1,allow_fixture=True)


@pytest.mark.parametrize('key',['loss','config','code','controls','endpoint','transform'])
def test_resume_identity_rejects_other_condition_config_or_manifest(tmp_path,key):
    man=manifest();path,_=checkpoint(tmp_path,man);changed=copy.deepcopy(man);changed[key]={'changed':True}
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,changed,0,allow_fixture=True)


@pytest.mark.parametrize('family',['squat-count-inner-cv-checkpoint-v1','squat-count-xy-ablation-checkpoint-v1','squat-count-velocity-checkpoint-v1'])
def test_cannot_finetune_other_checkpoint_family(tmp_path,family):
    man=manifest();_,payload=checkpoint(tmp_path,man);payload['version']=family
    path=tmp_path/'wrong.pt';save_checkpoint(path,payload)
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,man,0,allow_fixture=True)


def test_checkpoint_hash_history_no_overwrite(tmp_path):
    man=manifest();path,payload=checkpoint(tmp_path,man)
    with pytest.raises(FileExistsError):save_checkpoint(path,payload)
    bad=tmp_path/'bad.pt';bad.write_bytes(b'bad');write_json(str(bad)+'.sha256.json',{'sha256':'wrong','size_bytes':3})
    with pytest.raises(ValueError,match='byte hash'):rt.load_checkpoint(bad,man,0,allow_fixture=True)
    payload['history_sha256']='wrong';p=tmp_path/'history.pt';save_checkpoint(p,payload)
    with pytest.raises(ValueError,match='history'):rt.load_checkpoint(p,man,0,allow_fixture=True)


@pytest.mark.parametrize('case',['complete','wrong_epoch','duplicate','missing'])
def test_exact_epoch49_all_folds_without_training(tmp_path,monkeypatch,case):
    man=manifest();paths=[checkpoint(tmp_path,man,f,48 if case=='wrong_epoch' and f==1 else 49)[0] for f in range(3)]
    loader=rt.load_checkpoint;monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    if case=='duplicate':paths[1]=paths[0]
    if case=='missing':paths.pop()
    if case=='complete':
        reports,_=rt.endpoint_reports(man,paths);assert all(r['epoch']==49 for r in reports)
    else:
        with pytest.raises(ValueError):rt.endpoint_reports(man,paths)


def test_evaluator_common_metrics_counts_signed_error_paired_deltas_and_denominators():
    man=manifest();tr,held,controls=data(man);result=ev.compare(man,tr,held,controls,diagnosis.d)
    assert len(result['records'])==594 and len(result['paired_records'])==99
    assert result['results']['mse']['aggregate']['train_video_equal_MAE']==.25
    assert result['results']['mse']['aggregate']['held_out_pooled_MAE']==.25
    assert result['results']['mse']['aggregate']['train_appearances']==66
    assert result['results']['mse']['macro_group_MAE']==.25
    assert all(r['delta_AE_mse_minus_gru']==-.25 and r['delta_AE_mse_minus_duration']==-.25 for r in result['paired_records'])
    assert result['results']['mse']['held_out']['fixed_checkpoint_MSE']==.0625
    assert result['results']['mse']['held_out']['signed_error_mean']==.25
    assert result['results']['mse']['held_out']['over']==33
    assert sum(s['clips'] for s in result['results']['mse']['held_out']['by_count'].values())==33
    assert cv.unseal(json.loads(json.dumps(cv.sealed(result))))==result
    text=ev.report_markdown(result)
    assert 'not66independent' in text and 'AI count' in text and 'Pred min/max/mean/popSD' in text
    assert result['official_eligibility'] is False and result['assessment'] is False


@pytest.mark.parametrize('change',['missing_train','missing_fold','duplicate','wrong_fold','target','hash','null','inf','control_missing','control_duplicate','control_target'])
def test_evaluator_fail_closed_no_dropping(change):
    man=manifest();tr,held,ctrl=data(man);r=held[0][0]
    if change=='missing_train':tr[0].pop()
    if change=='missing_fold':del held[1]
    if change=='duplicate':held[0][-1]=copy.deepcopy(r)
    if change=='wrong_fold':r['fold']=1
    if change=='target':r['target']+=1
    if change=='hash':r['source_sha256']='wrong'
    if change=='null':r.update(prediction=None,status='failed',reason='failed')
    if change=='inf':r['prediction']=float('inf')
    if change=='control_missing':ctrl.pop()
    if change=='control_duplicate':ctrl[-1]=copy.deepcopy(ctrl[-2])
    if change=='control_target':ctrl[0]['target']+=1
    with pytest.raises(ValueError):ev.compare(man,tr,held,ctrl,diagnosis.d)


def test_train_aggregation_does_not_cancel_opposite_errors():
    man=manifest();tr,held,ctrl=data(man)
    v=next(r['source']['video_id'] for r in man['rows'] if r['target']>=1)
    visits=[r for rs in tr.values() for r in rs if r['video_id']==v]
    visits[0]['prediction']=visits[0]['target']-1;visits[1]['prediction']=visits[1]['target']+1
    result=ev.compare(man,tr,held,ctrl,diagnosis.d)['results']['mse']['aggregate']
    assert next(r for r in result['records'] if r['video_id']==v)['train_mean_AE']==1
    assert result['train_video_equal_MAE']==pytest.approx((32*.25+1)/33)


def test_forward_and_final_train_inference_keep_weights_without_optimizer_update(tmp_path,monkeypatch):
    man=manifest();path,saved=checkpoint(tmp_path,man,epoch=49)
    rows={r['source']['video_id']:r for r in man['rows']}
    def subset(root,data,ids):
        samples=[]
        for v in ids:
            s=fixtures.sample();s['video_id']=v;s['target']=rows[v]['target'];samples.append(s)
        return fixtures.MemoryDataset([rows[v] for v in ids],samples)
    monkeypatch.setattr(original,'subset',subset)
    check=rt.forward_only(None,None,man)
    assert len(check['checks'])==6 and check['parameters_unchanged']
    rs,receipt=ev.train_predictions(None,None,man,0,saved,diagnosis.d)
    assert len(rs)==20 and receipt['model_before_sha256']==receipt['model_after_sha256']==saved['model_sha256']
    assert not receipt['optimizer_restored']
    saved['epoch']=48
    with pytest.raises(ValueError):ev.train_predictions(None,None,man,0,saved,diagnosis.d)


@pytest.mark.parametrize('change',['none','loss','config','code','endpoint','controls','original_bytes'])
def test_reload_exact_manifest_controls_code(tmp_path,monkeypatch,change):
    base=fixtures.base_manifest();data_value={'rows':base['rows']}
    for name in ('original.json','diagnostic.py','control.json'):write_json(tmp_path/name,{'synthetic':True})
    links=dict(original_manifest=identity(tmp_path,tmp_path/'original.json'),diagnostic_tool=identity(tmp_path,tmp_path/'diagnostic.py'),
        diagnostic_files={'control':identity(tmp_path,tmp_path/'control.json')},original_gru_run='orig',diagnosis_dir='diag')
    frozen=m.build(base,links);ctrl=data(frozen)[2]
    monkeypatch.setattr(m,'verify_controls',lambda *a:(base,data_value,links,ctrl))
    if change not in ('none','original_bytes'):frozen[change]={'wrong':True}
    write_json(tmp_path/'manifest.json',cv.sealed(frozen))
    if change=='original_bytes':(tmp_path/'original.json').write_text('changed')
    if change=='none':assert m.load(tmp_path,tmp_path/'manifest.json')==(frozen,data_value,ctrl)
    else:
        with pytest.raises((ValueError,KeyError)):m.load(tmp_path,tmp_path/'manifest.json')


def test_cli_user_gate_no_overwrite_failure_receipt(tmp_path,monkeypatch):
    with pytest.raises(PermissionError):TRAIN(None,None,None,None,tmp_path/'train')
    with pytest.raises(SystemExit):cli.main(['train','--manifest','none.json','--fold','0','--output-dir',str(tmp_path/'a')])
    assert not (tmp_path/'a').exists()
    monkeypatch.setattr(cli,'environment',lambda:{'synthetic_test':True})
    def bad(*a):raise ValueError('synthetic mismatch')
    monkeypatch.setattr(m,'load',bad)
    assert cli.main(['verify','--root',str(tmp_path),'--manifest','bad.json','--output-dir',str(tmp_path/'error')])==2
    assert json.loads((tmp_path/'error/failure.json').read_text())['comparison_complete'] is False
    with pytest.raises(SystemExit):cli.main(['verify','--root',str(tmp_path),'--manifest','bad.json','--output-dir',str(tmp_path/'error')])


def test_light_import_and_source_discovery_no_guessed_control_paths():
    subprocess.run([sys.executable,'-c','import sys; import gym_coach.counting.mse.manifest; assert not any(n in sys.modules for n in ("torch","numpy","cv2","mediapipe"))'],check=True)
    assert m.command_arg(['cli','--gru-run','actual'],'--gru-run')=='actual'
    for bad in ([],['--gru-run'],['--gru-run','a','--gru-run','b']):
        with pytest.raises(ValueError):m.command_arg(bad,'--gru-run')


def test_saved_batch_order_and_nonfinite_training_log_rejected_without_updates():
    man=manifest();ident=m.fold_identity(man,0);r=report(man,0,49);ids=ident['partition']['train_ids']
    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(loss.Config().seed+49)).tolist()
    r.update(training_steps=150,training_loss=1.,batch_order_sha256=canonical_hash([ids[i] for i in order]))
    rt.validate_report(r,ident,man,49)
    r['batch_order_sha256']='wrong'
    with pytest.raises(ValueError):rt.validate_report(r,ident,man,49)
    r['training_loss']=float('nan')
    with pytest.raises(ValueError):rt.validate_report(r,ident,man,49)


def test_checkpoint_report_cannot_override_wrong_record_fold():
    man=manifest();r=report(man,0,49);r['records'][0]['fold']=1
    with pytest.raises(ValueError,match='fold'):
        rt.validate_report(r,m.fold_identity(man,0),man,49,fixture=True)


def test_recorded_initial_control_weights_must_match(monkeypatch):
    man=manifest();man['controls']['original_initializations']={'0':{'model_sha256':'wrong'}}
    with pytest.raises(ValueError,match='saved original control'):rt.forward_only(None,None,man)


def test_environment_does_not_mislabel_future_user_training_as_zero(monkeypatch):
    monkeypatch.setattr(cli,'environment',lambda:dict(python='fixture',training_runs=0,training_runtime='NOT_MEASURED'))
    assert cli.runtime_environment()=={'python':'fixture'}
