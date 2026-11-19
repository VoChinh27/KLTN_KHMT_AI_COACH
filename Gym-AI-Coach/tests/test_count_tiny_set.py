"""Tiny diagnostic preparation: no backward, optimizer update or fitting, even fixtures."""
import copy
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.tiny_set import manifest as m,runtime as rt,diagnostics as d,evaluation as ev,cli
from gym_coach.counting.tiny_set.contract import Config
from gym_coach.counting.tiny_set.safety import no_learning
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.inner_cv.oof import pair_identity
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.research.contract import Config as OriginalConfig
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.readiness import write_json
import test_count_xy_ablation as fixtures

TRAIN=rt.train


@pytest.fixture(autouse=True)
def forbid_learning():
    with no_learning():yield


def manifest():return m.build(fixtures.base_manifest(),{'synthetic_fixture':True},[])


def records(man,error=.25):
    return [dict(**pair_identity(r),prediction=r['target']+error,status='estimated',reason=None) for r in man['rows']]


def fake_history(man,model,epochs=2,error=.25):
    # Telemetry fabricated ONLY for serialization tests; no loss graph or update performed.
    for p in model.parameters():p.grad=torch.ones_like(p)
    grad=d.gradient_observation(model.named_parameters(),True)
    change=d.parameter_change(model.state_dict(),model.state_dict());obs=d.step_log(grad,change,grad['observed_preclip_norm'])
    result=[];h=original.state_hash(model.state_dict());ident=m.experiment_identity(man)
    for e in range(epochs):
        order=torch.randperm(6,generator=torch.Generator().manual_seed(Config().seed+e)).tolist()
        rs=records(man,error)
        result.append(cv.sealed(dict(epoch=e,training_steps=0,identity_sha256=canonical_hash(ident),
            initial_model_sha256=h,model_sha256=h,batch_size_actual=6,
            batch_order_sha256=canonical_hash([man['training_ids'][i] for i in order]),
            training_loss=.5,records=rs,evaluation=d.metrics(man,rs),step_observation=obs)))
    return result


def checkpoint(tmp_path,man,epoch=1):
    model,optimizer=original.fresh_components(Config())
    history=fake_history(man,model,epoch+1)
    payload=rt.checkpoint_payload(m.experiment_identity(man),model,optimizer,epoch,history,
        original.state_hash(model.state_dict()),fixture=True)
    path=tmp_path/'fixture.pt';save_checkpoint(path,payload)
    return path,payload


def test_selection_deterministic_exact_six_counts_from_fold1_train():
    base=fixtures.base_manifest();rows,ranking=m.select(base)
    assert [r['target'] for r in rows]==list(range(6))
    assert {r['source']['video_id'] for r in rows}<=set(base['folds'][1]['train_ids'])
    assert not {r['source']['video_id'] for r in rows}&set(base['folds'][1]['evaluation_ids'])
    for r in rows:
        candidates=[x for x in base['rows'] if x['target']==r['target'] and x['source']['video_id'] in base['folds'][1]['train_ids']]
        assert r==min(candidates,key=lambda x:(x['source']['sha256'],x['source']['video_id']))
    before=copy.deepcopy(base);base['rows'].reverse()
    assert m.select(base)==(rows,ranking) and before['rows']==list(reversed(base['rows']))
    man=m.build(base,{},[])
    assert man['training_ids']==man['evaluation_ids'] and man['epochs']==500 and man['steps_per_epoch']==1
    assert man['report_epoch_indices']==[49,499] and not man['held_out_fold_used']


def test_tie_break_uses_video_id_with_same_hash_in_same_component():
    base=fixtures.base_manifest();ids=set(base['folds'][1]['train_ids'])
    groups={}
    for r in base['rows']:
        if r['source']['video_id'] in ids:groups.setdefault(r['source']['group_id'],[]).append(r)
    a,b=next(v[:2] for v in groups.values() if len(v)>1)
    for r in base['rows']:
        if r['source']['video_id'] in ids and r['target']==0:r['target']=5
    a['target']=b['target']=0;a['source']['sha256']=b['source']['sha256']='0'*64
    # Keep all other counts supported in remaining candidates.
    rest=[r for r in base['rows'] if r['source']['video_id'] in ids and r is not a and r is not b]
    for i in range(5):rest[i]['target']=i+1
    rows,_=m.select(base)
    assert rows[0]['source']['video_id']==min(a['source']['video_id'],b['source']['video_id'])


def test_missing_target_never_falls_back_to_fold_evaluation():
    base=fixtures.base_manifest();ids=set(base['folds'][1]['train_ids'])
    for r in base['rows']:
        if r['source']['video_id'] in ids and r['target']==5:r['target']=4
    assert any(r['target']==5 for r in base['rows'])
    with pytest.raises(ValueError,match='no count5'):m.select(base)


@pytest.mark.parametrize('case',['role','input_status','supervised_eligible','duplicate_id','fold','source_leakage','group_leakage','component_leakage','duplicate_selected_hash'])
def test_selection_fails_on_admission_partition_or_linkage(case):
    base=fixtures.base_manifest();lookup={r['source']['video_id']:r for r in base['rows']}
    a=lookup[base['folds'][1]['train_ids'][0]];b=lookup[base['folds'][1]['evaluation_ids'][0]]
    if case=='role':a['role']='validation'
    if case=='input_status':a['input_status']='rejected'
    if case=='supervised_eligible':a['supervised_eligible']=False
    if case=='duplicate_id':a['source']['video_id']=b['source']['video_id']
    if case=='fold':base['folds'][1]['train_ids'][0]=b['source']['video_id']
    if case=='source_leakage':b['source']['sha256']=a['source']['sha256']
    if case=='group_leakage':b['source']['group_id']=a['source']['group_id']
    if case=='component_leakage':b['component_id']=a['component_id']
    if case=='duplicate_selected_hash':
        rows,_=m.select(base);rows[0]['source']['sha256']=rows[1]['source']['sha256']='0'*64
    with pytest.raises(ValueError):m.select(base)


def test_config_changes_only_budget_and_initialization_optimizer_match():
    a=OriginalConfig().model_dump();b=Config().model_dump()
    assert {k for k in b if b[k]!=a[k]}=={'epochs'} and b['epochs']==500 and b['batch_size']==8
    x,ox=original.fresh_components(OriginalConfig());y,oy=original.fresh_components(Config())
    assert original.state_hash(x.state_dict())==original.state_hash(y.state_dict())
    assert ox.state_dict()==oy.state_dict() and not ox.state and ox is not oy
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(x.parameters(),y.parameters()))
    assert b['loss']=='smooth_l1_beta1_video_mean'
    with pytest.raises(ValueError):Config(epochs=501)


def test_forward_original_inputs_masks_finite_no_mutation(monkeypatch):
    man=manifest();samples=[]
    for row in man['rows']:
        s=fixtures.sample();s.update(video_id=row['source']['video_id'],target=row['target']);samples.append(s)
    ds=fixtures.MemoryDataset(man['rows'],samples);before=copy.deepcopy(samples)
    monkeypatch.setattr(original,'subset',lambda *a:ds)
    result=rt.forward_only(None,None,man)
    assert result['input_shape']==[6,5,58] and result['output_shape']==[6]
    assert result['parameters_unchanged'] and result['finite'] and result['training_steps']==0
    assert result['initial_model_sha256']==result['original_initial_sha256']==result['after_model_sha256']
    assert samples==before and result['actual_batch_size']==6 and result['batches_per_epoch']==1
    ds.rows=ds.rows[::-1]
    with pytest.raises(ValueError):rt.subset(None,None,man)


def test_mask_reset_input_audit_is_not_a_pose_accuracy_claim(monkeypatch):
    man=manifest();rows=copy.deepcopy(man['rows']);samples=[]
    for r in rows:
        r['source']['frame_count']=5;r['input_stats']=dict(complete_frames=4,longest_incomplete_span=1)
        r['input_artifact']={'sha256':'fixture'}
        s=fixtures.sample();s['video_id']=r['source']['video_id'];samples.append(s)
    ds=fixtures.MemoryDataset(rows,samples);monkeypatch.setattr(original,'subset',lambda *a:ds)
    result=m.input_audit(None,None,rows)
    assert len(result)==6 and all(r['reset_frame_indices']==[1] and r['gap_start_indices']==[1] for r in result)
    assert all(r['complete_frames']==4 and r['incomplete_frames']==1 for r in result)
    rows[0]['input_stats']['complete_frames']=5
    with pytest.raises(ValueError,match='support'):m.input_audit(None,None,rows)


def test_metrics_formula_six_denominator_signed_and_serialization():
    man=manifest();rs=records(man,0);errors=[0,1,-2,1,-2,1]
    for r,e in zip(rs,errors):r['prediction']+=e
    q=d.metrics(man,rs)
    assert q['clips']==6 and q['MAE']==pytest.approx(7/6) and q['RMSE']==pytest.approx(math.sqrt(11/6))
    assert q['max_absolute_error']==2 and q['signed_error_mean']==pytest.approx(-1/6)
    assert cv.unseal(json.loads(json.dumps(cv.sealed(q))))==q
    assert len(q['records'])==6


@pytest.mark.parametrize('case',['missing','duplicate','id','target','source','group','component','row_hash','label','null','nan','inf','negative','failed','reason','fold'])
def test_metrics_never_drop_or_substitute_bad_prediction(case):
    man=manifest();rs=records(man);r=rs[0]
    if case=='missing':rs.pop()
    if case=='duplicate':rs[-1]=copy.deepcopy(r)
    if case=='id':r['video_id']='other'
    if case=='target':r['target']+=1
    if case in ('source','group','component','row_hash','label'):
        r[dict(source='source_sha256',group='group_id',component='component_id',row_hash='row_sha256',label='label_record_hash')[case]]='bad'
    if case in ('null','nan','inf','negative'):r['prediction']=dict(null=None,nan=float('nan'),inf=float('inf'),negative=-1)[case]
    if case=='failed':r['status']='failed'
    if case=='reason':r['reason']='unavailable'
    if case=='fold':r['fold']=0
    with pytest.raises(ValueError):d.metrics(man,rs)


def criterion_history():
    return [dict(epoch=e,evaluation=dict(clips=6,MAE=.10,RMSE=.12,max_absolute_error=.25)) for e in range(500)]


@pytest.mark.parametrize('case',['all','final_only','one_mae_fail','one_max_fail','earlier_streak_only','nineteen','wrong_count','short','duplicate','nonfinite'])
def test_exact_final20_consecutive_at500_no_best_epoch(case):
    h=criterion_history();expected=True
    if case=='final_only':
        for r in h[:480]:r['evaluation'].update(MAE=2.,max_absolute_error=5.)
    if case in ('one_mae_fail','one_max_fail','nineteen'):
        h[480]['evaluation']['MAE' if case!='one_max_fail' else 'max_absolute_error']=.100001 if case!='one_max_fail' else .250001;expected=False
    if case=='earlier_streak_only':h[-1]['evaluation']['MAE']=1.;expected=False
    if case=='wrong_count':h[-1]['evaluation']['clips']=5
    if case=='short':h.pop()
    if case=='duplicate':h[-1]['epoch']=498
    if case=='nonfinite':h[-1]['evaluation']['MAE']=float('nan')
    if case in ('wrong_count','short','duplicate','nonfinite'):
        with pytest.raises(ValueError):d.final_criterion(h)
    else:
        r=d.final_criterion(h);assert r['met'] is expected and r['checked_epoch_indices']==list(range(480,500))
        assert not r['independent_evaluation']


def test_logging_fixtures_without_autograd_or_optimizer_update():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));q=torch.nn.Parameter(torch.tensor([4.]))
    p.grad=torch.tensor([3.,4.]);q.grad=torch.tensor([0.])
    grad=d.gradient_observation([('p',p),('q',q)],True)
    assert grad['observed_preclip_norm']==5. and grad['zero_gradient_tensors']==1
    before={'p':p.detach().clone(),'q':q.detach().clone()}
    after={'p':torch.tensor([2.,4.]),'q':torch.tensor([4.])}
    change=d.parameter_change(before,after)
    assert change['delta_l2']==pytest.approx(math.sqrt(5)) and change['delta_max_abs']==2. and change['changed_tensors']==1
    log=d.step_log(grad,change,5.)
    assert log['exceeded_clip_threshold'] and log['steps']==1 and torch.equal(p,torch.tensor([1.,2.]))
    assert change['before_sha256']!=change['after_sha256']


@pytest.mark.parametrize('case',['missing','nan','no_path','norm_mismatch','inf_norm'])
def test_invalid_gradient_telemetry_rejected(case):
    p=torch.nn.Parameter(torch.tensor([1.]));p.grad=torch.tensor([1.])
    if case=='missing':p.grad=None
    if case=='nan':p.grad[0]=float('nan')
    grad=d.gradient_observation([('p',p)],case!='no_path')
    change=d.parameter_change({'p':p},{'p':p})
    norm=2. if case=='norm_mismatch' else float('inf') if case=='inf_norm' else 1.
    with pytest.raises(ValueError):d.step_log(grad,change,norm)


def test_clip_boundary_zero_gradient_and_unchanged_weights_are_recorded():
    p=torch.nn.Parameter(torch.tensor([1.]));p.grad=torch.tensor([1.])
    grad=d.gradient_observation([('p',p)],True);change=d.parameter_change({'p':p},{'p':p})
    assert not d.step_log(grad,change,1.)['exceeded_clip_threshold']
    p.grad.zero_();grad=d.gradient_observation([('p',p)],True)
    assert grad['gradients_finite'] and grad['zero_gradient_tensors']==1
    assert d.step_log(grad,change,0.)['update']['delta_l2']==0


def test_resume_fixture_restores_only_and_has_same_rng_optimizer(tmp_path):
    man=manifest();path,payload=checkpoint(tmp_path,man);saved=rt.load_checkpoint(path,man,allow_fixture=True)
    assert rt.resume_start(saved)==2
    model,optimizer=original.fresh_components(Config());before=original.state_hash(model.state_dict())
    assert restore_state(saved,model,optimizer)==(2,0)
    assert original.state_hash(model.state_dict())==before==saved['model_sha256']
    assert torch.equal(torch.get_rng_state(),payload['torch_rng']) and not optimizer.state
    with pytest.raises(ValueError,match='user-trained'):rt.load_checkpoint(path,man)
    with pytest.raises(ValueError,match='exhausted'):rt.resume_start(dict(epoch=499))
    assert rt.resume_start(dict(epoch=498))==499
    with pytest.raises(FileExistsError):save_checkpoint(path,payload)


@pytest.mark.parametrize('case',['config','training_ids','criterion','code','parent_manifest','parent_fold'])
def test_manifest_identity_rejects_wrong_resume(tmp_path,case):
    man=manifest();path,_=checkpoint(tmp_path,man);bad=copy.deepcopy(man);bad[case]={'wrong':True}
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,bad,allow_fixture=True)


@pytest.mark.parametrize('case',['family','epoch','steps','history_hash','model_hash','optimizer','initial','metric','batch_order','hash_chain','gradient','update_nonfinite'])
def test_checkpoint_telemetry_integrity(tmp_path,case):
    man=manifest();_,saved=checkpoint(tmp_path,man)
    if case=='family':saved['version']='squat-count-mse-checkpoint-v1'
    if case=='epoch':saved['epoch']=500
    if case=='steps':saved['training_steps']=2
    if case=='history_hash':saved['history_sha256']='bad'
    if case=='model_hash':saved['model_sha256']='bad'
    if case=='optimizer':saved['optimizer']['param_groups'][0]['lr']=.1
    if case=='initial':saved['initial_model_sha256']='bad'
    if case in ('metric','batch_order','hash_chain','gradient','update_nonfinite'):
        report=cv.unseal(saved['epoch_reports'][0])
        if case=='metric':report['evaluation']['MAE']=100.
        if case=='batch_order':report['batch_order_sha256']='bad'
        if case=='hash_chain':report['step_observation']['update']['before_sha256']='bad'
        if case=='gradient':report['step_observation']['gradient']['tensors_with_gradient']=5
        if case=='update_nonfinite':report['step_observation']['update']['delta_l2']=-1.
        saved['epoch_reports'][0]=cv.sealed(report);saved['history_sha256']=canonical_hash(saved['epoch_reports'])
    path=tmp_path/'bad.pt';save_checkpoint(path,saved)
    with pytest.raises(ValueError):rt.load_checkpoint(path,man,allow_fixture=True)


def test_full_budget_report_from_fixture_no_inference(tmp_path,monkeypatch):
    man=manifest();path,_=checkpoint(tmp_path,man,epoch=499)
    loader=rt.load_checkpoint;monkeypatch.setattr(rt,'load_checkpoint',lambda p,m:loader(p,m,allow_fixture=True))
    result=ev.evaluate(man,path)
    assert result['total_steps']==500 and result['clips']==6 and not result['criterion']['met']
    assert set(result['endpoints'])=={'49','499'} and len(result['history'])==500
    assert result['new_forward_calls']==result['new_training_steps']==0
    assert result['no_parameter_change_steps']==500 and result['fraction_steps_exceeding_clip']==1.
    text=ev.report_markdown(result)
    assert 'Chưa học khớp trong thiết lập và ngân sách này' in text and 'not3000 independent' in text
    assert not result['official_eligibility'] and not result['assessment']
    result['criterion']['met']=True
    assert 'Đạt quy ước' in ev.report_markdown(result)


def test_report_rejects_early_checkpoint(tmp_path,monkeypatch):
    man=manifest();path,_=checkpoint(tmp_path,man,epoch=49)
    loader=rt.load_checkpoint;monkeypatch.setattr(rt,'load_checkpoint',lambda p,m:loader(p,m,allow_fixture=True))
    with pytest.raises(ValueError,match='index499'):ev.evaluate(man,path)


def test_training_requires_user_opt_in_and_single_batch_eval_contract():
    with pytest.raises(PermissionError):TRAIN(None,None,None,None)
    source=inspect.getsource(TRAIN)
    assert 'randperm(6' in source and 'batch=collate([dataset[i] for i in order])' in source
    assert source.count('optimizer.step()')==1 and source.count('loss.backward()')==1
    assert 'original.predict(model,dataset)' in source and 'range(first,BUDGET)' in source
    assert 'best' not in source and 'early_stop' not in source and 'count_loss' in source


@pytest.mark.parametrize('args',[
    ['train'],['verify','--user-run'],['verify','--resume','x'],['report'],['verify','--checkpoint','x'],['prepare','--manifest','x']])
def test_cli_invalid_arguments_cannot_start_training(tmp_path,args):
    with pytest.raises(SystemExit) as e:cli.main(args+['--output-dir',str(tmp_path/'out')])
    assert e.value.code==2 and not (tmp_path/'out').exists()


def test_cli_no_overwrite_failure_receipt_and_pure_prepare_verify(tmp_path,monkeypatch):
    man=manifest();monkeypatch.setattr(m,'prepare_inputs',lambda *a:(man,{}));monkeypatch.setattr(m,'load',lambda *a:(man,{}))
    assert cli.main(['prepare','--original-manifest','fixture','--root',str(tmp_path),'--output-dir','new'])==0
    assert cv.unseal(json.loads((tmp_path/'new/manifest.json').read_text()))==man
    with pytest.raises(SystemExit):cli.main(['verify','--manifest','fixture','--root',str(tmp_path),'--output-dir','new'])
    assert cli.main(['verify','--manifest','fixture','--root',str(tmp_path),'--output-dir','verify'])==0
    def fail(*a):raise ValueError('source mismatch')
    monkeypatch.setattr(m,'load',fail)
    assert cli.main(['verify','--manifest','fixture','--root',str(tmp_path),'--output-dir','fail'])==2
    assert json.loads((tmp_path/'fail/failure.json').read_text())['expected_clips']==6


def test_contract_import_does_not_load_heavy_backends():
    code="import sys; from gym_coach.counting.tiny_set import contract; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
