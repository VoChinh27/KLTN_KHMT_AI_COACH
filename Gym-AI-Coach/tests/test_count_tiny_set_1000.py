"""Budget1000 preparation tests: no learning, including fake gradient/update fixtures."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.tiny_set_1000 import contract as c,manifest as m,runtime as rt,diagnostics as d,evaluation as ev,cli
from gym_coach.counting.tiny_set_1000.safety import no_learning
from gym_coach.counting.tiny_set import manifest as old_m,runtime as old_rt,diagnostics as old_d
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.research.contract import Config as OriginalConfig
import test_count_tiny_set as old_tests
import test_count_xy_ablation as fixtures

TRAIN=rt.train


@pytest.fixture(autouse=True)
def forbid_learning():
    with no_learning():yield


def manifest():
    base=old_tests.manifest()
    evidence=dict(endpoints={'499':dict(MAE=.2,max_absolute_error=.3)},criterion=dict(passing_final_epochs=0,conclusion='NOT_FIT_IN_THIS_SETUP_AND_BUDGET'))
    return m.build(base,dict(parent_manifest={'synthetic':True},checkpoint={'synthetic':True}),evidence)


def checkpoint(tmp_path,man,epoch=1):
    model,optimizer=original.fresh_components(c.Config());base=old_tests.manifest()
    history=old_tests.fake_history(base,model,epoch+1)
    for i,envelope in enumerate(history):
        r=cv.unseal(envelope);r['identity_sha256']=canonical_hash(m.experiment_identity(man));history[i]=cv.sealed(r)
    payload=rt.checkpoint_payload(m.experiment_identity(man),model,optimizer,epoch,history,original.state_hash(model.state_dict()),fixture=True)
    path=tmp_path/f'new-{epoch}.pt';save_checkpoint(path,payload)
    return path,payload


def test_budget_only_config_and_verbatim_copied_cohort():
    base=old_tests.manifest();before=copy.deepcopy(base);new=m.build(base,{'parent_manifest':{'synthetic':True}}, {})
    assert {k for k in new['config'] if new['config'][k]!=base['config'][k]}=={'epochs'}
    assert new['config']['epochs']==new['epochs']==new['expected_total_steps']==1000
    allowed={'version','code','config','config_sha256','criterion','epochs','expected_total_steps','report_epoch_indices'}
    assert {k for k in base if new[k]!=base[k]}<=allowed
    assert {k for k in base['criterion'] if base['criterion'][k]!=new['criterion'][k]}=={'final_epoch_index'}
    for key in ('rows','training_ids','evaluation_ids','input_audit','sampler','original_config','core_identity','parent_manifest','selection_rule','candidate_rankings'):
        assert new[key]==base[key]
    assert base==before and new['rows'] is not base['rows']
    assert new['report_epoch_indices']==[49,499,999] and new['steps_per_epoch']==1
    assert cv.unseal(json.loads(json.dumps(cv.sealed(new))))==new


@pytest.mark.parametrize('case',['budget','config','order','duplicate','target','evaluation'])
def test_parent_condition_or_six_ids_cannot_be_reselected(case):
    base=old_tests.manifest()
    if case=='budget':base['epochs']=1000
    if case=='config':base['config']['learning_rate']=.01
    if case=='order':base['rows'].reverse()
    if case=='duplicate':base['training_ids'][0]=base['training_ids'][1]
    if case=='target':base['rows'][0]['target']=5
    if case=='evaluation':base['evaluation_ids'].reverse()
    with pytest.raises(ValueError):m.build(base,{'parent_manifest':{}},{})


def test_fresh_init_optimizer_raw_inputs_and_mask_match500(monkeypatch):
    old=old_tests.manifest();new=manifest();before=copy.deepcopy(new)
    a,oa=original.fresh_components(old_tests.Config());b,ob=original.fresh_components(c.Config())
    assert original.state_hash(a.state_dict())==original.state_hash(b.state_dict())
    assert oa.state_dict()==ob.state_dict() and not oa.state and oa is not ob
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    samples=[]
    for row in new['rows']:
        s=fixtures.sample();s.update(video_id=row['source']['video_id'],target=row['target']);samples.append(s)
    ds=fixtures.MemoryDataset(new['rows'],samples);monkeypatch.setattr(original,'subset',lambda *args:ds)
    raw=copy.deepcopy(samples);old_result=old_rt.forward_only(None,None,old);result=rt.forward_only(None,None,new)
    assert result==old_result and result['training_steps']==0 and result['input_shape']==[6,5,58]
    assert samples==raw and new==before


def test_training_loop_and_shared_telemetry_identical_to_frozen500_ast():
    def loop(source):
        tree=ast.parse(source);train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        return ast.dump(next(n for n in train.body if isinstance(n,ast.For) and n.target.id=='epoch'),include_attributes=False)
    assert loop(Path(old_rt.__file__).read_text())==loop(Path(rt.__file__).read_text())
    assert all(getattr(d,name) is getattr(old_d,name) for name in ('metrics','gradient_observation','parameter_change','step_log'))
    assert 'range(first,BUDGET)' in inspect.getsource(TRAIN)
    with pytest.raises(PermissionError):TRAIN(None,None,None,None)


def history():
    return [dict(epoch=i,evaluation=dict(clips=6,MAE=.1,RMSE=.12,max_absolute_error=.25)) for i in range(1000)]


@pytest.mark.parametrize('case',['all','final_only','index979_fail','index980_fail','index999_fail','earlier_streak','short','extra','duplicate','bad_clips','nan','max_fail'])
def test_exact_indices980_to999_no_off_by_one(case):
    h=history();expected=True
    if case=='final_only':
        for r in h[:980]:r['evaluation'].update(MAE=2.,max_absolute_error=5.)
    if case=='index979_fail':h[979]['evaluation']['MAE']=2.
    if case=='index980_fail':h[980]['evaluation']['MAE']=.100001;expected=False
    if case=='index999_fail':h[999]['evaluation']['MAE']=.100001;expected=False
    if case=='max_fail':h[990]['evaluation']['max_absolute_error']=.250001;expected=False
    if case=='earlier_streak':
        for r in h[980:]:r['evaluation']['MAE']=1.
        expected=False
    if case=='short':h.pop()
    if case=='extra':h.append(copy.deepcopy(h[-1]))
    if case=='duplicate':h[999]['epoch']=998
    if case=='bad_clips':h[999]['evaluation']['clips']=5
    if case=='nan':h[999]['evaluation']['MAE']=float('nan')
    if case in ('short','extra','duplicate','bad_clips','nan'):
        with pytest.raises(ValueError):d.final_criterion(h)
    else:
        q=d.final_criterion(h);assert q['met'] is expected and q['checked_epoch_indices']==list(range(980,1000))
        assert q['required_final_consecutive_epochs']==20 and not q['independent_evaluation']


def test_checkpoints_not_interchangeable_with500_and_restore_only(tmp_path):
    man=manifest();new,payload=checkpoint(tmp_path,man)
    saved=rt.load_checkpoint(new,man,allow_fixture=True)
    model,optimizer=original.fresh_components(c.Config());h=original.state_hash(model.state_dict())
    assert restore_state(saved,model,optimizer)==(2,0) and original.state_hash(model.state_dict())==h
    assert torch.equal(torch.get_rng_state(),payload['torch_rng']) and not optimizer.state
    with pytest.raises(ValueError,match='identity'):old_rt.load_checkpoint(new,old_tests.manifest(),allow_fixture=True)
    old_path,_=old_tests.checkpoint(tmp_path,old_tests.manifest())
    assert old_rt.load_checkpoint(old_path,old_tests.manifest(),allow_fixture=True)['epoch']==1
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(old_path,man,allow_fixture=True)
    with pytest.raises(ValueError,match='user-trained'):rt.load_checkpoint(new,man)
    assert rt.resume_start(dict(epoch=498))==499 and rt.resume_start(dict(epoch=998))==999
    with pytest.raises(ValueError,match='exhausted'):rt.resume_start(dict(epoch=999))
    with pytest.raises(FileExistsError):save_checkpoint(new,payload)


@pytest.mark.parametrize('field',['config','code','criterion','training_ids','old500_sources','parent_tiny_manifest'])
def test_resume_identity_strict(tmp_path,field):
    man=manifest();path,_=checkpoint(tmp_path,man);bad=copy.deepcopy(man);bad[field]={'changed':True}
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,bad,allow_fixture=True)


@pytest.mark.parametrize('case',['epoch','steps','history','metric','missing_video','duplicate_video','target','prediction','loss','batch','model','gradient','optimizer','sidecar'])
def test_report_checkpoint_rejects_corrupt_history_or_nonfinite(tmp_path,case):
    man=manifest();_,saved=checkpoint(tmp_path,man)
    if case=='epoch':saved['epoch']=1000
    if case=='steps':saved['training_steps']=2
    if case=='history':saved['epoch_reports'].pop()
    if case=='model':saved['model_sha256']='bad'
    if case=='optimizer':saved['optimizer']['param_groups'][0]['lr']=.1
    if case in ('metric','missing_video','duplicate_video','target','prediction','loss','batch','gradient'):
        r=cv.unseal(saved['epoch_reports'][-1])
        if case=='metric':r['evaluation']['MAE']=10.
        if case=='missing_video':r['records'].pop()
        if case=='duplicate_video':r['records'][-1]=copy.deepcopy(r['records'][0])
        if case=='target':r['records'][0]['target']+=1
        if case=='prediction':r['records'][0]['prediction']=None
        if case=='loss':r['training_loss']=-1
        if case=='batch':r['batch_size_actual']=5
        if case=='gradient':r['step_observation']['gradient']['gradients_finite']=False
        saved['epoch_reports'][-1]=cv.sealed(r);saved['history_sha256']=canonical_hash(saved['epoch_reports'])
    path=tmp_path/'bad.pt';save_checkpoint(path,saved)
    if case=='sidecar':path.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError):rt.load_checkpoint(path,man,allow_fixture=True)


def reproduction_payloads():
    model,_=original.fresh_components(c.Config());base=old_tests.manifest();man=manifest()
    old_history=old_tests.fake_history(base,model,500);new_history=old_tests.fake_history(base,model,1000)
    for e,envelope in enumerate(new_history):
        r=cv.unseal(envelope);r['identity_sha256']=canonical_hash(m.experiment_identity(man));r['elapsed_segment_seconds']=float(e)+1
        new_history[e]=cv.sealed(r)
    h=original.state_hash(model.state_dict())
    old=dict(epoch=499,epoch_reports=old_history,model=model.state_dict(),model_sha256=h)
    middle=dict(epoch=499,epoch_reports=new_history[:500],model=copy.deepcopy(model.state_dict()),model_sha256=h)
    final=dict(epoch=999,epoch_reports=new_history,model=model.state_dict(),model_sha256=h)
    return old,middle,final


def test_reproduction_exact_meaningful_values_ignores_only_time_identity():
    a,b,f=reproduction_payloads();q=ev.compare_500(a,b,f)
    assert q['identical'] and q['compared_epochs']==500 and q['clips']==6
    assert not q['differing_fields'] and all(t['equal'] for t in q['tensors'])
    assert all(p['difference']==0 for p in q['predictions'])


@pytest.mark.parametrize('field',['training_loss','model_sha256','batch_order_sha256','records','step_observation'])
def test_reproduction_mismatch_reported_even_tiny_no_tolerance(field):
    a,b,f=reproduction_payloads();r=copy.deepcopy(cv.unseal(a['epoch_reports'][499]))
    if field=='training_loss':r[field]+=1e-12
    if field in ('model_sha256','batch_order_sha256'):r[field]='other'
    if field=='records':r[field][0]['prediction']+=1e-12
    if field=='step_observation':r[field]['preclip_norm']+=1e-12
    a['epoch_reports'][499]=cv.sealed(r)
    q=ev.compare_500(a,b,f)
    assert q['status']=='MISMATCH_REQUIRES_INVESTIGATION'
    assert any(x['epoch_index']==499 and x['field']==field for x in q['differing_fields'])


def test_tensor_reproduction_difference_not_hidden_by_history():
    a,b,f=reproduction_payloads();next(iter(b['model'].values())).flatten()[0]+=1.
    q=ev.compare_500(a,b,f);assert not q['identical'] and any(not t['equal'] for t in q['tensors'])


@pytest.mark.parametrize('case',['missing_epoch','wrong_endpoint','wrong_lineage','missing_field'])
def test_reproduction_missing_or_wrong_pair_fails(case):
    a,b,f=reproduction_payloads()
    if case=='missing_epoch':a['epoch_reports'].pop()
    if case=='wrong_endpoint':b['epoch']=498
    if case=='wrong_lineage':b['epoch_reports']=list(reversed(b['epoch_reports']))
    if case=='missing_field':
        r=cv.unseal(a['epoch_reports'][0]);del r['training_loss'];a['epoch_reports'][0]=cv.sealed(r)
    with pytest.raises(ValueError):ev.compare_500(a,b,f)


def test_report_fixture_full1000_and_epoch500_controls_no_inference(tmp_path,monkeypatch):
    man=manifest();final,saved=checkpoint(tmp_path,man,999)
    middle=tmp_path/'middle.pt';mid=copy.deepcopy(saved);mid['epoch']=499;mid['epoch_reports']=mid['epoch_reports'][:500]
    mid['history_sha256']=canonical_hash(mid['epoch_reports']);save_checkpoint(middle,mid)
    old_path,_=old_tests.checkpoint(tmp_path,old_tests.manifest(),499)
    loader=rt.load_checkpoint;oldloader=old_rt.load_checkpoint
    monkeypatch.setattr(rt,'load_checkpoint',lambda p,m:loader(p,m,allow_fixture=True))
    monkeypatch.setattr(old_rt,'load_checkpoint',lambda p,m:oldloader(p,m,allow_fixture=True))
    monkeypatch.setattr(ev,'check_identity',lambda *args:old_path)
    monkeypatch.setattr(old_m,'load',lambda *args:(old_tests.manifest(),{}))
    result=ev.evaluate(None,man,final,middle)
    assert result['reproducibility']['identical'] and result['new1000']['total_steps']==1000
    assert set(result['new1000']['endpoints'])=={'49','499','999'} and not result['new1000']['criterion']['met']
    assert result['new1000']['no_parameter_change_steps']==1000 and result['new_training_steps']==0
    assert 'Tái lập mốc500' in ev.report_markdown(result) and 'Chưa học khớp' in ev.report_markdown(result)
    assert not result['official_eligibility'] and not result['assessment']
    with pytest.raises(ValueError,match='indices499 and999'):ev.evaluate(None,man,middle,middle)


@pytest.mark.parametrize('args',[
    ['train'],['verify','--user-run'],['verify','--resume','old.pt'],['report','--manifest','m','--checkpoint','last.pt'],
    ['report','--manifest','m','--checkpoint-500','mid.pt'],['verify','--checkpoint-500','mid.pt'],['prepare','--old-manifest','old']])
def test_cli_rejects_invalid_options_before_execution(tmp_path,args):
    with pytest.raises(SystemExit) as ex:cli.main(args+['--output-dir',str(tmp_path/'new')])
    assert ex.value.code==2 and not (tmp_path/'new').exists()


def test_cli_prepare_verify_failures_and_no_overwrite(tmp_path,monkeypatch):
    man=manifest();monkeypatch.setattr(m,'prepare_inputs',lambda *a:(man,{}));monkeypatch.setattr(m,'load',lambda *a:(man,{}))
    assert cli.main(['prepare','--old-manifest','old','--old-report-dir','report','--old-review','review','--root',str(tmp_path),'--output-dir','new'])==0
    assert cv.unseal(json.loads((tmp_path/'new/manifest.json').read_text()))==man
    with pytest.raises(SystemExit):cli.main(['verify','--manifest','m','--root',str(tmp_path),'--output-dir','new'])
    assert cli.main(['verify','--manifest','m','--root',str(tmp_path),'--output-dir','verify'])==0
    def fail(*args):raise ValueError('source linkage changed')
    monkeypatch.setattr(m,'load',fail)
    assert cli.main(['verify','--manifest','m','--root',str(tmp_path),'--output-dir','bad'])==2


def test_import_contract_without_heavy_backend():
    code="import sys; from gym_coach.counting.tiny_set_1000 import contract; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cli_saves_mismatch_evidence_returns_nonzero_without_discarding(tmp_path,monkeypatch):
    man=manifest();monkeypatch.setattr(m,'load',lambda *a:(man,{}))
    (tmp_path/'final.pt').write_bytes(b'fixture');(tmp_path/'mid.pt').write_bytes(b'fixture')
    mismatch=dict(status='REPRODUCTION_MISMATCH_REQUIRES_INVESTIGATION',history=[],reproducibility={'identical':False})
    monkeypatch.setattr(ev,'evaluate',lambda *a:copy.deepcopy(mismatch))
    monkeypatch.setattr(ev,'report_markdown',lambda r:'fixture mismatch report')
    code=cli.main(['report','--root',str(tmp_path),'--manifest','m','--checkpoint','final.pt','--checkpoint-500','mid.pt','--output-dir','report'])
    assert code==3 and (tmp_path/'report/REPORT.md').exists()
    result=cv.unseal(json.loads((tmp_path/'report/result.json').read_text()))
    assert result['status']==mismatch['status'] and result['reproducibility']['identical'] is False
