"""Only forward/fixture restore/pure metrics; forbid learning throughout."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.inner_cv_500 import manifest as m,runtime as rt,evaluation as ev,reproduction as rep,cli
from gym_coach.counting.inner_cv_500.contract import Config
from gym_coach.counting.inner_cv_500.safety import no_learning
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.inner_cv.oof import pair_identity,metric
from gym_coach.counting.research.contract import Config as OriginalConfig
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json
from gym_coach.counting.research.data import identity
import test_count_xy_ablation as fixtures
import test_count_train_capacity as diagnosis

TRAIN=rt.train


@pytest.fixture(autouse=True)
def forbid_learning():
    with no_learning():yield


def base():return dict(fixtures.base_manifest(),epochs=50)
def manifest():return m.build(base(),{}, {}, {})


def records(man,f,side,error=.25):
    lookup={r['source']['video_id']:r for r in man['rows']}
    return [dict(**pair_identity(lookup[v]),fold=f,prediction=lookup[v]['target']+error,status='estimated',reason=None)
        for v in man['folds'][f][side+'_ids']]


def data(man,error=.25):
    train={f:records(man,f,'train',error) for f in range(3)};held={f:records(man,f,'evaluation',error) for f in range(3)}
    controls=[dict(r,method=method,side=side) for method in ('gru','duration','constant') for f in range(3)
        for side,key in (('train','train'),('held_out','evaluation')) for r in records(man,f,key,.5)]
    return train,held,controls


def report(man,f,e):
    rs=[{k:v for k,v in r.items() if k!='fold'} for r in records(man,f,'evaluation')]
    return dict(epoch=e,fold=f,manifest_sha256=canonical_hash(man),identity_sha256=canonical_hash(m.fold_identity(man,f)),
        records=rs,metric=metric(rs),training_steps=0,training_loss=None,model_sha256='fixture',elapsed_seconds=float(e))


def checkpoint(tmp_path,man,f=0,epoch=1):
    model,optimizer=original.fresh_components(Config());h=original.state_hash(model.state_dict());history=[]
    for e in range(epoch+1):
        r=report(man,f,e);r['model_sha256']=h;history.append(cv.sealed(r))
    payload=rt.checkpoint_payload(m.fold_identity(man,f),model,optimizer,epoch,0,history,fixture=True)
    path=tmp_path/f'new-{f}-{epoch}.pt';save_checkpoint(path,payload);return path,payload


def test_only_budget_identity_endpoints_change_preserve_all_rows_folds():
    a=base();before=copy.deepcopy(a);b=m.build(a,{}, {}, {})
    assert {k for k in b['config'] if b['config'][k]!=a['config'][k]}=={'epochs'}
    assert b['config']['loss']=='smooth_l1_beta1_video_mean' and b['epochs']==500
    for k in ('rows','folds','core_identity','cv_code'):assert a[k]==b[k]
    assert a==before and b['rows'] is not a['rows']
    assert b['expected_steps']==[1500,1500,2000] and b['steps_per_epoch']==[3,3,4]
    assert b['endpoints']==[49,499] and b['endpoint']['epoch_index']==499
    assert cv.unseal(json.loads(json.dumps(cv.sealed(b))))==b
    with pytest.raises(ValueError):Config(epochs=501)


@pytest.mark.parametrize('case',['group','source','component','fold','row','config','budget'])
def test_parent_cohort_or_config_cannot_change(case):
    b=base();look={r['source']['video_id']:r for r in b['rows']}
    a,z=[look[b['folds'][0][k+'_ids'][0]] for k in ('train','evaluation')]
    if case=='group':z['source']['group_id']=a['source']['group_id']
    if case=='source':z['source']['sha256']=a['source']['sha256']
    if case=='component':z['component_id']=a['component_id']
    if case=='fold':b['folds'][0]['train_ids'].reverse()
    if case=='row':b['rows'].pop()
    if case=='config':b['config']['loss']='mse_video_mean'
    if case=='budget':b['epochs']=500
    with pytest.raises(ValueError):m.build(b,{}, {}, {})


def test_initialization_optimizer_and_train_loop_match_original():
    a,oa=original.fresh_components(OriginalConfig());b,ob=original.fresh_components(Config())
    assert original.state_hash(a.state_dict())==original.state_hash(b.state_dict())
    assert oa.state_dict()==ob.state_dict() and not oa.state and oa is not ob
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    def update(source):
        tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        epoch=next(n for n in fn.body if isinstance(n,ast.For) and n.target.id=='epoch')
        stop=next(i for i,n in enumerate(epoch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='records' for t in n.targets))
        return [ast.dump(n,include_attributes=False) for n in epoch.body[:stop]]
    assert update(Path(original.__file__).read_text())==update(Path(rt.__file__).read_text())
    from gym_coach.counting.research.network import count_loss
    assert rt.count_loss is count_loss
    with pytest.raises(PermissionError):TRAIN(None,None,None,None,None)


def test_forward_and_train_inference_no_optimizer_no_weights_change(tmp_path,monkeypatch):
    man=manifest();lookup={r['source']['video_id']:r for r in man['rows']}
    def subset(root,data,ids):
        samples=[]
        for v in ids:
            s=fixtures.sample();s.update(video_id=v,target=lookup[v]['target']);samples.append(s)
        return fixtures.MemoryDataset([lookup[v] for v in ids],samples)
    monkeypatch.setattr(original,'subset',subset)
    q=rt.forward_only(None,None,man)
    assert len(q['checks'])==6 and q['parameters_unchanged'] and q['unique_clips']==33
    for epoch in (49,499):
        _,saved=checkpoint(tmp_path,man,epoch=epoch)
        rs,receipt=ev.train_predictions(None,None,man,0,saved,diagnosis.d)
        assert len(rs)==20 and receipt['model_before_sha256']==receipt['model_after_sha256']==saved['model_sha256']
        assert not receipt['optimizer_instantiated'] and not receipt['optimizer_restored']
    saved['epoch']=50
    with pytest.raises(ValueError):ev.train_predictions(None,None,man,0,saved,diagnosis.d)


@pytest.mark.parametrize('f,steps',[(0,1500),(1,1500),(2,2000)])
def test_step_counts_and_sampler_checked_without_updates(f,steps):
    man=manifest();r=report(man,f,499);r.update(training_steps=steps,training_loss=.5)
    ids=man['folds'][f]['train_ids'];order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(Config().seed+499)).tolist()
    r['batch_order_sha256']=canonical_hash([ids[i] for i in order]);ident=m.fold_identity(man,f)
    rt.validate_report(r,ident,man,499)
    r['training_steps']-=1
    with pytest.raises(ValueError):rt.validate_report(r,ident,man,499)
    r['training_steps']=steps;r['batch_order_sha256']='bad'
    with pytest.raises(ValueError):rt.validate_report(r,ident,man,499)


def test_resume_only_restores_fixture_state(tmp_path):
    man=manifest();path,payload=checkpoint(tmp_path,man);saved=rt.load_checkpoint(path,man,0,allow_fixture=True)
    model,optimizer=original.fresh_components(Config());before=original.state_hash(model.state_dict())
    assert restore_state(saved,model,optimizer)==(2,0)
    assert original.state_hash(model.state_dict())==before==saved['model_sha256']
    assert torch.equal(torch.get_rng_state(),payload['torch_rng']) and not optimizer.state
    with pytest.raises(ValueError,match='user-trained'):rt.load_checkpoint(path,man,0)
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,man,1,allow_fixture=True)
    with pytest.raises(ValueError,match='identity'):original.load_checkpoint(path,cv.fold_identity(base(),0),allow_fixture=True)
    with pytest.raises(FileExistsError):save_checkpoint(path,payload)
    # exhausted guard occurs before creating fresh components or restoring optimizer
    source=inspect.getsource(TRAIN)
    assert source.index('first_epoch >= config.epochs')<source.index('model, optimizer =')


@pytest.mark.parametrize('case',['family','config','fold','epoch','steps','history','model','optimizer','prediction','missing_video','target','sidecar'])
def test_loader_rejects_bad_condition_fold_hash_history(tmp_path,case):
    man=manifest();_,s=checkpoint(tmp_path,man)
    if case=='family':s['version']='squat-count-inner-cv-checkpoint-v1'
    if case=='config':s['identity']['config']['seed']=123
    if case=='fold':s['identity']['fold']=1
    if case=='epoch':s['epoch']=500
    if case=='steps':s['training_steps']=1
    if case=='history':s['history_sha256']='bad'
    if case=='model':s['model_sha256']='bad'
    if case=='optimizer':s['optimizer']['param_groups'][0]['lr']=.1
    if case in ('prediction','missing_video','target'):
        r=copy.deepcopy(cv.unseal(s['epoch_reports'][-1]))
        if case=='prediction':r['records'][0]['prediction']=None
        if case=='missing_video':r['records'].pop()
        if case=='target':r['records'][0]['target']+=1
        s['epoch_reports'][-1]=cv.sealed(r);s['history_sha256']=canonical_hash(s['epoch_reports'])
    path=tmp_path/'bad.pt';save_checkpoint(path,s)
    if case=='sidecar':path.write_bytes(b'bad')
    with pytest.raises(ValueError):rt.load_checkpoint(path,manifest(),0,allow_fixture=True)


@pytest.mark.parametrize('case',['valid49','valid499','wrong_epoch','duplicate','missing'])
def test_two_fixed_endpoints_exact_three_folds(tmp_path,monkeypatch,case):
    man=manifest();e=49 if case=='valid49' else 499
    paths=[checkpoint(tmp_path,man,f,e-1 if case=='wrong_epoch' and f==1 else e)[0] for f in range(3)]
    loader=rt.load_checkpoint;monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    if case=='duplicate':paths[1]=paths[0]
    if case=='missing':paths.pop()
    if case.startswith('valid'):
        rs,sources=rt.endpoint_reports(man,paths,e);assert len(rs)==3 and all(r['epoch']==e for r in rs)
    else:
        with pytest.raises(ValueError):rt.endpoint_reports(man,paths,e)
    with pytest.raises(ValueError):rt.endpoint_reports(man,paths,5)


def test_metrics_equal_video_aggregation_macro_groups_counts_and_pairing():
    man=manifest();tr,held,controls=data(man)
    result=ev.compare_endpoint(man,tr,held,controls,diagnosis.d,499)
    q=result['results']['cv500'];a=q['aggregate']
    assert a['train_video_equal_MAE']==a['held_out_pooled_MAE']==.25
    assert a['unique_videos']==33 and a['train_appearances']==66 and a['held_out_predictions']==33
    assert q['macro_group_MAE']==q['train_macro_group_MAE']==.25
    assert sum(v['unique_videos'] for v in q['train_unique_video_by_count'].values())==33
    assert q['held_out']['clips']==33 and q['train_appearance_descriptions']['clips']==66
    assert len(result['records'])==396 and len(result['paired_records'])==99
    assert all(p['delta_AE_cv500_minus_gru']==p['delta_AE_cv500_minus_duration']==-.25 for p in result['paired_records'])
    v=next(r['source']['video_id'] for r in man['rows'] if r['target']>0)
    hits=[r for rs in tr.values() for r in rs if r['video_id']==v]
    hits[0]['prediction']=hits[0]['target']-1;hits[1]['prediction']=hits[1]['target']+1
    q=ev.compare_endpoint(man,tr,held,controls,diagnosis.d,49)['results']['cv500']['aggregate']
    assert next(r for r in q['records'] if r['video_id']==v)['train_mean_AE']==1
    assert q['train_video_equal_MAE']==pytest.approx((32*.25+1)/33)


@pytest.mark.parametrize('case',['train_missing','fold_missing','held_duplicate','target','hash','fold','nan','failed','control_missing','control_duplicate','control_target','endpoint'])
def test_evaluator_never_drops_bad_video(case):
    man=manifest();tr,held,controls=data(man);r=held[0][0];e=499
    if case=='train_missing':tr[0].pop()
    if case=='fold_missing':del held[1]
    if case=='held_duplicate':held[0][-1]=copy.deepcopy(r)
    if case=='target':r['target']+=1
    if case=='hash':r['source_sha256']='bad'
    if case=='fold':r['fold']=1
    if case=='nan':r['prediction']=float('nan')
    if case=='failed':r.update(status='failed',prediction=None,reason='backend')
    if case=='control_missing':controls.pop()
    if case=='control_duplicate':controls[-1]=copy.deepcopy(controls[-2])
    if case=='control_target':controls[0]['target']+=1
    if case=='endpoint':e=50
    with pytest.raises(ValueError):ev.compare_endpoint(man,tr,held,controls,diagnosis.d,e)


def test_change_patterns_separate_train_oof_and_folds():
    a=dict(train_video_equal_MAE=1.,held_out_pooled_MAE=1.)
    assert ev.trend(a,dict(train_video_equal_MAE=.5,held_out_pooled_MAE=.6))['descriptive_pattern']=='train_and_oof_MAE_decreased'
    assert ev.trend(a,dict(train_video_equal_MAE=.5,held_out_pooled_MAE=1.1))['descriptive_pattern']=='train_MAE_decreased_OOF_not_decreased'
    man=manifest();tr,held,ctrl=data(man,.5);x=ev.compare_endpoint(man,tr,held,ctrl,diagnosis.d,49)
    tr,held,ctrl=data(man,.25)
    for r in held[1]:r['prediction']=r['target']+1.
    y=ev.compare_endpoint(man,tr,held,ctrl,diagnosis.d,499)
    q=ev.summarize_change({'49':x,'499':y});assert q['fold_patterns_differ']


def reproduction_payloads():
    man=manifest();model,opt=original.fresh_components(Config());h=original.state_hash(model.state_dict());old=[];new=[]
    for e in range(500):
        r=report(man,0,e);r['model_sha256']=h;r['training_loss']=.5;new.append(cv.sealed(r))
        if e<50:
            q=dict(r,manifest_sha256='old',identity_sha256='old',elapsed_seconds=9.);old.append(cv.sealed(q))
    a=dict(epoch=49,identity={'fold':0},epoch_reports=old,model=model.state_dict(),model_sha256=h)
    b=dict(epoch=49,identity={'fold':0},epoch_reports=new[:50],model=copy.deepcopy(model.state_dict()),model_sha256=h)
    f=dict(epoch=499,identity={'fold':0},epoch_reports=new,model=model.state_dict(),model_sha256=h)
    return a,b,f


def test_reproduction_checks_only_saved_fields_exactly():
    a,b,f=reproduction_payloads();q=rep.compare(a,b,f)
    assert q['identical'] and q['compared_epochs']==50
    assert any('not logged' in s for s in q['not_directly_checked'])
    assert 'batch_order_sha256' not in q['fields'] and 'training_steps' in q['fields']


@pytest.mark.parametrize('case',['loss','steps','records','tensor','lineage','missing','fold'])
def test_reproduction_detects_differences_without_tolerance(case):
    a,b,f=reproduction_payloads()
    if case in ('loss','steps','records'):
        r=copy.deepcopy(cv.unseal(a['epoch_reports'][-1]))
        if case=='loss':r['training_loss']+=1e-12
        if case=='steps':r['training_steps']+=1
        if case=='records':r['records'][0]['prediction']+=1e-12
        a['epoch_reports'][-1]=cv.sealed(r)
    if case=='tensor':next(iter(b['model'].values())).flatten()[0]+=1.
    if case=='lineage':b['epoch_reports']=b['epoch_reports'][::-1]
    if case=='missing':a['epoch_reports'].pop()
    if case=='fold':b['identity']['fold']=1
    if case in ('lineage','missing','fold'):
        with pytest.raises(ValueError):rep.compare(a,b,f)
    else:assert rep.compare(a,b,f)['status']=='MISMATCH_REQUIRES_INVESTIGATION'


@pytest.mark.parametrize('args',[
    ['train'],['train','--user-run'],['verify','--user-run'],['verify','--resume','old.pt'],
    ['aggregate','--manifest','m','--checkpoints-50','p'],['prepare','--original-manifest','old']])
def test_cli_rejects_before_learning(tmp_path,args):
    with pytest.raises(SystemExit) as exc:cli.main(args+['--output-dir',str(tmp_path/'new')])
    assert exc.value.code==2 and not (tmp_path/'new').exists()


def test_cli_verify_no_overwrite_and_mismatch_receipt(tmp_path,monkeypatch):
    man=manifest();monkeypatch.setattr(m,'load',lambda *a:(man,{},[]))
    assert cli.main(['verify','--manifest','m','--root',str(tmp_path),'--output-dir','verify'])==0
    with pytest.raises(SystemExit):cli.main(['verify','--manifest','m','--root',str(tmp_path),'--output-dir','verify'])
    result=dict(status='REPRODUCTION_MISMATCH_REQUIRES_INVESTIGATION',reproducibility={'identical':False},endpoints={})
    monkeypatch.setattr(ev,'evaluate',lambda *a:result);monkeypatch.setattr(ev,'report_markdown',lambda r:'fixture mismatch')
    assert cli.main(['aggregate','--manifest','m','--root',str(tmp_path),'--checkpoints-50','a','--checkpoints-500','b','--output-dir','report'])==3
    assert (tmp_path/'report/REPORT.md').exists()


def test_contract_light_import():
    code="import sys; from gym_coach.counting.inner_cv_500 import contract; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);assert r.returncode==0,r.stderr


def test_exhausted_resume_stops_before_initialization_or_update(tmp_path,monkeypatch):
    man=manifest();path,_=checkpoint(tmp_path,man,epoch=499)
    loader=rt.load_checkpoint
    monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    monkeypatch.setattr(rt,'subset',lambda *a:object())
    # loader may construct a reference model; train must stop before output creation.
    with pytest.raises(ValueError,match='budget exhausted'):
        TRAIN(None,None,man,0,tmp_path/'never-created',user_run=True,resume=path)
    assert not (tmp_path/'never-created').exists()


def test_complete_evaluator_two_endpoints_fixture_inference_only(tmp_path,monkeypatch):
    man=manifest();old_manifest_path=tmp_path/'old.json';old_manifest_path.write_text('{}')
    old_paths={f:tmp_path/f'old-{f}.pt' for f in range(3)}
    for p in old_paths.values():p.write_bytes(b'fixture old header; loader stubbed below')
    man['controls']=dict(original_manifest=identity(tmp_path,old_manifest_path),diagnostic_tool={},
        verified_lineage=dict(checkpoint_links={'gru':{str(f):identity(tmp_path,p) for f,p in old_paths.items()}}))
    paths={49:[],499:[]};payloads={};old_payloads={}
    for f in range(3):
        for e in (49,499):
            path,payload=checkpoint(tmp_path,man,f,e);paths[e].append(path);payloads[path]=payload
        old_payloads[old_paths[f]]=copy.deepcopy(payloads[paths[49][-1]])
    loader=rt.load_checkpoint;monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    monkeypatch.setattr(cv,'load',lambda *a:(base(),{}))
    monkeypatch.setattr(original,'load_checkpoint',lambda p,i:old_payloads[p])
    monkeypatch.setattr(ev,'diagnostic_tool',lambda *a:diagnosis.d)
    lookup={r['source']['video_id']:r for r in man['rows']}
    def subset(root,data,ids):
        samples=[]
        for v in ids:
            s=fixtures.sample();s.update(video_id=v,target=lookup[v]['target']);samples.append(s)
        return fixtures.MemoryDataset([lookup[v] for v in ids],samples)
    monkeypatch.setattr(original,'subset',subset)
    controls=data(man)[2]
    result=ev.evaluate(tmp_path,{},man,paths[49],paths[499],controls)
    assert result['reproducibility']['identical'] and len(result['checkpoint_sources'])==6
    assert len(result['inference_receipts'])==6 and result['train_forward_calls']==132 and result['held_out_forward_calls']==0
    assert result['new_training_steps']==result['new_baseline_fits']==0
    assert all(q['model_before_sha256']==q['model_after_sha256'] for q in result['inference_receipts'])
    assert set(result['endpoints'])=={'49','499'} and not result['official_eligibility'] and not result['assessment']
    text=ev.report_markdown(result)
    assert 'Errors by count, including0 and3–5' in text and 'not66independent' in text and 'not log realized batch order' in text
