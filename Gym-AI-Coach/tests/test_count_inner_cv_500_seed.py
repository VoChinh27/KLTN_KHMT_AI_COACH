"""No fitting/parameter updates, including synthetic fixtures and resume tests."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.inner_cv_500_seed import manifest as m,runtime as rt,evaluation as ev,cli
from gym_coach.counting.inner_cv_500_seed.contract import Config,OBSERVATIONS
from gym_coach.counting.inner_cv_500_seed.safety import no_learning
from gym_coach.counting.inner_cv_500 import runtime as old,manifest as old_m,evaluation as old_ev
from gym_coach.counting.inner_cv_500.contract import Config as OldConfig
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.inner_cv.oof import pair_identity,metric
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.research.network import count_loss
from gym_coach.counting.models import canonical_hash,STATUS
import test_count_xy_ablation as fixtures
import test_count_train_capacity as diagnosis

TRAIN=rt.train
OLD_TRAIN=old.train


@pytest.fixture(autouse=True)
def guard():
    with no_learning():yield


def parent():
    # Synthetic values/labels; only group sizes mirror the locked assignment for directional checks.
    sizes={'g05':3,'g06':5,'g08':1,'g09':4,'g10':1,'g11':3,'g12':1,'g17':2,'g19':2,'g20':4,'g21':4,'g23':3}
    rows=[]
    for g,n in sizes.items():
        for i in range(n):
            v=f'synthetic_{g}_{i}'
            rows.append(dict(source=dict(video_id=v,group_id=g,sha256=hashlib.sha256(v.encode()).hexdigest()),
                component_id='component-'+g,target=2+i%4,label_record_hash=canonical_hash(v),
                input_status='accepted',supervised_eligible=True,role='train'))
    rows,order,folds=cv.partition(rows)
    return dict(**STATUS,rows=rows,folds=folds,config=OldConfig().model_dump(mode='json'),epochs=500,
        config_sha256=canonical_hash(OldConfig().model_dump(mode='json')),
        core_identity={},cv_code={},code={},dataset={},group_order=order,assignment_rule='fixture',
        parent_cohort=112,parent_roles={},excluded_from_inner_cv=79,oof_clips=33,oof_groups=12,
        outer_validation_used=False,development_test_used=False,development_test_previously_exposed=True,target_kind='AI_weak_fixture')


def manifest():return m.build(parent(),{}, {})


def records(man,f,side,error=.25):
    lookup={r['source']['video_id']:r for r in man['rows']}
    return [dict(**pair_identity(lookup[v]),fold=f,prediction=lookup[v]['target']+error,status='estimated',reason=None)
        for v in man['folds'][f][side+'_ids']]


def inputs(man):
    tr={f:records(man,f,'train') for f in range(3)};held={f:records(man,f,'evaluation') for f in range(3)}
    saved=dict(epoch_index=499,records=[dict(r,method=method,side=side)
        for method in ('cv500','duration','constant') for f in range(3)
        for side,key in (('train','train'),('held_out','evaluation')) for r in records(man,f,key,.5)])
    return tr,held,saved


def report(man,f,e,fixture=True):
    rs=records(man,f,'evaluation');ids=man['folds'][f]['train_ids']
    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(Config().seed+e)).tolist()
    return dict(epoch=e,fold=f,manifest_sha256=canonical_hash(man),identity_sha256=canonical_hash(m.fold_identity(man,f)),
        records=rs,metric=metric(rs),training_steps=0 if fixture else (e+1)*((len(ids)+7)//8),
        training_loss=None if fixture else .5,model_sha256='fixture',batch_order_sha256=canonical_hash([ids[i] for i in order]))


def checkpoint(tmp_path,man,f=0,e=1):
    model,opt=original.fresh_components(Config());h=original.state_hash(model.state_dict())
    history=[cv.sealed(dict(report(man,f,i),model_sha256=h)) for i in range(e+1)]
    payload=rt.checkpoint_payload(m.fold_identity(man,f),model,opt,e,0,history,fixture=True)
    path=tmp_path/f'fixture-{f}-{e}.pt';save_checkpoint(path,payload);return path,payload


def memory_subset(man):
    lookup={r['source']['video_id']:r for r in man['rows']}
    def subset(root,data,ids):
        samples=[dict(fixtures.sample(),video_id=v,target=lookup[v]['target']) for v in ids]
        return fixtures.MemoryDataset([lookup[v] for v in ids],samples)
    return subset


def test_only_seed_changes_configuration_and_rows_folds_are_deep_copies():
    p=parent();before=copy.deepcopy(p);man=m.build(p,{}, {})
    assert {k for k in p['config'] if man['config'][k]!=p['config'][k]}=={'seed'}
    for k in ('rows','folds','dataset','core_identity','cv_code'):assert man[k]==p[k]
    assert p==before and man['rows'] is not p['rows']
    assert man['expected_steps']==[1500,1500,2000] and man['observations']==list(OBSERVATIONS)
    assert cv.unseal(json.loads(json.dumps(cv.sealed(man))))==man
    assert [len(f['train_ids']) for f in man['folds']]==[20,20,26]
    assert [len(f['evaluation_ids']) for f in man['folds']]==[13,13,7]
    for value in (20261026,0,20261028):
        with pytest.raises(ValueError):Config(seed=value)


@pytest.mark.parametrize('case',['fold','group','component','source','loss','budget','seed'])
def test_reject_parent_design_drift(case):
    p=parent();lookup={r['source']['video_id']:r for r in p['rows']};a,b=[lookup[p['folds'][0][s+'_ids'][0]] for s in ('train','evaluation')]
    if case=='fold':p['folds'][0]['train_ids'].reverse()
    if case=='group':b['source']['group_id']=a['source']['group_id']
    if case=='component':b['component_id']=a['component_id']
    if case=='source':b['source']['sha256']=a['source']['sha256']
    if case=='loss':p['config']['loss']='mse_video_mean'
    if case=='budget':p['epochs']=50
    if case=='seed':p['config']['seed']=20261027
    with pytest.raises(ValueError):m.build(p,{}, {})


def test_exact_code_reuse_isolation_seed_uses_loss_optimizer_clipping():
    assert TRAIN.__code__ is OLD_TRAIN.__code__
    assert rt.validate_report.__code__ is old.validate_report.__code__
    assert rt.load_checkpoint.__code__ is old.load_checkpoint.__code__
    assert rt.train_predictions.__code__ is old_ev.train_predictions.__code__
    assert old.Config is OldConfig and rt.namespace['Config'] is Config
    assert rt.namespace['fold_identity'] is m.fold_identity and rt.namespace['count_loss'] is count_loss
    assert 'clip_grad_norm_(model.parameters(), 1.0' in inspect.getsource(TRAIN)
    models=[original.fresh_components(Config()) for _ in range(3)]
    previous,op=original.fresh_components(OldConfig())
    hashes=[original.state_hash(model.state_dict()) for model,opt in models]
    assert len(set(hashes))==1 and hashes[0]!=original.state_hash(previous.state_dict())
    assert all(opt.state_dict()==op.state_dict() and not opt.state for model,opt in models)
    assert len({next(model.parameters()).data_ptr() for model,opt in models})==3
    with pytest.raises(PermissionError):TRAIN(None,None,None,None,None)


@pytest.mark.parametrize('f',[0,1,2])
@pytest.mark.parametrize('e',[0,49,499])
def test_batch_order_new_seed_full_coverage_and_old_seed_rejected(f,e):
    man=manifest();r=report(man,f,e,False);ids=man['folds'][f]['train_ids']
    rt.validate_report(r,m.fold_identity(man,f),man,e)
    a=torch.randperm(len(ids),generator=torch.Generator().manual_seed(Config().seed+e)).tolist()
    b=torch.randperm(len(ids),generator=torch.Generator().manual_seed(Config().seed+e)).tolist()
    old_order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(OldConfig().seed+e)).tolist()
    assert a==b and a!=old_order and sorted(a)==list(range(len(ids)))
    assert not set(ids)&set(man['folds'][f]['evaluation_ids'])
    r['batch_order_sha256']=canonical_hash([ids[i] for i in old_order])
    with pytest.raises(ValueError,match='batch order'):rt.validate_report(r,m.fold_identity(man,f),man,e)


def test_forward_only_same_inputs_masks_and_weights(tmp_path,monkeypatch):
    man=manifest();man['sources']['diagnostic_tool']={}
    monkeypatch.setattr(original,'subset',memory_subset(man))
    q=rt.forward_only(None,None,man)
    assert len(q['checks'])==6 and q['checked_appearances']==99 and q['unique_clips']==33
    assert all(r['seed']==20261027 and r['initial_model_sha256']!=r['old_initial_model_sha256'] for r in q['initializations'])
    assert all(r['weights_before']==r['weights_after'] and r['input_shape'][-1]==58 for r in q['checks'])
    path,s=checkpoint(tmp_path,man,e=49)
    rs,receipt=rt.train_predictions(None,None,man,0,s,diagnosis.d)
    assert len(rs)==20 and receipt['model_before_sha256']==receipt['model_after_sha256']
    assert not receipt['optimizer_instantiated'] and not receipt['optimizer_restored']


def test_fixture_resume_restores_without_updates_and_old_loader_rejects(tmp_path):
    man=manifest();path,s=checkpoint(tmp_path,man)
    loaded=rt.load_checkpoint(path,man,0,allow_fixture=True)
    model,opt=original.fresh_components(Config());before=original.state_hash(model.state_dict())
    assert restore_state(loaded,model,opt)==(2,0)
    assert before==original.state_hash(model.state_dict())==s['model_sha256'] and not opt.state
    assert torch.equal(torch.get_rng_state(),s['torch_rng'])
    with pytest.raises(ValueError):rt.load_checkpoint(path,man,0)
    with pytest.raises(ValueError):old.load_checkpoint(path,parent(),0,allow_fixture=True)
    with pytest.raises(FileExistsError):save_checkpoint(path,s)


@pytest.mark.parametrize('case',['family','seed','fold','manifest','history','epoch','steps','weights','optimizer','missing','duplicate','target','hash','sampler'])
def test_reject_wrong_checkpoint_condition_hash_history(tmp_path,case):
    man=manifest();_,s=checkpoint(tmp_path,man)
    if case=='family':s['version']=old.CHECKPOINT_VERSION
    if case=='seed':s['identity']['config']['seed']=20261026
    if case=='fold':s['identity']['fold']=1
    if case=='manifest':s['identity']['manifest_sha256']='bad'
    if case=='history':s['history_sha256']='bad'
    if case=='epoch':s['epoch']=500
    if case=='steps':s['training_steps']=1
    if case=='weights':s['model_sha256']='bad'
    if case=='optimizer':s['optimizer']['param_groups'][0]['lr']=1
    if case in ('missing','duplicate','target'):
        r=cv.unseal(s['epoch_reports'][-1])
        if case=='missing':r['records'].pop()
        if case=='duplicate':r['records'][-1]=copy.deepcopy(r['records'][0])
        if case=='target':r['records'][0]['target']+=1
        s['epoch_reports'][-1]=cv.sealed(r);s['history_sha256']=canonical_hash(s['epoch_reports'])
    if case=='sampler':
        s['kind']='user_trained';s['training_steps']=6
        s['epoch_reports']=[cv.sealed(dict(report(man,0,e,False),model_sha256=s['model_sha256'],batch_order_sha256='bad')) for e in range(2)]
        s['history_sha256']=canonical_hash(s['epoch_reports'])
    path=tmp_path/'bad.pt';save_checkpoint(path,s)
    if case=='hash':path.write_bytes(b'bad')
    with pytest.raises(ValueError):rt.load_checkpoint(path,manifest(),0,allow_fixture=True)


def test_exhausted_resume_before_output_or_parameter_updates(tmp_path,monkeypatch):
    man=manifest();path,_=checkpoint(tmp_path,man,e=499)
    monkeypatch.setitem(rt.namespace,'load_checkpoint',lambda p,m,f:rt.load_checkpoint(p,m,f,allow_fixture=True))
    monkeypatch.setitem(rt.namespace,'subset',lambda *a:object())
    with pytest.raises(ValueError,match='budget exhausted'):TRAIN(None,None,man,0,tmp_path/'never',user_run=True,resume=path)
    assert not (tmp_path/'never').exists()


def test_metrics_pairing_equal_video_and_three_separate_directions():
    man=manifest();tr,held,saved=inputs(man)
    for r in held[0]:r['prediction']=r['target']+(-1 if r['group_id']=='g20' else 1)
    q=ev.compare_endpoint(man,tr,held,saved,diagnosis.d,499)
    assert set(q['results'])==set(ev.METHODS) and len(q['records'])==396 and len(q['paired_records'])==99
    assert len(q['fold0_held_out'])==13
    assert all(x['condition_holds'] for x in q['separate_directional_observations'][ev.NEW].values())
    assert not q['separate_directional_observations'][ev.OLD][OBSERVATIONS[2]]['condition_holds']
    a=q['results'][ev.NEW]['aggregate'];assert a['train_video_equal_MAE']==.25
    assert a['unique_videos']==33 and a['train_appearances']==66
    assert q['equal_video_comparisons']['train'][ev.OLD]['summary']['new_smaller_AE']==33
    v=next(iter(tr[0]))['video_id'];hits=[r for rs in tr.values() for r in rs if r['video_id']==v]
    hits[0]['prediction']=hits[0]['target']-1;hits[1]['prediction']=hits[1]['target']+1
    q=ev.compare_endpoint(man,tr,held,saved,diagnosis.d,499)
    assert q['results'][ev.NEW]['aggregate']['train_video_equal_MAE']==pytest.approx((32*.25+1)/33)
    assert next(r for r in q['equal_video_comparisons']['train'][ev.OLD]['records'] if r['video_id']==v)['new_AE']==1


@pytest.mark.parametrize('case',['train_missing','held_missing','duplicate','fold_missing','target','source','fold','failed','nan','control_missing','control_duplicate','control_target','endpoint'])
def test_fail_comparison_without_dropping_video(case):
    man=manifest();tr,held,saved=inputs(man);r=held[0][0]
    if case=='train_missing':tr[0].pop()
    if case=='held_missing':held[1].pop()
    if case=='duplicate':held[0][-1]=copy.deepcopy(r)
    if case=='fold_missing':del held[2]
    if case=='target':r['target']+=1
    if case=='source':r['source_sha256']='bad'
    if case=='fold':r['fold']=1
    if case=='failed':r.update(status='failed',reason='fixture',prediction=None)
    if case=='nan':r['prediction']=float('nan')
    if case=='control_missing':saved['records'].pop()
    if case=='control_duplicate':saved['records'][-1]=copy.deepcopy(saved['records'][-2])
    if case=='control_target':saved['records'][0]['target']+=1
    if case=='endpoint':saved['epoch_index']=49
    with pytest.raises(ValueError):ev.compare_endpoint(man,tr,held,saved,diagnosis.d,499)


@pytest.mark.parametrize('case',['valid','duplicate','missing','epoch'])
def test_endpoint_dispatch_across_resume_directories(tmp_path,monkeypatch,case):
    man=manifest();paths=[]
    for f in range(3):
        folder=tmp_path/('resume-'+str(f));folder.mkdir();paths.append(checkpoint(folder,man,f,49 if case=='epoch' and f==1 else 499)[0])
    monkeypatch.setitem(rt.namespace,'load_checkpoint',lambda p,m,f:rt.load_checkpoint(p,m,f,allow_fixture=True))
    if case=='duplicate':paths[1]=paths[0]
    if case=='missing':paths.pop()
    if case=='valid':
        rs,links=rt.endpoint_reports(man,paths);assert len(rs)==len(links)==3
    else:
        with pytest.raises(ValueError):rt.endpoint_reports(man,paths)


@pytest.mark.parametrize('args',[['train'],['train','--user-run'],['verify','--user-run'],['aggregate','--manifest','m'],['prepare']])
def test_cli_missing_inputs_cannot_learn(tmp_path,args):
    with pytest.raises(SystemExit) as exc:cli.main(args+['--output-dir',str(tmp_path/'new')])
    assert exc.value.code==2 and not (tmp_path/'new').exists()


def test_contract_import_lightweight():
    r=subprocess.run([sys.executable,'-c',"import sys; import gym_coach.counting.inner_cv_500_seed.contract; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


@pytest.mark.parametrize('bad',['missing_source','config','code','rows','envelope'])
def test_manifest_tamper_never_rebuilds_implicitly(tmp_path,monkeypatch,bad):
    from gym_coach.counting.readiness import write_json
    man=manifest();links={k:dict(path='missing',sha256='bad',size_bytes=1) for k in ('parent_manifest','comparison','diagnosis_report','review')}
    if bad=='missing_source':man['sources']=links
    else:
        from gym_coach.counting.research.data import identity
        source=tmp_path/'source.json';source.write_text('{}');man['sources']={k:identity(tmp_path,source) for k in links}
    pristine=copy.deepcopy(man)
    monkeypatch.setattr(m,'prepare_inputs',lambda *a:(pristine,{},{}))
    if bad=='config':man['config']['seed']=20261026
    if bad=='code':man['code']['runtime.py']['sha256']='bad'
    if bad=='rows':man['rows'][0]['target']+=1
    envelope=cv.sealed(man)
    if bad=='envelope':envelope['sha256']='bad'
    path=tmp_path/'manifest.json';write_json(path,envelope)
    with pytest.raises((ValueError,FileNotFoundError)):m.load(tmp_path,path)


@pytest.mark.parametrize('mismatch',[False,True])
def test_full_evaluator_fixed_endpoints_inference_only_and_lineage(tmp_path,monkeypatch,mismatch):
    man=manifest();man['sources']['diagnostic_tool']={}
    paths={49:[],499:[]}
    for f in range(3):
        for e in (49,499):
            folder=tmp_path/f'resume-{f}-{e}';folder.mkdir()
            path,s=checkpoint(folder,man,f,e)
            if mismatch and e==49 and f==0:
                r=cv.unseal(s['epoch_reports'][0]);r['records'][0]['prediction']+=.1;r['metric']=metric(r['records'])
                s['epoch_reports'][0]=cv.sealed(r);s['history_sha256']=canonical_hash(s['epoch_reports'])
                path=folder/'different-lineage.pt';save_checkpoint(path,s)
            paths[e].append(path)
    loader=rt.load_checkpoint
    monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    monkeypatch.setitem(rt.namespace,'load_checkpoint',rt.load_checkpoint)
    monkeypatch.setattr(ev,'diagnostic_tool',lambda *a:diagnosis.d)
    calls=[];memory=memory_subset(man)
    def subset(root,data,ids):calls.append(ids);return memory(root,data,ids)
    monkeypatch.setattr(original,'subset',subset)
    saved=inputs(man)[2];controls=dict(endpoints={str(e):dict(saved,epoch_index=e) for e in (49,499)})
    if mismatch:
        with pytest.raises(ValueError,match='lineage'):ev.evaluate(tmp_path,{},man,paths[49],paths[499],controls)
        assert not calls
    else:
        result=ev.evaluate(tmp_path,{},man,paths[49],paths[499],controls)
        assert result['train_forward_calls']==132 and result['held_out_forward_calls']==0
        assert result['new_training_steps']==result['new_baseline_fits']==0
        assert calls==[man['folds'][f]['train_ids'] for e in (49,499) for f in range(3)]
        assert result['primary_endpoint']==499 and result['descriptive_endpoint']==49
        assert all(r['model_before_sha256']==r['model_after_sha256'] for r in result['inference_receipts'])
        text=ev.report_markdown(result)
        assert 'All13 fold0' in text and 'no combined stability label' in text and 'independent final' in text
        assert cv.unseal(json.loads(json.dumps(cv.sealed(result))))==result


def test_cli_verify_no_overwrite(tmp_path,monkeypatch):
    man=manifest();man['evidence']['source_files']={'files':0}
    monkeypatch.setattr(m,'load',lambda *a:(man,{},{}))
    args=['verify','--manifest','m','--root',str(tmp_path),'--output-dir','verify']
    assert cli.main(args)==0
    with pytest.raises(SystemExit):cli.main(args)
