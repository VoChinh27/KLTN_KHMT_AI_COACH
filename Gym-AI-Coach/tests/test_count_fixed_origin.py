"""Synthetic/forward only. All parameter learning is forbidden, including fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.fixed_origin import transform as t,manifest as m,runtime as rt,evaluation as ev,cli
from gym_coach.counting.fixed_origin.contract import SEEDS,config_type
from gym_coach.counting.fixed_origin.safety import no_learning
from gym_coach.counting.body_normalized import transform as old_t,runtime as old_rt
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.inner_cv.oof import metric
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.research.network import count_loss
from gym_coach.counting.readiness import write_json
import test_count_body_normalized as fixture
import test_count_train_capacity as diagnosis

TRAIN=rt.train


@pytest.fixture(autouse=True)
def guard():
    with no_learning():yield


def manifest():
    parent=fixture.manifest();audit={};checks={}
    source=fixture.memory(parent)(None,None,[r['source']['video_id'] for r in parent['rows']])
    for i in range(len(source)):
        raw=source[i];v=raw['video_id'];new,audit[v]=t.normalize(raw,parent['input_audit'][v])
        checks[v]=t.independent_check(raw,new,parent['input_audit'][v])
    evidence=dict(parent['sources'],parent_manifest={'path':'fixture.json','sha256':'fixture','size_bytes':0})
    return m.build(parent,evidence,audit,checks)


def inputs(man,error=.25,body_error=.5):
    train,held,raw=fixture.inputs(man,error)
    body=[dict(r,method='body_normalized') for r in raw if r['method']=='gru']
    for r in body:r['prediction']=r['target']+body_error
    return train,held,dict(raw=raw,body=body)


def checkpoint(folder,man,seed=SEEDS[0],fold=0,epoch=1):
    engine,selected,_=rt.engine(man,seed)
    model,opt=original.fresh_components(config_type(seed)());h=original.state_hash(model.state_dict())
    source=fixture.memory(man)(None,None,man['folds'][fold]['evaluation_ids'])
    dataset=t.FixedDataset(source,man['input_audit'],man['parent_input_audit'])
    records=original.predict(model,dataset)
    history=[cv.sealed(dict(epoch=e,fold=fold,manifest_sha256=canonical_hash(selected),identity_sha256=canonical_hash(m.fold_identity(selected,fold)),
                           records=records,metric=metric(records),training_steps=0,training_loss=None,model_sha256=h)) for e in range(epoch+1)]
    saved=engine.checkpoint_payload(m.fold_identity(selected,fold),model,opt,epoch,0,history,fixture=True)
    path=folder/f'{seed}-{fold}-{epoch}.pt';save_checkpoint(path,saved)
    return path,saved


@pytest.mark.parametrize('hips,mask,expected',[
    ([(0,0),(10,8),(2,4),(6,2)],[True]*4,[4,3]),
    ([(0,0),(999,999),(2,4),(6,2)],[True,False,True,True],[2,2]),
    ([(5,7)],[True],[5,7])])
def test_independent_median_motion_relation_preserved_channels_and_copy(hips,mask,expected):
    raw=fixture.sample(lengths=(2.,)*len(hips),complete=mask)
    for i,(x,y) in enumerate(hips):
        for j in range(8):raw['values'][i][7*j]+=x-2;raw['values'][i][7*j+1]+=y-5
    before=copy.deepcopy(raw);old,locked=old_t.normalize(raw);new,audit=t.normalize(raw,locked)
    assert audit['origin']==expected and audit['scale']==locked['scale']==2.
    assert raw==before and new['complete']==mask and new['duration_seconds']==raw['duration_seconds']
    for i,row in enumerate(raw['values']):
        for col,value in enumerate(row):
            if not mask[i] or col>=56 or col%7>1:assert new['values'][i][col]==value
            else:
                k=col%7;assert new['values'][i][col]-old['values'][i][col]==pytest.approx((hips[i][k]-expected[k])/2)
    independent=t.independent_check(raw,new,locked)
    assert independent['origin']==expected and independent['median_hip_residual']<1e-10
    assert independent['adjacent_complete_edges']==[[i-1,i] for i in range(1,len(mask)) if mask[i] and mask[i-1]]
    assert t.normalize(raw,locked)==(new,audit)
    with pytest.raises(ValueError):t.normalize(new,locked)


@pytest.mark.parametrize('case',['scale','origin_input','mask','nan','inf','empty','hash'])
def test_invalid_source_or_parent_scale_never_fallback(case):
    raw=fixture.sample();_,locked=old_t.normalize(raw)
    if case=='scale':locked['scale']*=2
    if case=='origin_input':raw['values'][0][14]+=1
    if case=='mask':raw['complete'][0]=False
    if case=='nan':raw['values'][0][0]=float('nan')
    if case=='inf':raw['values'][0][0]=float('inf')
    if case=='empty':raw['values']=[]
    if case=='hash':locked['normalized_sample_sha256']='bad'
    with pytest.raises((ValueError,TypeError)):t.normalize(raw,locked)


@pytest.mark.parametrize('seed',SEEDS)
def test_only_origin_config_changes_loop_initialization_optimizer_and_subsets(seed,monkeypatch):
    man=manifest();engine,selected,ns=rt.engine(man,seed)
    assert engine.train.__code__ is old_rt.TEMPLATE.train.__code__ and ns['count_loss'] is count_loss
    old_cfg=fixture.config_type(seed)().model_dump(mode='json');cfg=config_type(seed)().model_dump(mode='json')
    assert {k for k in cfg if cfg[k]!=old_cfg[k]}=={'normalization'}
    models=[original.fresh_components(config_type(seed)()) for _ in range(3)]
    reference,opt=original.fresh_components(fixture.config_type(seed)())
    assert len({next(model.parameters()).data_ptr() for model,_ in models})==3
    assert all(original.state_hash(model.state_dict())==original.state_hash(reference.state_dict()) and optimizer.state_dict()==opt.state_dict() and not optimizer.state for model,optimizer in models)
    monkeypatch.setattr(original,'subset',fixture.memory(man))
    a=engine.subset(None,None,man['folds'][0]['train_ids']);b=engine.subset(None,None,man['folds'][1]['evaluation_ids'])
    aa={a[i]['video_id']:a[i] for i in range(len(a))};bb={b[i]['video_id']:b[i] for i in range(len(b))}
    assert set(aa)&set(bb) and all(aa[v]==bb[v] for v in set(aa)&set(bb))
    with pytest.raises(PermissionError):TRAIN(None,None,man,seed,0,None)


def test_forward_all_subsets_and_inputs_finite_weights_unchanged(monkeypatch):
    man=manifest();monkeypatch.setattr(original,'subset',fixture.memory(man))
    q=rt.forward_only(None,None,man)
    assert q['forward_calls']==198 and q['unique_videos']==33
    assert all(x['initial_model_sha256']==x['model_after_sha256'] for x in q['checks'])
    assert sum(x['videos'] for x in q['checks'])==198


@pytest.mark.parametrize('seed',SEEDS)
def test_checkpoint_restore_without_learning(tmp_path,seed):
    man=manifest();path,saved=checkpoint(tmp_path,man,seed)
    engine,selected,_=rt.engine(man,seed);cp=engine.load_checkpoint(path,selected,0,allow_fixture=True)
    model,opt=original.fresh_components(config_type(seed)());h=original.state_hash(model.state_dict())
    assert restore_state(cp,model,opt)==(2,0)
    assert h==original.state_hash(model.state_dict()) and not opt.state
    with pytest.raises(ValueError):engine.load_checkpoint(path,selected,0)
    other,sel,_=rt.engine(man,SEEDS[1] if seed==SEEDS[0] else SEEDS[0])
    with pytest.raises(ValueError):other.load_checkpoint(path,sel,0,allow_fixture=True)


@pytest.mark.parametrize('case',['family','scale','origin','fold','seed','config','manifest','epoch','history','weights','hash','parent'])
def test_checkpoint_identity_rejects_incompatible_or_corrupt(tmp_path,case):
    man=manifest();path,saved=checkpoint(tmp_path,man)
    if case=='family':saved['version']='squat-count-body-normalized-gru500-v1-checkpoint'
    if case=='scale':saved['identity']['input_audit_sha256']='bad'
    if case=='origin':saved['identity']['transform']['origin']='per_frame'
    if case=='fold':saved['identity']['fold']=1
    if case=='seed':saved['identity']['seed']=SEEDS[1]
    if case=='config':saved['identity']['config']['learning_rate']=.5
    if case=='manifest':saved['identity']['manifest_sha256']='bad'
    if case=='epoch':saved['epoch']=500
    if case=='history':saved['history_sha256']='bad'
    if case=='weights':saved['model_sha256']='bad'
    if case=='parent':saved['identity']['parent_manifest']['sha256']='bad'
    bad=tmp_path/'bad.pt';save_checkpoint(bad,saved)
    if case=='hash':bad.write_bytes(b'corrupt')
    engine,selected,_=rt.engine(manifest(),SEEDS[0])
    with pytest.raises(ValueError):engine.load_checkpoint(bad,selected,0,allow_fixture=True)


def test_equal_video_error_metrics_seed_direction_and_focus_tables():
    man=manifest();tr,held,controls=inputs(man)
    v=tr[0][0]['video_id'];hits=[r for rs in tr.values() for r in rs if r['video_id']==v]
    hits[0]['prediction']=hits[0]['target']-1;hits[1]['prediction']=hits[1]['target']+1
    a=ev.compare(man,SEEDS[0],tr,held,controls,diagnosis.d)
    assert a['results']['fixed_origin']['aggregate']['train_video_equal_MAE']==pytest.approx((32*.25+1)/33)
    assert len(a['fold1_held_out'])==len(a['fold0_held_out'])==13 and len(a['paired_records'])==99
    tr,held,controls=inputs(man,error=.75)
    b=ev.compare(man,SEEDS[1],tr,held,controls,diagnosis.d);combined=ev.combine({str(SEEDS[0]):a,str(SEEDS[1]):b})
    assert combined['methods']['fixed_origin']['OOF_MAE']==.5
    assert combined['descriptive_result']==combined['fold1_direction']=='inconsistent_across_seeds'
    assert combined['methods']['fixed_origin']['macro_group_MAE']==.5
    assert a['paired_equal_video']['held_out']['body_normalized']['improved']==33


@pytest.mark.parametrize('case',['train_missing','held_missing','duplicate','target','source','fold','failed','nan','body_missing','raw_duplicate'])
def test_comparison_cannot_drop_or_mislink_video(case):
    man=manifest();tr,held,c=inputs(man);r=held[0][0]
    if case=='train_missing':tr[0].pop()
    if case=='held_missing':held[0].pop()
    if case=='duplicate':held[0][-1]=copy.deepcopy(r)
    if case=='target':r['target']+=1
    if case=='source':r['source_sha256']='bad'
    if case=='fold':r['fold']=1
    if case=='failed':r.update(prediction=None,status='failed',reason='fixture')
    if case=='nan':r['prediction']=float('nan')
    if case=='body_missing':c['body'].pop()
    if case=='raw_duplicate':c['raw'][-1]=copy.deepcopy(c['raw'][0])
    with pytest.raises(ValueError):ev.compare(man,SEEDS[0],tr,held,c,diagnosis.d)


@pytest.mark.parametrize('case',['valid','missing','duplicate','endpoint','replay_mismatch'])
def test_six_endpoint_aggregate_replays_exactly_no_training(tmp_path,monkeypatch,case):
    man=manifest();paths=[checkpoint(tmp_path,man,s,f,49 if case=='endpoint' and f==2 else 499)[0] for s in SEEDS for f in range(3)]
    if case=='missing':paths.pop()
    if case=='duplicate':paths[-1]=paths[0]
    create=rt.engine
    def fixture_engine(*a):
        engine,selected,ns=create(*a);load=engine.load_checkpoint
        engine.load_checkpoint=lambda p,m,f:load(p,m,f,allow_fixture=True)
        return engine,selected,ns
    monkeypatch.setattr(rt,'engine',fixture_engine);monkeypatch.setattr(original,'subset',fixture.memory(man));monkeypatch.setattr(ev,'diagnostic_tool',lambda *a:diagnosis.d)
    if case=='replay_mismatch':
        predict=rt.train_predictions
        def changed(*a,**kw):
            rs,q=predict(*a,**kw)
            if kw.get('side')=='evaluation':rs[0]['prediction']+=.01
            return rs,q
        monkeypatch.setattr(rt,'train_predictions',changed)
    controls={str(s):inputs(man)[2] for s in SEEDS}
    if case!='valid':
        with pytest.raises(ValueError):ev.evaluate(tmp_path,{},man,paths,controls)
    else:
        q=ev.evaluate(tmp_path,{},man,paths,controls)
        assert q['train_forward_calls']==132 and q['held_forward_calls']==66
        assert len(q['inference_receipts'])==12 and all(r['before_model_sha256']==r['after_model_sha256'] for r in q['inference_receipts'])
        assert cv.unseal(json.loads(json.dumps(cv.sealed(q))))==q
        text=ev.report(q);assert all(g in text for g in ('g11','g20','g17','g19','NOT_VALIDATED'))


@pytest.mark.parametrize('case',['code','config','environment','scale','origin','envelope'])
def test_manifest_fail_closed(tmp_path,monkeypatch,case):
    man=manifest();pristine=copy.deepcopy(man);monkeypatch.setattr(m,'prepare',lambda *a:(pristine,{},{}))
    if case=='code':man['code']['runtime.py']['sha256']='bad'
    if case=='config':man['configs'][str(SEEDS[0])]['seed']=SEEDS[1]
    if case=='environment':man['environment']['torch']='bad'
    v=next(iter(man['input_audit']))
    if case=='scale':man['input_audit'][v]['scale']*=2
    if case=='origin':man['input_audit'][v]['origin'][0]+=1
    env=cv.sealed(man)
    if case=='envelope':env['sha256']='bad'
    path=tmp_path/'manifest.json';write_json(path,env)
    with pytest.raises(ValueError):m.load(tmp_path,path)


@pytest.mark.parametrize('args',[['train'],['train','--user-run'],['verify','--user-run'],['aggregate','--manifest','missing']])
def test_cli_rejects_incomplete_learning_options(tmp_path,args):
    with pytest.raises(SystemExit):cli.main(args+['--output-dir',str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()


def test_lightweight_transform_import():
    r=subprocess.run([sys.executable,'-c',"import sys;import gym_coach.counting.fixed_origin.transform;assert not set(('torch','numpy','cv2','mediapipe','scipy'))&set(sys.modules)"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
