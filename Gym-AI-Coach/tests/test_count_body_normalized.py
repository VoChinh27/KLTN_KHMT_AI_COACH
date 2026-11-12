"""All tests prohibit learning, including fixtures and checkpoint resume."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from gym_coach.counting.body_normalized import transform as t,manifest as m,runtime as rt,evaluation as ev,cli
from gym_coach.counting.body_normalized.contract import SEEDS,EPSILON,TOLERANCE,config_type,ENDPOINT
from gym_coach.counting.body_normalized.safety import no_learning
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.inner_cv.oof import metric
from gym_coach.counting.inner_cv_500.contract import Config as Raw26
from gym_coach.counting.inner_cv_500_seed.contract import Config as Raw27
from gym_coach.counting.research.experiment import save_checkpoint,restore_state
from gym_coach.counting.research.network import count_loss
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.readiness import write_json
import test_count_inner_cv_500_seed as fixture
import test_count_xy_ablation as xy
import test_count_train_capacity as diagnosis

TRAIN=rt.train


@pytest.fixture(autouse=True)
def guarded():
    with no_learning():yield


def sample(v='fixture',lengths=(2.,4.,6.,8.),complete=None):
    rows=[]
    for i,length in enumerate(lengths):
        joints=[(1,5-length),(3,5-length),(1,5),(3,5),(1,7),(3,7),(1,9),(3,9)]
        row=[]
        for j,(x,y) in enumerate(joints):row.extend([float(x),float(y),.5+j*.05,1.,1.,0.,0.])
        rows.append(row+[i/30.,0. if i==0 else 1/30.])
    return dict(video_id=v,values=rows,complete=list(complete or [True]*len(rows)),duration_seconds=len(rows)/30.,target=3)


def manifest():
    base=fixture.parent();audit={};controls={}
    for seed,raw in zip(SEEDS,(Raw26,Raw27)):
        model,opt=original.fresh_components(raw())
        h=original.state_hash(model.state_dict())
        controls[str(seed)]=dict(raw_config=raw().model_dump(mode='json'),initializations={str(f):h for f in range(3)})
    for row in base['rows']:
        v=row['source']['video_id']
        _,audit[v]=t.normalize(dict(sample(v),target=row['target']))
    return m.build(base,dict(controls=controls,sources=[],diagnostic_tool={}),audit)


def memory(man):
    lookup={r['source']['video_id']:r for r in man['rows']}
    def subset(root,data,ids):
        return xy.MemoryDataset([lookup[v] for v in ids],[dict(sample(v),target=lookup[v]['target']) for v in ids])
    return subset


def inputs(man,error=.25):
    train={f:fixture.records(man,f,'train',error) for f in range(3)}
    held={f:fixture.records(man,f,'evaluation',error) for f in range(3)}
    controls=[dict(r,method=method,side=side) for method in ('gru','duration','constant') for f in range(3)
        for side,key in (('train','train'),('held_out','evaluation')) for r in fixture.records(man,f,key,.5)]
    return train,held,controls


def report(selected,f,e,h,fixture_only=True):
    ids=selected['folds'][f]['train_ids'];rs=fixture.records(selected,f,'evaluation')
    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(selected['seed']+e)).tolist()
    return dict(epoch=e,fold=f,manifest_sha256=canonical_hash(selected),identity_sha256=canonical_hash(m.fold_identity(selected,f)),
        records=rs,metric=metric(rs),training_steps=0 if fixture_only else (e+1)*((len(ids)+7)//8),
        training_loss=None if fixture_only else .5,model_sha256=h,batch_order_sha256=canonical_hash([ids[i] for i in order]))


def checkpoint(folder,man,seed=SEEDS[0],fold=0,epoch=1):
    engine,selected,_=rt.engine(man,seed)
    model,opt=original.fresh_components(config_type(seed)());h=original.state_hash(model.state_dict())
    history=[cv.sealed(report(selected,fold,e,h)) for e in range(epoch+1)]
    saved=engine.checkpoint_payload(m.fold_identity(selected,fold),model,opt,epoch,0,history,fixture=True)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{seed}-{fold}-{epoch}.pt';save_checkpoint(path,saved)
    return path,saved


def test_exact_median_joints_channels_mask_timeline_nonmutation():
    source=sample(complete=[True,False,True,True]);before=copy.deepcopy(source)
    value,q=t.normalize(source)
    assert q['scale']==6 and q['scale_frames']==q['complete_frames']==3 and q['excluded_scale_frames']==0
    assert source==before and value['values'][1]==source['values'][1]
    assert value['complete']==source['complete'] and value['duration_seconds']==source['duration_seconds']
    for i,valid in enumerate(source['complete']):
        for j in range(58):
            expected=source['values'][i][j]
            if valid and j in t.XY:expected=(expected-(2 if j%7==0 else 5))/6
            assert value['values'][i][j]==expected
        if valid:assert max(abs(v) for v in t.midpoint(value['values'][i],2,3))<=TOLERANCE
    assert q['original_sample_sha256']==canonical_hash(source) and q['normalized_sample_sha256']==canonical_hash(value)
    assert t.normalize(source)==(value,q)


@pytest.mark.parametrize('lengths,median',[((2.,4.),3.),((2.,4.,100.),4.),((0.,EPSILON,2.),2.),((2.,),2.)])
def test_video_scale_median_valid_samples_not_framewise(lengths,median):
    _,q=t.normalize(sample(lengths=lengths))
    assert q['scale']==pytest.approx(median)


@pytest.mark.parametrize('scale,shift',[(1.,(7.,-3.)),(3.,(0.,0.)),(.5,(1.,2.)),(7.,(-5.,2.))])
def test_complete_coordinates_translation_positive_scale_invariant(scale,shift):
    original_sample=sample();changed=copy.deepcopy(original_sample)
    for row in changed['values']:
        for j in range(8):
            for a in (0,1):row[j*7+a]=row[j*7+a]*scale+shift[a]
    a,qa=t.normalize(original_sample);b,qb=t.normalize(changed)
    assert qb['scale']==pytest.approx(qa['scale']*scale)
    for ar,br in zip(a['values'],b['values']):assert br==pytest.approx(ar,abs=TOLERANCE)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),None,'x',True])
@pytest.mark.parametrize('incomplete',[False,True])
def test_invalid_values_cannot_reach_collate(bad,incomplete):
    source=sample();source['values'][1][0]=bad;source['complete'][1]=not incomplete
    with pytest.raises(t.NormalizationError,match='nonfinite_or_invalid_input'):t.normalize(source)


@pytest.mark.parametrize('case',['zero','no_complete','tiny','shape','mask','nonfinite_output'])
def test_degenerate_scale_and_invalid_shape_explicit_failure(case):
    source=sample()
    if case=='zero':source=sample(lengths=(0.,0.))
    if case=='tiny':source=sample(lengths=(EPSILON/4,EPSILON/4))
    if case=='no_complete':source['complete']=[False]*4
    if case=='shape':source['values'][0].pop()
    if case=='mask':source['complete'][0]=1
    if case=='nonfinite_output':source['values'][0][28]=1e308
    with pytest.raises(t.NormalizationError):t.normalize(source)


def test_label_not_consulted_dataset_cache_copies_and_drift():
    source=sample();changed=dict(source,target=100)
    a,_=t.normalize(source);b,_=t.normalize(changed)
    assert a['values']==b['values'] and a['complete']==b['complete']
    man=manifest();raw=memory(man)(None,None,man['folds'][0]['train_ids'])
    dataset=t.BodyDataset(raw,man['input_audit']);before=canonical_hash(dataset[0])
    value=dataset[0];value['values'][0][0]+=1
    assert canonical_hash(dataset[0])==before
    altered=copy.deepcopy(man['input_audit']);altered[dataset[0]['video_id']]['scale']+=1
    with pytest.raises(ValueError,match='identity changed'):t.BodyDataset(raw,altered)


def test_config_cohort_fold_coverage_endpoint_and_roundtrip():
    man=manifest();base=fixture.parent()
    assert man['rows']==base['rows'] and man['folds']==base['folds'] and man['rows'] is not base['rows']
    assert len(man['input_audit'])==33 and len({r['source']['group_id'] for r in man['rows']})==12
    assert [len(f['train_ids']) for f in man['folds']]==[20,20,26]
    assert [len(f['evaluation_ids']) for f in man['folds']]==[13,13,7]
    assert len(set(v for f in man['folds'] for v in f['evaluation_ids']))==33
    assert man['endpoint']==ENDPOINT and man['epochs']==500
    for seed,raw in zip(SEEDS,(Raw26,Raw27)):
        a=config_type(seed)().model_dump();b=raw().model_dump()
        assert {k for k in a if a[k]!=b[k]}=={'normalization'}
    assert cv.unseal(json.loads(json.dumps(cv.sealed(man))))==man


@pytest.mark.parametrize('case',['fold','group','component','source','audit'])
def test_cohort_leakage_or_missing_audit_rejected(case):
    base=fixture.parent();man=manifest();lookup={r['source']['video_id']:r for r in base['rows']}
    a,b=[lookup[base['folds'][0][s+'_ids'][0]] for s in ('train','evaluation')]
    if case=='fold':base['folds'][0]['train_ids'].reverse()
    if case=='group':b['source']['group_id']=a['source']['group_id']
    if case=='component':b['component_id']=a['component_id']
    if case=='source':b['source']['sha256']=a['source']['sha256']
    if case=='audit':man['input_audit'].pop(next(iter(man['input_audit'])))
    with pytest.raises(ValueError):m.build(base,man['sources'],man['input_audit'])


@pytest.mark.parametrize('seed',SEEDS)
def test_same_code_initialization_optimizer_batch_order_and_transform_both_sides(seed,monkeypatch):
    man=manifest();engine,selected,ns=rt.engine(man,seed)
    assert engine.train.__code__ is rt.TEMPLATE.train.__code__ and ns['count_loss'] is count_loss
    assert ns['Config'] is config_type(seed) and engine.subset is ns['subset']
    assert rt.parent.Config is Raw26
    models=[original.fresh_components(config_type(seed)()) for f in range(3)]
    raw,opt=original.fresh_components((Raw26 if seed==SEEDS[0] else Raw27)())
    assert all(original.state_hash(x.state_dict())==original.state_hash(raw.state_dict()) for x,o in models)
    assert all(o.state_dict()==opt.state_dict() and not o.state for x,o in models)
    assert len({next(x.parameters()).data_ptr() for x,o in models})==3
    for f in range(3):
        for e in (0,49,499):engine.validate_report(report(selected,f,e,'fixture',False),m.fold_identity(selected,f),selected,e)
    monkeypatch.setattr(original,'subset',memory(man))
    tr=engine.subset(None,None,man['folds'][0]['train_ids']);evd=engine.subset(None,None,man['folds'][1]['evaluation_ids'])
    a={tr[i]['video_id']:tr[i] for i in range(len(tr))};b={evd[i]['video_id']:evd[i] for i in range(len(evd))}
    assert set(a)&set(b) and all(a[v]==b[v] for v in set(a)&set(b))
    with pytest.raises(PermissionError):TRAIN(None,None,man,seed,0,None)


def test_forward_checks_all33_both_seeds_no_weight_updates(monkeypatch):
    man=manifest();monkeypatch.setattr(original,'subset',memory(man))
    result=rt.forward_only(None,None,man)
    assert result['unique_videos']==33 and result['forward_calls']==198 and len(result['checks'])==12
    assert result['training_steps']==result['baseline_fits']==0
    assert all(q['initial_model_sha256']==q['model_after_sha256'] and math.isfinite(q['loss_max']) for q in result['checks'])


@pytest.mark.parametrize('case',['valid','scale','timeline','mask'])
def test_input_audit_retains_all33_or_reports_exact_video_failure(monkeypatch,case):
    man=manifest()
    for row in man['rows']:row['source'].update(frame_count=4,fps=30.)
    source=memory(man)(None,None,[r['source']['video_id'] for r in man['rows']])
    v=source[0]['video_id']
    samples=[source[i] for i in range(len(source))]
    if case=='scale':samples[0]['complete']=[False]*4
    if case=='scale':
        for row in samples[0]['values']:
            for j in range(8):row[j*7+3]=0.
    if case=='timeline':samples[0]['values'][1][-1]=1.
    if case=='mask':samples[0]['complete'][0]=False
    monkeypatch.setattr(original,'subset',lambda *a:xy.MemoryDataset(source.rows,samples))
    if case=='valid':assert m.audit_inputs(None,None,man)==man['input_audit']
    else:
        with pytest.raises(ValueError) as error:m.audit_inputs(None,None,man)
        assert set(error.value.diagnostics['errors'])=={v}
        assert len(error.value.diagnostics['eligible_audits'])==32
        assert error.value.diagnostics['expected_clips']==33


def test_exhausted_resume_stops_before_creating_run(tmp_path,monkeypatch):
    man=manifest();path,_=checkpoint(tmp_path,man,epoch=499)
    engine,selected,ns=rt.engine(man,SEEDS[0]);loader=engine.load_checkpoint
    monkeypatch.setitem(ns,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    monkeypatch.setitem(ns,'subset',lambda *a:object())
    with pytest.raises(ValueError,match='budget exhausted'):
        engine.train(None,None,selected,0,tmp_path/'never',user_run=True,resume=path)
    assert not (tmp_path/'never').exists()


@pytest.mark.parametrize('seed',SEEDS)
def test_fixture_resume_load_restore_without_step_and_cross_seed_reject(tmp_path,seed):
    man=manifest();path,saved=checkpoint(tmp_path,man,seed)
    engine,selected,_=rt.engine(man,seed)
    loaded=engine.load_checkpoint(path,selected,0,allow_fixture=True)
    model,opt=original.fresh_components(config_type(seed)());before=original.state_hash(model.state_dict())
    assert restore_state(loaded,model,opt)==(2,0)
    assert before==original.state_hash(model.state_dict())==saved['model_sha256'] and not opt.state
    other,sel,_=rt.engine(man,SEEDS[1] if seed==SEEDS[0] else SEEDS[0])
    with pytest.raises(ValueError):other.load_checkpoint(path,sel,0,allow_fixture=True)
    with pytest.raises(ValueError):engine.load_checkpoint(path,selected,1,allow_fixture=True)
    with pytest.raises(ValueError):engine.load_checkpoint(path,selected,0)


@pytest.mark.parametrize('case',['family','transform','config','seed','fold','manifest','epoch','history','weights','optimizer','hash','target','missing','duplicate'])
def test_checkpoint_rejects_wrong_condition_and_corrupt_linkage(tmp_path,case):
    man=manifest();path,saved=checkpoint(tmp_path,man)
    if case=='family':saved['version']=rt.parent.CHECKPOINT_VERSION
    if case=='transform':saved['identity']['transform']['epsilon']=1.
    if case=='config':saved['identity']['config']['normalization']='image_center_height_units_no_fit'
    if case=='seed':saved['identity']['seed']=SEEDS[1]
    if case=='fold':saved['identity']['fold']=1
    if case=='manifest':saved['identity']['manifest_sha256']='bad'
    if case=='epoch':saved['epoch']=500
    if case=='history':saved['history_sha256']='bad'
    if case=='weights':saved['model_sha256']='bad'
    if case=='optimizer':saved['optimizer']['param_groups'][0]['lr']=1
    if case in ('target','missing','duplicate'):
        r=cv.unseal(saved['epoch_reports'][-1])
        if case=='target':r['records'][0]['target']+=1
        if case=='missing':r['records'].pop()
        if case=='duplicate':r['records'][-1]=copy.deepcopy(r['records'][0])
        saved['epoch_reports'][-1]=cv.sealed(r);saved['history_sha256']=canonical_hash(saved['epoch_reports'])
    wrong=tmp_path/'bad.pt';save_checkpoint(wrong,saved)
    if case=='hash':wrong.write_bytes(b'bad')
    engine,selected,_=rt.engine(manifest(),SEEDS[0])
    with pytest.raises(ValueError):engine.load_checkpoint(wrong,selected,0,allow_fixture=True)


def test_equal_video_metrics_paired_fold0_and_two_seed_combination():
    man=manifest();tr,held,controls=inputs(man)
    video=tr[0][0]['video_id'];hits=[r for rs in tr.values() for r in rs if r['video_id']==video]
    hits[0]['prediction']=hits[0]['target']-1;hits[1]['prediction']=hits[1]['target']+1
    a=ev.compare(man,SEEDS[0],tr,held,controls,diagnosis.d)
    assert a['results']['body_normalized']['aggregate']['train_video_equal_MAE']==pytest.approx((32*.25+1)/33)
    assert len(a['fold0_held_out'])==13 and len(a['paired_records'])==99
    assert a['paired_equal_video']['held_out']['raw58']['improved']==33
    tr,held,controls=inputs(man,.75);b=ev.compare(man,SEEDS[1],tr,held,controls,diagnosis.d)
    combined=ev.combine({str(SEEDS[0]):a,str(SEEDS[1]):b})
    assert combined['descriptive_result']=='inconsistent_across_seeds'
    assert combined['methods']['body_normalized']['OOF_MAE']==.5
    assert combined['methods']['body_normalized']['train_MAE']==pytest.approx(((32*.25+1)/33+.75)/2)
    assert combined['methods']['body_normalized']['unique_videos']==33
    assert len(combined['methods']['body_normalized']['group_MAE'])==12


@pytest.mark.parametrize('case',['train_missing','held_missing','duplicate','target','source','fold','failed','nan','control_missing','control_duplicate','control_target'])
def test_incomplete_comparison_never_silently_drops_video(case):
    man=manifest();tr,held,controls=inputs(man);r=held[0][0]
    if case=='train_missing':tr[0].pop()
    if case=='held_missing':held[1].pop()
    if case=='duplicate':held[0][-1]=copy.deepcopy(r)
    if case=='target':r['target']+=1
    if case=='source':r['source_sha256']='bad'
    if case=='fold':r['fold']=1
    if case=='failed':r.update(status='failed',reason='fixture',prediction=None)
    if case=='nan':r['prediction']=float('nan')
    if case=='control_missing':controls.pop()
    if case=='control_duplicate':controls[-1]=copy.deepcopy(controls[-2])
    if case=='control_target':controls[0]['target']+=1
    with pytest.raises(ValueError):ev.compare(man,SEEDS[0],tr,held,controls,diagnosis.d)


@pytest.mark.parametrize('case',['missing','duplicate','group','target','fold','seed'])
def test_cross_seed_pairing_fail_closed(case):
    man=manifest();tr,held,controls=inputs(man)
    results={str(s):ev.compare(man,s,tr,held,controls,diagnosis.d) for s in SEEDS}
    rows=results[str(SEEDS[1])]['results']['body_normalized']['aggregate']['records']
    if case=='missing':rows.pop()
    if case=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if case=='group':rows[0]['group_id']='g00'
    if case=='target':rows[0]['target']+=1
    if case=='fold':rows[0]['held_out_fold']=10
    if case=='seed':results.pop(str(SEEDS[1]))
    with pytest.raises(ValueError):ev.combine(results)


@pytest.mark.parametrize('case',['valid','duplicate','missing','endpoint'])
def test_six_checkpoint_evaluator_fixed_endpoint_only_inference(tmp_path,monkeypatch,case):
    man=manifest();paths=[]
    for s in SEEDS:
        for f in range(3):paths.append(checkpoint(tmp_path,man,s,f,49 if case=='endpoint' and f==2 else 499)[0])
    if case=='duplicate':paths[-1]=paths[0]
    if case=='missing':paths.pop()
    create=rt.engine
    def fixture_engine(*args):
        engine,selected,ns=create(*args);load=engine.load_checkpoint
        engine.load_checkpoint=lambda p,m,f:load(p,m,f,allow_fixture=True)
        return engine,selected,ns
    monkeypatch.setattr(rt,'engine',fixture_engine)
    monkeypatch.setattr(original,'subset',memory(man))
    monkeypatch.setattr(ev,'diagnostic_tool',lambda *a:diagnosis.d)
    controls={str(s):inputs(man)[2] for s in SEEDS}
    if case!='valid':
        with pytest.raises(ValueError):ev.evaluate(tmp_path,{},man,paths,controls)
    else:
        q=ev.evaluate(tmp_path,{},man,paths,controls)
        assert q['train_forward_calls']==132 and q['held_forward_calls']==0
        assert all(c['before_model_sha256']==c['after_model_sha256'] and not c['optimizer_restored'] for c in q['inference_receipts'])
        assert cv.unseal(json.loads(json.dumps(cv.sealed(q))))==q
        text=ev.report(q);assert 'g11' in text and 'g20' in text and 'NOT_VALIDATED' in text


@pytest.mark.parametrize('args',[['train'],['train','--user-run'],['train','--user-run','--seed','20261026'],['verify','--user-run'],['aggregate','--manifest','x']])
def test_cli_missing_options_cannot_train(tmp_path,args):
    with pytest.raises(SystemExit) as e:cli.main(args+['--output-dir',str(tmp_path/'new')])
    assert e.value.code==2 and not (tmp_path/'new').exists()


def test_cli_no_overwrite_and_invalid_scale_failure_diagnostics(tmp_path,monkeypatch):
    man=manifest();monkeypatch.setattr(m,'load',lambda *a:(man,{},{}))
    args=['verify','--root',str(tmp_path),'--manifest','m','--output-dir','verify']
    assert cli.main(args)==0
    with pytest.raises(SystemExit):cli.main(args)
    def fail(*args):
        error=ValueError('no valid scale');error.diagnostics={'errors':{'fixture':'no_valid_video_scale'}};raise error
    monkeypatch.setattr(m,'prepare',fail)
    assert cli.main(['prepare','--root',str(tmp_path),'--output-dir','blocked'])==2
    assert json.loads((tmp_path/'blocked/failure.json').read_text())['diagnostics']['errors']['fixture']=='no_valid_video_scale'


@pytest.mark.parametrize('case',['config','code','input_hash','scale','envelope'])
def test_manifest_changes_rejected_even_if_resealed(tmp_path,monkeypatch,case):
    man=manifest();pristine=copy.deepcopy(man)
    monkeypatch.setattr(m,'prepare',lambda *a:(pristine,{},{}))
    if case=='config':man['configs'][str(SEEDS[0])]['seed']=SEEDS[1]
    if case=='code':man['code']['runtime.py']['sha256']='bad'
    first=next(iter(man['input_audit']))
    if case=='input_hash':man['input_audit'][first]['original_sample_sha256']='bad'
    if case=='scale':man['input_audit'][first]['scale']*=2
    obj=cv.sealed(man)
    if case=='envelope':obj['sha256']='bad'
    path=tmp_path/'manifest.json';write_json(path,obj)
    with pytest.raises(ValueError):m.load(tmp_path,path)


def test_lightweight_transform_import():
    cmd="import sys; import gym_coach.counting.body_normalized.transform; assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    result=subprocess.run([sys.executable,'-c',cmd],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
