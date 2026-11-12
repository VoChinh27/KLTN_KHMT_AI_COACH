"""Velocity research preparation: no fitting/gradients/optimizer updates, even in fixtures."""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from gym_coach.counting.inner_cv import runtime as original_rt, manifest as cv
from gym_coach.counting.inner_cv.oof import metric
from gym_coach.counting.models import canonical_hash
from gym_coach.counting.research.contract import Config
from gym_coach.counting.research.experiment import save_checkpoint, restore_state
from gym_coach.counting.readiness import write_json
from gym_coach.counting.research.data import identity
from gym_coach.counting.velocity import transform as tr, manifest as m, runtime as rt, evaluation as ev, audit, cli
from gym_coach.counting.velocity.safety import no_learning
import test_count_xy_ablation as fixtures


@pytest.fixture(autouse=True)
def no_parameter_learning():
    with no_learning():
        yield


def manifest():
    return m.build(fixtures.base_manifest(), {'synthetic_fixture': True}, {'synthetic_fixture': True})


def reports(man, epoch=49, error=.25):
    values = fixtures.endpoint_reports(man, epoch=epoch, error=error)
    for r in values:
        r['identity_sha256'] = canonical_hash(m.fold_identity(man, r['fold']))
    return values


def controls(man):
    result = fixtures.controls(man)
    result['zero_xy'] = [dict(r, fold=f['fold']) for f in reports(man, error=.6) for r in f['records']]
    return result


def sample(n=6, fps=25, mask=None, indices=None):
    indices = list(range(n)) if indices is None else indices
    values = []
    for index in indices:
        t = index/fps
        vector = []
        for j in range(8):
            vector.extend([j/10+2*t, -j/10-3*t, .9, 1., 1., 0., 0.])
        values.append(vector+[t, 0.0 if index == 0 else 1/fps])
    return dict(video_id='synthetic', values=values, complete=[True]*n if mask is None else mask,
                duration_seconds=n/fps, target=2)


def transform(s, fps=25, indices=None):
    return tr.velocity_xy(s, frame_indices=list(range(len(s['values']))) if indices is None else indices, fps=fps)


def source_for(s, fps=25):
    return dict(source=dict(video_id=s['video_id'], frame_count=len(s['values']), fps=fps))


def fixture_checkpoint(tmp_path, man, fold=0, epoch=1):
    import torch
    model, optimizer = original_rt.fresh_components(Config())
    param = next(model.parameters())
    optimizer.state[param] = dict(step=torch.tensor(3.), exp_avg=torch.ones_like(param), exp_avg_sq=torch.ones_like(param)*2)
    history = []
    for e in range(epoch+1):
        report = reports(man, epoch=e)[fold]
        report['model_sha256'] = original_rt.state_hash(model.state_dict())
        history.append(cv.sealed(report))
    payload = rt.checkpoint_payload(m.fold_identity(man, fold), model, optimizer, epoch, 0, history, fixture=True)
    path = tmp_path/f'synthetic-velocity-not-trained-{fold}.pt'; save_checkpoint(path, payload)
    return path, payload


@pytest.mark.parametrize('fps', [25, 30, 60])
def test_known_velocity_units_sign_and_first_frame(fps):
    s = sample(fps=fps); value = transform(s, fps)
    assert tr.descriptor()['unit'] == 'image_height_per_second'
    assert all(value['values'][0][c] == 0.0 for c in tr.XY_INDICES)
    for vector in value['values'][1:]:
        for j in range(8):
            assert vector[7*j] == pytest.approx(2., abs=1e-12)
            assert vector[7*j+1] == pytest.approx(-3., abs=1e-12)


def test_exact_sixteen_channels_copy_mask_metadata_unchanged():
    s = sample(); before = copy.deepcopy(s); value = transform(s)
    assert tr.XY_INDICES == tuple(j*7+k for j in range(8) for k in (0,1))
    assert s == before and len(tr.XY_INDICES) == 16
    assert {k:v for k,v in s.items() if k != 'values'} == {k:v for k,v in value.items() if k != 'values'}
    for a,b in zip(s['values'],value['values']):
        assert len(b) == 58 and all(a[c] == b[c] for c in range(58) if c not in tr.XY_INDICES)
    value['values'][0][2] = -10; value['complete'][0] = False
    assert s == before


def test_gap_first_after_gap_and_incomplete_current_are_boundary_placeholders():
    s=sample(mask=[True,True,False,False,True,True]); result=transform(s)
    for t in (0,2,3,4): assert all(result['values'][t][c] == 0.0 for c in tr.XY_INDICES)
    for t in (1,5): assert result['values'][t][0] == pytest.approx(2.)
    assert result['complete'] == s['complete']  # true at first-after-gap is retained
    assert tr.support_reason(4,s['complete'],list(range(6))) == 'first_after_incomplete'


def test_nonadjacent_frames_zero_and_no_interpolation():
    indices=[0,1,4,5,6,7]; s=sample(indices=indices)
    result=transform(s,indices=indices)
    assert all(result['values'][2][c] == 0.0 for c in tr.XY_INDICES)
    assert result['values'][3][0] == pytest.approx(2.) and result['complete'] == s['complete']


def test_no_future_frame_in_computation_and_actual_stationary_zero_distinct():
    s=sample(); original=transform(s)
    future=copy.deepcopy(s)
    for c in tr.XY_INDICES: future['values'][4][c] += 3
    assert transform(future)['values'][:4] == original['values'][:4]
    flat=sample()
    for row in flat['values']:
        for c in tr.XY_INDICES: row[c]=.2
    result=transform(flat)
    assert all(v[c] == 0. for v in result['values'] for c in tr.XY_INDICES)
    assert tr.support_reason(1,flat['complete'],list(range(6))) == 'computed_backward_difference'
    assert tr.support_reason(0,flat['complete'],list(range(6))) == 'first_sample'


@pytest.mark.parametrize('dt', [0., -1., float('inf'), float('-inf'), float('nan'), .03, None, True])
def test_invalid_dt_rejected_without_zero_repair(dt):
    s=sample();s['values'][2][-1]=dt
    with pytest.raises(ValueError):transform(s)


@pytest.mark.parametrize('fps', [0,-2,float('inf'),float('nan'),True,None])
def test_invalid_fps_rejected(fps):
    with pytest.raises(ValueError):transform(sample(),fps)


@pytest.mark.parametrize('change', ['first_dt','elapsed','short_vector','mask_length','mask_type','coordinate_nan','indices_duplicate','indices_reverse','indices_bool'])
def test_invalid_timeline_shape_or_coordinates_rejected(change):
    s=sample();indices=list(range(6))
    if change=='first_dt':s['values'][0][-1]=.04
    if change=='elapsed':s['values'][3][-2]=.121
    if change=='short_vector':s['values'][2].pop()
    if change=='mask_length':s['complete'].pop()
    if change=='mask_type':s['complete'][0]=1
    if change=='coordinate_nan':s['values'][0][0]=float('nan')
    if change=='indices_duplicate':indices[1]=0
    if change=='indices_reverse':indices.reverse()
    if change=='indices_bool':indices[0]=False
    with pytest.raises(ValueError):transform(s,indices=indices)


def test_nonfinite_difference_is_error_without_clipping_and_large_finite_is_preserved():
    s=sample();s['values'][0][0]=-1e308;s['values'][1][0]=1e308
    with pytest.raises(ValueError,match='Nonfinite velocity'):transform(s)
    s=sample();s['values'][1][0]=1000.
    assert transform(s)['values'][1][0] == 25000.  # no hidden clipping or normalizer


def test_wrapper_same_train_evaluation_and_source_linkage(monkeypatch):
    s=sample();source=fixtures.MemoryDataset([source_for(s)],[s])
    monkeypatch.setattr(original_rt,'subset',lambda *args:source)
    assert rt.subset(None,None,['train'])[0] == rt.subset(None,None,['eval'])[0] == transform(s)
    assert rt.subset(None,None,[]) .rows is source.rows
    source.rows[0]['source']['frame_count']+=1
    with pytest.raises(ValueError,match='linkage'):rt.subset(None,None,[])[0]


def test_audit_exact_denominators_boundary_counts_finiteness_no_source_mutation(monkeypatch):
    base=fixtures.base_manifest(); samples=[]
    for r in base['rows']:
        r['input_artifact']={'path':'synthetic-cache','sha256':'a'*64,'size_bytes':1}
        s=sample(n=100);s['video_id']=r['source']['video_id'];s['complete'][50]=False;samples.append(s)
    source=fixtures.MemoryDataset(base['rows'],samples);before=copy.deepcopy(samples)
    monkeypatch.setattr(audit,'subset',lambda *args:source)
    result=audit.audit_inputs(None,None,base)
    assert result['clips']==33 and result['frames']==3300 and len(result['records'])==33
    assert result['support_counts']=={'first_sample':33,'computed_backward_difference':3201,'incomplete_current':33,'first_after_incomplete':33}
    assert result['computed_components']['n']==3201*16 and result['including_boundary_placeholders']['n']==3300*16
    assert result['all_finite_float64_and_float32'] and result['model_forward_calls']==0 and samples==before
    assert result['per_channel']['left_shoulder_vx']['mean']==pytest.approx(2.)
    assert result['per_channel']['left_shoulder_vy']['mean']==pytest.approx(-3.)


def test_forward_only_shape_finite_weights_independent_same_initialization():
    import torch
    from gym_coach.counting.research.network import collate
    a,oa=original_rt.fresh_components(Config());b,ob=original_rt.fresh_components(Config())
    before=original_rt.state_hash(a.state_dict())
    assert before==original_rt.state_hash(b.state_dict()) and oa is not ob and not oa.state and not ob.state
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
    batch=collate([transform(sample())])
    with torch.inference_mode():output=a(batch)
    assert batch['x'].shape==(1,6,58) and output.shape==(1,) and torch.isfinite(output).all()
    assert original_rt.state_hash(a.state_dict())==before and all(p.grad is None for p in a.parameters())
    assert sum(p.numel() for p in a.parameters())==3666


def test_folds_config_and_update_loop_remain_original_without_executing_loop():
    base=fixtures.base_manifest();man=manifest()
    assert man['folds']==base['folds'] and man['rows']==base['rows'] and man['config']==base['config']
    assert [(len(f['train_ids']),len(f['evaluation_ids'])) for f in man['folds']]==[(20,13),(20,13),(26,7)]
    for f in man['folds']:
        assert not set(f['train_ids'])&set(f['evaluation_ids'])
        assert not set(f['train_groups'])&set(f['evaluation_groups'])
    def loop(source):
        tree=ast.parse(source);train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        epoch=next(n for n in train.body if isinstance(n,ast.For) and n.target.id=='epoch')
        stop=next(i for i,n in enumerate(epoch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='records' for t in n.targets))
        return [ast.dump(n,include_attributes=False) for n in epoch.body[:stop]]
    assert loop(Path(original_rt.__file__).read_text(encoding='utf-8'))==loop(inspect.getsource(rt))
    assert "evaluation = subset(root, data, identity['partition']['evaluation_ids'])" in inspect.getsource(rt.train)
    assert 'frozen_runtime.fresh_components(config)' in inspect.getsource(rt.train)


@pytest.mark.parametrize('key',['group_id','sha256','component_id'])
def test_fold_leakage_rejected(key):
    base=fixtures.base_manifest();lookup={r['source']['video_id']:r for r in base['rows']}
    a,b=[lookup[base['folds'][0][side+'_ids'][0]] for side in ('train','evaluation')]
    if key!='component_id':a,b=a['source'],b['source']
    b[key]=a[key]
    with pytest.raises(ValueError):m.build(base,{}, {})


def test_resume_identity_optimizer_rng_and_no_fixture_in_production(tmp_path):
    import torch
    man=manifest();path,payload=fixture_checkpoint(tmp_path,man)
    saved=rt.load_checkpoint(path,man,0,allow_fixture=True)
    model,optimizer=original_rt.fresh_components(Config())
    assert restore_state(saved,model,optimizer)==(2,0)
    assert original_rt.state_hash(model.state_dict())==payload['model_sha256']
    assert torch.equal(torch.get_rng_state(),payload['torch_rng'])
    assert len(optimizer.state)==1 and float(next(iter(optimizer.state.values()))['step'])==3
    with pytest.raises(ValueError,match='user-trained'):rt.load_checkpoint(path,man,0)
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,man,1,allow_fixture=True)


@pytest.mark.parametrize('key',['transform','config_sha256','code','controls','endpoint','input_audit_sha256'])
def test_resume_rejects_other_identity(tmp_path,key):
    man=manifest();path,_=fixture_checkpoint(tmp_path,man);changed=copy.deepcopy(man);changed[key]='changed'
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(path,changed,0,allow_fixture=True)


@pytest.mark.parametrize('version',['squat-count-inner-cv-checkpoint-v1','squat-count-xy-ablation-checkpoint-v1'])
def test_old_checkpoint_families_rejected(tmp_path,version):
    man=manifest();_,payload=fixture_checkpoint(tmp_path,man);payload['version']=version
    p=tmp_path/'wrong-family.pt';save_checkpoint(p,payload)
    with pytest.raises(ValueError,match='identity'):rt.load_checkpoint(p,man,0,allow_fixture=True)


def test_checkpoint_hash_history_and_no_overwrite(tmp_path):
    man=manifest();path,payload=fixture_checkpoint(tmp_path,man)
    with pytest.raises(FileExistsError):save_checkpoint(path,payload)
    path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='byte hash'):rt.load_checkpoint(path,man,0,allow_fixture=True)
    payload['history_sha256']='bad';p=tmp_path/'bad-history.pt';save_checkpoint(p,payload)
    with pytest.raises(ValueError,match='history'):rt.load_checkpoint(p,man,0,allow_fixture=True)


def test_metrics_all_five_methods_denominators_signed_errors_delta_and_sd():
    man=manifest();result=ev.compare_reports(man,reports(man),controls(man))
    assert set(result['metrics'])=={'velocity','gru','zero_xy','constant','duration'}
    assert result['clips']==33 and result['groups']==12 and result['delta_mean']==-.25
    assert result['metrics']['velocity']['pooled']['continuous_MAE']==.25
    assert result['metrics']['velocity']['macro_group_MAE']==.25
    for method,mtr in result['metrics'].items():
        for key in ('folds','groups','counts'):assert sum(v['clips'] for v in mtr[key].values())==33
        assert mtr['predictions']['population_sd']==statistics.pstdev(r['predictions'][method] for r in result['records'])
    for row in result['records']:
        assert row['signed_errors']['velocity']==.25 and row['absolute_errors']['velocity']==.25
        assert row['delta_AE_velocity_minus_gru']==-.25
    assert cv.unseal(json.loads(json.dumps(cv.sealed(result))))==result
    assert result['official_eligibility'] is False and result['assessment'] is False
    assert 'Zero-xy MAE' in ev.report_markdown(result)


@pytest.mark.parametrize('change',['epoch48','mixed_epoch','duplicate_fold','missing_clip','duplicate_clip','source','target','group','null','nonfinite','zero_missing','zero_hash','zero_duplicate','identity'])
def test_pairing_and_endpoint_fail_closed(change):
    man=manifest();rs=reports(man);ctrl=controls(man);first=rs[0]['records'][0]
    if change=='epoch48':
        for r in rs:r['epoch']=48
    if change=='mixed_epoch':rs[1]['epoch']=5
    if change=='duplicate_fold':rs[1]=copy.deepcopy(rs[0])
    if change=='missing_clip':rs[0]['records'].pop()
    if change=='duplicate_clip':rs[0]['records'][-1]=copy.deepcopy(first)
    if change=='source':first['source_sha256']='bad'
    if change=='target':first['target']+=1
    if change=='group':first['group_id']='bad'
    if change=='null':first.update(prediction=None,status='failed',reason='synthetic error')
    if change=='nonfinite':first['prediction']=float('nan')
    if change=='zero_missing':ctrl['zero_xy'].pop()
    if change=='zero_hash':ctrl['zero_xy'][0]['row_sha256']='bad'
    if change=='zero_duplicate':ctrl['zero_xy'][-1]=copy.deepcopy(ctrl['zero_xy'][0])
    if change=='identity':rs[0]['identity_sha256']='bad'
    with pytest.raises(ValueError):ev.compare_reports(man,rs,ctrl)


@pytest.mark.parametrize('case',['complete','wrong_epoch','duplicate'])
def test_endpoint_checkpoint_enforcement_without_learning(tmp_path,monkeypatch,case):
    man=manifest();paths=[fixture_checkpoint(tmp_path,man,f,48 if case=='wrong_epoch' and f==1 else 49)[0] for f in range(3)]
    loader=rt.load_checkpoint
    monkeypatch.setattr(rt,'load_checkpoint',lambda p,m,f:loader(p,m,f,allow_fixture=True))
    if case=='duplicate':paths[1]=paths[0]
    if case=='complete':
        rs,_=rt.endpoint_reports(man,paths);assert all(r['epoch']==49 for r in rs)
    else:
        with pytest.raises(ValueError):rt.endpoint_reports(man,paths)


def test_training_guard_cli_no_overwrite_and_failure_receipt(tmp_path,monkeypatch):
    with pytest.raises(PermissionError):rt.train(None,None,None,None,tmp_path/'train')
    with pytest.raises(SystemExit):cli.main(['train','--manifest','none.json','--fold','0','--output-dir',str(tmp_path/'a')])
    assert not (tmp_path/'a').exists()
    monkeypatch.setattr(cli,'environment',lambda:{'synthetic_test':True})
    def bad(*args):raise ValueError('synthetic bad hash')
    monkeypatch.setattr(m,'load',bad)
    assert cli.main(['verify','--root',str(tmp_path),'--manifest','bad.json','--output-dir',str(tmp_path/'error')])==2
    assert json.loads((tmp_path/'error/failure.json').read_text())['comparison_complete'] is False
    with pytest.raises(SystemExit):cli.main(['verify','--root',str(tmp_path),'--manifest','bad.json','--output-dir',str(tmp_path/'error')])


def test_light_import_no_backend_or_learning():
    code='import sys; import gym_coach.counting.velocity.transform; import gym_coach.counting.velocity.manifest; assert not any(n in sys.modules for n in ("torch","numpy","mediapipe","cv2"))'
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('change',['none','code','transform','config','endpoint','input_audit','zero_bytes'])
def test_manifest_reload_requires_exact_audit_code_and_saved_controls(tmp_path,monkeypatch,change):
    base=fixtures.base_manifest();data={'rows':base['rows']};saved_audit={'synthetic_fixture':True}
    for name in ('original.json','oof.json','baseline.json','zero.json','zero-manifest.json'):
        write_json(tmp_path/name,{'synthetic':True})
    links=dict(original_manifest=identity(tmp_path,tmp_path/'original.json'),gru_oof=identity(tmp_path,tmp_path/'oof.json'),
        comparison_dir='synthetic-controls',comparison_files={'baseline':identity(tmp_path,tmp_path/'baseline.json')},
        zero_xy=dict(manifest=identity(tmp_path,tmp_path/'zero-manifest.json'),directory='synthetic-zero',
                     files={'zero':identity(tmp_path,tmp_path/'zero.json')}))
    frozen=m.build(base,links,saved_audit);ctrl=controls(frozen)
    monkeypatch.setattr(m,'verify_controls',lambda *args:(base,data,links,ctrl))
    monkeypatch.setattr(audit,'audit_inputs',lambda *args:saved_audit)
    if change not in ('none','zero_bytes'):frozen[change]={'changed':True}
    write_json(tmp_path/'manifest.json',cv.sealed(frozen))
    if change=='zero_bytes':(tmp_path/'zero.json').write_text('bad hash')
    if change=='none':assert m.load(tmp_path,tmp_path/'manifest.json')==(frozen,data,ctrl)
    else:
        with pytest.raises(ValueError):m.load(tmp_path,tmp_path/'manifest.json')


def test_saved_sampler_hash_is_verified_without_learning():
    import torch
    man=manifest();ident=m.fold_identity(man,0);report=reports(man)[0];ids=ident['partition']['train_ids']
    report.update(training_steps=50*math.ceil(len(ids)/8),training_loss=.1)
    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(Config().seed+49)).tolist()
    report['batch_order_sha256']=canonical_hash([ids[i] for i in order])
    rt.validate_report(report,ident,man,49)
    report['batch_order_sha256']='wrong'
    with pytest.raises(ValueError,match='sampler'):rt.validate_report(report,ident,man,49)


def test_pooled_weighting_macro12_and_positive_delta():
    man=manifest();rs=reports(man)
    for r in rs:
        for item in r['records']:item['prediction']=item['target']+int(item['group_id'][1:])/2
    result=ev.compare_reports(man,rs,controls(man));errors=[r['absolute_errors']['velocity'] for r in result['records']]
    assert result['metrics']['velocity']['pooled']['continuous_MAE']==math.fsum(errors)/33
    assert result['metrics']['velocity']['macro_group_MAE']==sum(i/2 for i in range(1,13))/12
    assert result['delta_mean']==pytest.approx(math.fsum(errors)/33-.5) and result['delta_mean']>0


def test_float32_overflow_blocks_audit_without_clipping(monkeypatch):
    base=fixtures.base_manifest();samples=[]
    for r in base['rows']:
        r['input_artifact']={'synthetic':True};s=sample(n=100);s['video_id']=r['source']['video_id'];samples.append(s)
    samples[0]['values'][1][0]=1e38
    source=fixtures.MemoryDataset(base['rows'],samples)
    monkeypatch.setattr(audit,'subset',lambda *args:source)
    with pytest.raises((ValueError,RuntimeError)):audit.audit_inputs(None,None,base)
    assert samples[0]['values'][1][0]==1e38


def test_cli_saved_comparison_all_five_methods_hashes_and_no_fitting(tmp_path,monkeypatch):
    man=manifest();ctrl=controls(man);rs=reports(man)
    monkeypatch.setattr(cli,'environment',lambda:{'synthetic_test':True})
    monkeypatch.setattr(m,'load',lambda *args:(man,{},ctrl))
    paths=[]
    for i in range(3):
        p=tmp_path/f'synthetic-{i}.json';write_json(p,{'synthetic_fixture':True});paths.append(p)
    sources=[dict(path=str(p),fold=i,epoch=49) for i,p in enumerate(paths)]
    monkeypatch.setattr(rt,'endpoint_reports',lambda *args:(rs,sources))
    out=tmp_path/'comparison'
    assert cli.main(['aggregate','--root',str(tmp_path),'--manifest','mock.json','--checkpoints',
                     *map(str,paths),'--output-dir',str(out)])==0
    result=cv.unseal(json.loads((out/'comparison.json').read_text()))
    assert result['clips']==33 and len(result['metrics'])==5 and result['baseline_fits']==0
    assert len(cv.unseal(json.loads((out/'per-video-errors.json').read_text())))==33
    assert (out/'REPORT.md').is_file() and (out/'inventory.json').is_file()
