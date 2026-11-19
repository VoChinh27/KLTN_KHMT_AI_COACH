"""Synthetic contracts and inference-only checks. Learning is forbidden in tests."""
import ast
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from gym_coach.counting.models import canonical_hash, VideoIdentity
from gym_coach.counting.research.contract import Config, INPUT_SIZE, JOINTS
from gym_coach.counting.research.data import encode, split_index, summarize, identity, check_identity, local, CountDataset
from gym_coach.counting.research.evaluation import evaluate, rounded_count, better_checkpoint
from gym_coach.pose.models import PoseFrame, PoseExtractionConfig, PoseBackendIdentity, JointObservation
from gym_coach.pose.mapping import JointName
from gym_coach.contracts import Exercise, JointProvenance
from gym_coach.flags import FlagCode


@pytest.fixture(autouse=True)
def forbid_learning(monkeypatch):
    import torch
    from gym_coach.counting.research import baselines
    def blocked(*args,**kwargs):
        raise AssertionError('No parameter learning is allowed in this test suite')
    monkeypatch.setattr(torch.Tensor,'backward',blocked)
    monkeypatch.setattr(torch.autograd,'backward',blocked)
    monkeypatch.setattr(torch.optim.AdamW,'step',blocked)
    monkeypatch.setattr(baselines,'fit_baselines',blocked)


def source(n=20,g=1):
    name=f'v_BodyWeightSquats_g{g:02d}_c01.avi'
    return VideoIdentity(video_id=name,path='Data/UCF-101_GYM/BodyWeightSquats/'+name,
        sha256=str(g)*64,group_id=f'g{g:02d}',fps=25.0,frame_count=n,width=320,height=240).model_dump(mode='json')


def frames(s,bad=(),status='low_visibility',outside=False):
    result=[]
    for i in range(s['frame_count']):
        js=[]
        for j in JointName:
            issue=i in bad and j.value==JOINTS[0]
            missing=issue and status=='missing'
            js.append(JointObservation(joint=j,x=None if missing else (1.2 if outside and issue else 0.6),
                y=None if missing else 0.7,visibility=None if missing else (0.2 if issue and not outside else 0.9),
                status=JointProvenance(status) if issue and not outside else JointProvenance.OBSERVED,
                missing_reason=FlagCode.MISSING_JOINT if missing else None))
        result.append(PoseFrame(video_id=s['video_id'],exercise=Exercise.BODY_WEIGHT_SQUATS,
            frame_index=i,timestamp_seconds=i/s['fps'],has_pose=True,backend=PoseBackendIdentity(mediapipe_version='0.10.14'),
            config=PoseExtractionConfig(),joints=tuple(js)))
    return result


def sample(n=20,bad=()):
    s=source(n);data=encode(frames(s,bad),s)
    return dict(video_id=s['video_id'],values=data['values'],complete=data['complete'],
                duration_seconds=data['duration_seconds'],target=2)


def row(g,target=2,status='accepted',role='train'):
    return dict(source=source(g=g),role=role,target=target,input_status=status,
        supervised_eligible=status=='accepted' and target is not None,input_reasons=[],exclusion_reasons=[])


def test_input_axes_time_no_learned_normalization():
    s=source();result=encode(frames(s),s)
    assert result['status']=='accepted'
    assert result['values'][0][:2]==pytest.approx([.1*320/240,.2])
    assert result['values'][-1][-2:]==[19/25,1/25]
    assert result['duration_seconds']==20/25
    assert all(len(r)==INPUT_SIZE for r in result['values'])
    assert result['stats']['joint_samples']==160


@pytest.mark.parametrize('status', ['missing','low_visibility'])
def test_invalid_placeholder_is_masked_provenance_retained(status):
    s=source();result=encode(frames(s,[2],status=status),s)
    value=result['values'][2]
    assert value[0:2]==[0.,0.] and value[3]==0
    assert value[5 if status=='low_visibility' else 6]==1
    assert not result['complete'][2]


def test_outside_observed_is_not_usable():
    s=source();r=encode(frames(s,[2],outside=True),s)
    assert r['values'][2][3:7]==[0,1,0,0]


@pytest.mark.parametrize('bad,expected',[(range(6),'accepted'),(range(7),'rejected')])
def test_fraction_boundary(bad,expected):
    s=source();r=encode(frames(s,bad),s)
    assert r['status']==expected


@pytest.mark.parametrize('length,status',[(10,'accepted'),(11,'rejected')])
def test_gap_boundaries(length,status):
    s=source(100);r=encode(frames(s,range(length)),s)
    assert r['status']==status
    if status=='rejected':assert 'research_long_gap' in [x['code'] for x in r['reasons']]


@pytest.mark.parametrize('n,status',[(4,'rejected'),(5,'accepted'),(1501,'rejected')])
def test_length_limits(n,status):
    s=source(n);assert encode(frames(s),s)['status']==status


@pytest.mark.parametrize('code',['multi_person','side_switch_discontinuity','unstable_signal'])
def test_explicit_evidence_abstains(code):
    s=source();r=encode(frames(s),s,evidence_flags=[dict(code=code)])
    assert r['status']=='rejected'
    assert r['reasons'][0]['stage']=='count_research_input'
    assert r['reasons'][0]['source_tier']=='S3'


@pytest.mark.parametrize('mutation',['time','index','source','drop','viewpoint'])
def test_bad_timeline_rejected(mutation):
    s=source();fs=frames(s)
    changes={'time':{'timestamp_seconds':.123},'index':{'frame_index':8},
             'source':{'video_id':'other'},'viewpoint':{'viewpoint':'side'}}
    if mutation=='drop':fs.pop()
    else:fs[2]=PoseFrame.model_validate_json(json.dumps({**fs[2].model_dump(mode='json'),**changes[mutation]}))
    with pytest.raises(ValueError):encode(fs,s)


def test_config_frozen_roundtrip_and_no_quality_promotion():
    from pydantic import ValidationError
    c=Config();assert Config.model_validate_json(c.model_dump_json()).content_hash()==c.content_hash()
    for key,value in [('min_complete_frame_fraction',.5),('assessment',True),('official_eligibility',True),('epochs',2)]:
        with pytest.raises(ValidationError):Config.model_validate_json(json.dumps({**c.model_dump(mode='json'),key:value}))
    s=source();f=frames(s);before=[x.model_dump_json() for x in f]
    encode(f,s)
    assert before==[x.model_dump_json() for x in f]


def test_hash_and_path_guards(tmp_path):
    path=tmp_path/'input.json';path.write_text('{}')
    link=identity(tmp_path,path);assert check_identity(tmp_path,link)==path
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):check_identity(tmp_path,link)
    with pytest.raises(ValueError):local(tmp_path,'../outside.json')


@pytest.mark.parametrize('key',['group_id','sha256','component_id'])
def test_split_leakage_rejected(key):
    a,b=source(g=1),source(g=2)
    if key=='sha256':b['sha256']=a['sha256']
    # group identity is bound to filename; same group, distinct clip is needed.
    if key=='group_id':
        b=copy.deepcopy(a);b['video_id']=b['video_id'].replace('c01','c02');b['path']=b['path'].replace('c01','c02')
    rows=[dict(**{k:s[k] for k in ('video_id','path','sha256','group_id')},component_id='same' if key=='component_id' else str(i),role=role)
          for i,(s,role) in enumerate([(a,'train'),(b,'validation')])]
    with pytest.raises(ValueError,match='leakage'):
        split_index(dict(version='squat-count-dataset-v1',videos=[dict(source=a),dict(source=b)]),
                    dict(version='squat-count-split-proposal-v1',dataset_manifest_sha256='hash',assignments=rows),'hash')


def test_summary_full_denominators_and_null():
    rows=[row(1),row(2,target=None),row(3,status='rejected'),row(4,status='error'),row(5,status='unchecked')]
    result=summarize(rows)['all']
    assert result['clips']==5 and result['labels_available']==4 and result['labels_unresolved']==1
    assert result['supervised_clips']==1 and result['input_status']['accepted']==2


@pytest.mark.parametrize('value,result',[(-3,0),(0,0),(.49,0),(.5,1),(1.5,2),(2.5,3)])
def test_rounding(value,result):assert rounded_count(value)==result


@pytest.mark.parametrize('value',[True,'2',float('nan'),float('inf')])
def test_invalid_output(value):
    with pytest.raises(ValueError):rounded_count(value)


def test_evaluator_exact_denominators():
    rows=[row(1,target=2),row(2,target=None),row(3,target=1),row(4,status='rejected'),row(5),row(6)]
    p={rows[0]['source']['video_id']:2.5,rows[1]['source']['video_id']:3.,rows[2]['source']['video_id']:.6,
       rows[4]['source']['video_id']:float('nan')}
    r=evaluate(rows,p,'train')
    assert (r['cohort'],r['labels_available'],r['input_accepted'],r['predictions_available'],r['comparable'])==(6,5,5,3,2)
    assert r['continuous_MAE']==pytest.approx(.45) and r['signed_error']==pytest.approx(.05)
    assert r['rounded_exact_match']==.5
    assert r['coverage']['comparable_over_cohort']==2/6
    assert r['statuses']=={'estimated':3,'unavailable':1,'failed':1,'not_run':1}
    assert 'accuracy' not in r


def test_prediction_cannot_bypass_input():
    r=row(1,status='rejected')
    with pytest.raises(ValueError):evaluate([r],{r['source']['video_id']:2},'train')


def test_empty_metrics_are_null_not_zero():
    r=evaluate([row(1,target=None)],{},'train')
    assert r['continuous_MAE'] is None and r['rounded_exact_match'] is None


def test_model_failure_kept_in_denominator():
    r=row(1);out=evaluate([r],{r['source']['video_id']:dict(status='failed',reason='prediction_operational_error',detail='fixture')},'train')
    assert out['cohort']==1 and out['comparable']==0 and out['statuses']=={'failed':1}
    assert out['records'][0]['prediction'] is None and out['records'][0]['reason']=='prediction_operational_error'


def test_earliest_validation_tie():
    assert better_checkpoint(1.,None) and better_checkpoint(.9,1.)
    assert not better_checkpoint(1.,1.) and not better_checkpoint(float('nan'),1.)


def test_model_forward_padding_and_no_grad():
    import torch
    from gym_coach.counting.research.network import CountGRU,setup,collate
    setup();model=CountGRU();model.eval()
    before=[p.clone() for p in model.parameters()]
    with torch.inference_mode():
        single=model(collate([sample(10)]))
        combined=model(collate([sample(10),sample(20)]))
    assert single[0]==pytest.approx(float(combined[0]),abs=1e-6)
    assert torch.isfinite(combined).all() and (combined>=0).all() and not combined.requires_grad
    assert all(torch.equal(p,b) and p.grad is None for p,b in zip(model.parameters(),before))


def test_hidden_state_resets_on_gap():
    import torch
    from gym_coach.counting.research.network import CountGRU,collate,setup
    setup();model=CountGRU();batch=collate([sample(10,bad=[4])]);hidden=[]
    hook=model.cell.register_forward_pre_hook(lambda module,args:hidden.append(args[1].clone()))
    with torch.inference_mode():model(batch)
    hook.remove()
    assert torch.equal(hidden[5],torch.zeros_like(hidden[5]))


def test_loss_fixed_fixture_masks_null_no_backward():
    import torch
    from gym_coach.counting.research.network import collate,count_loss
    a,b=sample(),sample();b['target']=None
    batch=collate([a,b])
    with torch.no_grad():loss=count_loss(torch.tensor([3.,999.]),batch)
    assert float(loss)==.5
    batch['target_mask'][:]=False
    with pytest.raises(ValueError):count_loss(torch.tensor([3.,0.]),batch)


def test_frozen_baseline_predictions_without_fitting():
    from gym_coach.counting.research.baselines import predict
    r=row(1);parameters=dict(constant=2,duration_intercept=-1,duration_slope=2)
    assert list(predict(parameters,[r],'train','constant').values())==[2]
    assert list(predict(parameters,[r],'train','duration').values())==pytest.approx([.6])


def test_fit_code_train_only_static_inspection():
    # Fitting deliberately NOT executed, even on synthetic fixtures.
    import gym_coach.counting.research.baselines as baseline
    tree=ast.parse(Path(baseline.__file__).read_text())
    fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_baselines')
    text=ast.unparse(fit)
    assert "r['role'] == 'train'" in text and "r['supervised_eligible']" in text
    assert 'training_subset_hash' in text


def test_learning_commands_refuse_without_user_run():
    from gym_coach.counting.research.cli import main
    from gym_coach.counting.research.experiment import train,run_baselines
    for command in ('train','baseline'):
        with pytest.raises(SystemExit):main([command])
    with pytest.raises(PermissionError):train(None,None,None,None)
    with pytest.raises(PermissionError):run_baselines(None,None,None,None)


def test_test_partition_requires_final_flag():
    from gym_coach.counting.research.experiment import evaluate_saved
    with pytest.raises(PermissionError):evaluate_saved(None,None,None,None,'development_test',checkpoint='not_read')


def test_checkpoint_hash_identity_and_untrained_rejection(tmp_path):
    from gym_coach.counting.research.experiment import save_checkpoint,load_checkpoint
    p=tmp_path/'synthetic-not-trained.pt'
    data=dict(version='squat-count-checkpoint-v1',identity={'fixture':True},training_steps=0,best_model=None)
    save_checkpoint(p,data)
    with pytest.raises(ValueError,match='not a user-trained'):load_checkpoint(p,{'fixture':True})
    with pytest.raises(ValueError,match='identity'):load_checkpoint(p,{'fixture':False})
    with pytest.raises(FileExistsError):save_checkpoint(p,data)
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):load_checkpoint(p,{'fixture':True})


def test_import_light_and_no_backend():
    code="import sys;import gym_coach.counting.research.data;import gym_coach.counting.research.evaluation;assert not set(('torch','numpy','scipy','cv2','mediapipe')) & set(sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_restore_only_no_optimizer_step(tmp_path):
    import torch
    from gym_coach.counting.research.network import CountGRU,setup
    from gym_coach.counting.research.experiment import restore_state
    setup();model=CountGRU();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    state=copy.deepcopy(model.state_dict())
    # Simulated state, not a trained checkpoint. No fitting, backward or step.
    fixture=dict(model=state,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),epoch=3,training_steps=0)
    with torch.no_grad():
        for p in model.parameters():p.add_(1)
    epoch,steps=restore_state(fixture,model,optimizer)
    assert (epoch,steps)==(4,0)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    assert optimizer.state=={}


def test_cached_loader_null_target_exclusion(tmp_path):
    from gym_coach.counting.readiness import write_json
    s=source();payload=encode(frames(s),s);p=tmp_path/'cache.json';write_json(p,payload)
    a=row(1);a['input_artifact']=identity(tmp_path,p)
    b=copy.deepcopy(a);b['target']=None;b['supervised_eligible']=False
    dataset=CountDataset(tmp_path,{'rows':[a,b]},'train')
    assert len(dataset)==1 and dataset[0]['target']==2
    assert len(CountDataset(tmp_path,{'rows':[a,b]},'train',supervised=False))==2
    p.write_text('{}')
    with pytest.raises(ValueError,match='changed'):dataset[0]


@pytest.mark.parametrize('status',['unchecked','error'])
def test_readiness_rejects_unchecked_or_integrity_errors(status):
    from gym_coach.counting.research.experiment import require_ready
    with pytest.raises(ValueError):require_ready({'rows':[row(1,status=status)]})


@pytest.mark.parametrize('role',['train','validation'])
def test_readiness_requires_supervised_train_and_validation(role):
    from gym_coach.counting.research.experiment import require_ready
    rows=[row(1),row(2,role='validation')];rows[0 if role=='train' else 1]['supervised_eligible']=False
    with pytest.raises(ValueError):require_ready({'rows':rows,'summary':summarize(rows)})


def test_checkpoint_selection_never_uses_development_test_static():
    import gym_coach.counting.research.experiment as experiment
    tree=ast.parse(Path(experiment.__file__).read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    calls=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='predict_model']
    assert len(calls)==1 and calls[0].args[-1].value=='validation'


def test_collate_null_and_padding_separate():
    from gym_coach.counting.research.network import collate
    a,b=sample(5),sample(10);a['target']=None
    batch=collate([a,b])
    assert batch['target_mask'].tolist()==[False,True]
    assert batch['lengths'].tolist()==[5,10]
    assert not batch['complete'][0,5:].any()


def test_model_does_not_use_padding_values():
    import torch
    from gym_coach.counting.research.network import collate,CountGRU,setup
    setup();model=CountGRU();batch=collate([sample(5),sample(10)])
    with torch.inference_mode():
        before=model(batch);batch['x'][0,5:]=12345.;after=model(batch)
    assert torch.equal(before,after)


def test_full_112_denominator_keeps_rejected_unresolved_and_errors():
    rows=[]
    for i in range(112):
        r=row(1,role='train' if i<67 else 'validation' if i<88 else 'development_test',
              target=None if i<8 else 2,status='rejected' if i%2 else 'accepted')
        r['source']['video_id']=f'synthetic-{i}'
        rows.append(r)
    result=summarize(rows)
    assert [result[role]['clips'] for role in ('train','validation','development_test')]==[67,21,24]
    assert result['all']['labels_available']==104 and result['all']['labels_unresolved']==8
    assert result['all']['supervised_clips']==52


def test_saved_raw_and_quality_reuse_preserves_unknown_gate(tmp_path):
    from gym_coach.quality import QualityGate,QualityProfile
    from gym_coach.counting.research.data import saved_input
    from gym_coach.counting.readiness import write_json,file_info
    s=source();fs=frames(s)
    q=QualityGate().evaluate(fs).model_dump(mode='json')
    folder=tmp_path/'cache';folder.mkdir()
    raw=folder/'raw-pose-unknown.jsonl'
    raw.write_text('\n'.join(f.model_dump_json() for f in fs),encoding='utf-8')
    write_json(folder/'preprocessing.json',dict(quality=q,metadata=s))
    write_json(folder/'source.json',dict(candidate=dict(sha256=s['sha256'],path=s['path'])))
    planned=dict(action='reuse',snapshot=dict(directory='cache',files={p.name:file_info(p) for p in folder.iterdir()}))
    summary=dict(sequence=q['eligibility'],quality_viewpoint='unknown',metrics=q['metric_eligibility'])
    config=dict(quality_hash=QualityProfile().content_hash(),pose=fs[0].config.model_dump(mode='json'),
                mediapipe_version='0.10.14',model_asset={'sha256':None})
    result,report,links=saved_input(tmp_path,s,planned,summary,config)
    assert report==q and result==fs and len(links)==3
    assert encode(result,s)['status']=='accepted'
    raw.write_text('{}')
    with pytest.raises(ValueError,match='hash'):saved_input(tmp_path,s,planned,summary,config)
