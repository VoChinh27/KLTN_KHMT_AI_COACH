"""Diagnostic tooling only; no learning or real source inference."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('count_diagnosis',Path(__file__).resolve().parents[1]/'scripts/diagnose_squat_count_experiment.py')
diagnosis=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnosis)


def sample():
    return dict(video_id='synthetic-diagnostic',target=2,duration_seconds=.4,
        complete=[True,True,False,True,True],values=[[float(i+j/100) for j in range(58)] for i in range(5)])


@pytest.mark.parametrize('method',diagnosis.METHODS)
def test_perturbation_does_not_mutate_input_or_target(method):
    original=sample();before=copy.deepcopy(original)
    result=diagnosis.perturb(original,method)
    assert original==before and result['target']==original['target']
    assert result['video_id']==original['video_id']


@pytest.mark.parametrize('method',['freeze_xy','reverse_xy','shuffle_xy'])
def test_pose_probes_preserve_time_masks_and_visibility(method):
    original=sample();result=diagnosis.perturb(original,method)
    assert result['duration_seconds']==original['duration_seconds']
    assert result['complete']==original['complete']
    for a,b in zip(original['values'],result['values']):
        assert all(a[c]==b[c] for c in range(58) if c not in diagnosis.XY)
    assert result['values'][2]==original['values'][2]  # no bridging gap


def test_reverse_is_within_runs():
    s=sample();r=diagnosis.perturb(s,'reverse_xy')
    assert [v[0] for v in r['values']]==[1.,0.,2.,4.,3.]


def test_duplicate_is_explicit_ood_reset_probe():
    s=sample();r=diagnosis.perturb(s,'duplicate_reset_fixed_time_duration')
    assert r['complete']==s['complete']+[False]+s['complete']
    assert r['values'][:5]==r['values'][6:]==s['values']
    assert r['duration_seconds']==s['duration_seconds']


def test_descriptive_population_sd_and_null_denominator():
    d=diagnosis.describe([0.,2.,None])
    assert d==dict(n=2,null=1,minimum=0.,maximum=2.,mean=1.,median=1.,population_sd=1.)
    assert diagnosis.describe([])['mean'] is None


def test_only_supported_perturbations():
    with pytest.raises(ValueError):diagnosis.perturb(sample(),'change_target')


def test_saved_report_target_or_metric_tampering_rejected():
    from gym_coach.counting.research.evaluation import evaluate
    row=dict(role='validation',source={'video_id':'synthetic','sha256':'a'*64},input_status='accepted',target=2)
    report=evaluate([row],{'synthetic':1.2},'validation')
    assert diagnosis.verify_report([row],report,'validation')==report
    changed=copy.deepcopy(report);changed['records'][0]['target']=1
    with pytest.raises(ValueError):diagnosis.verify_report([row],changed,'validation')
    changed=copy.deepcopy(report);changed['continuous_MAE']=0.
    with pytest.raises(ValueError):diagnosis.verify_report([row],changed,'validation')


def test_mean_pool_duplicate_run_probe_on_actual_network():
    import torch
    from gym_coach.counting.research.network import CountGRU,collate,setup
    setup();model=CountGRU();model.eval();s=sample()
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():
        a=model(collate([s]));b=model(collate([diagnosis.perturb(s,'duplicate_reset_fixed_time_duration')]))
    assert float(a[0])==pytest.approx(float(b[0]),abs=1e-6)
    assert diagnosis.parameter_equal(before,model.state_dict())
    assert all(p.grad is None for p in model.parameters())
