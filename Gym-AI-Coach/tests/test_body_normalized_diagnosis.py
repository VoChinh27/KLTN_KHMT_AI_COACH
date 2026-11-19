import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from gym_coach.counting import body_normalized_diagnosis as d
from gym_coach.counting.body_normalized.transform import normalize
from gym_coach.counting.body_normalized.safety import no_learning
from gym_coach.counting.inner_cv import runtime as original
from gym_coach.counting.research.network import setup,CountGRU,collate
from gym_coach.counting.body_normalized.contract import Config26
import test_count_body_normalized as fixture


@pytest.fixture(autouse=True)
def guard():
    with no_learning():yield


def test_independent_geometry_median_and_preserved_fields():
    raw=fixture.sample(complete=[True,False,True,True]);expected,scale=d.independent_geometry(raw,1e-8)
    assert scale==6
    assert expected['values'][0][14]==-1/6 and expected['values'][0][15]==0
    prod,_=normalize(raw);check=d.check_transform(raw,prod,1e-8,1e-10)
    assert check['max_abs_coordinate_difference']==0 and check['incomplete_and_nonxy_exact']
    assert expected['values'][1]==raw['values'][1]


@pytest.mark.parametrize('case',['mask','time','duration','channel','incomplete','double_normalized','nonfinite'])
def test_geometry_drift_detected(case):
    raw=fixture.sample(complete=[True,False,True,True]);prod,_=normalize(raw)
    if case=='mask':prod['complete'][0]=False
    if case=='time':prod['values'][0][-1]=1
    if case=='duration':prod['duration_seconds']+=1
    if case=='channel':prod['values'][0][2]=0
    if case=='incomplete':prod['values'][1][0]+=1
    # Artificially scale normalized coordinates a second time; never feed to model.
    if case=='double_normalized':prod['values'][0][0]/=6
    if case=='nonfinite':prod['values'][0][0]=float('nan')
    with pytest.raises(ValueError):d.check_transform(raw,prod,1e-8,1e-10)


def test_support_steps_do_not_cross_gap_and_singletons_are_counted():
    raw=fixture.sample(complete=[True,False,True,True]);prod,a=normalize(raw)
    q=d.measure(raw,prod,a['scale'],30.)
    assert q['complete_frames']==3 and q['hidden_zero_samples']==1
    assert q['resets_after_complete']==q['restarts_after_gap']==1 and q['adjacent_complete_edges']==1
    assert q['gaps'][0]['start']==q['gaps'][0]['end']==1
    assert q['largest_steps'][0]['previous_frame']==2 and q['largest_steps'][0]['frame_index']==3
    assert q['per_frame'][1]['hip'] is None


def test_torso_variation_and_removed_translation_are_descriptions():
    raw=fixture.sample();prod,a=normalize(raw);q=d.measure(raw,prod,a['scale'],30.)
    assert q['torso']['mean']==5 and q['torso']['minimum']==2 and q['torso']['maximum']==8
    assert q['hip']['y_range']==0 and q['hip']['path_adjacent_complete']==0
    assert q['torso']['population_cv']>0 and 'not inferable' in q['identity_switch']


def paired_fixture():
    ids=['a','b','c'];metadata={v:dict(target=2,group_id='g19' if v!='c' else 'g06',duration_seconds=4.,source={'sha256':v}) for v in ids}
    rows=[]
    for v in ids:
        for method,error in [('body_normalized',2 if v!='c' else 0),('raw58',.5),('duration',.25)]:
            rows.append(dict(video_id=v,method=method,target=2,source_sha256=v,fold=1,status='estimated',reason=None,prediction=2+error))
    return rows,metadata,ids


def test_paired_contribution_uses_full_fold_denominator_including_improvements():
    rows,meta,ids=paired_fixture();q=d.paired_table(rows,meta,ids)
    assert q['n']==3 and q['groups']['g19']['n']==2
    assert q['groups']['g19']['total_delta_AE']==3 and q['groups']['g19']['contribution_to_fold_MAE_increase']==1
    assert q['groups']['g06']['contribution_to_fold_MAE_increase']==pytest.approx(-.5/3)
    assert sum(g['contribution_to_fold_MAE_increase'] for g in q['groups'].values())==pytest.approx(q['delta_MAE'])


@pytest.mark.parametrize('case',['missing','duplicate','target','source','fold','failed','nan'])
def test_pairing_rejects_corruption(case):
    rows,meta,ids=paired_fixture()
    if case=='missing':rows.pop()
    if case=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if case=='target':rows[0]['target']=1
    if case=='source':rows[0]['source_sha256']='bad'
    if case=='fold':rows[0]['fold']=0
    if case=='failed':rows[0].update(status='failed',prediction=None)
    if case=='nan':rows[0]['prediction']=float('nan')
    with pytest.raises(ValueError):d.paired_table(rows,meta,ids)


def test_range_overlap_denominators_do_not_assert_joint_support():
    raw=fixture.sample();norm,a=normalize(raw);pose=d.measure(raw,norm,a['scale'],30.)
    records=[dict(video_id=str(i),target=i+1,duration_seconds=float(i+1),pose=pose) for i in range(3)]
    q=d.compare_distribution(records[:2],records[2:])
    assert q['summaries']['duration']['held_inside_train_minmax']==0
    assert q['summaries']['scale']['held_inside_train_minmax']==1 and q['held_n']==1
    assert 'multivariate' in q['interpretation']


def script():
    path=Path(__file__).parents[1]/'scripts/diagnose_squat_body_normalized_fold1.py'
    spec=importlib.util.spec_from_file_location('body_diagnosis_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


@pytest.mark.parametrize('case',['exact','mismatch','weight_mismatch'])
def test_replay_hook_observes_tensor_exactness_and_weights_unchanged(case):
    s=script();man=fixture.manifest();dataset=fixture.memory(man)(None,None,man['folds'][1]['evaluation_ids'])
    setup(Config26());model=CountGRU(Config26());model.eval()
    expected={r['video_id']:r['prediction'] for r in original.predict(model,dataset)}
    before=original.state_hash(model.state_dict())
    if case=='mismatch':expected[next(iter(expected))]+=.1
    if case=='weight_mismatch':
        with pytest.raises(ValueError,match='model hash'):s.replay(model,dataset,expected,'bad')
    else:
        result=s.replay(model,dataset,expected,before)
        assert result['before']==result['after']==before
        assert len(result['actual_inputs'])==13 and result['exact']==(case=='exact')
        assert len(result['differences'])==(0 if case=='exact' else 1)


def test_script_no_training_or_parameter_mutation_calls():
    import ast
    tree=ast.parse(Path(script().__file__).read_text())
    forbidden={'train','backward','step','fit','fit_baselines','fit_inner_baselines','restore_state'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
