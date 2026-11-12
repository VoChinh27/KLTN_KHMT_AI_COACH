"""Pure saved-data diagnosis checks; learning and model forward are forbidden."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('fold0_diagnosis',Path(__file__).parents[1]/'scripts/diagnose_squat_cv500_fold0.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


@pytest.fixture(autouse=True)
def no_learning():
    with d.no_learning(),d.previous_tool().no_learning_or_forward():yield


def rows():
    return [dict(video_id=f'v{i}',group_id=f'g{i%4}',target=i%5,duration_seconds=2+i/10,
        gru50=i%5+1,gru500=i%5+0.25,duration=i%5+0.5) for i in range(13)]


def test_full_denominator_contributions_and_no_mutation():
    r=rows();before=copy.deepcopy(r);out=d.analyze(r,[x['video_id'] for x in r])
    assert r==before and out['overall']['n']==13
    assert out['overall']['models']['gru500']['MAE']==.25
    assert out['overall']['delta_MAE_500_minus_50']==-.75
    assert out['overall']['improved']==13
    assert sum(x['fold_AE_fraction'] for x in out['records'])==pytest.approx(1)
    assert sum(x['fold_MAE_contribution'] for x in out['records'])==pytest.approx(.25)
    assert sum(x['n'] for x in out['by_count'].values())==13
    assert sum(x['n'] for x in out['by_group'].values())==13


@pytest.mark.parametrize('mutation',['missing','duplicate','extra','unexpected','expected_duplicate'])
def test_pairing_fails_closed(mutation):
    r=rows();ids=[x['video_id'] for x in r]
    if mutation=='missing':r.pop()
    elif mutation=='duplicate':r[-1]=r[0]
    elif mutation=='extra':r.append(dict(r[0],video_id='extra'))
    elif mutation=='unexpected':r[-1]=dict(r[-1],video_id='other')
    else:ids[-1]=ids[0]
    with pytest.raises(ValueError,match='denominator'):d.analyze(r,ids)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),None,'1'])
def test_invalid_predictions_not_dropped(invalid):
    r=rows();r[0]['gru500']=invalid
    with pytest.raises((ValueError,TypeError)):d.analyze(r,[x['video_id'] for x in r])


def test_direction_no_rounding_and_zero_error_share_null():
    r=rows()[:3];r[0]['gru500']=r[0]['target']+.001;r[1]['gru500']=r[1]['target']-.001;r[2]['gru500']=r[2]['target']
    m=d.measure(r)['models']['gru500'];assert (m['over'],m['under'],m['exact'])==(1,1,1)
    for x in r:x['gru500']=x['target']
    out=d.analyze(r,[x['video_id'] for x in r]);assert all(x['fold_AE_fraction'] is None for x in out['records'])


@pytest.mark.parametrize('mask,gaps,runs',[
    ([True]*3,[],[(0,2)]),([False]*3,[(0,2)],[]),
    ([False,True,False,False,True],[(0,0),(2,3)],[(1,1),(4,4)]),
    ([True,False],[(1,1)],[(0,0)]),([],[],[])])
def test_interval_boundaries(mask,gaps,runs):
    assert [(r['start'],r['end']) for r in d.intervals(mask,False)]==gaps
    assert [(r['start'],r['end']) for r in d.intervals(mask,True)]==runs
    assert sum(r['frames'] for v in (True,False) for r in d.intervals(mask,v))==len(mask)


def test_source_hash_change_and_escape(tmp_path):
    p=tmp_path/'source.json';p.write_text('1');s=d.Sources(tmp_path);s.add(p)
    assert s.preserve()['files']==1
    p.write_text('2')
    with pytest.raises(ValueError,match='hash mismatch'):s.preserve()
    with pytest.raises(ValueError,match='outside'):s.add(tmp_path.parent/'escape')


def test_hash_expected_mismatch(tmp_path):
    p=tmp_path/'source.json';p.write_text('1');s=d.Sources(tmp_path)
    with pytest.raises(ValueError,match='hash'):s.add(p,{'sha256':'0'*64})


def test_guard_blocks_model_forward_and_backward():
    import torch
    from gym_coach.counting.research.network import CountGRU
    with pytest.raises(AssertionError):CountGRU()(None)
    with pytest.raises(AssertionError):torch.tensor(1.,requires_grad=True).backward()


def test_support_weighted_vs_clip_denominators():
    r=[]
    for n,c in [(100,100),(10,7)]:
        r.append(dict(support=dict(frames=n,complete_frames=c,complete_fraction=c/n,
            incomplete_frames=n-c,longest_gap_frames=n-c,valid_runs=1,hidden_zero_samples=n-c,
            internal_reset_transitions=int(n!=c),raw_without_pose=0)))
    s=d.support_summary(r)
    assert s['frame_weighted_complete_fraction']==107/110
    assert s['clip_mean_complete_fraction']==.85
    assert s['total_hidden_zero_samples']==3


def test_no_training_or_forward_call_in_diagnostic_source():
    import ast
    tree=ast.parse(Path(d.__file__).read_text(encoding='utf8'))
    forbidden={'train','backward','step','fit','fit_baselines','train_predictions','evaluate','predict'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
