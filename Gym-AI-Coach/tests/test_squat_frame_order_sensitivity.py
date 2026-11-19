"""Pure permutations and synthetic inference. No fitting or parameter updates."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch

from gym_coach.counting import frame_order_sensitivity as policy
from gym_coach.counting.inner_cv import runtime as shared
from gym_coach.counting.inner_cv_500_seed.safety import no_learning
from gym_coach.counting.models import canonical_hash

path=Path(__file__).resolve().parents[1]/'scripts/diagnose_squat_frame_order.py'
spec=importlib.util.spec_from_file_location('squat_frame_order_test_cli',path)
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def guard():
    with no_learning():yield


def sample(mask=(True,True,False,True,False,True,True,True)):
    rows=[]
    for i,valid in enumerate(mask):
        row=[]
        for j in range(8):
            row.extend([i+j/10,i-j/10,.5+i/100,int(valid),int(valid),int(not valid),0])
        rows.append(row+[i/25,1/25 if i else 0])
    return dict(video_id='fixture',complete=list(mask),duration_seconds=len(mask)/25,target=2,values=rows)


def test_blocks_gaps_timestamps_singleton_and_no_mutation():
    s=sample();before=copy.deepcopy(s);order=[1,0,2,3,4,7,5,6]
    q=policy.apply_order(s,order)
    assert s==before and q['target'] is None and q['duration_seconds']==s['duration_seconds']
    assert list(policy.runs(s['complete']))==[(0,2),(3,4),(5,8)]
    for i,j in enumerate(order):
        assert q['values'][i][:56]==s['values'][j][:56]
        assert q['values'][i][56:]==s['values'][i][56:]
    q['values'][0][3]=0;q['complete'][3]=False
    assert s==before


@pytest.mark.parametrize('order',[[5,1,2,3,4,0,6,7],[2,1,0,3,4,5,6,7],[0,1,2,3,4,5,6,6],[0,1]])
def test_nonbijective_or_cross_gap_rejected(order):
    with pytest.raises(ValueError):policy.apply_order(sample(),order)


def test_identity_and_unmovable_run_kept_all_twenty_times():
    s=sample((False,True,False,True,False))
    for r in range(20):
        order=policy.permutation(s['video_id'],s['complete'],r)
        assert order==list(range(5))
        assert policy.apply_order(s,order,identity_control=True)==s
        assert policy.apply_order(s,order)['target'] is None
    with pytest.raises(ValueError):policy.permutation('fixture',s['complete'],20)


def test_reproducibility_by_video_not_by_seed_or_external_rng():
    mask=[True]*12+[False]+[True]*4
    orders=[policy.permutation('a',mask,r) for r in range(20)]
    assert orders==[policy.permutation('a',mask,r) for r in range(20)]
    assert orders!=[policy.permutation('b',mask,r) for r in range(20)]
    assert all(set(q[:12])==set(range(12)) and q[12]==12 and set(q[13:])==set(range(13,17)) for q in orders)


def test_shared_tensor_for_both_models_and_frozen_weights():
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.inputs=[]
        def forward(self,batch):
            self.inputs.append((id(batch),cli.batch_hash(batch)))
            return self.weight+batch['x'][:,0,0]
    s=sample();v=s['video_id'];models={seed:Probe().eval() for seed in policy.SEEDS}
    conditions={seed:dict(models={0:model},links={0:{'sha256':'fixture'}},hashes={0:shared.state_hash(model.state_dict())})
        for seed,model in models.items()}
    originals={(v,seed):dict(prediction=1.) for seed in policy.SEEDS}
    orders=[policy.permutation(v,s['complete'],r) for r in range(20)]
    plans={v:[dict(repeat=r,source_for_destination=o,sha256=canonical_hash(o)) for r,o in enumerate(orders)]}
    info={v:dict(fold=0,sample_sha256=canonical_hash(s))}
    with torch.inference_mode():draws=cli.measure(conditions,{v:s},info,plans,originals)
    assert len(draws)==40 and models[policy.SEEDS[0]].inputs==models[policy.SEEDS[1]].inputs
    assert len(cli.weight_receipts(conditions))==2
    for i in range(0,40,2):assert draws[i]['model_input_sha256']==draws[i+1]['model_input_sha256']


def layout():
    ids=[f'fixture_{i}' for i in range(33)]
    groups=[ids[:13],ids[13:26],ids[26:]]
    folds=[dict(fold=f,evaluation_ids=g,train_ids=[v for v in ids if v not in g]) for f,g in enumerate(groups)]
    held=policy.held_map(folds)
    info={v:dict(video_id=v,fold=f,target=1) for v,f in held.items()}
    originals={(v,s):dict(prediction=1.,fold=f) for v,f in held.items() for s in policy.SEEDS}
    draws=[]
    for i,(v,f) in enumerate(held.items()):
        for s in policy.SEEDS:
            for r in range(20):
                delta=(i+1)*(1 if r%2 else -1)*(1 if s==policy.SEEDS[0] else 2)
                draws.append(dict(video_id=v,seed=s,fold=f,repeat=r,original_prediction=1.,shuffled_prediction=1.+delta,
                    delta=delta,abs_delta=abs(delta),permutation_sha256=str(r),model_input_sha256=v+str(r),
                    positions_moved=r,positions_changed=r))
    return folds,info,originals,draws


def test_equal_video_statistics_and_signed_cancellation():
    _,info,originals,draws=layout()
    rows=policy.summarize(draws,originals,info);q=policy.describe(rows)
    assert q['videos']==33 and q['20261026']['mean_abs_delta']==q['20261026']['median_abs_delta']==17
    assert q['20261026']['mean_signed_delta']==0 and q['20261027']['mean_abs_delta']==34
    assert q['paired_new_minus_old']['higher']==33


@pytest.mark.parametrize('case',['missing_draw','duplicate_draw','wrong_video','wrong_seed','wrong_fold','delta','hash','missing_original'])
def test_fixed_denominators_fail_closed(case):
    _,info,originals,draws=layout()
    if case=='missing_draw':draws.pop()
    if case=='duplicate_draw':draws.append(draws[0])
    if case=='wrong_video':draws[0]['video_id']='extra'
    if case=='wrong_seed':draws[0]['seed']=1
    if case=='wrong_fold':draws[0]['fold']=2
    if case=='delta':draws[0]['delta']+=1
    if case=='hash':draws[0]['model_input_sha256']='bad'
    if case=='missing_original':originals.pop(next(iter(originals)))
    with pytest.raises(ValueError):policy.summarize(draws,originals,info)


@pytest.mark.parametrize('bad',['missing','overlap','leak'])
def test_held_out_partition(bad):
    folds,_,_,_=layout()
    if bad=='missing':folds[0]['evaluation_ids'].pop()
    if bad=='overlap':folds[0]['evaluation_ids'][0]=folds[1]['evaluation_ids'][0]
    if bad=='leak':folds[0]['train_ids'].append(folds[0]['evaluation_ids'][0])
    with pytest.raises(ValueError):policy.held_map(folds)


def test_original_mismatch_saves_all_66_and_failing_video_before_measure(tmp_path,monkeypatch):
    class One(torch.nn.Module):
        def forward(self,batch):
            return torch.ones_like(batch['duration'])
    folds,info,_,_=layout()
    samples={v:dict(sample(),video_id=v) for v in info}
    datasets={f['fold']:f['evaluation_ids'] for f in folds}
    monkeypatch.setattr(shared,'predict',lambda model,ids:[dict(video_id=v,prediction=1.,reason=None) for v in ids])
    conditions={}
    for s in policy.SEEDS:
        models={f:One().eval() for f in datasets}
        oof={v:dict(prediction=1.,checkpoint_prediction=1.) for v in info}
        conditions[s]=dict(models=models,oof=oof,links={f:dict(sha256=f'fixture-{s}-{f}') for f in datasets},
            hashes={f:shared.state_hash(m.state_dict()) for f,m in models.items()})
    conditions[policy.SEEDS[0]]['oof']['fixture_0']['prediction']=1.25
    with torch.inference_mode(), pytest.raises(ValueError,match='Stopped before perturbation'):
        cli.reproduce(tmp_path,conditions,datasets,samples)
    replay=cli.cv.unseal(cli.read(tmp_path/'original-replay.json'))
    assert replay['comparisons']==66 and len(replay['mismatches'])==1
    mismatch=replay['mismatches'][0]
    assert mismatch['video_id']=='fixture_0' and mismatch['fold']==0 and mismatch['delta_vs_oof']==-.25
    assert mismatch['checkpoint']['sha256']=='fixture-20261026-0'


def test_wrong_weight_hash_stops_verification():
    model=torch.nn.Linear(1,1).eval()
    with pytest.raises(ValueError,match='Weights'):
        cli.weight_receipts({20261027:dict(models={0:model},hashes={0:'wrong hash'})})


def test_run_length_histogram_survives_sealed_json_roundtrip():
    import json
    histogram=cli.run_length_histogram([{'run_lengths':[2,11,2,100,1]}])
    assert cli.cv.unseal(json.loads(json.dumps(cli.cv.sealed(histogram))))=={'1':1,'2':2,'11':1,'100':1}
