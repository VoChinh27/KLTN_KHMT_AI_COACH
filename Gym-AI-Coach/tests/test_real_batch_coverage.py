"""Batch scope/denominators/renderer tests with synthetic inputs only."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from gym_coach import geometry
from gym_coach.features import GEOMETRY_REGISTRY
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import load_preprocessing_profile
from gym_coach.reporting import real_batch_coverage as batch
from gym_coach.reporting import real_descriptive_demo as single
from gym_coach.reporting import real_descriptive_render as renderer
from test_real_descriptive_demo import source, result_from


def pairs40():
    result=[]
    for i in range(40):
        for side in ('left','right'):
            n=5+i
            action='allow' if (i+int(side=='right'))%3==0 else 'degrade' if i%2==0 else 'skip'
            key=single.KEYS[int(side=='right')]
            result.append(dict(path=f'Data/UCF-101_GYM/BodyWeightSquats/clip-{i:02d}.avi',side=side,
                feature_id=key,total_frames=n,observed_valid_frames=n if action=='allow' else n-1,
                observed_valid_ratio=1. if action=='allow' else (n-1)/n,
                metric=dict(action=action,reasons=[] if action=='allow' else ['partial_joint_coverage']),
                sequence=dict(action='skip'),candidate=action=='allow'))
    return result


def measured(pairs):
    result=[]
    for p in pairs:
        stat=batch.base_stat(p)
        n=p['total_frames'];available=n if p['metric']['action']=='allow' else 0
        stat.update(processing_status='completed',evaluated_frames=n,available_frames=available,unavailable_frames=n-available)
        result.append(stat)
    return result


def test_all80_pairs_whole_denominators_micro_macro_and_sides():
    pairs=pairs40();stats=measured(pairs);before=copy.deepcopy(stats)
    summary=batch.aggregate(pairs,stats)
    n=sum(5+i for i in range(40))
    assert summary['clip_count']==40 and len(summary['pairs'])==80 and summary['source_frames']==n
    assert summary['total']['frame_pair_denominator']==2*n
    assert summary['total']['evaluated_frame_pairs']==2*n and summary['total']['unevaluated_frame_pairs']==0
    assert summary['total']['available']+summary['total']['unavailable']==2*n
    assert summary['total']['numeric_micro']==sum(s['available_frames'] for s in stats)/(2*n)
    assert summary['clip_macro_numeric']==pytest.approx(sum(s['available_frames']/s['total_frames'] for s in stats)/80)
    assert summary['total']['numeric_micro']!=summary['clip_macro_numeric']
    assert sum(summary['total']['metric_actions'].values())==80
    assert all(s['pair_count']==40 and s['frame_pair_denominator']==n for s in summary['by_side'].values())
    assert summary['total']['raw_micro']>summary['total']['numeric_micro']
    assert summary['official_eligibility'] is summary['assessment'] is False
    assert stats==before


def test_missing_failed_and_partial_pairs_are_not_dropped():
    pairs=pairs40();stats=measured(pairs);lost=stats.pop()
    stats[0].update(processing_status='processing_error',evaluated_frames=0,available_frames=0,unavailable_frames=0,error='test failure')
    summary=batch.aggregate(pairs,stats)
    assert len(summary['pairs'])==80 and summary['total']['processing_errors']==2
    assert summary['total']['unevaluated_frame_pairs']==pairs[0]['total_frames']+lost['total_frames']
    assert summary['total']['numeric_micro']==summary['total']['available']/summary['total']['frame_pair_denominator']


@pytest.mark.parametrize('fault',['duplicate','missing_expected_side','unexpected','wrong_count','wrong_feature','partial_completed','negative'])
def test_bad_scope_counts_or_side_rejected(fault):
    pairs=pairs40();stats=measured(pairs)
    if fault=='duplicate':stats.append(copy.deepcopy(stats[0]))
    elif fault=='missing_expected_side':pairs.pop()
    elif fault=='unexpected':stats[0]['path']='not-in-pool'
    elif fault=='wrong_count':stats[0]['total_frames']+=1
    elif fault=='wrong_feature':stats[0]['feature_id']=single.KEYS[1]
    elif fault=='partial_completed':stats[0].update(evaluated_frames=0,available_frames=0,unavailable_frames=0)
    else:stats[0].update(available_frames=-1,unavailable_frames=stats[0]['total_frames']+1)
    with pytest.raises(ValueError):batch.aggregate(pairs,stats)


def test_frozen_examples_use_gate_then_path_and_side_never_numeric():
    pairs=pairs40();before=copy.deepcopy(pairs)
    chosen=batch.select_examples(list(reversed(pairs)))
    for action,pair in chosen.items():
        expected=sorted([p for p in pairs if p['metric']['action']==action],key=lambda p:(p['path'],p['side']))[0]
        assert pair==expected
    assert pairs==before
    missing=batch.select_examples([p for p in pairs if p['metric']['action']=='allow'])
    assert missing['degrade'] is missing['skip'] is None


@pytest.mark.parametrize('case',['global_skip','degrade','skip'])
def test_adapter_stats_nulls_provenance_and_vietnamese_without_recalculation(case,monkeypatch):
    processed=source(case);result=result_from(processed)
    pair=single.rank_pair('p','left',processed.raw_frames,processed.quality);stat=batch.base_stat(pair)
    from gym_coach.features import GeometryFeatureObservation
    for r in result['frames']:
        obs=GeometryFeatureObservation.model_validate_json(json.dumps(r['observation']),strict=True)
        batch.account(stat,obs)
    assert stat['evaluated_frames']==31 and sum(stat['input_status'].values())==93
    if case!='global_skip':
        assert stat['available_frames']==0 and stat['unavailable_frames']==31 and stat['reasons']
    def forbidden(*a,**k):raise AssertionError('Renderer recalculates domain result')
    monkeypatch.setattr(geometry,'joint_angle',forbidden)
    monkeypatch.setattr(batch,'observe_geometry_feature',forbidden)
    text=renderer.overlay_text(result['frames'][0],'left',pair['sequence'],localized_unavailable=True)
    if case=='global_skip':assert text['angle'].endswith('°')
    else:
        assert text['angle']=='Không đủ điều kiện đo' and 'Gate toàn clip' in text['reason']
        assert renderer.overlay_text(result['frames'][0],'left',pair['sequence'])['angle']=='Unavailable'


@pytest.mark.parametrize('reuse',[False,True])
def test_one_factory_call_shared_two_independent_adapters(tmp_path,monkeypatch,reuse):
    processed=source();raw=tuple(f.model_copy(update={'viewpoint':'unknown'}) for f in processed.raw_frames)
    source_data=dict(assumed=processed.raw_frames,raw=raw,quality=processed.quality,metadata=processed.metadata,
        candidate=dict(sha256='1'*64,review_id='test'),payloads={n:b'copied test payload' for n in
            ('raw-pose-unknown.jsonl','pose-assumed-side.jsonl','quality-assumed-side.json','quality-unknown.json','source.json')})
    pairs=[single.rank_pair('path',side,raw,processed.quality) for side in ('left','right')]
    prior=None
    if reuse:
        prior=tmp_path/'prior.json';batch.write_preprocessing_result(processed,prior)
        def forbidden(*a,**k):raise AssertionError('Validated preprocessing should be reused')
        monkeypatch.setattr(batch,'preprocess_pose',forbidden)
    else:monkeypatch.setattr(batch,'preprocess_pose',lambda *a,**k:processed)
    calls=[];original=batch.build_feature_input_context
    def wrapped(*a,**k):calls.append(k['frame_index']);return original(*a,**k)
    monkeypatch.setattr(batch,'build_feature_input_context',wrapped)
    done=batch._process_clip(tmp_path,tmp_path/'worker-new',source_data,pairs,processed.profile,prior)
    assert calls==list(range(31)) and done['timing']['factory_calls']==31 and done['timing']['adapter_calls']==62
    assert all(s['processing_status']=='completed' and s['available_frames']==31 for s in done['pairs'])
    for side,key in zip(('left','right'),single.KEYS):
        records=[json.loads(x) for x in (tmp_path/'worker-new'/f'{side}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==31 and all(r['side']==side and r['observation']['feature_id']==key for r in records)


def test_worker_error_returns_two_whole_denominators(tmp_path,monkeypatch):
    processed=source();pairs=[single.rank_pair('path',side,processed.raw_frames,processed.quality) for side in ('left','right')]
    def failed(*a,**k):raise RuntimeError('fixture processing error')
    monkeypatch.setattr(batch,'preprocess_pose',failed)
    source_data=dict(payloads={n:b'' for n in ('raw-pose-unknown.jsonl','pose-assumed-side.jsonl','quality-assumed-side.json','quality-unknown.json','source.json')},
        assumed=processed.raw_frames,quality=processed.quality,metadata=processed.metadata,candidate=dict(sha256='1'*64,review_id='test'))
    done=batch._process_clip(tmp_path,tmp_path/'failed-new',source_data,pairs,processed.profile)
    assert len(done['pairs'])==2 and all(p['processing_status']=='processing_error' and p['total_frames']==31 for p in done['pairs'])
    assert (tmp_path/'failed-new/processing-error.json').is_file()


@pytest.mark.parametrize('fault',[None,'preprocessing_bytes','selection_bytes','pool_hash'])
def test_prior_cache_requires_unchanged_hashes_and_pool_linkage(tmp_path,fault):
    prior=tmp_path/'outputs/prior/demo';prior.mkdir(parents=True)
    (prior/'preprocessing.json').write_bytes(b'validated snapshot bytes')
    single.write_json(prior/'selection.json',dict(path='clip',sha256='1'*64))
    single.write_json(prior/'manifest.json',dict(files={p.name:dict(sha256=single.sha(p),size_bytes=p.stat().st_size)
        for p in prior.iterdir()}))
    loaded={'clip':dict(candidate=dict(sha256='1'*64))}
    if fault=='preprocessing_bytes':(prior/'preprocessing.json').write_bytes(b'changed snapshot bytes')
    elif fault=='selection_bytes':(prior/'selection.json').write_bytes(b'changed selection')
    elif fault=='pool_hash':loaded['clip']['candidate']['sha256']='2'*64
    if fault:
        with pytest.raises(ValueError):batch.validate_prior_preprocessing(tmp_path,Path('outputs/prior'),loaded)
    else:
        assert batch.validate_prior_preprocessing(tmp_path,Path('outputs/prior'),loaded)=={'clip':str(prior/'preprocessing.json')}


@pytest.mark.parametrize('fault',['side','observation_hash'])
def test_example_rejects_saved_pair_linkage_before_rendering(tmp_path,monkeypatch,fault):
    processed=source();result=result_from(processed)
    folder=tmp_path/'batch/clips/test';folder.mkdir(parents=True)
    batch.write_preprocessing_result(processed,folder/'preprocessing.json')
    rows=[dict(side='left',observation=r['observation'],source_linkage=r['source_linkage']) for r in result['frames']]
    if fault=='side':rows[0]['side']='right'
    else:rows[0]['source_linkage']['observation_hash']='0'*64
    (folder/'left.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    def forbidden(*a,**k):raise AssertionError('Saved-only rendering recalculates geometry')
    monkeypatch.setattr(batch,'observe_geometry_feature',forbidden)
    monkeypatch.setattr(geometry,'joint_angle',forbidden)
    pair=single.rank_pair('clip','left',processed.raw_frames,processed.quality)
    with pytest.raises(ValueError,match='Saved pair/frame linkage'):
        batch.materialize_example(tmp_path,tmp_path/'batch',pair,dict(candidate=dict(review_id='test')),'1'*64)
    assert not (tmp_path/'batch/examples').exists()


def test_landing_has_all40_clips_local_links_and_escaped_metadata():
    pairs=pairs40();summary=batch.aggregate(pairs,measured(pairs));summary['clips'][0]['path']='<script>&.avi'
    page=batch.opening_page(summary,[dict(action='allow',path='p',side='left',link='examples/allow/START_HERE.html')])
    assert page.count('<tr>')==41 and '&lt;script&gt;&amp;.avi' in page and '<script>' not in page
    for text in ('NOT_VALIDATED','official_eligibility=false','assessment=false','browser/playback QA NOT_RUN','Không đủ điều kiện đo'):
        assert text in page
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k=='href')
    p=Links();p.feed(page)
    assert all(':' not in n and '..' not in n and not Path(n).is_absolute() for n in p.links)
