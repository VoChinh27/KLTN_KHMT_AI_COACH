"""Synthetic tests only. References are explicit test labels, never model output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.counting.models import (AnnotationBundle, AnnotationPass, Bottom, Cycle, Decision,
    EvaluationPolicy, Moment, Partial, Prediction, Span, Uncertain, VideoIdentity, canonical_hash, reason)
from gym_coach.counting.evaluation import checked, disagreement, evaluate, match_cycles
from gym_coach.counting.split import propose_split
from gym_coach.counting.readiness import file_info, safe_video_paths, write_json, SCOPE, decode_video

P = 'a'*64
NOW = datetime(2026, 10, 24, tzinfo=timezone.utc)


def source(g=1, c=1, fps=25., sha=None):
    name = f'v_BodyWeightSquats_g{g:02}_c{c:02}.avi'
    return VideoIdentity(video_id=name, path=f'{SCOPE.as_posix()}/{name}',
        sha256=sha or hashlib.sha256(name.encode()).hexdigest(), group_id=f'g{g:02}',
        fps=fps, frame_count=251, width=320, height=240)


def moment(i, fps=25.): return Moment(frame_index=i, timestamp_seconds=i/fps)
def span(a, b, fps=25.): return Span(start=moment(a, fps), end=moment(b, fps))
def cycle(a=25, b=50, c=75, *, fps=25., identity='r1', dwell=False):
    return Cycle(cycle_id=identity, interval=span(a, c, fps),
        bottom=Bottom(status='interval' if dwell else 'event', support=span(b, b+5 if dwell else b, fps)))


def annotation(s=None, *, actor='A', cycles=None, **kwargs):
    if cycles is None: cycles=(cycle(fps=(s or source()).fps),)
    data = dict(source=s or source(), protocol_sha256=P, status='completed', annotator=actor,
        method='synthetic_fixture', annotated_at=NOW, independent_of_predictions=True,
        blinded_to_other_pass=actor != 'C', provenance_note='Explicit synthetic test reference; not athlete data',
        subject_description='synthetic', count_status='available', count=len(cycles), count_reason=None,
        boundaries_status='exhaustive', boundaries_reason=None, cycles=cycles)
    data.update(kwargs)
    return AnnotationPass(**data)


def bundle(s=None, **kwargs):
    a, b, f = (annotation(s, actor=actor, **kwargs) for actor in ('A', 'B', 'C'))
    decision=Decision(field='review_all', rationale='Synthetic acceptance case', reviewer='C', decided_at=NOW,
                      pass_a_sha256=canonical_hash(a), pass_b_sha256=canonical_hash(b), final_sha256=canonical_hash(f))
    return AnnotationBundle(data_kind='synthetic', pass_a=a, pass_b=b, final=f, decisions=(decision,))


def prediction(s=None, *, count=1, cycles=None, status='estimated', **kwargs):
    data=dict(source=s or source(), status=status, count=count,
              cycles=cycles if cycles is not None else (cycle(),), events_exhaustive=True,
              algorithm_version='synthetic-test', config_sha256=P, evidence_ref='explicit synthetic fixture')
    data.update(kwargs)
    return Prediction(**data)


def ev(sources, labels, predictions, **kwargs):
    return evaluate(sources, labels, predictions, protocol_sha256=P, data_kind='synthetic', **kwargs)


@pytest.mark.parametrize('field,value', [('fps',float('nan')),('fps',float('inf')),('fps',0.),
    ('fps',True),('frame_count',True),('frame_count',-1),('width',0),('sha256','x'),
    ('group_id','g02'),('video_id','other.avi'),('path','../escape.avi'),('fps',None)])
def test_invalid_source(field,value):
    data=source().model_dump(mode='json'); data[field]=value
    with pytest.raises(ValueError): checked(VideoIdentity,data)


@pytest.mark.parametrize('fps',[25.,30.,60.])
def test_frame_time_and_json_roundtrip(fps):
    s=source(fps=fps); b=bundle(s,cycles=(cycle(fps=fps),))
    assert checked(AnnotationBundle,b)==b
    p=prediction(s,cycles=(cycle(fps=fps),))
    result=ev([s],[b],[p])
    assert result['count_metrics']['mae']==0
    assert result['event_metrics']['timing_seconds']['bottom']['mae']==0


@pytest.mark.parametrize('index,time',[(251,10.04),(1,.05),(1,float('nan')),(-1,0.),(True,0.)])
def test_reject_bad_sample(index,time):
    data=annotation().model_dump(mode='json')
    data['cycles'][0]['interval']['start']={'frame_index':index,'timestamp_seconds':time}
    with pytest.raises(ValueError): checked(AnnotationPass,data)


@pytest.mark.parametrize('field,value',[('count',0),('annotator','fake'),('status','completed'),
    ('subject_description','person'),('annotated_at',NOW.isoformat())])
def test_pending_never_prefills(field,value):
    data=AnnotationPass(source=source(),protocol_sha256=P).model_dump(mode='json');data[field]=value
    with pytest.raises(ValueError): checked(AnnotationPass,data)


@pytest.mark.parametrize('field,value',[('method',None),('annotated_at','2026-10-24T00:00:00'),
    ('independent_of_predictions',False),('provenance_note',''),('annotator',''),('subject_description',None)])
def test_completed_requires_provenance(field,value):
    data=annotation().model_dump(mode='json');data[field]=value
    with pytest.raises(ValueError): checked(AnnotationPass,data)


def test_count_known_without_boundaries_and_zero_vs_unavailable():
    known=annotation(cycles=(),count=3,boundaries_status='unavailable',boundaries_reason='exact boundaries unclear')
    assert known.count==3
    zero=bundle(cycles=()); predicted=prediction(count=0,cycles=())
    assert ev([source()],[zero],[predicted])['count_metrics']['exact_count_match']==1
    unknown=annotation(cycles=(),count=None,count_status='unavailable',count_reason='occlusion',
                       boundaries_status='unavailable',boundaries_reason='occlusion')
    assert unknown.count is None


def test_dwell_unavailable_bottom_shared_top():
    cycles=(cycle(dwell=True),cycle(75,100,125,identity='r2'))
    ref=bundle(cycles=cycles)
    result=ev([source()],[ref],[prediction(count=2,cycles=cycles)])
    assert result['event_metrics']['true_positive']==2
    assert result['event_metrics']['matched_without_point_bottom']==1
    assert result['event_metrics']['timing_seconds']['bottom']['n']==1
    assert Bottom(status='unavailable',reason='occluded').support is None


@pytest.mark.parametrize('kind,a,b',[('clip_start',1,10),('clip_end',200,249),('both_edges',0,249)])
def test_partial_requires_clip_edge(kind,a,b):
    with pytest.raises(ValueError): annotation(partials=(Partial(kind=kind,interval=span(a,b),reason='cut'),))


def test_partial_mask_reported_but_count_error_kept():
    ref=bundle(partials=(Partial(kind='clip_start',interval=span(0,15),reason='starts midway'),))
    pred=prediction(count=2,cycles=(cycle(1,5,10,identity='partial'),cycle()))
    result=ev([source()],[ref],[pred])
    assert result['count_metrics']['mae']==1
    assert result['event_metrics']['masked_predictions']==1
    assert result['event_metrics']['true_positive']==1
    assert result['rows'][0]['excluded_prediction_indices']==[0]


def test_gap_uncertain_never_becomes_known_total():
    u=Uncertain(interval=span(100,130),affects_count=True,reason='cut or occlusion')
    with pytest.raises(ValueError): annotation(uncertain=(u,))
    a=annotation(uncertain=(u,),count=None,count_status='unavailable',count_reason='unknown missing rep',
                 boundaries_status='partial',boundaries_reason='visible cycles only')
    assert len(a.cycles)==1 and a.count is None
    with pytest.raises(ValueError): annotation(uncertain=(Uncertain(interval=span(45,55),affects_count=False,reason='missing'),))


@pytest.mark.parametrize('mutation',['same_rater','not_blind','synthetic_real','wrong_hash','missing_log','early_log','wrong_source'])
def test_adjudication_guards(mutation):
    data=bundle().model_dump(mode='json')
    if mutation=='same_rater': data['pass_b']['annotator']='A'
    elif mutation=='not_blind': data['pass_b']['blinded_to_other_pass']=False
    elif mutation=='synthetic_real': data['data_kind']='real_development'
    elif mutation=='wrong_hash': data['decisions'][0]['final_sha256']='b'*64
    elif mutation=='missing_log': data['decisions']=[]
    elif mutation=='early_log': data['decisions'][0]['decided_at']='2020-01-01T00:00:00+00:00'
    else: data['pass_b']['source']['sha256']='b'*64
    with pytest.raises(ValueError): checked(AnnotationBundle,data)


def test_disagreement_requires_adjudication_even_when_counts_equal():
    a=annotation();b=annotation(actor='B',cycles=(cycle(26,51,76),))
    review=AnnotationBundle(data_kind='synthetic',pass_a=a,pass_b=b)
    assert disagreement(review)['status']=='REVIEW_REQUIRED'
    assert 'cycles' in disagreement(review)['differences']
    assert ev([source()],[review],[prediction()])['count_metrics']['mae'] is None


@pytest.mark.parametrize('state',['abstained','failed','not_run','candidates_only'])
def test_abstention_null_reasons_and_coverage(state):
    kw=dict(status=state,count=None,cycles=(cycle(),) if state=='candidates_only' else (),
            events_exhaustive=False,reasons=(reason('blocked','Synthetic unavailable'),))
    p=prediction(**kw)
    assert checked(Prediction,p)==p
    result=ev([source()],[bundle()],[p])
    assert result['count_metrics']['mae'] is None
    assert result['coverage']['prediction']==0
    assert result['event_metrics']['precision'] is None
    kw['count']=0
    with pytest.raises(ValueError): prediction(**kw)
    kw['count']=None;kw['reasons']=()
    with pytest.raises(ValueError): prediction(**kw)


def test_denominators_include_failed_absent_candidate_and_empty_labels():
    sources=[source(g=i) for i in range(1,6)]
    preds=[prediction(sources[0]),prediction(sources[1],count=None,status='failed',cycles=(),events_exhaustive=False,
        reasons=(reason('backend_error','synthetic fault','operational_error'),))]
    result=ev(sources,[],preds)
    assert result['denominator']['video_paths']==5
    assert result['coverage']==dict(prediction=.2,label=0.,comparable=0.,abstain=0.,failure=.2,not_run=.6,candidates_only=0.)
    assert result['evaluation_status']=='NOT_EVALUATED' and result['count_metrics']['mae'] is None
    assert len(result['rows'])==5


def test_hand_computed_count_statistics():
    sources=[source(g=i) for i in range(1,4)]
    labels=[bundle(s,cycles=(),count=c,boundaries_status='unavailable',boundaries_reason='count-only test')
            for s,c in zip(sources,(2,3,0))]
    preds=[prediction(s,count=c,cycles=(),events_exhaustive=False) for s,c in zip(sources,(3,1,0))]
    r=ev(sources,labels,preds)
    assert r['count_metrics']==dict(n=3,mae=1.,signed_error=-1/3,exact_count_match=1/3)
    assert r['event_metrics']['precision'] is None


def test_boundary_tolerance_misses_extras_signed_timing():
    ref=bundle(cycles=(cycle(),cycle(100,125,150,identity='r2')))
    pred=prediction(count=2,cycles=(cycle(27,52,77),cycle(175,200,225,identity='r3')))
    r=ev([source()],[ref],[pred])
    assert r['event_metrics']['true_positive']==1
    assert r['event_metrics']['false_positive']==r['event_metrics']['false_negative']==1
    assert r['event_metrics']['precision']==r['event_metrics']['recall']==.5
    assert r['event_metrics']['timing_seconds']['start']['mae']==pytest.approx(.08)
    assert r['event_metrics']['timing_frames']['end']['signed_error']==2
    strict=ev([source()],[ref],[pred],policy=EvaluationPolicy(boundary_tolerance_seconds=0.))
    assert strict['event_metrics']['true_positive']==0
    assert strict['event_metrics']['timing_seconds']['start']['mae'] is None


def test_matching_max_cardinality_not_greedy_reuse():
    # First prediction can match either; second only the second. Both must match.
    refs=(cycle(20,30,40),cycle(41,51,61,identity='r2'))
    preds=(cycle(30,40,50),cycle(52,62,72,identity='p2'))
    assert match_cycles(preds,refs,.5)==((0,0),(1,1))


@pytest.mark.parametrize('case',['duplicate_source','extra_prediction','wrong_hash','wrong_protocol','unknown_kind','duplicate_label'])
def test_intake_no_silent_drops(case):
    sources=[source()];labels=[bundle()];preds=[prediction()];kwargs={}
    if case=='duplicate_source':sources*=2
    if case=='extra_prediction':preds.append(prediction(source(2)))
    if case=='wrong_hash':preds=[prediction(source(sha='b'*64))]
    if case=='wrong_protocol':kwargs['protocol_sha256']='b'*64
    if case=='unknown_kind':kwargs['data_kind']='real_holdout'
    if case=='duplicate_label':labels*=2
    with pytest.raises(ValueError):evaluate(sources,labels,preds,protocol_sha256=kwargs.get('protocol_sha256',P),data_kind=kwargs.get('data_kind','synthetic'))


def test_nested_mutation_and_version_dispatch():
    d=bundle().model_dump(mode='json');d['version']='unknown'
    with pytest.raises(ValueError):checked(AnnotationBundle,d)
    p=prediction().model_copy(update={'count':True})
    with pytest.raises(ValueError):ev([source()],[bundle()],[p])


def test_group_duplicate_source_union_and_order_independence():
    sources=[source(g=i,c=j) for i in range(1,11) for j in (1,2)]
    sources[-1]=source(g=10,c=2,sha=sources[0].sha256)
    edges=[dict(a=sources[2].video_id,b=sources[4].video_id,evidence_ref='synthetic verified source relationship')]
    a=propose_split(sources,exposed_ids=[s.video_id for s in sources],related_sources=edges)
    b=propose_split(list(reversed(sources)),exposed_ids=[s.video_id for s in sources],related_sources=edges)
    assert a==b and len(a['assignments'])==20
    assert a['component_count']==8 and not a['pristine_holdout_established']
    for field in ('sha256','group_id','component_id'):
        for value in {r[field] for r in a['assignments']}:
            assert len({r['role'] for r in a['assignments'] if r[field]==value})==1


@pytest.mark.parametrize('kwargs',[{'exposed_ids':['unknown']}, {'exposed_ids':[], 'related_sources':[{'a':'x','b':'y'}]}])
def test_invalid_split_input(kwargs):
    with pytest.raises(ValueError):propose_split([source()],**kwargs)


def test_all_112_paths_40_subset_does_not_filter_and_split_denominators():
    sources=[source(g=(i//5)+1,c=i%5+1) for i in range(112)]
    result=propose_split(sources,exposed_ids=[s.video_id for s in sources])
    assert len(result['assignments'])==112
    assert len({r['video_id'] for r in result['assignments']})==112
    assert set(r['role'] for r in result['assignments'])=={'train','validation','development_test'}


def test_paths_hashes_no_overwrite(tmp_path):
    folder=tmp_path/SCOPE;folder.mkdir(parents=True)
    p=folder/source().video_id;p.write_bytes(b'synthetic file')
    (folder/'notes.txt').write_text('not a video')
    videos,other=safe_video_paths(tmp_path)
    assert videos==[p] and other==[(SCOPE/'notes.txt').as_posix()]
    assert file_info(p)['sha256']==hashlib.sha256(b'synthetic file').hexdigest()
    out=tmp_path/'receipt.json';write_json(out,{'a':1})
    with pytest.raises(FileExistsError):write_json(out,{'a':2})
    assert json.loads(out.read_bytes())=={'a':1}


def test_decode_failure_remains_in_inventory(tmp_path):
    path=tmp_path/'bad.avi';path.write_bytes(b'not a video')
    result=decode_video(path)
    assert result['status']=='failed' and result['reason']['category']=='input_error'
    assert result['frames_decoded']==0


def test_missing_metadata_pending_is_representable():
    d=source().model_dump(mode='json')
    for k in ('fps','frame_count','width','height'):d[k]=None
    s=checked(VideoIdentity,d)
    assert AnnotationPass(source=s,protocol_sha256=P).count is None
    with pytest.raises(ValueError):annotation(s,cycles=(),count=0)


def test_public_contract_evaluator_import_no_backend():
    code="""
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('cv2','numpy','scipy','mediapipe','torch','tkinter'):
   raise AssertionError('heavy import '+name)
sys.meta_path.insert(0,Block())
from gym_coach.counting import models, evaluation, split, readiness
assert evaluation.evaluate([],[],[],protocol_sha256='a'*64,data_kind='synthetic')['count_metrics']['mae'] is None
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_two_pass_intake_never_automatically_resolves():
    from gym_coach.counting.annotations import assemble
    s=source();a=annotation();b=annotation(actor='B')
    manifest=dict(version='squat-count-dataset-v1',data_kind='synthetic',protocol_sha256=P,videos=[dict(source=s.model_dump(mode='json'))])
    wrap=lambda row:dict(version='squat-count-pass-collection-v1',dataset_manifest_sha256=P,annotations=[row.model_dump(mode='json')])
    joined,reviews=assemble(manifest,P,wrap(a),wrap(b))
    assert joined['bundles'][0]['final'] is None
    assert reviews[0]['status']=='AGREEMENT_REQUIRES_LOG'
    bad=wrap(b);bad['annotations']=[]
    with pytest.raises(ValueError):assemble(manifest,P,wrap(a),bad)
    bad=wrap(b);bad['dataset_manifest_sha256']='b'*64
    with pytest.raises(ValueError):assemble(manifest,P,wrap(a),bad)


@pytest.mark.parametrize('failure',[False,True])
def test_prepare_retains_all_sources_blinds_package_and_failure(tmp_path,monkeypatch,failure):
    from gym_coach.counting import readiness as prep
    from gym_coach.counting.evaluation import main as evaluate_cli
    from gym_coach.counting.annotations import main as annotation_cli
    monkeypatch.chdir(tmp_path)
    folder=tmp_path/SCOPE;folder.mkdir(parents=True)
    video=folder/source().video_id;video.write_bytes(b'fake bytes; decoding is mocked')
    for name in (prep.VISUAL, prep.REPLAY_PLAN, prep.AUDIT, prep.PROTOCOL_FILE):
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
    write_json(tmp_path/prep.VISUAL,dict(videos=[dict(relative_path=video.relative_to(tmp_path).as_posix(),
        video_sha256=file_info(video)['sha256'],decoder_metadata=dict(fps=25.,frame_count=251,width=320,height=240),
        visual_viewpoint='oblique',samples=[dict(agent_viewed=True)])]))
    write_json(tmp_path/prep.REPLAY_PLAN,{})
    (tmp_path/prep.PROTOCOL_FILE).write_text('synthetic protocol')
    (tmp_path/prep.AUDIT).write_text('class_name,file_name,camera_view\nBodyWeightSquats,'+video.name+',UNKNOWN\n')
    monkeypatch.setattr(prep,'decode_video',lambda p:dict(status='failed' if failure else 'complete',
        frames_decoded=0 if failure else 251,metadata=None if failure else dict(fps=25.,frame_count=251,width=320,height=240),
        reason={'code':'decode_error'} if failure else None))
    monkeypatch.setattr(prep,'saved_compatibility',lambda r,s:({},[]))
    result=prep.prepare(Path('outputs/new'))
    assert result['video_paths']==1 and result['decode_status']==({'failed':1} if failure else {'complete':1})
    new=tmp_path/'outputs/new';blind=new/'annotation_package'
    assert file_info(blind/'videos'/video.name)==file_info(video)
    text=(blind/'manifest.json').read_text()
    assert 'metric' not in text and 'oblique' not in text and 'quality' not in text
    a=json.loads((blind/'pass_a.blank.json').read_bytes())['annotations'][0]
    assert a['count'] is None and a['annotator'] is None
    assert not list(blind.rglob('*pose*'))
    with pytest.raises(FileExistsError):prep.prepare(Path('outputs/new'))
    annotation_cli(['--manifest',str(new/'dataset-manifest.json'),'--pass-a',str(blind/'pass_a.blank.json'),
        '--pass-b',str(blind/'pass_b.blank.json'),'--output-dir','outputs/review'])
    evaluate_cli(['--manifest',str(new/'dataset-manifest.json'),'--annotations','outputs/review/adjudication.pending.json',
        '--predictions',str(new/'predictions.not_run.json'),'--output-dir','outputs/evaluation'])
    result=json.loads((tmp_path/'outputs/evaluation/evaluation.json').read_bytes())
    assert result['denominator']['video_paths']==1 and result['count_metrics']['mae'] is None
    assert result['prediction_states']=={'not_run':1}
    if failure:assert a['source']['fps'] is None
    # A second preparation may reuse verified decode evidence, never silently
    # trust stale metadata or decode again despite compatible evidence.
    monkeypatch.setattr(prep,'decode_video',lambda p: (_ for _ in ()).throw(AssertionError('unexpected decode')))
    reused=prep.prepare(Path('outputs/reused'),reuse_decode_from=Path('outputs/new'))
    assert reused['video_paths']==1
    report=json.loads((tmp_path/'outputs/reused/dataset-manifest.json').read_bytes())
    assert report['actual_decoded_frames_this_run']==0 and report['reused_decode_clips']==1
    video.write_bytes(b'changed video')
    with pytest.raises(ValueError,match='no longer match'):prep.prepare(Path('outputs/stale'),reuse_decode_from=Path('outputs/new'))
    (new/'dataset-manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):prep.load_decode_reuse(tmp_path,Path('outputs/new'))


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,True,'0.2'])
def test_matching_tolerance_is_explicit_and_finite(value):
    with pytest.raises(ValueError):EvaluationPolicy(boundary_tolerance_seconds=value)


def test_discovered_snapshot_rejects_traversal(tmp_path):
    from gym_coach.counting.readiness import verify_additional_snapshot
    from gym_coach.counting.models import STATUS
    from gym_coach.reporting.single_video import VERSION
    folder=tmp_path/'outputs/snapshot';folder.mkdir(parents=True)
    write_json(folder/'manifest.json',dict(version=VERSION,status='completed',**STATUS,files={'../escape.json':{}}))
    with pytest.raises(ValueError,match='Unsafe snapshot path'):
        verify_additional_snapshot(tmp_path,source(),{'path':'outputs/snapshot/manifest.json'})
