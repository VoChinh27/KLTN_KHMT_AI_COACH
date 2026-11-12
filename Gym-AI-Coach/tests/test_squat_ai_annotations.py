import copy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from gym_coach.counting.ai_annotations import AIAnnotation, save_pass, load_pass, compare_passes, verify_evidence, conservative_pair
from gym_coach.counting.models import AnnotationPass
from gym_coach.counting.readiness import file_info
from gym_coach.counting.ai_annotations import AIViewpoint, summarize_pairs


@pytest.fixture
def payload(tmp_path):
    video=tmp_path/'Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi'
    video.parent.mkdir(parents=True);video.write_bytes(b'test source, not real video')
    sheet=tmp_path/'sheet.jpg';sheet.write_bytes(b'test evidence')
    result = dict(protocol_sha256='a'*64,semantics_protocol_sha256='b'*64,
        source=dict(video_id=video.name,path=video.relative_to(tmp_path).as_posix(),sha256=file_info(video)['sha256'],group_id='g01',fps=25.0,frame_count=5,width=320,height=240),
        pass_id='A',recorded_at='2026-10-26T00:00:00Z',evidence=[dict(path='sheet.jpg',sha256=file_info(sheet)['sha256'],frame_indices=[0,1,2,3,4])],
        subject_description='Synthetic schema test only',whole_clip_count=1,observed_complete_count=1,count_reason=None,
        boundaries_reason='Unknown exact event samples',observations='Synthetic fixture, no actual visual review claim')
    manifest=dict(version='squat-ai-source-sheets-v1',source=parse(result).source.model_dump(mode='json'),decoded_frames=5,
        sheets=[dict(file='sheet.jpg',identity=file_info(sheet),frame_indices=list(range(5)),timestamps_seconds=[i/25 for i in range(5)])])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return result


def parse(p):return AIAnnotation.model_validate_json(json.dumps(p))
def span(a,b):return dict(start=dict(frame_index=a,timestamp_seconds=a/25),end=dict(frame_index=b,timestamp_seconds=b/25))


def test_roundtrip_and_no_overwrite(payload,tmp_path):
    a=parse(payload);p=tmp_path/'a.json';save_pass(p,a,tmp_path)
    assert load_pass(p)==a
    with pytest.raises(FileExistsError):save_pass(p,a,tmp_path)
    d=json.loads(p.read_bytes());d['payload']['whole_clip_count']=99;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='hash'):load_pass(p)


@pytest.mark.parametrize('key,value',[('method','manual_video_review'),('author_kind','human'),('human_validated',True),('independent_ground_truth',True),('independent_raters',True),('blinded_to_other_pass',True),('official_eligibility',True),('assessment',True),('whole_clip_count',True),('whole_clip_count',-1),('whole_clip_count','1')])
def test_reject_claims_and_counts(payload,key,value):
    payload[key]=value
    with pytest.raises(ValidationError):parse(payload)


def test_ai_not_accepted_as_human(payload):
    with pytest.raises(ValidationError):AnnotationPass.model_validate_json(parse(payload).model_dump_json())


def test_partial_coverage_null_not_zero(payload):
    payload['evidence'][0]['frame_indices']=[0,2,4]
    with pytest.raises(ValidationError):parse(payload)
    payload.update(whole_clip_count=None,count_reason='unviewed samples')
    assert parse(payload).whole_clip_count is None
    payload['whole_clip_count']=0;payload['observed_complete_count']=0;payload['count_reason']=None
    with pytest.raises(ValidationError):parse(payload)


def test_zero_with_complete_evidence(payload):
    payload.update(whole_clip_count=0,observed_complete_count=0)
    assert parse(payload).whole_clip_count==0


@pytest.mark.parametrize('indices',[[0,1,1,2,3,4],[4,3,2,1,0],[-1,0,1,2,3,4],[0,1,2,3,4,5],[]])
def test_invalid_evidence(payload,indices):
    payload['evidence'][0]['frame_indices']=indices
    with pytest.raises(ValidationError):parse(payload)


def test_uncertain_affects_count(payload):
    payload['uncertain']=[dict(interval=span(1,3),affects_count=True,reason='Occlusion')]
    with pytest.raises(ValidationError):parse(payload)
    payload.update(whole_clip_count=None,count_reason='Occlusion',observed_complete_count=None)
    assert parse(payload).whole_clip_count is None


def test_hint_order_timeline_count(payload):
    h=dict(start_range=span(0,1),bottom_range=span(2,2),end_range=span(3,4),reason='Uncertain envelopes')
    payload['cycle_hints']=[h];assert len(parse(payload).cycle_hints)==1
    h['bottom_range']=span(1,2)
    with pytest.raises(ValidationError):parse(payload)
    h['bottom_range']=span(2,2);h['end_range']['end']['timestamp_seconds']=.99
    with pytest.raises(ValidationError):parse(payload)
    h['end_range']=span(3,4);payload['observed_complete_count']=0;payload['whole_clip_count']=0
    with pytest.raises(ValidationError):parse(payload)


def test_partial_edge_and_hint_overlap(payload):
    payload['partials']=[dict(kind='clip_start',interval=span(1,2),reason='Partial')]
    with pytest.raises(ValidationError):parse(payload)
    payload['partials'][0]['interval']=span(0,1)
    payload['cycle_hints']=[dict(start_range=span(0,1),bottom_range=span(2,2),end_range=span(3,4),reason='Hint')]
    with pytest.raises(ValidationError):parse(payload)


@pytest.mark.parametrize('target',['sheet.jpg','Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi'])
def test_hash_linkage(payload,tmp_path,target):
    (tmp_path/target).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash linkage'):save_pass(tmp_path/'a.json',parse(payload),tmp_path)


def test_count_comparison_only_eligible_pairs(payload):
    a=parse(payload);payload['pass_id']='B';b=parse(payload)
    assert compare_passes(a,b)['count_equal'] is True
    payload.update(whole_clip_count=None,count_reason='Ambiguous')
    assert compare_passes(a,parse(payload))['count_equal'] is None
    payload['source']['sha256']='c'*64
    with pytest.raises(ValueError):compare_passes(a,parse(payload))


@pytest.mark.parametrize('change',['source','indices','timestamps','duplicate','decoded','version'])
def test_sheet_manifest_linkage(payload,tmp_path,change):
    path=tmp_path/'manifest.json';m=json.loads(path.read_bytes())
    if change=='source':m['source']['sha256']='c'*64
    elif change=='indices':m['sheets'][0]['frame_indices']=[0,2,4]
    elif change=='timestamps':m['sheets'][0]['timestamps_seconds'][2]=.09
    elif change=='duplicate':m['sheets']*=2
    elif change=='decoded':m['decoded_frames']=4
    else:m['version']='unknown'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='linkage'):verify_evidence(parse(payload),tmp_path)


def test_conservative_pair_does_not_average_or_promote_unknown(payload):
    a=parse(payload);payload['pass_id']='B';payload.update(whole_clip_count=2,observed_complete_count=2)
    r=conservative_pair(a,parse(payload))
    assert r['whole_clip_count'] is None and r['observed_complete_count'] is None
    assert r['count_reason'] and r['observed_reason'] and r['boundaries'] is None
    payload.update(whole_clip_count=None,count_reason='unclear',observed_complete_count=1)
    r=conservative_pair(a,parse(payload))
    assert r['whole_clip_count'] is None and r['observed_complete_count']==1
    assert r['official_eligibility'] is False and r['independent_ground_truth'] is False


def test_resume_rechecks_bytes_without_overwrite(payload,tmp_path,monkeypatch):
    from scripts.squat_ai_annotation_intake import intake
    monkeypatch.chdir(tmp_path)
    run=tmp_path/'outputs/run';run.mkdir(parents=True)
    docs=tmp_path/'docs';docs.mkdir();(docs/'SQUAT_AI_ANNOTATION_PROTOCOL.md').write_text('AI')
    protocol=tmp_path/'PROTOCOL.md';protocol.write_text('semantics')
    dataset=tmp_path/'dataset-manifest.json';dataset.write_text(json.dumps({'videos':[{'source':parse(payload).source.model_dump(mode='json')}]}))
    (run/'decision.json').write_text(json.dumps({'inputs':{p.as_posix():file_info(p) for p in (protocol,dataset)}}))
    sheets=run/'source-sheets'/payload['source']['video_id'];sheets.mkdir(parents=True)
    (sheets/'sheet.jpg').write_bytes((tmp_path/'sheet.jpg').read_bytes())
    (sheets/'manifest.json').write_bytes((tmp_path/'manifest.json').read_bytes())
    observations=run/'rows.json';observations.write_text(json.dumps([dict(video_id=payload['source']['video_id'],viewed_sheets=['sheet.jpg'],subject='test',count=1,observed=1,notes='test')]))
    assert intake(run,observations,'A')[0]['status']=='created'
    assert intake(run,observations,'A')[0]['status']=='verified_resume'
    output=next((run/'pass-A').glob('*.json'));before=output.read_bytes()
    (sheets/'sheet.jpg').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash linkage'):intake(run,observations,'A')
    assert output.read_bytes()==before


def view_payload(payload):
    return dict(protocol_sha256='a'*64,rubric_sha256='b'*64,source=payload['source'],
        recorded_at=payload['recorded_at'],evidence=payload['evidence'],
        reviewed_intervals=[dict(interval=span(0,4),viewpoint='front',reason='Synthetic schema test')],
        declared_whole_clip_viewpoint='front',subject_description='Synthetic',camera_changes=False,
        camera_direction_notes='Front',framing_notes='Test',occlusion_notes='Test',uncertainty_notes='Unverified')


def test_ai_viewpoint_roundtrip_and_mixed_not_promoted(payload):
    p=view_payload(payload);v=AIViewpoint.model_validate_json(json.dumps(p))
    assert AIViewpoint.model_validate_json(v.model_dump_json())==v
    p['reviewed_intervals']=[dict(interval=span(0,2),viewpoint='side',reason='Test'),dict(interval=span(3,4),viewpoint='unknown',reason='Mixed')]
    p['declared_whole_clip_viewpoint']='side'
    with pytest.raises(ValidationError):AIViewpoint.model_validate_json(json.dumps(p))
    p['declared_whole_clip_viewpoint']='unknown'
    assert AIViewpoint.model_validate_json(json.dumps(p)).documented is False


@pytest.mark.parametrize('key,value',[('documented',True),('source_camera_view','side'),('human_validated',True),('official_eligibility',True),('assessment',True),('evidence_status','documented')])
def test_viewpoint_rejects_promoted_claims(payload,key,value):
    p=view_payload(payload);p[key]=value
    with pytest.raises(ValidationError):AIViewpoint.model_validate_json(json.dumps(p))


@pytest.mark.parametrize('bad',['gap','time','partial'])
def test_viewpoint_requires_full_linked_timeline(payload,bad):
    p=view_payload(payload)
    if bad=='gap':p['reviewed_intervals'][0]['interval']=span(1,4)
    elif bad=='time':p['reviewed_intervals'][0]['interval']['end']['timestamp_seconds']=.3
    else:p['evidence'][0]['frame_indices']=[0,1,2]
    with pytest.raises(ValidationError):AIViewpoint.model_validate_json(json.dumps(p))


def test_full_cohort_summary_includes_null_and_zero(payload):
    a=parse(payload);p=copy.deepcopy(payload);p['pass_id']='B';b=parse(p)
    p=copy.deepcopy(payload);p['source']['video_id']='v_BodyWeightSquats_g01_c02.avi';p['source']['path']=p['source']['path'].replace('c01','c02')
    p.update(whole_clip_count=None,observed_complete_count=0,count_reason='unclear');a2=parse(p);p['pass_id']='B';b2=parse(p)
    ids=[a.source.video_id,a2.source.video_id];pairs=[(a,b),(a2,b2)]
    s=summarize_pairs(pairs,ids)['summary']
    assert (s['clips'],s['passes'],s['unique_frames'])==(2,4,10)
    assert (s['comparable_count_pairs'],s['matching_count_pairs'],s['count_unresolved_clips'])==(1,1,1)
    assert s['known_count_sum']==1 and s['exact_exhaustive_boundaries_available']==0
    for invalid in (pairs[:1],pairs+pairs[:1]):
        with pytest.raises(ValueError,match='cohort'):summarize_pairs(invalid,ids)
    p.update(whole_clip_count=0,count_reason=None);a0=parse({**p,'pass_id':'A'});b0=parse(p)
    s=summarize_pairs([(a0,b0)],[a0.source.video_id])['summary']
    assert s['count_available_clips']==1 and s['known_count_sum']==0


def test_ai_contract_import_has_no_video_or_training_backend():
    import subprocess,sys
    code="import sys; import gym_coach.counting.ai_annotations; assert not any(x in sys.modules for x in ('cv2','mediapipe','numpy','scipy','torch'))"
    subprocess.run([sys.executable,'-c',code],check=True)
