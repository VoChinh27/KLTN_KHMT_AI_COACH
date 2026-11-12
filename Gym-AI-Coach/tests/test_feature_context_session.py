"""Strict intake, detached nested state and exact legacy identity equivalence."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import pickle
import subprocess
import sys

import pytest

from gym_coach.contracts import ErrorCategory, SourceTier, Stage
from gym_coach.features import (
    FeatureContextSession, FeatureInputContext, LinkedFeatureInputContext,
    build_feature_input_context, observe_geometry_feature,
)
from gym_coach.features.models import content_hash
from gym_coach.flags import FlagCode, GymError
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import PreprocessingResult, PreprocessingProfile
from gym_coach.preprocessing.models import content_hash as source_hash
from gym_coach.reporting.real_descriptive_demo import KEYS
from stage5_6_fixture import fixture_result
from test_feature_observation import fixture_context
from test_preprocessing import missing
from test_real_descriptive_demo import source


def legacy_projection(linked):
    out=linked.model_dump(mode='json',exclude={'frame','quality','preprocessing_result'})
    out.update(frame_index=linked.frame.frame_index,timestamp_seconds=linked.frame.timestamp_seconds,
               measurement_input_hash=linked.measurement_input_hash)
    return out


@pytest.mark.parametrize('case',['global_skip','degrade','skip','imputed','missing','fallback','degenerate'])
def test_entire_observation_and_every_reference_linkage_field_equal(case):
    if case in ('global_skip','degrade','skip'):result=source(case)
    elif case=='degenerate':result=fixture_context(ankle=(.4,.4))[1]
    else:result=fixture_context(changes=missing([5],J.LEFT_KNEE) if case!='fallback' else missing(range(31),J.RIGHT_HIP),
        profile=PreprocessingProfile(max_gap_seconds=0.) if case=='missing' else None)[1]
    session=FeatureContextSession(result)
    for index in (0,5,15,30):
        old=build_feature_input_context(result,frame_index=index)
        selected=session.select(frame_index=index);new=selected.input_context()
        assert type(new) is FeatureInputContext and not isinstance(new,LinkedFeatureInputContext)
        assert new.frame==old.frame and new.quality==old.quality and selected.raw_frame==old.raw_frame
        assert selected.reference_linkage()==legacy_projection(old)
        assert selected.measurement_input_hash==content_hash(new)==old.measurement_input_hash
        for key in KEYS:
            reference=observe_geometry_feature(key,FeatureInputContext(frame=old.frame,quality=old.quality))
            optimized=observe_geometry_feature(key,new)
            assert optimized.model_dump(mode='json')==reference.model_dump(mode='json')
            assert optimized.model_dump_json()==reference.model_dump_json()
            if case in ('degrade','skip') and key==KEYS[0]:
                assert optimized.value is None and optimized.reasons and optimized.flags


@pytest.mark.parametrize('index',[-1,True,False,1.,'1',None,[],float('nan'),float('inf'),31,100])
def test_index_rejects_without_coercion(index):
    session=FeatureContextSession(source())
    with pytest.raises(GymError) as e:session.select(frame_index=index)
    assert e.value.flag.code==FlagCode.INVALID_FEATURE_CONTEXT_INDEX
    assert e.value.flag.stage==Stage.FEATURES and e.value.flag.category==ErrorCategory.INPUT_ERROR
    assert e.value.flag.source_tier==SourceTier.S3


@pytest.mark.parametrize('field',['raw_sequence_hash','quality_hash','profile_hash','raw_frame_hash',
    'raw_timestamp','processed_timestamp','processed_index','raw_video','metadata_fps','profile',
    'resolved_profile','quality','quality_core','raw_length','smoothing_runs','source_status'])
def test_complete_audit_rejects_corruption_like_reference(field):
    result=source();updates={}
    if field in ('raw_sequence_hash','quality_hash','profile_hash'):updates[field]='0'*64
    elif field=='profile':updates['profile']=result.profile.model_copy(update={'profile_version':'different'})
    elif field=='resolved_profile':updates[field]=result.resolved_profile.model_copy(update={'fps':60.})
    elif field=='metadata_fps':updates['metadata']=result.metadata.model_copy(update={'fps':'30.0'})
    elif field=='quality':updates['quality']=result.quality.model_copy(update={'total_frames':30})
    elif field=='quality_core':
        q=source('skip').quality;updates.update(quality=q,quality_hash=source_hash(q))
    elif field=='raw_length':updates['raw_frames']=result.raw_frames[:-1]
    elif field=='smoothing_runs':updates[field]=()
    else:
        raw=field in ('raw_timestamp','raw_video');key='raw_frames' if raw else 'frames'
        frames=list(getattr(result,key));frame=frames[5]
        change={
            'raw_frame_hash':{'raw_frame_hash':'0'*64},'raw_timestamp':{'timestamp_seconds':.6},
            'processed_timestamp':{'timestamp_seconds':.6},'processed_index':{'frame_index':6},
            'raw_video':{'video_id':'other.avi'},
        }.get(field,{})
        if field=='source_status':change['joints']=tuple(j.model_copy(update={'source_status':'missing'}) if j.joint==J.LEFT_KNEE else j for j in frame.joints)
        frames[5]=frame.model_copy(update=change);updates[key]=tuple(frames)
    unsafe=result.model_copy(update=updates)
    for intake in (lambda:build_feature_input_context(unsafe,frame_index=5),lambda:FeatureContextSession(unsafe)):
        with pytest.raises(GymError) as e:intake()
        assert e.value.flag.code in (FlagCode.FEATURE_CONTEXT_MISMATCH,FlagCode.INVALID_FEATURE_INPUT)
        assert e.value.flag.stage==Stage.FEATURES and e.value.flag.category==ErrorCategory.INPUT_ERROR
        assert e.value.flag.evidence['reason'] and 'value' not in e.value.flag.evidence


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),True,'0.5'])
def test_construct_and_nullable_nonfinite_or_bad_type_not_hidden(value):
    result=source('skip');frames=list(result.frames)
    frames[5]=frames[5].model_copy(update={'joints':tuple(j.model_copy(update={'x':value}) if j.joint==J.LEFT_KNEE else j for j in frames[5].joints)})
    unsafe=PreprocessingResult.model_construct(**{**result.__dict__,'frames':tuple(frames)})
    with pytest.raises(GymError) as e:FeatureContextSession(unsafe)
    assert e.value.flag.code==FlagCode.INVALID_FEATURE_INPUT


@pytest.mark.parametrize('value',[None,{},(),True,'already-validated'])
def test_no_caller_supplied_validated_flag_or_token(value):
    with pytest.raises(GymError):FeatureContextSession(value)
    if not isinstance(value,str):
        with pytest.raises(GymError):FeatureContextSession.from_json(value)


def test_external_json_rechecks_and_empty_result():
    result=source();s=FeatureContextSession.from_json(result.model_dump_json())
    assert s.select(frame_index=0).input_context().frame==result.frames[0]
    p=result.model_dump(mode='json');p['raw_sequence_hash']='0'*64
    with pytest.raises(GymError):FeatureContextSession.from_json(json.dumps(p))
    p=result.model_dump(mode='json');p['frames'][0]['joints'][0]['x']=True
    with pytest.raises(GymError):FeatureContextSession.from_json(json.dumps(p))
    from test_preprocessing import make_frames,metadata
    from gym_coach.quality import evaluate_quality
    from gym_coach.preprocessing import preprocess_pose
    raw=make_frames(n=0);empty=preprocess_pose(raw,evaluate_quality(raw),metadata(raw))
    s=FeatureContextSession(empty);assert len(s)==0
    with pytest.raises(GymError):s.select(frame_index=0)


def test_nested_caller_and_returned_context_mutation_cannot_change_snapshot():
    result=source('skip');session=FeatureContextSession(result);selection=session.select(frame_index=15)
    before=selection.input_context().model_dump_json();audit=selection.reference_linkage()
    result.quality.metric_eligibility.clear();result.quality.by_joint.clear()
    result.quality.flags[0].evidence.clear()
    returned=selection.input_context();returned.quality.metric_eligibility.clear();returned.quality.by_joint.clear()
    returned.quality.flags[0].evidence.clear()
    projection=selection.reference_linkage();projection['quality_hash']='0'*64
    session.metadata.clear()
    assert selection.input_context().model_dump_json()==before
    assert session.select(frame_index=15).input_context().model_dump_json()==before
    assert session.select(frame_index=15).reference_linkage()==audit
    with pytest.raises(AttributeError):session.extra=True
    with pytest.raises(AttributeError):selection._input_json=b'{}'
    with pytest.raises(TypeError):pickle.dumps(session)


def test_raw_flag_evidence_and_nested_frame_flags_are_detached():
    from gym_coach.flags import StructuredFlag,Severity
    from gym_coach.preprocessing import preprocess_pose
    from gym_coach.quality import evaluate_quality
    from test_preprocessing import make_frames,metadata
    flag=StructuredFlag(code=FlagCode.VIEWPOINT_UNKNOWN,category=ErrorCategory.MEASUREMENT_ERROR,
        stage=Stage.POSE,source_tier=SourceTier.S3,severity=Severity.INFO,message='audit',evidence={'marker':'before'})
    raw=make_frames(n=5);raw=(raw[0].model_copy(update={'flags':(flag,)}),*raw[1:])
    result=preprocess_pose(raw,evaluate_quality(raw),metadata(raw));s=FeatureContextSession(result)
    selected=s.select(frame_index=0);old=selected.raw_frame.model_dump_json()
    processed_before=selected.input_context().frame.model_dump_json()
    result.raw_frames[0].flags[0].evidence['marker']='caller edit'
    result.frames[0].flags[0].evidence['marker']='processed edit'
    selected.raw_frame.flags[0].evidence['marker']='returned edit'
    selected.input_context().frame.flags[0].evidence['marker']='returned frame edit'
    assert s.select(frame_index=0).raw_frame.model_dump_json()==old
    assert s.select(frame_index=0).input_context().frame.model_dump_json()==processed_before


def worker_snapshot(payload):
    import os
    s=FeatureContextSession.from_json(payload);c=s.select(frame_index=5)
    return os.getpid(),s.metadata,c.reference_linkage(),c.input_context().model_dump_json()


def test_no_shared_cache_between_clips_profiles_threads_or_workers():
    a=fixture_result()
    from stage5_6_fixture import fixture_input,knee_requirements,knee_viewpoints
    from gym_coach.preprocessing import preprocess_pose
    from gym_coach.quality import QualityGate
    from pathlib import Path
    raw,info,_=fixture_input(changes={(i,J.RIGHT_ANKLE):('observed',.8,.8) for i in range(31)})
    raw=tuple(f.model_copy(update={'video_id':'other.avi'}) for f in raw)
    info=info.model_copy(update={'video_id':'other.avi','source_path':Path('other.avi')})
    quality=QualityGate().evaluate(raw,metric_requirements=knee_requirements(),metric_viewpoints=knee_viewpoints())
    b=preprocess_pose(raw,quality,info,profile=a.profile)
    # A legitimate distinct configuration must have its own complete identity.
    data=b.model_dump(mode='json');data['profile']['profile_version']='other-config'
    p=PreprocessingProfile.model_validate(data['profile']);data['profile_hash']=source_hash(p)
    b=PreprocessingResult.model_validate_json(json.dumps(data),strict=True)
    payloads=[a.model_dump_json(),b.model_dump_json()]*2
    with ThreadPoolExecutor(max_workers=2) as ex:thread=list(ex.map(worker_snapshot,payloads))
    with ProcessPoolExecutor(max_workers=2) as ex:process=list(ex.map(worker_snapshot,payloads))
    for results in (thread,process):
        assert results[0][1:]==results[2][1:] and results[1][1:]==results[3][1:]
        assert results[0][1]['preprocessing_result_hash']!=results[1][1]['preprocessing_result_hash']
        assert results[0][2]['profile_hash']!=results[1][2]['profile_hash']
        assert results[0][3]!=results[1][3]


def test_full_source_validation_once_not_per_frame_and_no_gate_recompute(monkeypatch):
    result=source();original=PreprocessingResult.model_validate_json;calls=[]
    def tracked(*a,**k):calls.append(k);return original(*a,**k)
    monkeypatch.setattr(PreprocessingResult,'model_validate_json',tracked)
    from gym_coach.quality import gate
    def forbidden(*a,**k):raise AssertionError('Must preserve saved quality decisions')
    monkeypatch.setattr(gate,'_eligibility',forbidden)
    s=FeatureContextSession(result)
    for index in range(len(s)):
        context=s.select(frame_index=index).input_context()
        for key in KEYS:observe_geometry_feature(key,context)
    assert calls==[{'strict':True}]
    assert s.snapshot_size_bytes>0


def test_legacy_linked_validator_unchanged_and_programming_errors_not_hidden(monkeypatch):
    r=source();linked=build_feature_input_context(r,frame_index=0)
    payload=linked.model_dump(mode='json');payload['quality_hash']='0'*64
    from pydantic import ValidationError
    with pytest.raises(ValidationError):LinkedFeatureInputContext.model_validate_json(json.dumps(payload),strict=True)
    def broken(*a,**k):raise RuntimeError('programming error')
    monkeypatch.setattr(PreprocessingResult,'model_validate_json',broken)
    with pytest.raises(RuntimeError):FeatureContextSession(r)


def test_session_import_and_saved_intake_are_lightweight(tmp_path):
    path=tmp_path/'source.json';path.write_text(source().model_dump_json(),encoding='utf-8')
    code=r'''
import importlib.abc,sys,pathlib
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.features import FeatureContextSession,observe_geometry_feature
s=FeatureContextSession.from_json(pathlib.Path(sys.argv[1]).read_bytes())
assert observe_geometry_feature('squat_knee_angle_left_deg',s.select(frame_index=0).input_context()).status=='available'
'''
    p=subprocess.run([sys.executable,'-I','-B','-c',code,str(path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
