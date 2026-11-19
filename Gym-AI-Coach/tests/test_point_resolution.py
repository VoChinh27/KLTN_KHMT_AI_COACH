"""Frozen acceptance before opt-in producer; identical-observation noise proof."""
import builtins
import io
import hashlib
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import geometry, axis_geometry
from gym_coach.contracts import ErrorCategory
from gym_coach.flags import FlagCode, GymError
from gym_coach.segmentation.models import strict_json
from gym_coach.segmentation.squat import segment_squat, temporal_flag
from gym_coach.segmentation.temporal_v2 import segment_squat_v2
from gym_coach.segmentation.temporal_v2_models import Support, read_temporal_json
from gym_coach.segmentation.point_resolution_models import (
    SquatPointResolutionAudit, read_point_resolution_json, read_temporal_experiment_json,
)
from gym_coach.segmentation.point_resolution import inspect_point_support, segment_squat_point_resolution
from gym_coach.reporting.point_resolution_demo import write_point_demo, render_point_audit
from point_resolution_cases import make_source


def codes(check):
    return {r.reason for r in check.reasons}


@pytest.mark.parametrize('fps',[25.,30.,60.])
def test_smooth_strict_minimum_necessary_but_noise_not_identifiable(fps):
    source=make_source(fps=fps);before=strict_json(source)
    result=segment_squat_point_resolution(source)
    assert result.checks and result.necessary_condition_count==len(result.checks)
    assert result.accepted_point_count==0 and strict_json(source)==before
    for check in result.checks:
        b=check.bottom_frame
        assert check.strict_local_minimum is True and codes(check)=={'noise_not_identifiable'}
        assert source.samples[b].value<min(source.samples[b-1].value,source.samples[b+1].value)
        assert check.point_turn_event is check.arrival is check.departure is check.pause_duration_seconds is None
        assert check.status=='unavailable'


def test_identical_observations_from_reversal_and_noisy_hold_cannot_be_separated():
    source=make_source();rep=segment_squat(source).repetitions[0]
    b=rep.bottom_frame
    observed=[s.value for s in source.samples]
    latent_hold=list(observed)
    for i in range(b-1,b+2):latent_hold[i]=observed[b]
    noise=[o-h for o,h in zip(observed,latent_hold)]
    assert any(noise) and [h+n for h,n in zip(latent_hold,noise)]==observed
    # Neither history/noise assertion is an input to this or any other producer.
    same_observations=type(source).model_validate_json(strict_json(source),strict=True)
    a=segment_squat_point_resolution(source);other=segment_squat_point_resolution(same_observations)
    assert a==other and a.checks[0].necessary_conditions_met
    assert a.accepted_point_count==0 and 'noise_not_identifiable' in codes(a.checks[0])


@pytest.mark.parametrize('case',['plateau','jitter'])
def test_plateau_and_jitter_remain_unavailable(case):
    result=segment_squat_point_resolution(make_source(case))
    assert result.accepted_point_count==0 and result.checks
    if case=='plateau':
        assert all(c.strict_local_minimum is False and 'raw_minimum_not_strict' in codes(c) for c in result.checks)
    assert all(c.point_turn_event is None and c.pause_duration_seconds is None for c in result.checks)


@pytest.mark.parametrize('case,reason',[
    ('missing_neighbor','neighbor_unavailable'),('gap','neighbor_unavailable'),
    ('imputed_neighbor','imputed_support'),('imputed_candidate','imputed_support'),
    ('side_switch','side_switch')])
def test_direct_raw_support_rejects_missing_imputed_and_side(case,reason):
    source=make_source(case)
    check=inspect_point_support(source,60,Support(start_frame=30,end_frame=90))
    assert not check.necessary_conditions_met and reason in codes(check)
    assert check.point_turn_event is None


@pytest.mark.parametrize('case',['gap','missing_neighbor','side_switch'])
def test_pipeline_never_creates_candidate_across_discontinuity(case):
    result=segment_squat_point_resolution(make_source(case))
    for check in result.checks:
        assert not check.candidate_support.start_frame<60<=check.candidate_support.end_frame
    assert result.accepted_point_count==0


@pytest.mark.parametrize('start,end,overlaps',[(59,61,True),(30,30,True),(90,90,True),(91,100,False),(None,None,True)])
def test_unstable_signal_inclusive_candidate_overlap_or_global(start,end,overlaps):
    source=make_source()
    flag=temporal_flag(FlagCode.UNSTABLE_SIGNAL,'Synthetic operational-flag contract',source.video_id,start,end)
    check=inspect_point_support(source,60,Support(start_frame=30,end_frame=90),flags=(flag,))
    assert ('unstable_signal_overlap' in codes(check))==overlaps
    assert check.necessary_conditions_met is (not overlaps)
    assert check.point_turn_event is None


@pytest.mark.parametrize('frame',[0,180])
def test_neighbor_clip_edge_is_explicit(frame):
    source=make_source()
    check=inspect_point_support(source,frame,Support(start_frame=0,end_frame=180))
    assert not check.necessary_conditions_met
    assert {'neighbor_unavailable','candidate_at_clip_boundary'}<=codes(check)


def test_clip_edge_no_full_pvp_does_not_fake_event():
    result=segment_squat_point_resolution(make_source('clip_edge'))
    assert not result.checks and result.accepted_point_count==0
    assert any(r.reason=='no_full_pvp_candidate' for r in result.diagnostics)


def test_bad_timestamp_or_reference_wrapper_is_structured_input_error():
    source=make_source()
    broken=source.samples[59].model_copy(update={'timestamp_seconds':2.})
    bad=source.model_copy(update={'samples':(*source.samples[:59],broken,*source.samples[60:])})
    for item in (bad,{'signal':source,'references':[]},object()):
        with pytest.raises(GymError) as caught:segment_squat_point_resolution(item)
        assert caught.value.flag.category==ErrorCategory.INPUT_ERROR
        assert caught.value.flag.stage=='segmentation' and caught.value.flag.source_tier=='S3'
    with pytest.raises(GymError):inspect_point_support(source,-1,Support(start_frame=0,end_frame=180))


def test_serialization_dispatch_and_byte_compatible_v1_v2():
    source=make_source();old=segment_squat(source);base=segment_squat_v2(source)
    result=segment_squat_point_resolution(source)
    assert strict_json(result.base_result)==strict_json(base)
    assert strict_json(result.base_result.legacy)==strict_json(old)
    restored=read_point_resolution_json(strict_json(result))
    assert restored==result and restored.content_hash()==result.content_hash()
    assert read_temporal_experiment_json(strict_json(old))==old
    assert read_temporal_experiment_json(strict_json(base))==base
    assert read_temporal_experiment_json(strict_json(result))==result
    with pytest.raises(ValueError):read_temporal_json(strict_json(result))
    assert not result.official_eligibility and not result.base_result.assessment.enabled


@pytest.mark.parametrize('field',['policy_hash','base_hash','input_hash','accepted','pause','version','reason'])
def test_hash_status_null_and_version_tampering_rejected(field):
    result=segment_squat_point_resolution(make_source());data=result.model_dump(mode='json')
    if field in ('policy_hash','base_hash'):data[field]='0'*64
    elif field=='input_hash':data['checks'][0]['input_hash']='0'*64
    elif field=='accepted':data['accepted_point_count']=1
    elif field=='pause':data['checks'][0]['pause_duration_seconds']=0.
    elif field=='version':data['point_resolution_schema_version']='99.0.0'
    else:data['checks'][0]['reasons']=[]
    with pytest.raises(ValueError):read_point_resolution_json(json.dumps(data))


def test_renaming_fixture_cannot_remove_abstention():
    source=make_source();named=source.model_copy(update={'video_id':'synthetic-noisefree-approved-reversal'})
    a=segment_squat_point_resolution(source);b=segment_squat_point_resolution(named)
    assert a.accepted_point_count==b.accepted_point_count==0
    assert [codes(c) for c in a.checks]==[codes(c) for c in b.checks]


def test_no_heavy_imports_data_io_geometry_or_source_mutation(monkeypatch):
    source=make_source();before=strict_json(source)
    def forbidden(*args,**kwargs):raise AssertionError('No geometry calculation')
    monkeypatch.setattr(geometry,'joint_angle',forbidden)
    monkeypatch.setattr(axis_geometry,'midpoint_axis_angle_vertical',forbidden)
    for module in (builtins,io):
        original=module.open
        def guarded(file,*args,_open=original,**kwargs):
            if 'data' in str(file).replace('\\','/').lower().split('/'):raise AssertionError('No Data access')
            return _open(file,*args,**kwargs)
        monkeypatch.setattr(module,'open',guarded)
    result=segment_squat_point_resolution(source)
    assert strict_json(source)==before
    assert 'noise_not_identifiable' in render_point_audit(result)
    script="""
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'} or name.endswith('synthetic_fixtures'):raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.segmentation.point_resolution import segment_squat_point_resolution
"""
    done=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert done.returncode==0,done.stderr


def test_demo_artifacts_hashes_no_positive_event_and_exclusive_output(tmp_path):
    target=tmp_path/'new-demo';manifest=write_point_demo(target)
    assert manifest['accepted_point_count']==0 and manifest['synthetic_only'] is True
    assert any(entry['necessary_condition_count']>0 for entry in manifest['cases'])
    assert any(entry['necessary_condition_count']==0 for entry in manifest['cases'])
    for entry in manifest['cases']:
        p=target/entry['result_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['result_sha256']
        assert isinstance(read_point_resolution_json(p.read_text(encoding='utf-8')),SquatPointResolutionAudit)
    before={p:p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):write_point_demo(target)
    assert all(p.read_bytes()==raw for p,raw in before.items())
