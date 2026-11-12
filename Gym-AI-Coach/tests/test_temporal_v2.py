"""Frozen v2 acceptance: contract examples are NOT producer ground truth."""
import builtins
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.contracts import JointProvenance
from gym_coach.flags import FlagCode, GymError
from gym_coach.segmentation.models import SquatSignalSample, SquatTemporalInput, strict_json
from gym_coach.segmentation.squat import segment_squat
from gym_coach.segmentation.temporal_v2_models import (
    PROFILE_HASH, Bottom, Cycle, Event, EventComponent, Evidence, MotionInterval,
    PauseComponent, PauseValue, PhaseComponent, SquatTemporalResultV2, Support,
    V2Reason, event_id, read_temporal_json,
)
from gym_coach.segmentation.temporal_v2 import segment_squat_v2, phase_at_v2
from gym_coach.reporting.temporal_v2_renderer import render_temporal_v2
from gym_coach.reporting.temporal_v2_demo import write_demo


def source(*, fps=30., dwell=False, gap=(), switch=None, end=None):
    samples = []
    count = round(6*fps)+1 if end is None else end
    for i in range(count):
        t = i/fps
        value = 140.-30.*math.cos(math.pi*t)
        if dwell and any(abs(t-bottom) <= .2 for bottom in (2., 4.)):
            value = 110.
        samples.append(SquatSignalSample(frame_index=i, timestamp_seconds=t,
            value=None if i in gap else value, side_source='right' if switch is not None and i >= switch else 'left',
            joint_provenance=(JointProvenance.MISSING,)*3 if i in gap else (JointProvenance.OBSERVED,)*3,
            reasons=(FlagCode.MISSING_JOINT,) if i in gap else ()))
    digest = hashlib.sha256(json.dumps([s.model_dump(mode='json') for s in samples],sort_keys=True).encode()).hexdigest()
    return SquatTemporalInput(video_id='synthetic-v2-acceptance', fps=fps, samples=tuple(samples), source_hash=digest)


def contract_example(mode):
    """Hand-authored OUTPUT records, testing schema only; never producer inputs."""
    actual = segment_squat_v2(source())
    data = actual.model_dump(mode='json')
    data['producer_version'] = 'contract_acceptance_only'
    for cycle in data['cycles']:
        events = {e['event_id']:e for e in data['events']}
        s, b, e = (events[key] for key in (cycle['start_event_id'],cycle['bottom']['anchor_event_id'],cycle['end_event_id']))
        if mode == 'point_turn_at_resolution':
            b['evidence']['method'] = 'declared_point_evidence'
            reason = V2Reason(reason='pause_unresolved_at_resolution').model_dump(mode='json')
            cycle['bottom'] = dict(mode=mode,anchor_event_id=b['event_id'],
                arrival=dict(status='unavailable',event=None,reasons=[reason]),
                departure=dict(status='unavailable',event=None,reasons=[reason]),
                pause=dict(status='unavailable',value=None,reasons=[reason]))
            spans=[('Descent',s,b),('Ascent',b,e)]
            basis='point_turn_marker_span'
        else:
            bounds=[]
            for kind, frame in (('bottom_arrival',b['frame_index']-3),('bottom_departure',b['frame_index']+3)):
                bound=dict(b,kind=kind,frame_index=frame,timestamp_seconds=frame/actual.legacy.input.fps,
                    event_id=event_id(actual.legacy.input_hash,b['run_id'],kind,frame),
                    evidence=dict(method='declared_dwell_evidence',input_hash=actual.legacy.input_hash,
                        support=dict(start_frame=frame-1,end_frame=frame+1)))
                data['events'].append(bound);bounds.append(bound)
            a,d=bounds
            cycle['bottom']=dict(mode='dwell_candidate',anchor_event_id=b['event_id'],
                arrival=dict(status='candidate',event=a,reasons=[]), departure=dict(status='candidate',event=d,reasons=[]),
                pause=dict(status='candidate',value=dict(arrival_event_id=a['event_id'],departure_event_id=d['event_id'],
                    duration_seconds=d['timestamp_seconds']-a['timestamp_seconds'],unit='seconds',closure='[)'),reasons=[]))
            spans=[('Descent',s,a),('Bottom',a,d),('Ascent',d,e)];basis='dwell_bounds'
        cycle['motion']=dict(status='candidate',reasons=[],intervals=[dict(phase=p,start_event_id=x['event_id'],
            end_event_id=y['event_id'],closure='[)',basis=basis) for p,x,y in spans])
    return SquatTemporalResultV2.model_validate_json(json.dumps(data),strict=True)


def test_dwell_half_open_boundaries_and_independent_anchor():
    result=contract_example('dwell_candidate');cycle=result.cycles[0]
    a,d=cycle.bottom.arrival.event,cycle.bottom.departure.event
    assert phase_at_v2(result,a.frame_index-1).motion_phase=='Descent'
    assert phase_at_v2(result,a.frame_index).motion_phase=='Bottom'
    assert phase_at_v2(result,d.frame_index-1).motion_phase=='Bottom'
    assert phase_at_v2(result,d.frame_index).motion_phase=='Ascent'
    assert cycle.bottom.pause.value.duration_seconds==pytest.approx(.2,abs=1e-9)
    assert a.event_id!=cycle.bottom.anchor_event_id!=d.event_id
    assert any(e.kind=='bottom_arrival' for e in phase_at_v2(result,a.frame_index).events)


def test_point_turn_event_is_not_a_pause_or_physical_bound():
    result=contract_example('point_turn_at_resolution');cycle=result.cycles[0]
    b=next(e for e in result.events if e.event_id==cycle.bottom.anchor_event_id)
    frame=phase_at_v2(result,b.frame_index)
    assert frame.motion_phase=='Ascent' and b in frame.events
    assert cycle.bottom.arrival.event is cycle.bottom.departure.event is cycle.bottom.pause.value is None
    assert cycle.motion.intervals[1].basis=='point_turn_marker_span'


@pytest.mark.parametrize('dwell',[False,True])
@pytest.mark.parametrize('fps',[25.,30.,60.])
def test_producer_bounds_unresolved_never_infers_pause_from_anchor_or_fixture(fps,dwell):
    supplied=source(fps=fps,dwell=dwell);before=strict_json(supplied)
    result=segment_squat_v2(supplied)
    assert result.candidate_count==result.legacy.candidate_count>0
    assert strict_json(supplied)==before
    assert result.producer_version=='squat-v2-conservative-adapter-v1'
    for cycle in result.cycles:
        assert cycle.bottom.mode=='bounds_unresolved'
        assert cycle.bottom.arrival.event is cycle.bottom.departure.event is cycle.bottom.pause.value is None
        frame=phase_at_v2(result,next(e.frame_index for e in result.events if e.event_id==cycle.bottom.anchor_event_id))
        assert frame.motion_phase is None and frame.motion_status=='unavailable' and frame.reasons
        assert any(e.kind=='bottom_anchor' for e in frame.events)


def test_shared_top_one_event_two_roles_and_next_phase_ownership():
    result=contract_example('dwell_candidate');left,right=result.cycles[:2]
    assert left.end_event_id==right.start_event_id
    top=next(e for e in result.events if e.event_id==left.end_event_id)
    frame=phase_at_v2(result,top.frame_index)
    assert frame.motion_phase=='Descent' and frame.cycle_id==right.cycle_id
    assert {r.role for r in frame.event_roles}=={'top_start','top_end'}
    assert len(frame.events)==1
    final=next(e for e in result.events if e.event_id==right.end_event_id)
    assert phase_at_v2(result,final.frame_index).motion_phase is None
    assert phase_at_v2(result,final.frame_index).events==(final,)


@pytest.mark.parametrize('index',[30,60,90,120])
@pytest.mark.parametrize('kind',['gap','side'])
def test_gap_side_switch_split_runs_even_at_top(kind,index):
    supplied=source(gap=[index]) if kind=='gap' else source(switch=index)
    result=segment_squat_v2(supplied)
    assert any(d.reason==('missing_signal' if kind=='gap' else 'side_switch') for d in result.diagnostics)
    for cycle in result.cycles:
        events={e.event_id:e for e in result.events}
        s,e=events[cycle.start_event_id],events[cycle.end_event_id]
        assert not s.frame_index<index<=e.frame_index
    if kind=='gap':
        frame=phase_at_v2(result,index)
        assert frame.motion_phase is None and any(r.reason=='missing_signal' for r in frame.reasons)


def test_clip_ends_in_pause_no_departure_or_fake_full_cycle():
    result=segment_squat_v2(source(dwell=True,end=61))
    assert result.candidate_count==0 and result.events==() and result.cycles==()
    assert any(r.reason=='partial_end' for r in result.diagnostics)
    assert phase_at_v2(result,60).motion_phase is None


@pytest.mark.parametrize('status,reason',[('unavailable','bottom_bounds_unresolved'),('not_run','stage_not_run'),('blocked','temporal_gate_blocked')])
def test_abstention_null_and_reason_required_per_component(status,reason):
    flag=V2Reason(reason=reason)
    assert flag.stage=='segmentation' and flag.source_tier=='S3' and flag.category=='insufficient_data'
    for cls,field in ((EventComponent,'event'),(PauseComponent,'value'),(PhaseComponent,'intervals')):
        value=cls(status=status,reasons=(flag,))
        assert getattr(value,field) in (None,())
        with pytest.raises(ValidationError):cls(status=status)


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_pause_rejects_zero_negative_nonfinite(value):
    with pytest.raises(ValidationError):PauseValue(arrival_event_id='a',departure_event_id='d',duration_seconds=value)


@pytest.mark.parametrize('mode',['dwell_candidate','point_turn_at_resolution','bounds_unresolved'])
def test_json_roundtrip_preserves_modes_reasons_hashes_and_provenance(mode):
    result=segment_squat_v2(source()) if mode=='bounds_unresolved' else contract_example(mode)
    restored=read_temporal_json(strict_json(result))
    assert type(restored) is SquatTemporalResultV2 and restored==result
    assert restored.content_hash()==result.content_hash()
    assert restored.official_eligibility is False and restored.assessment.enabled is False


def test_version_dispatch_and_v1_backwards_compatibility():
    supplied=source();old=segment_squat(supplied);old_json=strict_json(old)
    new=segment_squat_v2(supplied)
    assert strict_json(new.legacy)==old_json
    assert type(read_temporal_json(old_json)) is type(old)
    assert strict_json(read_temporal_json(old_json))==old_json
    assert new.legacy.profile_hash==PROFILE_HASH
    data=json.loads(old_json);data['segmentation_schema_version']='0.2.0'
    with pytest.raises(ValidationError):read_temporal_json(json.dumps(data))
    for version in ('9.0.0',None):
        with pytest.raises(ValueError,match='Unsupported'):read_temporal_json(json.dumps({'segmentation_schema_version':version}))
    with pytest.raises(ValueError):read_temporal_json('{"segmentation_schema_version":"0.2.0","x":NaN}')


@pytest.mark.parametrize('change',['time','hash','duplicate','closure','padding','point_proxy','producer_promotion','provenance'])
def test_corrupt_semantics_rejected(change):
    result=contract_example('point_turn_at_resolution' if change=='point_proxy' else 'dwell_candidate')
    data=result.model_dump(mode='json')
    if change=='time':data['events'][0]['timestamp_seconds']+=.01
    elif change=='hash':data['legacy_hash']='0'*64
    elif change=='duplicate':data['events'].append(data['events'][0])
    elif change=='closure':data['cycles'][0]['motion']['intervals'][0]['closure']='[]'
    elif change=='padding':data['cycles'][0]['bottom']['pause']['value']['duration_seconds']+=1/30
    elif change=='point_proxy':
        for e in data['events']:
            e['evidence']['method']='legacy_pvp_proxy'
    elif change=='producer_promotion':data['producer_version']='squat-v2-conservative-adapter-v1'
    else:
        # Stale nested source mutation is rejected, not silently blessed by hash.
        data['legacy']['input']['samples'][58]['joint_provenance']=['missing']*3
    with pytest.raises(ValidationError):read_temporal_json(json.dumps(data))


def test_producer_rejects_fixture_reference_wrapper_and_bad_models():
    supplied=source()
    for bad in ({'signal':supplied,'references':[]},object(),supplied.model_copy(update={'fps':float('nan')})):
        with pytest.raises(GymError) as caught:segment_squat_v2(bad)
        assert caught.value.flag.category=='input_error' and caught.value.flag.stage=='segmentation'
    result=segment_squat_v2(supplied)
    with pytest.raises(GymError):phase_at_v2(result,-1)


def test_fixture_labels_cannot_control_producer():
    supplied=source();renamed=supplied.model_copy(update={'video_id':'synthetic-dwell-ground-truth-label'})
    a,b=segment_squat_v2(supplied),segment_squat_v2(renamed)
    assert [c.bottom.mode for c in a.cycles]==[c.bottom.mode for c in b.cycles]
    assert a.candidate_count==b.candidate_count


def test_light_import_and_no_geometry_backend_or_fixture_module():
    script="""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'} or fullname.endswith('synthetic_fixtures'):
   raise AssertionError('Forbidden import: '+fullname)
sys.meta_path.insert(0,Block())
from gym_coach.segmentation.temporal_v2_models import SquatTemporalResultV2
from gym_coach.segmentation.temporal_v2 import segment_squat_v2
from gym_coach.reporting.temporal_v2_renderer import render_temporal_v2
"""
    completed=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_producer_renderer_never_open_data_or_mutate_source(monkeypatch):
    supplied=source();before=strict_json(supplied)
    original=builtins.open
    def guarded(file,*args,**kwargs):
        if 'data' in str(file).replace('\\','/').lower().split('/'):
            raise AssertionError('Data read forbidden')
        return original(file,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guarded)
    result=segment_squat_v2(supplied)
    html=render_temporal_v2(result)
    assert 'NOT VALIDATED' in html and 'Unavailable' in html
    assert strict_json(supplied)==before


def test_renderer_null_pause_separate_motion_events_and_safe_json():
    result=segment_squat_v2(source().model_copy(update={'video_id':'synthetic-</script><img src=x>'}))
    html=render_temporal_v2(result)
    assert 'Motion phase' in html and 'Event' in html and 'Pause' in html
    assert '</script><img' not in html and '\\u003c' in html
    assert 'innerHTML' not in html and 'Unavailable' in html
    assert '0 s' not in html


def test_offline_demo_exclusive_artifacts_and_readback(tmp_path):
    target=tmp_path/'new-run'
    manifest=write_demo(target)
    assert manifest['synthetic_only'] is True and manifest['assessment_enabled'] is False
    for entry in manifest['cases']:
        path=target/entry['result_path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['result_sha256']
        assert isinstance(read_temporal_json(raw.decode()),SquatTemporalResultV2)
        assert (target/entry['html_path']).is_file()
    before={p:p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):write_demo(target)
    assert all(p.read_bytes()==raw for p,raw in before.items())
