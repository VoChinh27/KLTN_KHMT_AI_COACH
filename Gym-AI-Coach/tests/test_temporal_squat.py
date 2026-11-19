"""Synthetic temporal tests first; these are not real repetition accuracy."""
import hashlib
import json
import math
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.contracts import JointProvenance
from gym_coach.flags import FlagCode, GymError
from gym_coach.segmentation import (
    SquatSignalSample, SquatTemporalInput, SquatSegmentationProfile,
    SquatSegmentationResult, segment_squat,
)


def signal(fps=30.0, *, duration=6.0, gap=(), switch=None, values=None):
    n=int(duration*fps)+1 if values is None else len(values)
    samples=[]
    for i in range(n):
        value=140+30*math.cos(math.pi*(i/fps-1)) if values is None else values[i]
        if i in gap:value=None
        samples.append(SquatSignalSample(frame_index=i,timestamp_seconds=i/fps,value=value,
            side_source='right' if switch is not None and i>=switch else 'left',
            joint_provenance=(JointProvenance.MISSING,)*3 if value is None else (JointProvenance.OBSERVED,)*3,
            reasons=(FlagCode.MISSING_JOINT,) if value is None else ()))
    return SquatTemporalInput(video_id='synthetic-temporal-test',fps=fps,samples=tuple(samples),
        source_hash=hashlib.sha256(json.dumps(values or [n,fps]).encode()).hexdigest())


@pytest.mark.parametrize('fps',[25.0,30.0,60.0])
def test_known_synthetic_pvp_and_fps_aware_boundaries(fps):
    source=signal(fps);before=source.model_dump_json()
    result=segment_squat(source)
    assert result.status=='experimental_candidates' and result.candidate_count==2
    for rep,times in zip(result.repetitions,[(1.,2.,3.),(3.,4.,5.)]):
        assert (rep.start_seconds,rep.bottom_seconds,rep.end_seconds)==pytest.approx(times,abs=1/fps,rel=0)
        assert rep.descent_seconds==pytest.approx(1,abs=1/fps,rel=0)
        assert rep.ascent_seconds==pytest.approx(1,abs=1/fps,rel=0)
        assert rep.start_frame<rep.bottom_frame<rep.end_frame
    assert result.experimental_label=='EXPERIMENTAL / NOT VALIDATED'
    assert not result.official_eligibility and not result.assessment_enabled
    assert result.profile_hash==result.profile.content_hash() and result.input_hash==source.content_hash()
    assert source.model_dump_json()==before
    assert SquatSegmentationResult.model_validate_json(result.model_dump_json())==result


@pytest.mark.parametrize('gap',[range(58,63),range(30,151),[90]])
def test_no_rep_or_smoothing_bridges_any_unresolved_gap(gap):
    source=signal(gap=gap);result=segment_squat(source)
    for rep in result.repetitions:assert not set(range(rep.start_frame,rep.end_frame+1))&set(gap)
    assert all(result.smoothed_values[i] is None for i in gap)
    assert any(f.code==FlagCode.MISSING_JOINT for f in result.flags)


@pytest.mark.parametrize('switch',[30,60,90,120,150])
def test_side_switch_splits_run_and_is_explicit(switch):
    result=segment_squat(signal(switch=switch))
    assert any(f.code==FlagCode.SIDE_SWITCH_DISCONTINUITY for f in result.flags)
    assert all(not (rep.start_frame<switch<=rep.end_frame) for rep in result.repetitions)


@pytest.mark.parametrize('values',[[],[140.0]*5,[140.0]*181,[140.0+.2*math.sin(i) for i in range(181)]])
def test_empty_short_flat_and_small_jitter_do_not_invent_reps(values):
    result=segment_squat(signal(values=values))
    assert result.candidate_count==0 and result.status=='unavailable' and result.flags
    assert not result.official_eligibility


def test_clip_edge_cycles_are_partial_not_complete():
    source=signal(duration=2)
    result=segment_squat(source)
    assert result.candidate_count==0
    assert any(f.code in (FlagCode.PARTIAL_REP_START,FlagCode.PARTIAL_REP_END) for f in result.flags)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),True,-1.0,181.0])
def test_sample_rejects_invalid_numeric_values(bad):
    with pytest.raises(ValidationError):SquatSignalSample(frame_index=0,timestamp_seconds=0.,value=bad,side_source='left')


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf'),True])
def test_input_rejects_invalid_fps(bad):
    with pytest.raises(ValidationError):SquatTemporalInput(video_id='synthetic-test',fps=bad,samples=(),source_hash='a'*64)


def test_invalid_timeline_and_framewise_observation_cannot_authorize_temporal():
    source=signal()
    damaged=source.model_copy(update={'samples':(source.samples[1],*source.samples[1:])})
    with pytest.raises(GymError) as caught:segment_squat(damaged)
    assert caught.value.flag.stage=='segmentation' and caught.value.flag.category=='input_error'
    with pytest.raises(GymError):segment_squat(object())
    with pytest.raises(ValidationError):SquatTemporalInput.model_validate({**source.model_dump(),'video_id':'real.avi'})


def test_profile_and_audit_tampering_rejected():
    result=segment_squat(signal())
    for field in ('input_hash','profile_hash'):
        data=result.model_dump(mode='json');data[field]='0'*64
        with pytest.raises(ValidationError):SquatSegmentationResult.model_validate_json(json.dumps(data))
    with pytest.raises(ValidationError):SquatSegmentationProfile(min_excursion_degrees=0.)


def test_all_missing_and_high_frequency_reversals_are_unavailable():
    missing=segment_squat(signal(gap=range(181)))
    assert missing.candidate_count==0 and missing.smoothed_values==(None,)*181
    fast=signal(values=[120. if i%2 else 160. for i in range(181)])
    result=segment_squat(fast,profile=SquatSegmentationProfile(smoothing_span_seconds=0.))
    assert result.candidate_count==0 and any(f.code==FlagCode.UNSTABLE_SIGNAL for f in result.flags)


def test_gap_endpoint_and_smoothed_signal_cannot_be_forged():
    result=segment_squat(signal(gap=[60]))
    data=result.model_dump(mode='json');data['smoothed_values'][60]=140.
    with pytest.raises(ValidationError):SquatSegmentationResult.model_validate_json(json.dumps(data),strict=True)
    result=segment_squat(signal())
    data=result.model_dump(mode='json');rep=data['repetitions'][0]
    data['smoothed_values'][rep['bottom_frame']]=180.
    with pytest.raises(ValidationError):SquatSegmentationResult.model_validate_json(json.dumps(data),strict=True)


def test_plateau_extrema_have_stable_midpoint_and_no_duplicate_reps():
    # Explicit top/bottom plateaus, including partial leading/trailing cycles.
    values=[120.,140.,160.,160.,160.,140.,120.,100.,100.,100.,120.,140.,160.,160.,160.,140.,120.]
    result=segment_squat(signal(fps=10.,values=values),profile=SquatSegmentationProfile(smoothing_span_seconds=0.))
    assert [(r.start_frame,r.bottom_frame,r.end_frame) for r in result.repetitions]==[(3,8,13)]


def test_public_temporal_import_and_execution_are_light(tmp_path):
    code="""
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.segmentation import SquatTemporalInput,segment_squat
s=SquatTemporalInput(video_id='synthetic-empty',fps=30.,source_hash='a'*64,samples=())
assert segment_squat(s).candidate_count==0
"""
    p=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
