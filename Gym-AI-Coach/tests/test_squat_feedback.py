"""Synthetic integration, gate separation, safe presentation and readback."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach.contracts import ErrorCategory, Stage
from gym_coach.features import GEOMETRY_REGISTRY
from gym_coach.flags import FlagCode, GymError
from gym_coach.reporting.demo import synthetic_preprocessing, build_blocked_video_feedback, main
from gym_coach.reporting.renderer import render_feedback
from gym_coach.reporting.squat_builder import build_squat_feedback, synthetic_temporal_gate
from gym_coach.reporting.squat_models import SquatFeedbackResult
from gym_coach.segmentation import SquatSegmentationProfile, segment_squat


@pytest.fixture(scope='module')
def source():
    return synthetic_preprocessing(fps=10.)


@pytest.fixture(scope='module')
def feedback(source):
    return build_squat_feedback(source)


def validate(data):
    return SquatFeedbackResult.model_validate_json(json.dumps(data, allow_nan=False), strict=True)


def test_five_real_registry_calculations_and_synthetic_temporal_capability(source, feedback):
    assert feedback.segmentation.candidate_count == 2
    assert feedback.assessment.enabled is False and feedback.official_eligibility is False
    assert feedback.form_score is None and feedback.technical_feedback is None
    assert {f.phase for f in feedback.frames} == {'Unavailable','Start/Top','Descent','Bottom','Ascent','Return/Top'}
    assert feedback.segmentation.input.evidence_basis == 'genuine_synthetic_qualitygate'
    assert feedback.source.preprocessing_hash == feedback.segmentation.input.source_hash
    assert all(m.status == 'available' and not m.temporal_analysis_eligible
               for frame in feedback.frames for m in frame.metrics)
    # The public numeric observation's temporal false is neither mutated nor accepted.
    from gym_coach.features import observe_geometry_feature, FeatureInputContext
    obs = observe_geometry_feature('squat_knee_angle_left_deg', FeatureInputContext(frame=source.frames[20],quality=source.quality))
    with pytest.raises(GymError):
        segment_squat(obs)
    assert feedback.frames[20].metrics[2].observation_hash == obs.content_hash()
    before = source.model_dump_json()
    assert build_squat_feedback(source).content_hash() == feedback.content_hash()
    assert source.model_dump_json() == before


def test_roundtrip_preserves_metadata_provenance_phase_flags_and_hash(feedback):
    restored = SquatFeedbackResult.model_validate_json(feedback.model_dump_json(), strict=True)
    assert restored == feedback and restored.content_hash() == feedback.content_hash()
    assert restored.segmentation.flags
    assert all(m.joint_inputs for m in restored.frames[0].metrics)


@pytest.mark.parametrize('kind', ['registry_hash','source_hash','segmentation_hash','phase','phase_time','value',
                                  'assessment','score','temporal_promotion','timeline','quality_promotion',
                                  'missing_metric','unit','unsupported_exercise','real_authority'])
def test_tampered_contract_rejected(feedback, kind):
    data = json.loads(feedback.model_dump_json())
    if kind.endswith('_hash'): data[kind] = '0'*64
    elif kind == 'phase': data['frames'][0]['phase']='Descent'
    elif kind == 'phase_time': data['frames'][11]['phase_evidence']['end_frame']=999
    elif kind == 'value': data['frames'][0]['metrics'][0]['value']=999.
    elif kind == 'assessment': data['assessment']['enabled']=True
    elif kind == 'score': data['form_score']=100.
    elif kind == 'temporal_promotion': data['frames'][0]['metrics'][0]['temporal_analysis_eligible']=True
    elif kind == 'timeline': data['frames'][1]['timestamp_seconds'] += .05
    elif kind == 'quality_promotion': data['frames'][0]['metrics'][0]['quality_action']='skip'
    elif kind == 'missing_metric': data['frames'][0]['metrics'].pop()
    elif kind == 'unit': data['frames'][0]['metrics'][0]['unit']='meters'
    elif kind == 'unsupported_exercise': data['exercise']='HandstandPushups'
    elif kind == 'real_authority': data['official_eligibility']=True
    with pytest.raises(ValidationError): validate(data)


@pytest.mark.parametrize('side', ['left','right'])
def test_fixed_side_retained(source, side):
    result = build_squat_feedback(source, side=side)
    assert all(s.side_source==side for s in result.segmentation.input.samples)
    assert all(rep.side_source==side for rep in result.segmentation.repetitions)


@pytest.mark.parametrize('fault', ['global','missing_left','missing_right','wrong_joints','metric_degrade','viewpoint'])
def test_separate_gate_requires_sequence_and_exact_bilateral_knees(source, fault):
    q=source.quality
    metrics=dict(q.metric_eligibility)
    if fault=='global': q=q.model_copy(update={'eligibility':q.eligibility.model_copy(update={'eligible':False,'action':'skip'})})
    elif fault=='viewpoint':
        from gym_coach.contracts import Viewpoint
        q=q.model_copy(update={'viewpoint':Viewpoint.FRONT})
    elif fault.startswith('missing_'): metrics.pop('squat_knee_angle_'+fault.removeprefix('missing_')+'_deg')
    elif fault=='wrong_joints': metrics['squat_knee_angle_left_deg']=metrics['squat_knee_angle_right_deg']
    else: metrics['squat_knee_angle_right_deg']=metrics['squat_knee_angle_right_deg'].model_copy(update={'eligible':False,'action':'degrade'})
    q=q.model_copy(update={'metric_eligibility':metrics})
    assert synthetic_temporal_gate(source.model_copy(update={'quality':q}))


def test_fresh_quality_gap_is_blocked_even_after_explicit_imputation():
    source=synthetic_preprocessing(fps=10.,missing_frames=(20,))
    before=source.model_dump_json()
    result=build_squat_feedback(source)
    assert source.quality.eligibility.action=='degrade'
    assert result.segmentation.status=='blocked' and not result.segmentation.repetitions
    assert result.segmentation.input is None
    assert all(f.phase=='Unavailable' for f in result.frames)
    assert all(m.value is None and m.flags and m.reasons for f in result.frames for m in f.metrics if m.status=='unavailable')
    assert any(j.status=='imputed' for j in source.frames[20].joints)
    assert source.model_dump_json()==before


def test_builder_rejects_unsafe_source_and_profile(source):
    for bad in (None, {}, source.model_copy(update={'metadata':source.metadata.model_copy(update={'video_id':'real.avi'})})):
        with pytest.raises(GymError) as caught: build_squat_feedback(bad)
        assert caught.value.flag.category==ErrorCategory.INPUT_ERROR
        assert caught.value.flag.stage==Stage.REPORTING
    with pytest.raises(GymError): build_squat_feedback(source,profile=SquatSegmentationProfile.model_construct(min_excursion_degrees=float('nan')))


def test_renderer_safe_embedding_and_no_external_dependencies(feedback):
    dangerous=feedback.messages[0].model_copy(update={'text':'</script><img src=x onerror=alert(1)>'})
    result=feedback.model_copy(update={'messages':(dangerous,)})
    html=render_feedback(result)
    assert '</script><img' not in html and '\\u003c/script' in html
    assert 'textContent' in html and 'innerHTML' not in html
    assert '<script src=' not in html and 'https://' not in html
    assert 'EXPERIMENTAL / NOT VALIDATED' in html
    for definition in GEOMETRY_REGISTRY.for_exercise('BodyWeightSquats'):
        assert definition.feature_id+":" in html


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),True])
def test_render_refuses_unsafe_nonfinite_or_boolean_model_copy(feedback,bad):
    metric=feedback.frames[0].metrics[0].model_copy(update={'value':bad})
    frame=feedback.frames[0].model_copy(update={'metrics':(metric,*feedback.frames[0].metrics[1:])})
    with pytest.raises(GymError) as caught: render_feedback(feedback.model_copy(update={'frames':(frame,*feedback.frames[1:])}))
    assert caught.value.flag.code==FlagCode.FEEDBACK_SERIALIZATION_ERROR


@pytest.mark.parametrize('path', ['Data/UCF-101/BodyWeightSquats/test.avi','Data/UCF-101_GYM/HandstandPushups/test.avi','../outside.avi'])
def test_real_preview_refuses_nonpermitted_sources_before_decode(tmp_path,path):
    with pytest.raises(GymError): build_blocked_video_feedback(Path(path),Path.cwd(),tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_cli_refuses_existing_output_without_overwrite(tmp_path):
    sentinel=tmp_path/'sentinel';sentinel.write_text('keep')
    assert main(['--output',str(tmp_path)])==1
    assert sentinel.read_text()=='keep' and list(tmp_path.iterdir())==[sentinel]


def test_persisted_numeric_observations_match_feedback_hashes(tmp_path,source,feedback):
    from gym_coach.reporting.demo import write_case
    from gym_coach.features import GeometryFeatureObservation
    write_case(tmp_path,feedback,source)
    observations=[GeometryFeatureObservation.model_validate_json(line,strict=True)
                  for line in (tmp_path/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    projected=[m for frame in feedback.frames for m in frame.metrics]
    assert len(observations)==len(projected)==len(source.frames)*5
    for obs,metric in zip(observations,projected):
        assert obs.content_hash()==metric.observation_hash
        assert obs.joint_inputs==metric.joint_inputs and obs.reasons==metric.reasons and obs.flags==metric.flags
        assert not obs.temporal_analysis_eligible
    assert SquatFeedbackResult.model_validate_json((tmp_path/'feedback.json').read_text(encoding='utf-8'),strict=True)==feedback
    before=(tmp_path/'feedback.json').read_bytes()
    with pytest.raises(FileExistsError):write_case(tmp_path,feedback,source)
    assert (tmp_path/'feedback.json').read_bytes()==before


def test_public_contract_renderer_and_cli_help_are_light():
    code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.reporting.squat_models import SquatFeedbackResult
from gym_coach.reporting.renderer import render_feedback
from gym_coach.reporting.demo import main
from gym_coach.segmentation import segment_squat
main(['--help'])
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.integration
def test_permitted_real_preview_no_pipeline_inference(tmp_path,monkeypatch):
    import gym_coach.features as features
    def forbidden(*a,**k): raise AssertionError('Numeric adapters must not run for blocked real preview')
    monkeypatch.setattr(features,'observe_geometry_feature',forbidden)
    path=Path('Data/UCF-101_GYM/BodyWeightSquats/v_BodyWeightSquats_g25_c01.avi')
    from gym_coach.reporting.demo import byte_hash
    before=byte_hash(path)
    result=build_blocked_video_feedback(path,Path.cwd(),tmp_path/'new')
    assert result.status=='blocked' and result.source.video_byte_sha256==before==byte_hash(path)
    assert [f.frame_index for f in result.frames]==[0,51,102]
    assert result.source.documented is False and result.quality is None
    assert all(m.value is None and m.quality_action=='not_run' for f in result.frames for m in f.metrics)
    assert all(f.overlay_joints==() and f.phase=='Unavailable' for f in result.frames)
    assert SquatFeedbackResult.model_validate_json(result.model_dump_json(),strict=True)==result
    assert 'UNKNOWN' in (tmp_path/'new/dataset_row.json').read_text()
