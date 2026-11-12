"""Synthetic calculations and versioned policy contracts, not real accuracy."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gym_coach import geometry
from gym_coach.axis_geometry import midpoint_axis_angle_vertical
from gym_coach.contracts import Exercise, Viewpoint, JointProvenance
from gym_coach.features import (ANGLE_FEATURE_IDS, DEFAULT_REGISTRY, GEOMETRY_REGISTRY,
    FeatureInputContext, GeometryFeatureObservation, GeometryFeatureRegistry,
    build_feature_input_context, observe_feature, observe_geometry_feature)
from gym_coach.features.models import content_hash
from gym_coach.flags import GymError
from gym_coach.pose.mapping import JointName as J, REQUIRED_JOINTS
from gym_coach.preprocessing import ProcessedJoint, InterpolationSource, preprocess_pose
from gym_coach.quality import QualityGate
from test_preprocessing import make_frames, metadata


def fixture(feature_id, expected=90, *, changes=None, viewpoint=Viewpoint.SIDE, n=7):
    definition = GEOMETRY_REGISTRY.get(feature_id)
    if definition.measurement_method == "joint_angle_abc":
        end = {0:(.4,.2), 45:(.6,.3), 90:(.6,.5), 180:(.4,.7)}[expected]
        coords = ((.4,.3),(.4,.5),end)
    else:
        offset = {0:(0,-.2),45:(.2,-.2),90:(.2,0),180:(0,.2)}[expected]
        coords = ((.3,.5),(.5,.5),(.3+offset[0],.5+offset[1]),(.5+offset[0],.5+offset[1]))
    edits = {(i,j):("observed",*p) for i in range(n) for j,p in zip(definition.required_joints,coords)}
    edits.update(changes or {})
    exercise = definition.exercises[0]
    raw = tuple(f.model_copy(update=dict(exercise=exercise,viewpoint=viewpoint)) for f in make_frames(n=n,changes=edits))
    quality = QualityGate().evaluate(raw, metric_requirements=GEOMETRY_REGISTRY.quality_requirements(exercise),
                                     metric_viewpoints=GEOMETRY_REGISTRY.quality_viewpoints(exercise))
    result = preprocess_pose(raw,quality,metadata(raw,width=320,height=320))
    return FeatureInputContext(frame=result.frames[3],quality=quality),result


@pytest.mark.parametrize("feature_id", ANGLE_FEATURE_IDS)
@pytest.mark.parametrize("expected", [0,45,90,180])
def test_every_feature_known_geometry_and_order(feature_id,expected):
    ctx,_ = fixture(feature_id,expected)
    definition = GEOMETRY_REGISTRY.get(feature_id)
    obs = observe_geometry_feature(feature_id,ctx)
    assert obs.status == "available" and obs.value == pytest.approx(expected, abs=1e-10, rel=0)
    points = tuple((j.x,j.y) for j in obs.joint_inputs)
    primitive = geometry.joint_angle if len(points)==3 else midpoint_axis_angle_vertical
    assert obs.value == primitive(*points)
    assert tuple(j.joint for j in obs.joint_inputs) == definition.required_joints
    assert obs.unit == "degrees" and obs.quantity_kind == "angle"
    assert obs.measurement_tier == definition.measurement_tier and obs.source_tiers == definition.source_tiers
    assert obs.measurement_validation_status == "unvalidated"
    assert obs.assessment_status == "blocked_pending_expert_validation" and obs.temporal_analysis_eligible is False
    assert (obs.video_id,obs.frame_index,obs.timestamp_seconds) == (ctx.frame.video_id,3,ctx.frame.timestamp_seconds)
    assert GeometryFeatureObservation.model_validate_json(obs.model_dump_json()) == obs


@pytest.mark.parametrize("feature_id", ANGLE_FEATURE_IDS)
def test_local_allow_does_not_promote_skipped_sequence(feature_id):
    definition = GEOMETRY_REGISTRY.get(feature_id)
    other = next(j for j in REQUIRED_JOINTS[definition.exercises[0]] if j not in definition.required_joints)
    ctx,result = fixture(feature_id,changes={(i,other):("missing",None,None) for i in range(7)})
    before = result.model_dump_json(), ctx.quality.model_dump_json(), GEOMETRY_REGISTRY.model_dump_json()
    assert ctx.quality.eligibility.action == "skip" and ctx.quality.metric_eligibility[feature_id].action == "allow"
    obs = observe_geometry_feature(feature_id,build_feature_input_context(result,frame_index=3))
    assert obs.status == "available" and obs.availability.input.quality.sequence_status == "skipped"
    assert not obs.temporal_analysis_eligible and obs.assessment_status == "blocked_pending_expert_validation"
    assert before == (result.model_dump_json(),ctx.quality.model_dump_json(),GEOMETRY_REGISTRY.model_dump_json())
    if feature_id in ("squat_knee_angle_left_deg","squat_knee_angle_right_deg"):
        old = observe_feature(feature_id,ctx)
        assert old.value is None and any(f.evidence.get("scope")=="movement_analysis" for f in old.flags)


@pytest.mark.parametrize("feature_id", ANGLE_FEATURE_IDS)
@pytest.mark.parametrize("mode", ["degrade","skip","missing_key","wrong_joints","unknown","front","oblique"])
def test_exact_gate_and_viewpoint_blocks(feature_id,mode,monkeypatch):
    definition = GEOMETRY_REGISTRY.get(feature_id)
    changes = {}
    view = Viewpoint(mode) if mode in ("unknown","front","oblique") else Viewpoint.SIDE
    if mode in ("degrade","skip"):
        indices = [0] if mode=="degrade" else range(7)
        changes = {(i,definition.required_joints[0]):("low_visibility",.4,.4) for i in indices}
    ctx,_ = fixture(feature_id,changes=changes,viewpoint=view)
    if mode in ("missing_key","wrong_joints"):
        entries = dict(ctx.quality.metric_eligibility)
        if mode=="missing_key": del entries[feature_id]
        else: entries[feature_id] = entries[feature_id].model_copy(update={"required_joints":(J.NOSE,)})
        ctx = ctx.model_copy(update={"quality":ctx.quality.model_copy(update={"metric_eligibility":entries})})
    monkeypatch.setattr(geometry,"joint_angle",lambda *a: pytest.fail("Blocked calculation executed"))
    monkeypatch.setattr("gym_coach.axis_geometry.midpoint_axis_angle_vertical",lambda *a: pytest.fail("Blocked axis executed"))
    obs = observe_geometry_feature(feature_id,ctx)
    assert obs.status == "unavailable" and obs.value is None and obs.flags and obs.reasons
    assert obs.computation_status == "not_computed"
    if mode in ("degrade","skip"): assert ctx.quality.metric_eligibility[feature_id].action == mode
    assert GeometryFeatureObservation.model_validate_json(obs.model_dump_json()) == obs


@pytest.mark.parametrize("feature_id", ANGLE_FEATURE_IDS)
def test_degenerate_geometry_has_exact_structured_reason(feature_id):
    ctx,_ = fixture(feature_id)
    definition = GEOMETRY_REGISTRY.get(feature_id)
    points = dict((j.joint,j) for j in ctx.frame.joints)
    if len(definition.required_joints)==3:
        a,b,_ = definition.required_joints
        points[a] = points[a].model_copy(update={"x":points[b].x,"y":points[b].y})
    else:
        lh,rh,ls,rs = definition.required_joints
        points[ls] = points[ls].model_copy(update={"x":points[lh].x,"y":points[lh].y})
        points[rs] = points[rs].model_copy(update={"x":points[rh].x,"y":points[rh].y})
    ctx = ctx.model_copy(update={"frame":ctx.frame.model_copy(update={"joints":tuple(points.values())})})
    obs = observe_geometry_feature(feature_id,ctx)
    assert obs.value is None and obs.computation_status == "failed_measurement"
    assert obs.reasons == ("degenerate_geometry",)
    flag = obs.flags[0]
    assert flag.category == "measurement_error" and flag.stage == "features" and flag.source_tier == "S3"
    assert flag.location.feature == feature_id and flag.location.frame_start == 3
    assert all(flag.evidence[f"input_joint_{i}"]==j.value for i,j in enumerate(definition.required_joints))
    assert GeometryFeatureObservation.model_validate_json(obs.model_dump_json()) == obs
    data = obs.model_dump(mode="json"); data["flags"][0]["evidence"]["value"] = 0
    with pytest.raises(ValidationError): GeometryFeatureObservation.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("status", ["missing","low_visibility"])
@pytest.mark.parametrize("feature_id", ANGLE_FEATURE_IDS)
def test_null_unusable_provenance_blocks_even_synthetic_metric_allow(status,feature_id):
    ctx,_ = fixture(feature_id)
    target=GEOMETRY_REGISTRY.get(feature_id).required_joints[1]
    replacement = ProcessedJoint(joint=target,source_status=status,status=status,usable=False,
        unavailable_reason="missing_joint" if status=="missing" else "low_pose_visibility",smoothing_status="unavailable")
    joints = tuple(replacement if j.joint==target else j for j in ctx.frame.joints)
    ctx = ctx.model_copy(update={"frame":ctx.frame.model_copy(update={"joints":joints})})
    obs = observe_geometry_feature(feature_id,ctx)
    assert obs.value is None and "feature_provenance_rejected" in obs.reasons and "feature_joint_unusable" in obs.reasons


def test_imputation_references_retained_without_raw_gate_promotion():
    key = "squat_knee_angle_left_deg"
    ctx,_ = fixture(key)
    refs = InterpolationSource(left_frame_index=2,right_frame_index=4,left_timestamp_seconds=2/30,right_timestamp_seconds=4/30)
    joints = tuple(j.model_copy(update={"status":JointProvenance.IMPUTED,"source_status":"missing","interpolation":refs})
                   if j.joint==J.LEFT_KNEE else j for j in ctx.frame.joints)
    # Detached synthetic input contract, deliberately not a genuine raw-gap gate.
    synthetic = ctx.model_copy(update={"frame":ctx.frame.model_copy(update={"joints":joints})})
    obs = observe_geometry_feature(key,synthetic)
    assert obs.joint_inputs[1].status == "imputed" and obs.joint_inputs[1].interpolation == refs
    badrefs = refs.model_copy(update={"left_frame_index":3})
    badjoints = tuple(j.model_copy(update={"interpolation":badrefs}) if j.joint==J.LEFT_KNEE else j for j in joints)
    with pytest.raises(GymError): observe_geometry_feature(key,synthetic.model_copy(update={"frame":synthetic.frame.model_copy(update={"joints":badjoints})}))
    real,_ = fixture(key,changes={(3,J.LEFT_KNEE):("missing",None,None)})
    real_obs = observe_geometry_feature(key,real)
    assert real_obs.value is None and real.quality.metric_eligibility[key].action == "degrade"


@pytest.mark.parametrize("bad", [None,True,float('nan'),float('inf'),-float('inf')])
def test_unsafe_nonfinite_or_null_coordinates_are_structured_input_error(bad):
    ctx,_ = fixture("squat_knee_angle_left_deg")
    joints = tuple(j.model_copy(update={"x":bad}) if j.joint==J.LEFT_KNEE else j for j in ctx.frame.joints)
    with pytest.raises(GymError) as caught:
        observe_geometry_feature("squat_knee_angle_left_deg",ctx.model_copy(update={"frame":ctx.frame.model_copy(update={"joints":joints})}))
    assert caught.value.flag.category == "input_error" and caught.value.flag.code == "invalid_feature_input"


def test_registry_v1_stays_identical_and_v2_rejects_forgery():
    assert DEFAULT_REGISTRY.content_hash() == "bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a"
    assert len(ANGLE_FEATURE_IDS)==10 and len(set(ANGLE_FEATURE_IDS))==10
    assert GeometryFeatureRegistry.model_validate_json(GEOMETRY_REGISTRY.model_dump_json()) == GEOMETRY_REGISTRY
    data = GEOMETRY_REGISTRY.model_dump(mode="json"); data["features"][0]["formula"] = "fake"
    with pytest.raises(ValidationError): GeometryFeatureRegistry.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("field", ["video_id","frame_index","timestamp_seconds","quality_report_hash","quality_policy_hash","feature_id","primitive_version","unit"])
def test_serialized_audit_tampering_rejected(field):
    ctx,_ = fixture("squat_knee_angle_left_deg")
    data = observe_geometry_feature("squat_knee_angle_left_deg",ctx).model_dump(mode="json")
    data[field] = {"frame_index":4,"timestamp_seconds":.2,"quality_report_hash":"0"*64,"quality_policy_hash":"0"*64,
        "feature_id":"squat_hip_angle_left_deg","primitive_version":None,"unit":"torso_lengths"}.get(field,"other.avi")
    with pytest.raises((ValidationError,GymError)): GeometryFeatureObservation.model_validate_json(json.dumps(data))


def test_unknown_wrong_exercise_and_canonical_definition_handling():
    ctx,_ = fixture("squat_knee_angle_left_deg")
    with pytest.raises(GymError) as caught: observe_geometry_feature("unknown",ctx)
    assert caught.value.flag.code == "unknown_feature_id"
    assert observe_geometry_feature("hspu_elbow_angle_left_deg",ctx).value is None
    definition = GEOMETRY_REGISTRY.get("squat_knee_angle_left_deg")
    assert observe_geometry_feature(definition,ctx).value == 90
    with pytest.raises(GymError): observe_geometry_feature(definition.model_copy(update={"formula":"fake"}),ctx)
    for field in ("exercise","viewpoint"):
        bad = ctx.model_copy(update={"quality":ctx.quality.model_copy(update={field:"garbage"})})
        with pytest.raises(GymError): observe_geometry_feature(definition,bad)


def test_ratios_remain_blocked_with_null_fallback_scale():
    ctx,_ = fixture("squat_knee_angle_left_deg")
    fallback = ctx.frame.normalization.model_copy(update={"scale_status":"fallback_unit","torso_scale":None,"distance_ratio_eligible":False})
    ctx = ctx.model_copy(update={"frame":ctx.frame.model_copy(update={"normalization":fallback})})
    for definition in GEOMETRY_REGISTRY.features:
        if definition.quantity_kind == "ratio":
            obs = observe_geometry_feature(definition,ctx)
            assert obs.value is None and obs.primitive_name is None
            assert "feature_definition_blocked" in obs.reasons and "normalization_scale_fallback" in obs.reasons
            assert obs.measurement_validation_status == "blocked"


def test_legacy_knee_schema_roundtrips_are_unchanged():
    for key,version in (("squat_knee_angle_left_deg","0.2.0"),("squat_knee_angle_right_deg","0.3.0")):
        ctx,_ = fixture(key)
        old = observe_feature(key,ctx)
        assert old.numeric_feature_schema_version==version
        assert type(old).model_validate_json(old.model_dump_json())==old
        new = observe_geometry_feature(key,ctx)
        assert new.numeric_feature_schema_version=="0.4.0" and new.value==old.value
        with pytest.raises(ValidationError): GeometryFeatureObservation.model_validate_json(old.model_dump_json())


def test_public_feature_import_has_no_heavy_backends(tmp_path):
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from gym_coach.features import observe_geometry_feature,GEOMETRY_REGISTRY,GeometryFeatureObservation
assert len(GEOMETRY_REGISTRY.features)==12
assert not {'numpy','scipy','cv2','mediapipe','torch'} & sys.modules.keys()
"""
    proc = subprocess.run([sys.executable,"-I","-c",code],cwd=tmp_path,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr


@pytest.mark.parametrize('feature_id',[f.feature_id for f in GEOMETRY_REGISTRY.features if f.measurement_method=='joint_angle_abc'])
@pytest.mark.parametrize('case',['c_equals_b','a_equals_c'])
def test_other_joint_coincidence_cases(feature_id,case):
    ctx,_=fixture(feature_id)
    a,b,c=GEOMETRY_REGISTRY.get(feature_id).required_joints
    points={j.joint:j for j in ctx.frame.joints}
    ref=points[b if case=='c_equals_b' else a]
    points[c]=points[c].model_copy(update={'x':ref.x,'y':ref.y})
    ctx=ctx.model_copy(update={'frame':ctx.frame.model_copy(update={'joints':tuple(points.values())})})
    obs=observe_geometry_feature(feature_id,ctx)
    if case=='c_equals_b': assert obs.value is None and obs.flags[0].evidence['vector']=='BC'
    else: assert obs.value==0 and obs.status=='available'


@pytest.mark.parametrize('feature_id',ANGLE_FEATURE_IDS)
def test_feature_translation_and_positive_scale_invariance(feature_id):
    ctx,_=fixture(feature_id,45)
    before=observe_geometry_feature(feature_id,ctx)
    joints=tuple(j.model_copy(update={'x':j.x*3.0+15.0,'y':j.y*3.0-8.0}) for j in ctx.frame.joints)
    shifted=ctx.model_copy(update={'frame':ctx.frame.model_copy(update={'joints':joints})})
    after=observe_geometry_feature(feature_id,shifted)
    assert after.value==pytest.approx(before.value,abs=1e-10,rel=0)


@pytest.mark.parametrize('field',['registry_hash','definition_hash','input_hash'])
def test_availability_hash_tampering_is_rejected(field):
    ctx,_=fixture('squat_knee_angle_left_deg')
    data=observe_geometry_feature('squat_knee_angle_left_deg',ctx).model_dump(mode='json')
    data['availability'][field]='0'*64
    with pytest.raises(ValidationError): GeometryFeatureObservation.model_validate_json(json.dumps(data))


def test_adapter_does_not_recompute_gate_or_swallow_programming_exception(monkeypatch):
    ctx,_=fixture('squat_hip_angle_left_deg')
    monkeypatch.setattr(QualityGate,'evaluate',lambda *a,**k:pytest.fail('Adapter recomputed gate'))
    assert observe_geometry_feature('squat_hip_angle_left_deg',ctx).status=='available'
    def fail(*a): raise RuntimeError('programming failure')
    monkeypatch.setattr(geometry,'joint_angle',fail)
    with pytest.raises(RuntimeError,match='programming failure'): observe_geometry_feature('squat_hip_angle_left_deg',ctx)


def test_linked_source_hash_mismatch_is_not_accepted():
    ctx,result=fixture('squat_hip_angle_left_deg')
    linked=build_feature_input_context(result,frame_index=3)
    with pytest.raises(GymError) as caught:
        observe_geometry_feature('squat_hip_angle_left_deg',linked.model_copy(update={'raw_sequence_hash':'0'*64}))
    assert caught.value.flag.code=='feature_context_mismatch'
