"""Unified orchestration boundaries; synthetic backend/reader, genuine gates."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gym_coach.contracts import Viewpoint
from gym_coach.features import build_feature_input_context, FeatureInputContext, observe_geometry_feature
from gym_coach.pose.mapping import JointName as J
from gym_coach.reporting import single_video as pipe
from gym_coach.reporting import real_descriptive_render as render
from gym_coach.video.models import VideoFrame
from stage5_6_fixture import fixture_input


@pytest.fixture
def environment(tmp_path,monkeypatch):
    raw,metadata,_=fixture_input(n=9,viewpoint=Viewpoint.UNKNOWN)
    video=tmp_path/metadata.video_id;video.write_bytes(b"synthetic source bytes, not actual video")
    metadata=metadata.model_copy(update={"source_path":video})
    counter=dict(extractors=0,frames=0)
    class Reader:
        def __init__(self,path):self.metadata=metadata
        def frames(self):
            for i in range(9):yield VideoFrame(i,i/metadata.fps,"synthetic image")
    class Extractor:
        def __init__(self,*a,**kw):counter["extractors"]+=1
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def extract_frames(self,frames):
            for frame in frames:
                counter["frames"]+=1;yield raw[frame.frame_index]
    runtime=dict(package_version="fixture-only",backend=raw[0].backend.model_dump(mode="json"),
                 installed_versions={},implementation_files={"synthetic":"a"*64})
    monkeypatch.setattr(pipe,"OpenCVVideoReader",Reader)
    monkeypatch.setattr(pipe,"MediaPipePoseExtractor",Extractor)
    monkeypatch.setattr(pipe,"runtime_identity",lambda:runtime)
    def fake_render(output):
        result=pipe.load_saved(output)
        (output/"overlay-fixture.mp4").write_bytes(b"not video; orchestration fixture only")
        (output/"START_HERE.html").write_text(render.single_video_html(result,"overlay-fixture.mp4"),encoding="utf-8")
        pipe.write_json(output/"video-verification.json",dict(status="FIXTURE_ONLY_NOT_DECODED"))
    monkeypatch.setattr(render,"render_single_video_bundle",fake_render)
    return dict(video=video,raw=raw,metadata=metadata,counter=counter,runtime=runtime,root=tmp_path)


def run_case(env,name,**options):
    defaults=dict(side="left",mode="strict",cache_dir=env["root"]/"cache")
    defaults.update(options)
    result=pipe.run(env["video"],env["root"]/name,**defaults)
    return result,json.loads((env["root"]/name/"result.json").read_bytes())


def test_strict_default_and_exploratory_explicit_are_separate(environment):
    env=environment;strict,s=run_case(env,"strict")
    exploratory,e=run_case(env,"exploratory",mode="exploratory",assume_side=True)
    assert strict["inference_run"] and strict["cache"]["status"]=="miss"
    assert not exploratory["inference_run"] and exploratory["cache"]["status"]=="hit"
    assert env["counter"]==dict(extractors=1,frames=9)
    assert s["coverage"]["available_frames"]==0 and s["assumption"] is None
    assert all(r["observation"]["value"] is None and r["observation"]["reasons"] for r in s["frames"])
    assert e["coverage"]["available_frames"]==9
    assert e["assumption"]["status"]=="caller_assumed_unverified" and not e["documented"]
    for name in ("strict","exploratory"):
        frames=[json.loads(line) for line in (env["root"]/name/"raw-pose.jsonl").read_bytes().splitlines()]
        assert all(f["viewpoint"]=="unknown" for f in frames)
    assert s["camera_view"]==e["camera_view"]=="UNKNOWN"
    assert s["reference_status"]==e["reference_status"]=="REFERENCE_PENDING"
    assert not e["official_eligibility"] and not e["assessment"]
    assert all(r["observation"]["temporal_analysis_eligible"] is False for r in e["frames"])


def test_same_mode_fresh_cache_results_exact(environment):
    a,first=run_case(environment,"fresh",mode="exploratory",assume_side=True)
    b,second=run_case(environment,"cached",mode="exploratory",assume_side=True)
    assert first==second and first["result_hash"]==second["result_hash"]
    assert a["cache"]["status"]=="miss" and b["cache"]["status"]=="hit"
    assert environment["counter"]["extractors"]==1


@pytest.mark.parametrize("side",["left","right"])
def test_session_matches_legacy_without_numerical_baseline(environment,side):
    _,result=run_case(environment,side,side=side,mode="exploratory",assume_side=True)
    processed=pipe.PreprocessingResult.model_validate_json((environment["root"]/side/"preprocessing.json").read_bytes(),strict=True)
    for frame,row in zip(processed.frames,result["frames"]):
        linked=build_feature_input_context(processed,frame_index=frame.frame_index)
        expected=observe_geometry_feature(result["feature_id"],FeatureInputContext(frame=linked.frame,quality=linked.quality))
        assert expected.model_dump(mode="json")==row["observation"]
        assert linked.measurement_input_hash==row["source_linkage"]["measurement_input_hash"]
        assert row["side"]==side


@pytest.mark.parametrize("gate",["degrade","skip"])
def test_metric_gates_never_reopened(environment,monkeypatch,gate):
    from gym_coach import geometry
    raw=list(environment["raw"])
    indices=[4] if gate=="degrade" else list(range(9))
    for i in indices:
        joints=tuple(j.model_copy(update={"visibility":.1,"status":"low_visibility"}) if j.joint==J.LEFT_KNEE else j for j in raw[i].joints)
        raw[i]=raw[i].model_copy(update={"joints":joints})
    # Keep strict intake and real gate rules, change only synthetic source data.
    class Extractor:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def extract_frames(self,frames):return iter(raw)
    monkeypatch.setattr(pipe,"MediaPipePoseExtractor",Extractor)
    monkeypatch.setattr(geometry,"joint_angle",lambda *a:(_ for _ in ()).throw(AssertionError("Blocked metric reached geometry")))
    receipt,result=run_case(environment,gate,mode="exploratory",assume_side=True)
    assert result["metric"]["action"]==gate
    assert result["coverage"]["unavailable_frames"]==9
    assert all(r["observation"]["value"] is None and r["observation"]["reasons"] for r in result["frames"])


@pytest.mark.parametrize("mode,assume",[("exploratory",False),("strict",True)])
def test_no_implicit_assumption_or_mode_switch(environment,mode,assume):
    with pytest.raises(pipe.PipelineError,match="assume-side"):
        run_case(environment,"bad",mode=mode,assume_side=assume)
    assert not (environment["root"]/"bad").exists()


def test_disabled_cache_and_existing_output_are_explicit(environment):
    receipt,_=run_case(environment,"once",cache_dir=None)
    assert receipt["cache"]["status"]=="disabled" and receipt["inference_run"]
    before=(environment["root"]/"once/result.json").read_bytes()
    with pytest.raises(FileExistsError):run_case(environment,"once")
    assert (environment["root"]/"once/result.json").read_bytes()==before


@pytest.mark.parametrize("payload",["raw-pose.jsonl","metadata.json","manifest.json"])
def test_changed_cache_is_rejected_without_fresh_inference(environment,payload):
    receipt,_=run_case(environment,"first")
    entry=Path(receipt["cache"]["entry"])
    with (entry/payload).open("ab") as f:f.write(b"corruption")
    with pytest.raises(pipe.PipelineError,match="Cache rejected"):run_case(environment,"rejected")
    assert environment["counter"]["extractors"]==1
    failure=json.loads((environment["root"]/"rejected/failure.json").read_bytes())
    assert failure["status"]=="failed" and failure["cache"]["status"]=="rejected"
    assert not failure["inference_run"] and not (environment["root"]/"rejected/manifest.json").exists()


@pytest.mark.parametrize("field",["video","metadata","runtime","pose_config","preprocessing_profile_hash","quality_profile_hash","registry_hash","cache_version"])
def test_cache_key_depends_on_all_declared_identities(environment,field):
    receipt,_=run_case(environment,"first");entry=Path(receipt["cache"]["entry"])
    identity=json.loads((entry/"manifest.json").read_bytes())["identity"]
    changed=copy.deepcopy(identity)
    changed[field]="different"
    assert pipe.artifact_hash(changed)!=pipe.artifact_hash(identity)
    assert pipe.read_cache(environment["root"]/"cache",changed) is None


@pytest.mark.parametrize("stage",["pose","quality","preprocessing","session_geometry","render_decode"])
def test_partial_failure_has_error_receipt_not_success(environment,monkeypatch,stage):
    def fail(*a,**kw):raise RuntimeError("synthetic stage failure")
    if stage=="pose":monkeypatch.setattr(pipe,"MediaPipePoseExtractor",fail)
    elif stage=="quality":monkeypatch.setattr(pipe,"quality_for",fail)
    elif stage=="preprocessing":monkeypatch.setattr(pipe,"preprocess_pose",fail)
    elif stage=="session_geometry":monkeypatch.setattr(pipe,"observations",fail)
    else:monkeypatch.setattr(render,"render_single_video_bundle",fail)
    with pytest.raises(RuntimeError,match="stage failure"):run_case(environment,"failed")
    path=environment["root"]/"failed"
    failure=json.loads((path/"failure.json").read_bytes())
    assert failure["error"]["stage"]==stage and failure["status"]=="failed"
    assert stage in failure["timings"] and not (path/"manifest.json").exists() and not (path/"receipt.json").exists()
    assert (path/"FAILED.txt").exists()


def test_unreadable_video_uses_real_reader_and_keeps_failure(tmp_path):
    video=tmp_path/"not-video.avi";video.write_bytes(b"not a video")
    with pytest.raises(Exception):pipe.run(video,tmp_path/"failed",side="left")
    failure=json.loads((tmp_path/"failed/failure.json").read_bytes())
    assert failure["error"]["stage"]=="metadata"
    assert failure["error"]["upstream_flag"]["category"]=="input_error"


@pytest.mark.parametrize("fault",["side","mode","hash","frame","time","null","official","reference","sequence","source_joint"])
def test_saved_readback_rejects_tampering(environment,fault):
    _,r=run_case(environment,"source",mode="exploratory",assume_side=True)
    if fault=="side":r["side"]="right"
    elif fault=="mode":r["mode"]="strict"
    elif fault=="hash":r["frames"][0]["source_linkage"]["measurement_input_hash"]="0"*64
    elif fault=="frame":r["frames"][0]["frame_index"]=1
    elif fault=="time":r["frames"][0]["timestamp_seconds"]=.1
    elif fault=="null":r["frames"][0]["observation"]["value"]=None
    elif fault=="official":r["official_eligibility"]=True
    elif fault=="reference":r["reference_status"]="VALIDATED"
    elif fault=="sequence":r["sequence"]["action"]="skip"
    else:r["frames"][0]["raw_joints"][0]["x"]+=.01
    r.pop("result_hash");r["result_hash"]=pipe.artifact_hash(r)
    path=environment["root"]/"source/result.json";path.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(ValueError):pipe.load_saved(environment["root"]/"source")


def test_saved_render_does_not_compute_geometry_or_quality(environment,monkeypatch):
    from gym_coach import geometry
    _,r=run_case(environment,"source",mode="exploratory",assume_side=True)
    def fail(*a,**kw):raise AssertionError("Renderer recomputed measurement/policy")
    monkeypatch.setattr(geometry,"joint_angle",fail)
    monkeypatch.setattr(pipe,"observe_geometry_feature",fail)
    monkeypatch.setattr(pipe,"preprocess_pose",fail)
    monkeypatch.setattr(pipe.QualityGate,"evaluate",fail)
    assert pipe.load_saved(environment["root"]/"source")==r
    html=render.single_video_html(r,"overlay.mp4")
    assert "REFERENCE_PENDING" in html and "Unavailable" in html


def test_html_escaping_and_no_remote_requests(environment):
    _,r=run_case(environment,"source")
    r["source"]["video_id"]='</video><img src=x onerror="bad">&'
    html=render.single_video_html(r,"overlay.mp4")
    assert "&lt;/video&gt;&lt;img" in html and '<img src=x' not in html
    assert "<script" not in html and "fetch(" not in html and "connect-src 'none'" in html
    assert "Strict" in html and "Không đủ điều kiện đo" in html


def test_public_cli_import_does_not_load_backend():
    p=subprocess.run([sys.executable,"-B","-c","import sys;from gym_coach.reporting import single_video;assert not ({'cv2','numpy','scipy','mediapipe','torch'} & set(sys.modules))"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_cli_exits_nonzero_for_preflight_failure(environment,capsys):
    assert pipe.main([str(environment["video"]),"--side","left","--mode","exploratory","--output-dir",str(environment["root"]/"bad")])==1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_changed_bytes_same_filename_is_cache_miss(environment):
    first,_=run_case(environment,"old")
    environment["video"].write_bytes(b"different synthetic video bytes at same path")
    second,_=run_case(environment,"new")
    assert second["inference_run"] and second["cache"]["status"]=="miss"
    assert first["cache"]["key"]!=second["cache"]["key"] and environment["counter"]["extractors"]==2


def test_source_changed_during_extraction_does_not_commit_cache(environment,monkeypatch):
    class Extractor:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def extract_frames(self,frames):
            environment["video"].write_bytes(b"changed during synthetic inference")
            return iter(environment["raw"])
    monkeypatch.setattr(pipe,"MediaPipePoseExtractor",Extractor)
    with pytest.raises(pipe.PipelineError,match="Source bytes changed"):run_case(environment,"failed")
    assert not (environment["root"]/"cache").exists()


def test_new_renderer_preserves_non_4_by_3_aspect_ratio(environment):
    import numpy as np
    _,result=run_case(environment,"source")
    image=np.full((480,240,3),255,dtype=np.uint8)
    painted=render.paint_frame(image,result["frames"][0],"left",result["sequence"],
        localized_unavailable=True,presentation_mode="strict",preserve_aspect=True)
    assert painted.shape==(720,1280,3)
    assert tuple(painted[150,25])==(34,21,12)  # left letterbox: unchanged canvas BGR.
    assert tuple(painted[150,225])==(255,255,255)  # native portrait begins at x224.
