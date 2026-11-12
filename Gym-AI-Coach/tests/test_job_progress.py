"""Truthful stage events and headless subprocess supervision fixtures."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest

from gym_coach.reporting import single_video as pipe
from gym_coach.reporting import real_descriptive_render as render
from gym_coach.reporting.progress import ProgressEmitter,ProgressEvent,EventStream,STAGES
from gym_coach.job_controller import JobController,VerifiedRun,command_for,local_path,verify_completed
from test_single_video_pipeline import environment


@pytest.fixture
def instrumented(environment,monkeypatch):
    original=render.render_single_video_bundle
    def fake(output,*,progress=None):
        if progress is not None:progress.emit("render","started",message="Synthetic render boundary")
        original(output)
        (output/"HUONG_DAN_MO_DEMO.txt").write_text("Synthetic test only",encoding="utf-8")
        verification=dict(status="PASSED",video_file="overlay-fixture.mp4",decoded_frames=9,geometry_calculated_by_renderer=False)
        (output/"video-verification.json").write_text(json.dumps(verification),encoding="utf-8")
        if progress is not None:
            progress.emit("render","completed")
            progress.emit("video_validation","started",message="Synthetic validation boundary")
            progress.emit("video_validation","completed")
    monkeypatch.setattr(render,"render_single_video_bundle",fake)
    return environment


def emitter_for(path,run_id="test"):
    events=[]
    return ProgressEmitter(run_id,str(path.resolve()),events.append),events


def test_events_do_not_change_numerics_and_only_complete_after_manifest(instrumented):
    e=instrumented;output=e["root"]/"events";progress,events=emitter_for(output)
    def accept(event):
        events.append(event)
        if event.stage=="pipeline" and event.status=="completed":assert (output/"manifest.json").is_file()
    progress.sink=accept
    a=pipe.run(e["video"],e["root"]/"legacy",side="left",mode="exploratory",assume_side=True,cache_dir=e["root"]/"cache")
    b=pipe.run(e["video"],output,side="left",mode="exploratory",assume_side=True,cache_dir=e["root"]/"cache",progress=progress)
    assert json.loads((e["root"]/"legacy/result.json").read_bytes())==json.loads((output/"result.json").read_bytes())
    assert [x.stage for x in events if x.status=="completed" and x.stage!="pipeline"]==list(STAGES)
    assert [x.sequence for x in events]==list(range(1,len(events)+1))
    assert any(x.stage=="pose" and x.cache_status=="hit" for x in events)
    assert events[-1].stage=="pipeline" and events[-1].status=="completed"
    assert all(x.total is None for x in events if x.stage in ("metadata","quality","preprocessing","pipeline"))
    assert a["cache"]["status"]=="miss" and b["cache"]["status"]=="hit"


@pytest.mark.parametrize("legacy",[True,False])
def test_cli_legacy_summary_or_jsonl_only(instrumented,capsys,legacy):
    e=instrumented;args=[str(e["video"]),"--side","left","--output-dir",str(e["root"]/"cli")]
    if not legacy:args += ["--events-jsonl","--run-id","wire-test"]
    assert pipe.main(args)==0
    lines=capsys.readouterr().out.splitlines()
    if legacy:assert len(lines)==1 and json.loads(lines[0])["ok"] is True
    else:
        events=[ProgressEvent.model_validate_json(line,strict=True) for line in lines]
        assert events[-1].status=="completed" and all(x.run_id=="wire-test" for x in events)


def test_preflight_collision_has_terminal_failure_without_overwrite(instrumented):
    e=instrumented;out=e["root"]/"existing";out.mkdir();(out/"keep").write_bytes(b"old")
    progress,events=emitter_for(out)
    with pytest.raises(FileExistsError):pipe.run(e["video"],out,side="left",progress=progress)
    assert len(events)==2 and events[-1].status=="failed" and events[-1].error
    assert (out/"keep").read_bytes()==b"old" and not (out/"manifest.json").exists()


@pytest.mark.parametrize("bad",["count","total","nan","mixed_count","pipeline_count","no_error","version"])
def test_event_schema_rejects_invalid_progress(bad):
    d=dict(run_id="x",sequence=1,stage="pose",status="progress",elapsed_seconds=0.,output_dir="local",message="fixture")
    if bad=="count":d.update(completed=2,total=1,unit="frames")
    elif bad=="total":d.update(completed=0,total=0,unit="frames")
    elif bad=="nan":d["elapsed_seconds"]=float("nan")
    elif bad=="mixed_count":d.update(completed=1)
    elif bad=="pipeline_count":d.update(stage="pipeline",completed=1,total=1,unit="frames")
    elif bad=="no_error":d["status"]="failed"
    else:d["event_version"]="future"
    with pytest.raises(ValueError):ProgressEvent.model_validate(d)


@pytest.mark.parametrize("bad",["sequence","run","output","elapsed","order","count_regression","denominator","missing_stages","after_terminal"])
def test_event_stream_rejects_reorder_or_false_terminal(tmp_path,bad):
    progress,events=emitter_for(tmp_path);progress.emit("pipeline","started");progress.emit("metadata","started")
    progress.emit("metadata","completed");progress.emit("pose","started");progress.emit("pose","progress",completed=2,total=9,unit="frames")
    stream=progress.stream
    d=dict(run_id="test",output_dir=str(tmp_path.resolve()),sequence=stream.sequence+1,stage="pose",status="progress",elapsed_seconds=stream.elapsed+.01,message="fixture",completed=3,total=9,unit="frames")
    if bad=="sequence":d["sequence"]+=1
    elif bad=="run":d["run_id"]="wrong"
    elif bad=="output":d["output_dir"]="https://external"
    elif bad=="elapsed":d["elapsed_seconds"]=0.;stream.elapsed=1.
    elif bad=="order":d.update(stage="render",status="started")
    elif bad=="count_regression":d["completed"]=1
    elif bad=="denominator":d["total"]=10
    elif bad=="missing_stages":d.update(stage="pipeline",status="completed",completed=None,total=None,unit=None)
    else:
        progress.emit("pipeline","failed",error=dict(reason="fixture"));d["sequence"]=stream.sequence+1
    with pytest.raises(ValueError):stream.accept(ProgressEvent.model_validate(d))


def test_failed_backend_event_is_not_success(instrumented,monkeypatch):
    e=instrumented;progress,events=emitter_for(e["root"]/"failed")
    def fail(*a,**kw):raise RuntimeError("fixture backend failure")
    monkeypatch.setattr(pipe,"quality_for",fail)
    with pytest.raises(RuntimeError):pipe.run(e["video"],e["root"]/"failed",side="left",progress=progress)
    assert events[-1].status=="failed" and events[-1].error["stage"]=="quality"
    assert not (e["root"]/"failed/manifest.json").exists()


def fake_worker_command(tmp_path,scenario="good"):
    # A real child process with fake events; no backend/video/model operation.
    worker=tmp_path/("fake-"+scenario+".py")
    worker.write_text('''import sys,json,os,time
from gym_coach.reporting.progress import ProgressEmitter,jsonl_sink,STAGES
run_id,output,scenario=sys.argv[1:]
p=ProgressEmitter(run_id,output,jsonl_sink(sys.stdout))
if scenario=='stderr':os.write(2,b'x'*600000)
if scenario=='slow':time.sleep(.25)
if scenario=='bad':print('not JSONL',flush=True);os.write(2,b'z'*600000);sys.exit(0)
if scenario=='crash':sys.exit(7)
p.emit('pipeline','started')
if scenario=='failed':p.emit('pipeline','failed',error={'reason':'fixture'});sys.exit(0)
for stage in STAGES:p.emit(stage,'started');p.emit(stage,'completed')
if scenario=='missing':sys.exit(0)
p.emit('pipeline','completed')
if scenario=='nonzero':sys.exit(3)
''',encoding="utf-8")
    return lambda job:[sys.executable,"-u","-B",str(worker),job.run_id,str(job.output),scenario]


def fake_verified(job,source_hash):
    return VerifiedRun(job.output,"a"*64,(),tuple(dict(total_frames=9,available_frames=0,unavailable_frames=9,available_ratio=0.).items()),"hit",False)


def wait(controller):
    deadline=time.monotonic()+20
    while controller.active and time.monotonic()<deadline:controller.poll();time.sleep(.01)
    controller.poll();assert not controller.active,"worker did not finish"


@pytest.mark.parametrize("scenario",["good","stderr","bad","crash","missing","failed","nonzero"])
def test_worker_stream_exit_and_large_stderr(tmp_path,scenario):
    video=tmp_path/"video có dấu & khoảng trắng.avi";video.write_bytes(b"fixture")
    controller=JobController(command_factory=fake_worker_command(tmp_path,scenario),verifier=fake_verified)
    job=controller.start(video,tmp_path/"kết quả có dấu",use_cache=False)
    wait(controller)
    assert controller.status==("completed" if scenario in ("good","stderr") else "failed")
    assert (job.directory/"events.jsonl").is_file() and (job.directory/"backend-stderr.txt").is_file()
    if scenario in ("bad","stderr"):assert (job.directory/"backend-stderr.txt").stat().st_size==600000
    if scenario not in ("good","stderr"):
        with pytest.raises(RuntimeError):controller.open_result("report",opener=lambda p:None)


def test_default_strict_ack_double_launch_and_close_state(tmp_path):
    video=tmp_path/"x.avi";video.write_bytes(b"fixture")
    controller=JobController(command_factory=fake_worker_command(tmp_path,"slow"),verifier=fake_verified)
    with pytest.raises(ValueError,match="xác nhận"):controller.start(video,tmp_path/"runs",exploratory=True)
    job=controller.start(video,tmp_path/"runs")
    assert not job.exploratory and job.side=="left" and controller.active
    with pytest.raises(RuntimeError,match="job"):controller.start(video,tmp_path/"runs")
    args=command_for(job)
    assert args[0]==sys.executable and str(video) in args and "--events-jsonl" in args
    assert "--assume-side" not in args and "--mode" not in args
    wait(controller);assert controller.status=="completed"
    assert len(list((tmp_path/"runs").glob("run-*")))==1


def test_explicit_exploratory_arguments(tmp_path):
    video=tmp_path/"x.avi";video.write_bytes(b"fixture")
    controller=JobController(command_factory=fake_worker_command(tmp_path),verifier=fake_verified)
    job=controller.start(video,tmp_path/"runs",side="right",exploratory=True,acknowledged=True)
    args=command_for(job);assert "--assume-side" in args and args[args.index("--mode")+1]=="exploratory"
    wait(controller)


def test_worker_success_without_artifacts_is_failure(tmp_path):
    video=tmp_path/"x.avi";video.write_bytes(b"fixture")
    controller=JobController(command_factory=fake_worker_command(tmp_path))
    controller.start(video,tmp_path/"runs");wait(controller)
    assert controller.status=="failed" and controller.verified is None


def test_worker_construction_error_recovers(tmp_path):
    video=tmp_path/"x.avi";video.write_bytes(b"fixture")
    def fail(job):raise OSError("fake spawn failure")
    controller=JobController(command_factory=fail)
    controller.start(video,tmp_path/"runs");wait(controller)
    assert controller.status=="failed" and "spawn failure" in controller.error["message"]


def test_output_collision_is_preserved(tmp_path):
    video=tmp_path/"x.avi";video.write_bytes(b"fixture");parent=tmp_path/"runs";(parent/"run-constant").mkdir(parents=True)
    (parent/"run-constant/keep").write_bytes(b"old")
    controller=JobController()
    with pytest.raises(FileExistsError):controller.start(video,parent,run_id="constant")
    assert not controller.active and (parent/"run-constant/keep").read_bytes()==b"old"


def test_safe_open_only_verified_fixed_local_paths(instrumented):
    e=instrumented;out=e["root"]/"result"
    pipe.run(e["video"],out,side="left")
    from gym_coach.job_controller import Job
    job=Job("test",e["video"],e["root"],out,"left",False,None)
    verified=verify_completed(job,pipe.sha(e["video"]))
    controller=JobController();controller.status="completed";controller.verified=verified
    opened=[];assert controller.open_result("report",opener=opened.append)==out/"START_HERE.html"
    assert opened==[str(out/"START_HERE.html")]
    with pytest.raises(ValueError):controller.open_result("calc.exe",opener=opened.append)
    (out/"START_HERE.html").write_text("tampered",encoding="utf-8")
    with pytest.raises(ValueError,match="đã thay đổi"):controller.open_result("report",opener=opened.append)
    assert len(opened)==1


@pytest.mark.parametrize("path",["https://host/a","//host/share/a","\\\\host\\share\\a"])
def test_network_paths_never_open(path):
    with pytest.raises(ValueError):local_path(path)


def test_cache_status_survives_uncounted_progress_events():
    controller=JobController()
    base=dict(run_id="fixture",output_dir="local",elapsed_seconds=0.,message="fixture")
    controller.events.put(("progress",ProgressEvent(**base,sequence=1,stage="pose",status="started",cache_status="miss")))
    controller.events.put(("progress",ProgressEvent(**base,sequence=2,stage="pose",status="progress",completed=1,total=9,unit="frames")))
    controller.poll()
    assert controller.cache_status=="miss" and controller.last_event.cache_status is None


@pytest.mark.parametrize("stage,unit",[("metadata","frames"),("quality","frames"),("preprocessing","frames"),("geometry","output_frames"),("save","frames")])
def test_stage_unit_semantics(stage,unit):
    with pytest.raises(ValueError):ProgressEvent(run_id="x",sequence=1,stage=stage,status="progress",elapsed_seconds=0.,output_dir="local",message="fixture",completed=1,total=1,unit=unit)


def test_manifest_write_failure_cannot_emit_success(instrumented,monkeypatch):
    e=instrumented;progress,events=emitter_for(e["root"]/"failed");write=pipe.write_json
    def fail_manifest(path,value):
        if Path(path).name=="manifest.json":raise OSError("fixture manifest write failure")
        return write(path,value)
    monkeypatch.setattr(pipe,"write_json",fail_manifest)
    with pytest.raises(OSError):pipe.run(e["video"],e["root"]/"failed",side="left",progress=progress)
    assert events[-1].status=="failed" and events[-1].error["stage"]=="finalize"
    assert not any(x.stage=="pipeline" and x.status=="completed" for x in events)


def test_rejected_cache_is_explicit_on_failure_event(instrumented):
    e=instrumented;cache=e["root"]/"cache"
    receipt=pipe.run(e["video"],e["root"]/"first",side="left",cache_dir=cache)
    path=Path(receipt["cache"]["entry"])/"raw-pose.jsonl"
    with path.open("ab") as f:f.write(b"tampered fixture copy")
    progress,events=emitter_for(e["root"]/"second")
    with pytest.raises(pipe.PipelineError):pipe.run(e["video"],e["root"]/"second",side="left",cache_dir=cache,progress=progress)
    assert events[-1].status=="failed" and events[-1].cache_status=="rejected"
