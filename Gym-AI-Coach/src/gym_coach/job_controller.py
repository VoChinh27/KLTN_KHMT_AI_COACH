"""UI-independent thread/subprocess supervision and artifact verification."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from .reporting.progress import EventStream,ProgressEvent


def local_path(value):
    value=os.fspath(value)
    if value.startswith(("\\\\","//")) or "://" in value:raise ValueError("Chỉ hỗ trợ đường dẫn local")
    path=Path(value).absolute()
    for part in (path,*path.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(),"st_file_attributes",0)&0x400):
            raise ValueError("Không mở đường dẫn symlink/reparse")
    return path.resolve()


@dataclass(frozen=True)
class Job:
    run_id: str
    video: Path
    directory: Path
    output: Path
    side: str
    exploratory: bool
    cache: Path | None


@dataclass(frozen=True)
class VerifiedRun:
    directory: Path
    manifest_sha256: str
    targets: tuple[tuple[str,str,str],...]
    coverage: tuple[tuple[str,int|float],...]
    cache_status: str
    inference_run: bool

    def target(self,kind):
        from .reporting.single_video import sha
        if sha(self.directory/"manifest.json")!=self.manifest_sha256:raise ValueError("Manifest đã thay đổi sau kiểm tra")
        if kind=="folder":
            if local_path(self.directory)!=self.directory:raise ValueError("Thư mục kết quả đã đổi")
            return self.directory
        choices={k:(n,h) for k,n,h in self.targets}
        if kind not in choices:raise ValueError("Loại kết quả không được phép mở")
        name,expected=choices[kind];path=local_path(self.directory/name)
        if path.parent!=self.directory or sha(path)!=expected:raise ValueError("File kết quả đã thay đổi")
        return path


def verify_completed(job,source_sha256):
    """Never trust a log/event pathname as an executable or as completion proof."""
    from .reporting.single_video import DISABLED,VERSION,file_identity,sha,load_saved
    out=local_path(job.output);manifest=json.loads((out/"manifest.json").read_bytes())
    required={"source.json","raw-pose.jsonl","preprocessing.json","result.json","frames.jsonl","coverage.json",
              "quality-source.json","quality-processing.json","processing-metadata.json","receipt.json",
              "START_HERE.html","HUONG_DAN_MO_DEMO.txt","video-verification.json"}
    if manifest["version"]!=VERSION or manifest["status"]!="completed" or not required<=set(manifest["files"]):
        raise ValueError("Thiếu artifacts/manifest hoàn tất")
    for name,expected in manifest["files"].items():
        if Path(name).name!=name or ":" in name or "\\" in name or "/" in name:raise ValueError("Manifest path không an toàn")
        path=local_path(out/name)
        if path.parent!=out or file_identity(path)!=expected:raise ValueError("Hash/size artifact không khớp: "+name)
    receipt=json.loads((out/"receipt.json").read_bytes())
    if receipt["status"]!="completed" or any(manifest[k]!=v or receipt[k]!=v for k,v in DISABLED.items()):
        raise ValueError("Trạng thái khoa học/manifest không khớp")
    result=load_saved(out)
    if (local_path(result["source"]["source_path"])!=job.video or result["source"]["video_sha256"]!=source_sha256 or
            result["side"]!=job.side or result["mode"]!=("exploratory" if job.exploratory else "strict") or
            result["coverage"]!=receipt["coverage"]):raise ValueError("Kết quả không liên kết với video/bên/chế độ đã chọn")
    verification=json.loads((out/"video-verification.json").read_bytes());video=verification["video_file"]
    if (Path(video).name!=video or video not in manifest["files"] or not video.startswith("overlay-") or
            Path(video).suffix.lower()!=".mp4" or verification["status"]!="PASSED" or
            verification["decoded_frames"]!=result["coverage"]["total_frames"] or
            verification["geometry_calculated_by_renderer"] is not False):raise ValueError("Video output chưa được kiểm tra")
    status=receipt["cache"]["status"]
    if status not in ("hit","miss","disabled"):raise ValueError("Cache status không phù hợp run hoàn tất")
    return VerifiedRun(out,sha(out/"manifest.json"),
        tuple((k,n,manifest["files"][n]["sha256"]) for k,n in (("report","START_HERE.html"),("video",video))),
        tuple(result["coverage"].items()),status,receipt["inference_run"])


def command_for(job):
    command=[sys.executable,"-u","-B","-m","gym_coach.reporting.single_video",str(job.video),
        "--side",job.side,"--output-dir",str(job.output),"--events-jsonl","--run-id",job.run_id]
    if job.exploratory:command.extend(["--mode","exploratory","--assume-side"])
    if job.cache is not None:command.extend(["--cache-dir",str(job.cache)])
    return command


class JobController:
    def __init__(self,*,command_factory=command_for,verifier=verify_completed):
        self.command_factory=command_factory;self.verifier=verifier;self.lock=threading.Lock()
        self.active=False;self.job=None;self.started=None;self.events=queue.Queue();self.thread=None
        self.status="idle";self.last_event=None;self.verified=None;self.error=None;self.log=[]
        self.cache_status=None

    def start(self,video,parent,*,side="left",exploratory=False,acknowledged=False,use_cache=True,run_id=None):
        with self.lock:
            if self.active:raise RuntimeError("Đang có một job chạy; không tạo run thứ hai")
            if side not in ("left","right"):raise ValueError("Bên gối phải là trái hoặc phải")
            if exploratory and not acknowledged:raise ValueError("Cần xác nhận giả định side-view chưa kiểm chứng")
            video=local_path(video);parent=local_path(parent)
            if not video.is_file():raise ValueError("Chưa chọn video local tồn tại")
            run_id=run_id or uuid.uuid4().hex
            # Validate correlation before allocating folders/spawning anything.
            import re
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",run_id):raise ValueError("Run ID không hợp lệ")
            parent.mkdir(parents=True,exist_ok=True);directory=parent/("run-"+run_id)
            directory.mkdir(exist_ok=False)
            job=Job(run_id,video,directory,directory/"result",side,exploratory,parent/"pose-cache" if use_cache else None)
            self.job=job;self.active=True;self.started=time.monotonic();self.status="running"
            self.verified=None;self.error=None;self.last_event=None;self.log=[]
            self.cache_status=None
            self.thread=threading.Thread(target=self._work,args=(job,),daemon=False)
            try:self.thread.start()
            except Exception:
                self.active=False;self.status="failed"
                raise
            return job

    def _work(self,job):
        stream=EventStream(job.run_id,str(job.output));protocol_error=[];io_errors=[];stderr_tail=[b""]
        command=None;exit_code=None;source_sha=None;outcome=None
        def fail_protocol(message):
            if not protocol_error:protocol_error.append(message)
        try:
            command=self.command_factory(job)
            from .reporting.single_video import sha,write_json
            source_sha=sha(job.video)
            write_json(job.directory/"launcher.json",dict(command=command,shell=False,python=sys.executable,
                run_id=job.run_id,video=str(job.video),video_sha256=source_sha,output=str(job.output),
                side=job.side,mode="exploratory" if job.exploratory else "strict",cancellation_supported=False))
            env=os.environ.copy();env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8")
            flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
            with (job.directory/"events.jsonl").open("xb") as output_log,(job.directory/"backend-stderr.txt").open("xb") as error_log:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    shell=False,cwd=Path(__file__).resolve().parents[2],env=env,creationflags=flags)
                def stdout_reader():
                    try:
                        while True:
                            line=process.stdout.readline(65537)
                            if not line:break
                            output_log.write(line);output_log.flush()
                            if len(line)>65536:
                                fail_protocol("JSONL event quá dài");continue
                            if protocol_error:continue  # keep draining even after a bad protocol
                            try:
                                event=ProgressEvent.model_validate_json(line,strict=True);stream.accept(event)
                                self.events.put(("progress",event))
                            except Exception as exc:fail_protocol("Event không hợp lệ: "+str(exc)[:500])
                    except Exception as exc:io_errors.append(str(exc))
                    finally:process.stdout.close()
                def stderr_reader():
                    try:
                        while True:
                            chunk=process.stderr.read(4096)
                            if not chunk:break
                            error_log.write(chunk);error_log.flush();stderr_tail[0]=(stderr_tail[0]+chunk)[-2000:]
                    except Exception as exc:io_errors.append(str(exc))
                    finally:process.stderr.close()
                readers=[threading.Thread(target=f,daemon=False) for f in (stdout_reader,stderr_reader)]
                for thread in readers:thread.start()
                exit_code=process.wait()
                for thread in readers:thread.join()
            if io_errors:raise RuntimeError("Không đọc được worker stream: "+"; ".join(io_errors))
            if protocol_error:raise RuntimeError(protocol_error[0])
            if exit_code!=0:raise RuntimeError("Worker exit "+str(exit_code)+": "+(stream.terminal.message if stream.terminal else "không hoàn tất"))
            if stream.terminal is None or stream.terminal.status!="completed":raise RuntimeError("Worker thoát nhưng thiếu event hoàn tất hợp lệ")
            self.events.put(("verifying",None))
            outcome=self.verifier(job,source_sha)
            self.events.put(("completed",outcome))
        except Exception as exc:
            terminal_error=stream.terminal.error if stream.terminal is not None and stream.terminal.status=="failed" else None
            detail=dict(message=str(exc),exit_code=exit_code,run_id=job.run_id,evidence_dir=str(job.directory),
                protocol_error=protocol_error[0] if protocol_error else None,worker_error=terminal_error,
                backend_tail=stderr_tail[0].decode("utf-8",errors="replace"))
            self.events.put(("failed",detail))
        finally:
            from .reporting.single_video import write_json
            try:write_json(job.directory/"launcher-outcome.json",dict(status="completed" if outcome else "failed",exit_code=exit_code,
                event_count=stream.sequence,terminal=stream.terminal.model_dump(mode="json") if stream.terminal else None,
                protocol_errors=protocol_error,stream_errors=io_errors,validated_artifacts=outcome is not None))
            except Exception as exc:self.events.put(("log","Không ghi được launcher receipt: "+str(exc)))

    def poll(self):
        items=[]
        while True:
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            items.append((kind,value))
            if kind=="progress":
                self.last_event=value;self.log.append(value.message)
                if value.cache_status is not None:self.cache_status=value.cache_status
                if value.status=="failed":self.status="finishing_failure"
            elif kind=="verifying":self.status="verifying"
            elif kind=="completed":self.verified=value;self.status="completed"
            elif kind=="failed":self.error=value;self.status="failed"
            elif kind=="log":self.log.append(value)
        self.log=self.log[-80:]
        # Completion messages do not allow another launch until worker/log threads end.
        if self.active and self.thread is not None and not self.thread.is_alive() and self.status in ("completed","failed"):
            self.active=False
        return items

    def open_result(self,kind,*,opener=None):
        if self.active or self.status!="completed" or self.verified is None:raise RuntimeError("Chưa có kết quả đã kiểm tra để mở")
        path=self.verified.target(kind)
        if opener is None:
            if os.name!="nt":raise RuntimeError("Chức năng mở desktop hiện dùng Windows")
            opener=os.startfile
        opener(str(path));return path
