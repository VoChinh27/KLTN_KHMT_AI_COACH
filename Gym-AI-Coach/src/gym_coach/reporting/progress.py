"""Versioned execution events, independent of numerical/quality contracts."""
import json
import time
from typing import Literal

from pydantic import Field, model_validator

from ..contracts import ContractModel

VERSION = "squat-progress-v1"
STAGES = ("metadata", "pose", "quality", "preprocessing", "geometry", "save",
          "render", "video_validation", "finalize")
LABELS = dict(metadata="Đọc metadata và kiểm tra nguồn",pose="Pose / cache",quality="Kiểm tra chất lượng",
    preprocessing="Tiền xử lý",geometry="Đo hình học",save="Lưu kết quả",render="Tạo video overlay",
    video_validation="Giải mã kiểm tra video",finalize="Ghi manifest",pipeline="Luồng phân tích")


class ProgressEvent(ContractModel):
    event_version: Literal["squat-progress-v1"] = VERSION
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    sequence: int = Field(ge=1,strict=True)
    stage: Literal["pipeline","metadata","pose","quality","preprocessing","geometry","save","render","video_validation","finalize"]
    status: Literal["started","progress","completed","failed"]
    elapsed_seconds: float = Field(ge=0,strict=True)
    completed: int | None = Field(default=None,ge=0,strict=True)
    total: int | None = Field(default=None,ge=1,strict=True)
    unit: Literal["frames","jsonl_frames","output_frames","decoded_frames"] | None = None
    cache_status: Literal["disabled","not_checked","miss","hit","rejected"] | None = None
    message: str = Field(min_length=1,max_length=1000)
    output_dir: str = Field(min_length=1)
    error: dict[str,str] | None = None

    @model_validator(mode="after")
    def truthful_counts_and_terminal(self):
        if (self.completed is None)!=(self.total is None) or (self.completed is None)!=(self.unit is None):
            raise ValueError("Counts and unit must be supplied together; unknown denominator stays null")
        if self.completed is not None and self.completed>self.total:raise ValueError("Completed exceeds total")
        units={"pose":"frames","geometry":"frames","save":"jsonl_frames","render":"output_frames","video_validation":"decoded_frames"}
        if self.unit is not None and units.get(self.stage)!=self.unit:raise ValueError("Count unit is not meaningful for this stage")
        if self.status=="failed" and not self.error:raise ValueError("Failure requires a structured error")
        if self.status!="failed" and self.error is not None:raise ValueError("Only failure carries an error")
        if self.stage=="pipeline" and (self.completed is not None or self.status=="progress"):
            raise ValueError("No invented overall pipeline percentage/count")
        return self


class EventStream:
    """Reject reorder, mixed runs, regression, missing stages and late events."""
    def __init__(self,run_id,output_dir):
        self.run_id=run_id;self.output_dir=str(output_dir);self.sequence=0;self.elapsed=0.
        self.active=None;self.finished=[];self.last_count=None;self.last_total=None;self.last_unit=None
        self.started=False;self.terminal=None

    def accept(self,event):
        if (event.run_id!=self.run_id or event.output_dir!=self.output_dir or
                event.sequence!=self.sequence+1 or event.elapsed_seconds<self.elapsed or self.terminal):
            raise ValueError("Event run/path/order/elapsed mismatch or event after terminal")
        if not self.started:
            if (event.stage,event.status)!=("pipeline","started"):raise ValueError("Pipeline start required")
            self.started=True
        elif event.stage=="pipeline":
            if event.status=="completed":
                if self.active or self.finished!=list(STAGES):raise ValueError("Required stages not completed")
            elif event.status!="failed":raise ValueError("Duplicate pipeline start")
            self.terminal=event
        else:
            if event.status=="started":
                if self.active or len(self.finished)>=len(STAGES) or event.stage!=STAGES[len(self.finished)]:
                    raise ValueError("Stage order/start mismatch")
                self.active=event.stage;self.last_count=None;self.last_total=None;self.last_unit=None
            elif self.active!=event.stage:raise ValueError("Stage event without matching start")
            if event.completed is not None:
                if ((self.last_count is not None and event.completed<self.last_count) or
                        (self.last_total is not None and (event.total,event.unit)!=(self.last_total,self.last_unit))):
                    raise ValueError("Stage count/denominator regression")
                self.last_count,self.last_total,self.last_unit=event.completed,event.total,event.unit
            if event.status=="completed":
                if event.completed is not None and event.completed!=event.total:raise ValueError("Incomplete counted stage")
                self.finished.append(event.stage);self.active=None
            elif event.status=="failed":self.active=None
        self.sequence=event.sequence;self.elapsed=event.elapsed_seconds


class ProgressEmitter:
    def __init__(self,run_id,output_dir,sink,*,clock=time.perf_counter):
        self.run_id=run_id;self.output_dir=str(output_dir);self.sink=sink;self.clock=clock
        self.start=clock();self.stream=EventStream(run_id,self.output_dir)

    def emit(self,stage,status,*,completed=None,total=None,unit=None,cache_status=None,message=None,error=None):
        event=ProgressEvent(run_id=self.run_id,output_dir=self.output_dir,sequence=self.stream.sequence+1,
            stage=stage,status=status,elapsed_seconds=max(0.,self.clock()-self.start),completed=completed,total=total,
            unit=unit,cache_status=cache_status,message=message or LABELS[stage],error=error)
        self.stream.accept(event)
        self.sink(event)
        return event

    def count(self,stage,completed,total,unit):
        if completed==1 or completed==total or completed%10==0:
            self.emit(stage,"progress",completed=completed,total=total,unit=unit)


def jsonl_sink(stream):
    def write(event):
        stream.write(event.model_dump_json()+"\n");stream.flush()
    return write
