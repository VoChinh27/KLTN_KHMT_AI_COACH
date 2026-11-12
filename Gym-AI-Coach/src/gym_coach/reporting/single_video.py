"""One-command BodyWeightSquats descriptive pipeline; strict by default.

Reuses Stage1--4, validated clip sessions and the canonical framewise adapter.
Pose-only cache never carries exploratory viewpoint or measurement eligibility.
"""
import argparse
from contextlib import redirect_stdout
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import uuid

from .. import __version__
from ..contracts import Exercise, Viewpoint
from ..features import FeatureContextSession, GEOMETRY_REGISTRY, GeometryFeatureObservation, observe_geometry_feature
from ..features.models import content_hash
from ..flags import GymError
from ..pose import MediaPipePoseExtractor
from ..pose.models import PoseBackendIdentity, PoseExtractionConfig, PoseFrame
from ..pose.serialization import write_pose_frames
from ..preprocessing import PreprocessingProfile, preprocess_pose, write_preprocessing_result
from ..preprocessing.models import PreprocessingResult, content_hash as source_hash
from ..preprocessing.pipeline import validate_sequence
from ..quality import QualityGate, QualityProfile
from ..video import OpenCVVideoReader
from ..video.models import VideoMetadata
from .real_descriptive_demo import STATUS, artifact_hash, sha, write_json

VERSION = "squat-single-video-v1"
CACHE_VERSION = "squat-source-pose-cache-v1"
DISABLED = dict(**STATUS, reference_status="REFERENCE_PENDING")


class PipelineError(RuntimeError):
    """Workflow reasons are versioned receipts, not new upstream flag semantics."""
    def __init__(self, reason, message, *, category="input_error"):
        self.reason=reason;self.category=category
        super().__init__(message)


def _plain(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(),"st_file_attributes",0)&0x400):
            raise PipelineError("redirected_path","Symlink/reparse paths are not supported")
    return path.resolve()


def file_identity(path):
    return dict(sha256=sha(path),size_bytes=Path(path).stat().st_size)


def runtime_identity():
    """Identify installed bundled model without creating a backend or downloading."""
    spec=importlib.util.find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        raise PipelineError("backend_unavailable","Installed MediaPipe package not found")
    model=Path(next(iter(spec.submodule_search_locations)))/"modules/pose_landmark/pose_landmark_full.tflite"
    version=importlib.metadata.version("mediapipe")
    if version!="0.10.14" or not model.is_file():
        raise PipelineError("backend_incompatible","MediaPipe0.10.14 bundled full model required; downloads disabled")
    backend=PoseBackendIdentity(mediapipe_version=version,model_asset_sha256=sha(model))
    package=Path(__file__).resolve().parents[1]
    return dict(package_version=__version__,backend=backend.model_dump(mode="json"),
        installed_versions={n:importlib.metadata.version(n) for n in
            ("mediapipe","opencv-contrib-python","numpy","scipy","pydantic")},
        implementation_files={p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob("*.py"))})


def source_identity(video, metadata, runtime, config, quality_profile, preprocessing_profile):
    return dict(cache_version=CACHE_VERSION,video=file_identity(video),metadata=metadata.model_dump(mode="json"),
        exercise="BodyWeightSquats",exercise_source="caller_declared_not_action_recognition",
        raw_viewpoint="unknown",camera_view="UNKNOWN",runtime=runtime,
        pose_config=config.model_dump(mode="json"),pose_config_hash=config.content_hash(),
        quality_profile=quality_profile.model_dump(mode="json"),quality_profile_hash=quality_profile.content_hash(),
        preprocessing_profile=preprocessing_profile.model_dump(mode="json"),
        preprocessing_profile_hash=source_hash(preprocessing_profile),registry_hash=GEOMETRY_REGISTRY.content_hash())


def validate_raw(raw, metadata, identity):
    validate_sequence(raw,metadata)
    for frame in raw:
        if (frame.exercise!=Exercise.BODY_WEIGHT_SQUATS or frame.viewpoint!=Viewpoint.UNKNOWN or
                frame.backend.model_dump(mode="json")!=identity["runtime"]["backend"] or
                frame.config.model_dump(mode="json")!=identity["pose_config"]):
            raise PipelineError("pose_identity_mismatch","Raw pose model/config/exercise/viewpoint differs")


def read_cache(directory, identity):
    """Exact identity, all payload bytes and typed timeline validation; fail closed."""
    key=artifact_hash(identity);entry=_plain(Path(directory)/key)
    if not entry.exists():return None
    try:
        manifest=json.loads((entry/"manifest.json").read_bytes())
        if (manifest["version"]!=CACHE_VERSION or manifest["key"]!=key or manifest["identity"]!=identity or
                set(manifest["files"])!={"raw-pose.jsonl","metadata.json"}):
            raise ValueError("Cache identity/manifest mismatch")
        payload={}
        for name,expected in manifest["files"].items():
            path=_plain(entry/name)
            if file_identity(path)!=expected:raise ValueError("Cache payload bytes changed: "+name)
            payload[name]=path.read_bytes()
        metadata=VideoMetadata.model_validate_json(payload["metadata.json"],strict=True)
        if metadata.model_dump(mode="json")!=identity["metadata"]:raise ValueError("Cache metadata differs")
        raw=tuple(PoseFrame.model_validate_json(line,strict=True) for line in payload["raw-pose.jsonl"].splitlines())
        validate_raw(raw,metadata,identity)
        return raw,dict(status="hit",key=key,entry=str(entry),manifest_identity=file_identity(entry/"manifest.json"),
                        scope="raw unknown-viewpoint pose only; quality/preprocessing/geometry recomputed")
    except Exception as exc:
        raise PipelineError("cache_rejected",f"Cache rejected; no fallback/inference: {type(exc).__name__}: {exc}") from exc


def write_cache(directory,identity,raw):
    key=artifact_hash(identity);entry=_plain(Path(directory)/key)
    entry.mkdir(parents=True,exist_ok=False)
    write_pose_frames(entry/"raw-pose.jsonl",raw)
    write_json(entry/"metadata.json",identity["metadata"])
    write_json(entry/"manifest.json",dict(version=CACHE_VERSION,key=key,identity=identity,
        files={n:file_identity(entry/n) for n in ("raw-pose.jsonl","metadata.json")},**DISABLED))
    return dict(status="miss",key=key,entry=str(entry),stored=True,
                manifest_identity=file_identity(entry/"manifest.json"),scope="raw unknown-viewpoint pose only")


def quality_for(frames,profile):
    return QualityGate(profile).evaluate(frames,
        metric_requirements=GEOMETRY_REGISTRY.quality_requirements(Exercise.BODY_WEIGHT_SQUATS),
        metric_viewpoints=GEOMETRY_REGISTRY.quality_viewpoints(Exercise.BODY_WEIGHT_SQUATS))


def observations(processed,raw,side,*,progress=None):
    """Session intake once; unchanged adapter and schema for the fixed side."""
    session=FeatureContextSession(processed);rows=[];timings=Counter()
    for frame in raw:
        t=time.perf_counter();selected=session.select(frame_index=frame.frame_index);context=selected.input_context()
        timings["context_seconds"]+=time.perf_counter()-t
        t=time.perf_counter();obs=observe_geometry_feature(f"squat_knee_angle_{side}_deg",context)
        timings["adapter_seconds"]+=time.perf_counter()-t
        if obs.availability.input_hash!=selected.measurement_input_hash:
            raise PipelineError("observation_linkage_mismatch","Adapter/session input identity differs")
        rows.append(dict(video_id=frame.video_id,side=side,frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,raw_pose_hash=source_hash(frame),
            raw_joints=[j.model_dump(mode="json") for j in frame.joints],
            observation=obs.model_dump(mode="json"),
            quality_action=str(processed.quality.metric_eligibility[obs.feature_id].action),
            source_linkage=dict(selected.reference_linkage(),observation_hash=content_hash(obs))))
        if progress is not None:progress.count("geometry",len(rows),len(raw),"frames")
    return rows,session.metadata,dict(timings)


def make_result(source,mode,side,metadata,processed,rows,session):
    available=sum(r["observation"]["status"]=="available" for r in rows)
    result=dict(pipeline_version=VERSION,**DISABLED,synthetic=False,source=source,mode=mode,side=side,
        exercise="BodyWeightSquats",exercise_source="caller_declared_not_action_recognition",
        camera_view="UNKNOWN",source_viewpoint="unknown",documented=False,
        assumption=dict(viewpoint="side",source="caller_cli_assume_side",status="caller_assumed_unverified",
                        visual_assessment_performed_by_pipeline=False) if mode=="exploratory" else None,
        metadata=metadata.model_dump(mode="json"),feature_id=f"squat_knee_angle_{side}_deg",
        sequence=processed.quality.eligibility.model_dump(mode="json"),
        metric=processed.quality.metric_eligibility[f"squat_knee_angle_{side}_deg"].model_dump(mode="json"),
        preprocessing_result_hash=source_hash(processed),session_execution=session,
        coverage=dict(total_frames=len(rows),available_frames=available,unavailable_frames=len(rows)-available,
                      available_ratio=available/len(rows) if rows else None),frames=rows,
        temporal=dict(status="unavailable",reason="not_run_no_independent_temporal_evidence",
                      phase=None,repetitions=None,point_turn=None,pause=None),
        coordinate_note="Projected2D estimate from processed coordinates; skeleton from raw source pose; not physical3D or form judgment")
    result["result_hash"]=artifact_hash(result)
    return validate_result(result)


def validate_result(result):
    """Read-only renderer boundary: validate saved contracts/linkage, never measure."""
    data=dict(result);expected=data.pop("result_hash")
    if expected!=artifact_hash(data) or data["pipeline_version"]!=VERSION:raise ValueError("Result version/hash mismatch")
    if any(data[k]!=v for k,v in DISABLED.items()):raise ValueError("Official/reference/assessment status changed")
    if (data["side"] not in ("left","right") or data["feature_id"]!=f"squat_knee_angle_{data['side']}_deg" or
            data["exercise"]!="BodyWeightSquats" or data["camera_view"]!="UNKNOWN" or
            data["source_viewpoint"]!="unknown" or data["documented"] is not False):
        raise ValueError("Source/side/exercise metadata mismatch")
    if data["mode"]=="strict":
        if data["assumption"] is not None:raise ValueError("Strict cannot acquire side assumption")
        viewpoint="unknown"
    elif data["mode"]=="exploratory":
        if data["assumption"]!=dict(viewpoint="side",source="caller_cli_assume_side",status="caller_assumed_unverified",
                                    visual_assessment_performed_by_pipeline=False):raise ValueError("Unverified assumption required")
        viewpoint="side"
    else:raise ValueError("Unknown pipeline mode")
    if data["temporal"]!=dict(status="unavailable",reason="not_run_no_independent_temporal_evidence",
                              phase=None,repetitions=None,point_turn=None,pause=None):raise ValueError("Temporal remains unavailable")
    meta=VideoMetadata.model_validate_json(json.dumps(data["metadata"]),strict=True)
    if len(data["frames"])!=meta.frame_count:raise ValueError("Timeline denominator differs")
    available=0
    for i,row in enumerate(data["frames"]):
        obs=GeometryFeatureObservation.model_validate_json(json.dumps(row["observation"]),strict=True)
        link=row["source_linkage"]
        if ((row["video_id"],row["frame_index"],row["timestamp_seconds"],row["side"])!=(meta.video_id,i,i/meta.fps,data["side"]) or
                (obs.video_id,obs.frame_index,obs.timestamp_seconds,str(obs.viewpoint))!=(meta.video_id,i,i/meta.fps,viewpoint) or
                obs.feature_id!=data["feature_id"] or obs.temporal_analysis_eligible is not False or
                link["observation_hash"]!=content_hash(obs) or link["measurement_input_hash"]!=obs.availability.input_hash or
                link["preprocessing_result_hash"]!=data["preprocessing_result_hash"] or
                row["quality_action"]!=obs.availability.input.quality.metric_eligibility[obs.feature_id].action or
                obs.availability.input.quality.eligibility.model_dump(mode="json")!=data["sequence"] or
                obs.availability.input.quality.metric_eligibility[obs.feature_id].model_dump(mode="json")!=data["metric"]):
            raise ValueError("Saved frame/side/gate/observation linkage mismatch")
        available+=obs.status=="available"
    if data["coverage"]!=dict(total_frames=meta.frame_count,available_frames=available,
            unavailable_frames=meta.frame_count-available,available_ratio=available/meta.frame_count if meta.frame_count else None):
        raise ValueError("Coverage mismatch")
    return result


def load_saved(output):
    """Bind raw skeleton and Stage4 to saved observations before rendering."""
    output=Path(output);result=validate_result(json.loads((output/"result.json").read_bytes()))
    raw_data=(output/"raw-pose.jsonl").read_bytes()
    if sha(output/"raw-pose.jsonl")!=result["source"]["raw_pose_sha256"]:raise ValueError("Raw source bytes differ")
    raw=tuple(PoseFrame.model_validate_json(line,strict=True) for line in raw_data.splitlines())
    processed=PreprocessingResult.model_validate_json((output/"preprocessing.json").read_bytes(),strict=True)
    if source_hash(processed)!=result["preprocessing_result_hash"]:raise ValueError("Stage4 source hash mismatch")
    validate_sequence(raw,processed.metadata)
    for pose,assumed,frame,row in zip(raw,processed.raw_frames,processed.frames,result["frames"]):
        a,b=pose.model_dump(mode="json"),assumed.model_dump(mode="json")
        if a.pop("viewpoint")!="unknown" or b.pop("viewpoint")!=("side" if result["mode"]=="exploratory" else "unknown") or a!=b:
            raise ValueError("Processing copy changed source pose beyond viewpoint")
        if (source_hash(pose)!=row["raw_pose_hash"] or [j.model_dump(mode="json") for j in pose.joints]!=row["raw_joints"] or
                row["observation"]["availability"]["input"]["frame"]!=frame.model_dump(mode="json")):
            raise ValueError("Skeleton/processed frame linkage mismatch")
    return result


def _run(video,output,*,side,mode="strict",assume_side=False,cache_dir=None,progress=None):
    if side not in ("left","right") or mode not in ("strict","exploratory"):
        raise PipelineError("invalid_options","Expected fixed left/right and strict/exploratory")
    if (mode=="exploratory") is not assume_side:
        raise PipelineError("explicit_assumption_required","Use --mode exploratory --assume-side together; strict never assumes side")
    video=_plain(video);out=_plain(output);cache=_plain(cache_dir) if cache_dir is not None else None
    if out.exists():raise FileExistsError(f"Output already exists: {out}")
    if video.is_relative_to(out) or (cache is not None and (out.is_relative_to(cache) or cache.is_relative_to(out) or video.is_relative_to(cache))):
        raise PipelineError("overlapping_paths","Source, output and cache must be separate")
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();timings={};stage="metadata"
    cache_receipt=dict(status="not_checked" if cache is not None else "disabled")
    inference_run=False;inference_attempted=False;session=None
    @contextmanager
    def timed(name):
        nonlocal stage
        stage=name;t=time.perf_counter()
        event_stage={"metadata":"metadata","pose":"pose","quality":"quality","preprocessing":"preprocessing",
                     "session_geometry":"geometry","save_validate":"save","finalize":"finalize"}.get(name)
        if progress is not None and event_stage:progress.emit(event_stage,"started")
        try:
            yield
            if progress is not None and event_stage:
                last=progress.stream
                progress.emit(event_stage,"completed",completed=last.last_count,total=last.last_total,unit=last.last_unit,
                              cache_status=cache_receipt["status"] if event_stage=="pose" else None)
        finally:
            # Keep the legacy receipt timing snapshot; finalize elapsed is on events.
            if name!="finalize":timings[name]=time.perf_counter()-t
    try:
        with timed("metadata"):
            reader=OpenCVVideoReader(video);metadata=reader.metadata
            config=PoseExtractionConfig();qp=QualityProfile();pp=PreprocessingProfile()
            runtime=runtime_identity();identity=source_identity(video,metadata,runtime,config,qp,pp)
            write_json(out/"source.json",dict(version=VERSION,**DISABLED,identity=identity,
                declared_exercise="BodyWeightSquats",camera_view="UNKNOWN",source_viewpoint="unknown",documented=False))
        with timed("pose"):
            cached=read_cache(cache,identity) if cache is not None else None
            if cached is None:
                if cache is not None:cache_receipt=dict(status="miss",key=artifact_hash(identity),stored=False)
                if progress is not None:progress.emit("pose","progress",completed=0,total=metadata.frame_count,unit="frames",cache_status=cache_receipt["status"])
                inference_attempted=True
                with MediaPipePoseExtractor(metadata.video_id,Exercise.BODY_WEIGHT_SQUATS,viewpoint=Viewpoint.UNKNOWN,config=config) as extractor:
                    def extracted():
                        for i,frame in enumerate(extractor.extract_frames(reader.frames()),1):
                            yield frame  # write_pose_frames writes before iteration resumes.
                            if progress is not None:progress.count("pose",i,metadata.frame_count,"frames")
                    write_pose_frames(out/"raw-pose.jsonl",extracted())
                raw=tuple(PoseFrame.model_validate_json(line,strict=True) for line in (out/"raw-pose.jsonl").read_bytes().splitlines())
                validate_raw(raw,metadata,identity)
                if file_identity(video)!=identity["video"]:
                    raise PipelineError("source_changed","Source bytes changed during pose extraction; cache not committed")
                inference_run=True
                if cache is not None:cache_receipt=write_cache(cache,identity,raw)
            else:
                raw,cache_receipt=cached;write_pose_frames(out/"raw-pose.jsonl",raw)
                if progress is not None:progress.emit("pose","progress",completed=len(raw),total=len(raw),unit="frames",cache_status="hit",message="Pose cache đã kiểm tra; không inference mới")
        with timed("quality"):
            original_quality=quality_for(raw,qp)
            selected=tuple(f.model_copy(update={"viewpoint":Viewpoint.SIDE}) for f in raw) if mode=="exploratory" else raw
            quality=quality_for(selected,qp) if mode=="exploratory" else original_quality
            write_json(out/"quality-source.json",original_quality.model_dump(mode="json"))
            write_json(out/"quality-processing.json",quality.model_dump(mode="json"))
            write_json(out/"processing-metadata.json",dict(mode=mode,viewpoint="side" if assume_side else "unknown",
                assumption_source="caller_cli_assume_side" if assume_side else None,
                assumption_status="caller_assumed_unverified" if assume_side else None,
                documented=False,source_metadata_preserved=True,visual_assessment_performed=False))
        with timed("preprocessing"):
            processed=preprocess_pose(selected,quality,metadata,profile=pp)
            write_preprocessing_result(processed,out/"preprocessing.json")
        with timed("session_geometry"):
            rows,session,parts=observations(processed,raw,side,**({"progress":progress} if progress is not None else {}))
            timings.update(parts)
        with timed("save_validate"):
            source=dict(video_id=metadata.video_id,source_path=str(video),video_sha256=identity["video"]["sha256"],
                raw_pose_sha256=sha(out/"raw-pose.jsonl"),original_file="original"+(video.suffix.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,12}",video.suffix) else ".video"))
            result=make_result(source,mode,side,metadata,processed,rows,session)
            write_json(out/"result.json",result);write_json(out/"coverage.json",result["coverage"])
            with (out/"frames.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
                for i,row in enumerate(rows,1):
                    stream.write(json.dumps(dict(row,source_video_sha256=source["video_sha256"]),ensure_ascii=False,allow_nan=False)+"\n")
                    if progress is not None:progress.count("save",i,len(rows),"jsonl_frames")
            data=video.read_bytes()
            if artifact_hash(identity)!=artifact_hash(source_identity(video,metadata,runtime,config,qp,pp)):
                raise PipelineError("source_changed","Source bytes changed during processing")
            with (out/source["original_file"]).open("xb") as stream:stream.write(data)
        with timed("render_decode"):
            from .real_descriptive_render import render_single_video_bundle
            render_single_video_bundle(out,**({"progress":progress} if progress is not None else {}))
        receipt=dict(version=VERSION,status="completed",**DISABLED,mode=mode,side=side,
            inference_run=inference_run,inference_attempted=inference_attempted,cache=cache_receipt,session_execution=session,timings=timings,
            total_seconds=time.perf_counter()-started,coverage=result["coverage"],result_hash=result["result_hash"],
            browser_qa="NOT_RUN",error=None)
        with timed("finalize"):
            write_json(out/"receipt.json",receipt)
            write_json(out/"manifest.json",dict(version=VERSION,status="completed",**DISABLED,
                source_video_sha256=identity["video"]["sha256"],files={p.name:file_identity(p) for p in out.iterdir() if p.is_file()}))
        return receipt
    except Exception as exc:
        if isinstance(exc,PipelineError) and exc.reason=="cache_rejected":cache_receipt=dict(status="rejected",reason=str(exc))
        error=dict(reason=getattr(exc,"reason","stage_failed"),category=getattr(exc,"category","operational_error"),
            stage=stage,source_tier="S3",exception_type=type(exc).__name__,message=str(exc),
            upstream_flag=exc.flag.model_dump(mode="json") if isinstance(exc,GymError) else None)
        write_json(out/"failure.json",dict(version=VERSION,status="failed",**DISABLED,error=error,inference_run=inference_run,inference_attempted=inference_attempted,
            cache=cache_receipt,timings=timings,total_seconds=time.perf_counter()-started))
        (out/"FAILED.txt").write_text("PIPELINE FAILED — xem failure.json; partial artifacts không phải kết quả hoàn tất.\n",encoding="utf-8")
        raise


def run(video,output,*,side,mode="strict",assume_side=False,cache_dir=None,progress=None):
    if progress is None:return _run(video,output,side=side,mode=mode,assume_side=assume_side,cache_dir=cache_dir)
    progress.emit("pipeline","started",message="Bắt đầu; không có phần trăm ước lượng toàn pipeline")
    try:
        receipt=_run(video,output,side=side,mode=mode,assume_side=assume_side,cache_dir=cache_dir,progress=progress)
    except Exception as exc:
        progress.emit("pipeline","failed",cache_status="rejected" if getattr(exc,"reason",None)=="cache_rejected" else None,
            message="Không hoàn tất. Giữ nguyên partial output và nhật ký.",
            error=dict(reason=getattr(exc,"reason","pipeline_failed"),stage=progress.stream.active or "preflight",
                exception_type=type(exc).__name__,message=str(exc)[:800],evidence_path=str(Path(output).absolute()/"failure.json")))
        raise
    progress.emit("pipeline","completed",cache_status=receipt["cache"]["status"],
        message=f"Đã ghi manifest và artifacts; có số đo {receipt['coverage']['available_frames']}/{receipt['coverage']['total_frames']} frame. Báo cáo: START_HERE.html")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",type=Path);parser.add_argument("--side",choices=("left","right"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=("strict","exploratory"),default="strict")
    parser.add_argument("--assume-side",action="store_true",help="Explicit unverified side hypothesis; exploratory only")
    parser.add_argument("--cache-dir",type=Path,help="Optional immutable content/config-addressed source-pose cache")
    parser.add_argument("--events-jsonl",action="store_true",help="Versioned progress only on stdout; backend diagnostics on stderr")
    parser.add_argument("--run-id",help="Progress correlation ID (default random UUID)")
    args=parser.parse_args(argv)
    if args.events_jsonl:
        from .progress import ProgressEmitter,jsonl_sink
        progress=ProgressEmitter(args.run_id or uuid.uuid4().hex,str(args.output_dir.absolute().resolve()),jsonl_sink(sys.stdout))
        try:
            with redirect_stdout(sys.stderr):
                run(args.video,args.output_dir,side=args.side,mode=args.mode,assume_side=args.assume_side,cache_dir=args.cache_dir,progress=progress)
        except Exception:return 1
        return 0
    try:
        receipt=run(args.video,args.output_dir,side=args.side,mode=args.mode,assume_side=args.assume_side,cache_dir=args.cache_dir)
    except Exception as exc:
        print(json.dumps(dict(ok=False,reason=getattr(exc,"reason","pipeline_failed"),exception_type=type(exc).__name__,message=str(exc)),ensure_ascii=False))
        return 1
    print(json.dumps(dict(ok=True,output=str(args.output_dir),inference_run=receipt["inference_run"],cache=receipt["cache"]["status"],
        coverage=receipt["coverage"],seconds=receipt["total_seconds"]),ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
