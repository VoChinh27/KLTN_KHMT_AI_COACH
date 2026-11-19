"""Freeze a group-spread development sample and export clean native frames.

Reuses saved batch/replay identities and typed validators, never inference.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from . import squat_reference as ref

BATCH = Path("outputs/squat_real_batch_coverage/run-5ed2bf870a594400a25367bdbd18734f")
REPLAY = Path("outputs/squat_context_performance/run-efc12e2dfc244629b5657416ff12ea20")
VIDEO_SCOPE = Path("Data/UCF-101_GYM/BodyWeightSquats")


def file_info(path):
    data=Path(path).read_bytes()
    return dict(sha256=ref.digest(data),size_bytes=len(data))


def select_clips(pairs, limit=12):
    """Metadata-only, one clip/group; missing action coverage first, then path."""
    if not 1 <= limit <= 12:raise ValueError("Sample cap must be 1..12")
    clips={}
    for pair in pairs:
        path=pair["path"]
        match=re.fullmatch(r"v_BodyWeightSquats_(g\d{2})_c\d{2}\.avi",Path(path).name)
        if not match or not Path(path).is_relative_to(VIDEO_SCOPE):raise ValueError("Not a scoped squat clip")
        clip=clips.setdefault(path,dict(path=path,group=match.group(1),pairs=[]))
        clip["pairs"].append(pair)
    for c in clips.values():
        if sorted(p["side"] for p in c["pairs"])!=["left","right"]:
            raise ValueError("Exactly two unique fixed sides per clip")
        c["pairs"].sort(key=lambda p:p["side"])
    selected=[];groups=set();covered=set()
    def add(clip):
        selected.append(clip);groups.add(clip["group"])
        covered.update(p["metric"]["action"] for p in clip["pairs"])
    for action in ("allow","degrade","skip"):
        if action not in covered and len(selected)<limit:
            candidates=[c for _,c in sorted(clips.items()) if c["group"] not in groups
                        and any(p["metric"]["action"]==action for p in c["pairs"])]
            if candidates:add(candidates[0])
    for _,clip in sorted(clips.items()):
        if len(selected)>=limit:break
        if clip["group"] not in groups:add(clip)
    return selected


def sample_indices(count):
    if isinstance(count,bool) or not isinstance(count,int) or count<=0:raise ValueError("Positive frame count required")
    return sorted({(count-1)*k//4 for k in range(5)})


def verify_inventory(root, relative):
    folder=root/relative
    inventory=json.loads((folder/"artifact-inventory.json").read_bytes())
    for name,expected in inventory["run_files"].items():ref.checked_file(folder,name,expected)
    return dict(path=relative.as_posix(),inventory=file_info(folder/"artifact-inventory.json"),
                checked_files=len(inventory["run_files"]))


def prepare(output, *, workspace=None):
    # Imports at execution only: the evaluator remains backend-free.
    from . import context_performance as perf
    from ..features import GeometryFeatureObservation
    from ..features.models import content_hash
    from ..pose.models import PoseFrame
    from ..video import OpenCVVideoReader
    import cv2

    root=Path(workspace or Path.cwd()).resolve()
    out=(root/output).resolve()
    if not out.is_relative_to(root/"outputs"):raise ValueError("New output under workspace/outputs required")
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter()
    verification=[verify_inventory(root,p) for p in (BATCH,REPLAY)]
    plan=json.loads((root/REPLAY/"plan.json").read_bytes())
    jobs,pairs,profile,base=perf.verified_jobs(root,plan)
    opt=root/REPLAY/"optimized"
    replay_manifest=json.loads((opt/"manifest.json").read_bytes())
    for name,expected in replay_manifest["files"].items():ref.checked_file(opt,name,expected)
    for key,value in ref.STATUS.items():
        if replay_manifest[key]!=value:raise ValueError("Replay validation/gates differ")
    selected=select_clips(pairs)
    job_by_path={j["path"]:j for j in jobs}
    # Verify all40 source/config/quality identities before inheriting sampling metadata.
    checked=[]
    for job in jobs:
        result=perf.load_source(job,profile)
        checked.append(dict(path=job["path"],review_id=job["review_id"],frames=len(result.frames),
                            backend=result.raw_frames[0].backend.model_dump(mode="json"),
                            pose_config_hash=result.raw_frames[0].config.content_hash()))
    protocol=root/"docs/SQUAT_2D_REFERENCE_PROTOCOL.md"
    selection=[]
    for clip in selected:
        job=job_by_path[clip["path"]]
        source=json.loads((Path(job["directory"])/"source.json").read_bytes())
        metadata=source["decoder_metadata"]
        selection.append(dict(path=clip["path"],group=clip["group"],review_id=job["review_id"],
            video_id=metadata["video_id"],video_sha256=job["source_video_sha256"],
            width=metadata["width"],height=metadata["height"],fps=metadata["fps"],
            frame_count=metadata["frame_count"],timestamp_policy=metadata["timestamp_policy"],
            frame_indices=sample_indices(metadata["frame_count"]),
            metric_actions={p["side"]:p["metric"]["action"] for p in clip["pairs"]}))
    sampling=dict(version=ref.VERSION,**ref.STATUS,data_kind="real_development",created_at=datetime.now(timezone.utc).isoformat(),
        rule="protocol sampling v1: missing actions allow/degrade/skip then path; one clip/group; <=12; floor((N-1)*k/4), k0..4",
        protocol_version=ref.PROTOCOL,protocol_sha256=ref.digest(protocol.read_bytes()),
        source_batch=BATCH.as_posix(),source_replay=REPLAY.as_posix(),verification=verification,
        input_clip_count=len(jobs),input_pair_count=len(pairs),selection=selection,
        extraction_started=False,evaluation_started=False,reselection_allowed=False)
    out.mkdir(parents=True,exist_ok=False)
    ref.write_json(out/"sampling.json",sampling)  # Immutable pre-decode commitment.
    sampling_hash=ref.digest((out/"sampling.json").read_bytes())
    ref.write_json(out/"source-verification.json",dict(verified=checked,manifests=verification,
        replay_manifest=file_info(opt/"manifest.json"),current_profile_hash=perf.source_hash(profile),
        registry_hash=perf.GEOMETRY_REGISTRY.content_hash(),seconds=time.perf_counter()-start))
    blind=out/"annotation_package";blind.mkdir();(blind/"images").mkdir()
    (blind/"PROTOCOL.md").write_bytes(protocol.read_bytes())
    targets=[];reference_frames=[];model_frames=[];decode=[]
    for chosen in selection:
        job=job_by_path[chosen["path"]];folder=Path(job["directory"])
        video=perf.single.scoped(root,Path(chosen["path"]),VIDEO_SCOPE)
        if ref.digest(video.read_bytes())!=chosen["video_sha256"]:raise ValueError("Video bytes changed")
        raw_rows=[PoseFrame.model_validate_json(line,strict=True) for line in (folder/"raw-pose-unknown.jsonl").read_bytes().splitlines()]
        picked=set(chosen["frame_indices"]);records={}
        for side in ref.SIDES:
            name=f"clips/{job['review_id']}/{side}.jsonl"
            data=ref.checked_file(opt,name,replay_manifest["files"][name])
            lines=data.splitlines();old=(folder/(side+".jsonl")).read_bytes().splitlines()
            if len(lines)!=chosen["frame_count"] or len(old)!=len(lines):raise ValueError("Saved timeline size mismatch")
            records[side]={}
            for i in picked:
                row=json.loads(lines[i]);baseline=json.loads(old[i])
                if row!=baseline:raise ValueError("Replay record differs from verified baseline")
                obs=GeometryFeatureObservation.model_validate_json(json.dumps(row["observation"]),strict=True)
                if (row["side"]!=side or row["frame_index"]!=i or row["timestamp_seconds"]!=i/chosen["fps"] or
                        row["video_id"]!=chosen["video_id"] or row["source_video_sha256"]!=chosen["video_sha256"] or
                        obs.feature_id!=f"squat_knee_angle_{side}_deg" or obs.frame_index!=i or
                        content_hash(obs)!=row["source_linkage"]["observation_hash"]):
                    raise ValueError("Saved numerical observation/linkage mismatch")
                records[side][i]=(row,obs,file_info(opt/name))
        reader=OpenCVVideoReader(video)
        if (reader.metadata.width,reader.metadata.height,reader.metadata.frame_count,reader.metadata.fps)!=(
                chosen["width"],chosen["height"],chosen["frame_count"],chosen["fps"]):
            raise ValueError("Native decoder metadata differs")
        count=0
        for frame in reader.frames():
            count+=1
            if frame.frame_index not in picked:continue
            i=frame.frame_index;sample=f"{job['review_id']}-f{i:06d}";name=f"images/{sample}.png"
            if frame.image.shape[:2]!=(chosen["height"],chosen["width"]):raise ValueError("Native dimensions changed")
            ok,encoded=cv2.imencode(".png",frame.image)
            if not ok:raise ValueError("PNG encoding failed")
            with (blind/name).open("xb") as stream:stream.write(encoded.tobytes())
            decoded=cv2.imread(str(blind/name),cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.shape!=frame.image.shape or not (decoded==frame.image).all():
                raise ValueError("PNG not lossless native frame")
            identity=ref.FrameIdentity(sample_id=sample,video_id=chosen["video_id"],video_path=chosen["path"],
                video_sha256=chosen["video_sha256"],frame_index=i,timestamp_seconds=frame.timestamp_seconds,
                width=chosen["width"],height=chosen["height"],image_sha256=ref.digest(encoded.tobytes()))
            targets.append(dict(identity=identity.model_dump(mode="json"),image_path=name,group=chosen["group"],
                fps=chosen["fps"],frame_count=chosen["frame_count"],timestamp_policy=chosen["timestamp_policy"]))
            reference_frames.append(ref.ReferenceFrame(identity=identity,annotation_status="pending",
                landmarks={j:ref.LandmarkLabel(state="pending",reason="annotation_pending") for j in ref.JOINTS}))
            raw=raw_rows[i]
            if raw.frame_index!=i or raw.timestamp_seconds!=frame.timestamp_seconds or raw.video_id!=chosen["video_id"]:
                raise ValueError("Raw pose/native timeline mismatch")
            processed={}
            for side in ref.SIDES:
                row,obs,record_file=records[side][i]
                gate=obs.availability.input.quality.metric_eligibility[obs.feature_id]
                processed[side]=dict(feature_id=obs.feature_id,side=side,status=str(obs.status),value=obs.value,
                    unit=str(obs.unit),gate_action=str(gate.action),sequence_action=str(obs.availability.input.quality.eligibility.action),
                    reasons=[str(x) for x in obs.reasons],flags=[f.model_dump(mode="json") for f in obs.flags],
                    joint_inputs=[p.model_dump(mode="json") for p in obs.joint_inputs],
                    source_linkage=row["source_linkage"],source_file=(REPLAY/"optimized"/f"clips/{job['review_id']}/{side}.jsonl").as_posix(),
                    source_file_identity=record_file)
            model_frames.append(dict(identity=identity.model_dump(mode="json"),
                raw_joints={str(j.joint):j.model_dump(mode="json") for j in raw.joints if str(j.joint) in ref.JOINTS},
                processed=processed,raw_pose_file=(BATCH/"batch/clips"/job["review_id"]/"raw-pose-unknown.jsonl").as_posix(),
                raw_pose_file_identity=file_info(folder/"raw-pose-unknown.jsonl"),
                raw_frame_hash=content_hash(raw),backend=raw.backend.model_dump(mode="json"),pose_config_hash=raw.config.content_hash()))
        if count!=chosen["frame_count"]:raise ValueError("Full decode frame count differs")
        if ref.digest(video.read_bytes())!=chosen["video_sha256"]:raise ValueError("Video changed during decode")
        decode.append(dict(path=chosen["path"],decoded_frames=count,exported=len(picked),native_lossless_pixels=True))
    manifest=dict(version=ref.VERSION,**ref.STATUS,data_kind="real_development",protocol_version=ref.PROTOCOL,
        protocol_sha256=sampling["protocol_sha256"],sampling_sha256=sampling_hash,samples=targets)
    ref.write_json(blind/"manifest.json",manifest)
    annotation=ref.ReferenceDocument(protocol_sha256=manifest["protocol_sha256"],sampling_sha256=sampling_hash,frames=tuple(reference_frames))
    ref.write_json(blind/"annotations.blank.json",annotation.model_dump(mode="json"))
    evaluation=out/"evaluation_inputs";evaluation.mkdir()
    ref.write_json(evaluation/"predictions.json",dict(version=ref.VERSION,**ref.STATUS,
        manifest_hash=ref.canonical_hash(manifest),frames=model_frames))
    ref.write_json(out/"decode-receipt.json",dict(clips=decode,total_frames=sum(x["decoded_frames"] for x in decode),
        exported_images=len(targets),sampling_sha256=sampling_hash,inference_run=False,seconds=time.perf_counter()-start))
    guide=("GOI GAN NHAN MU: chi gui annotation_package.\n"
           "Doc PROTOCOL.md. Anh PNG nguyen kich thuoc; khong skeleton/predictions/gates.\n"
           "Sao chep annotations.blank.json thanh file moi truoc khi dien. Khong sua template/anh/manifest.\n"
           "Ben trai/phai giai phau; pixel anh goc. Khong thay uncertain/occluded bang toa do doan.\n"
           "Chua co nhan => REFERENCE_PENDING, sai so null, khong la accuracy=0.\n"
           "NOT_VALIDATED; official=false; assessment=false; G5.7 blocked; G6 IN_PROGRESS.\n")
    (blind/"HUONG_DAN.txt").write_text(guide,encoding="utf-8")
    ref.write_json(out/"package-integrity.json",dict(version=ref.VERSION,**ref.STATUS,
        files={p.relative_to(out).as_posix():file_info(p) for folder in (blind,evaluation) for p in folder.rglob("*") if p.is_file()},
        sampling=file_info(out/"sampling.json"),source_verification=file_info(out/"source-verification.json")))
    return dict(clip_count=len(selected),sample_count=len(targets),metric_actions=dict(Counter(
        action for c in selection for action in c["metric_actions"].values())),groups=[c["group"] for c in selection])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.output_dir)))


if __name__=="__main__":main()
