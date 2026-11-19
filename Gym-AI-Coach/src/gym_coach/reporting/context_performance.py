"""Controlled saved-source replay: identical reference/session work, no inference.

Session execution metadata lives in receipts; every numerical record and every
field of its reference linkage is compared, with nothing removed or tolerated.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from ..features import FeatureContextSession, GEOMETRY_REGISTRY, GeometryFeatureObservation
from ..features.models import content_hash
from ..pose.models import PoseFrame
from ..preprocessing import PreprocessingResult, load_preprocessing_profile
from ..preprocessing.models import content_hash as source_hash
from . import real_batch_coverage as batch
from . import real_descriptive_demo as single

VERSION = "squat-context-performance-v1"


def memory_usage():
    """Windows working-set samples, not allocator/session attribution or CPU time."""
    if os.name != "nt":
        return dict(available=False, reason="Windows working-set API unavailable")
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters();c.cb = ctypes.sizeof(c)
    if not fn(process(), ctypes.byref(c), c.cb):
        return dict(available=False, reason="GetProcessMemoryInfo failed")
    return dict(available=True, working_set_bytes=c.WorkingSetSize,
                process_peak_working_set_bytes=c.PeakWorkingSetSize)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    data = Path(path).read_bytes()
    if len(data) != expected["size_bytes"] or digest_bytes(data) != expected["sha256"]:
        raise ValueError(f"Saved artifact hash/size mismatch: {path}")
    return data


@contextmanager
def count_full_validations():
    """Measure actual full-source strict intakes, including Linked after-validator."""
    original = PreprocessingResult.model_validate_json
    counter = Counter()
    def tracked(*args, **kwargs):
        counter["full_source_validations"] += 1
        return original(*args, **kwargs)
    with patch.object(PreprocessingResult, "model_validate_json", new=tracked):
        yield counter


def load_source(job, profile):
    folder = Path(job["directory"])
    def read(name):
        return read_verified(folder/name, job["files"][name])
    processed = PreprocessingResult.model_validate_json(read("preprocessing.json"), strict=True)
    info = json.loads(read("source.json"))
    if (processed.profile != profile or info["decoder_metadata"] != processed.metadata.model_dump(mode="json")
            or info["candidate"]["path"] != job["path"]
            or info["candidate"]["sha256"] != job["source_video_sha256"]
            or info["camera_view"] != "UNKNOWN" or info["assumption_status"] != "visually_assessed_unverified"):
        raise ValueError("Source/profile/metadata identity mismatch")
    raw = [PoseFrame.model_validate_json(line, strict=True) for line in read("raw-pose-unknown.jsonl").splitlines()]
    if len(raw) != len(processed.frames):raise ValueError("Raw source frame count mismatch")
    for original, assumed in zip(raw, processed.raw_frames):
        a,b=original.model_dump(mode="json"),assumed.model_dump(mode="json")
        if a.pop("viewpoint")!="unknown" or b.pop("viewpoint")!="side" or a!=b:
            raise ValueError("Only the explicit viewpoint hypothesis may differ")
    if [p["side"] for p in job["pairs"]] != ["left","right"]:
        raise ValueError("Exactly two ordered, independent sides required")
    for pair,key in zip(job["pairs"],single.KEYS):
        metric=processed.quality.metric_eligibility.get(key)
        count=sum(all(single.observed_in_frame(j) for j in f.joints if j.joint in GEOMETRY_REGISTRY.get(key).required_joints) for f in raw)
        if (pair["path"]!=job["path"] or pair["feature_id"]!=key or pair["total_frames"]!=len(raw)
                or metric is None or metric.required_joints!=GEOMETRY_REGISTRY.get(key).required_joints
                or metric.model_dump(mode="json")!=pair["metric"]
                or processed.quality.eligibility.model_dump(mode="json")!=pair["sequence"]
                or count!=pair["observed_valid_frames"]):
            raise ValueError("Pair/gate/raw-coverage linkage mismatch")
    return processed


def replay_clip(job, output, mode, profile):
    """Same work on both paths; no baseline comparison inside timed scope."""
    if mode not in ("reference","session"):raise ValueError("Unknown context mode")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    gc.collect();memory_before=memory_usage();start=time.perf_counter()
    stats=[batch.base_stat(p) for p in job["pairs"]]
    timings=dict(load_validation_seconds=0.,session_setup_seconds=0.,context_seconds=0.,
                 adapter_seconds=0.,output_seconds=0.)
    session_metadata=None
    with count_full_validations() as validations:
        t=time.perf_counter();processed=load_source(job,profile)
        timings["load_validation_seconds"]=time.perf_counter()-t
        load_count=validations["full_source_validations"]
        t=time.perf_counter()
        context_source=FeatureContextSession(processed) if mode=="session" else processed
        if mode=="session":session_metadata=context_source.metadata
        timings["session_setup_seconds"]=time.perf_counter()-t
        setup_count=validations["full_source_validations"]-load_count
        memory_after_setup=memory_usage()
        streams=[(output/(side+".jsonl")).open("x",encoding="utf-8",newline="\n") for side in ("left","right")]
        try:
            for frame in processed.frames:
                t=time.perf_counter();context,link=batch._frame_input(context_source,frame.frame_index)
                timings["context_seconds"]+=time.perf_counter()-t
                for stream,stat in zip(streams,stats):
                    t=time.perf_counter();obs=batch.observe_geometry_feature(stat["feature_id"],context)
                    if obs.availability.input_hash!=link["measurement_input_hash"]:
                        raise ValueError("Measurement input identity changed")
                    timings["adapter_seconds"]+=time.perf_counter()-t
                    t=time.perf_counter()
                    record=dict(video_id=frame.video_id,frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,side=stat["side"],
                        source_video_sha256=job["source_video_sha256"],
                        viewpoint_evidence="visually_assessed_unverified",camera_view="UNKNOWN",
                        source_linkage=dict(link,observation_hash=content_hash(obs)),
                        observation=obs.model_dump(mode="json"))
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
                    batch.account(stat,obs)
                    timings["output_seconds"]+=time.perf_counter()-t
        finally:
            t=time.perf_counter()
            for stream in streams:stream.close()
            timings["output_seconds"]+=time.perf_counter()-t
        context_count=validations["full_source_validations"]-load_count-setup_count
        validation_count=validations["full_source_validations"]
    timings["total_wall_seconds"]=time.perf_counter()-start
    expected_context=2*len(processed.frames) if mode=="reference" else 0
    if load_count!=1 or setup_count!=int(mode=="session") or context_count!=expected_context:
        raise AssertionError("Unexpected full-source validation scope")
    for stat in stats:stat["processing_status"]="completed"
    receipt=dict(version=VERSION,mode=mode,pid=os.getpid(),**single.STATUS,
        frame_count=len(processed.frames),factory_calls=len(processed.frames),adapter_calls=2*len(processed.frames),
        validation_counts=dict(load=load_count,session_setup=setup_count,per_frame=context_count,total=validation_count),
        timing=timings,memory_before=memory_before,memory_after_setup=memory_after_setup,memory_after=memory_usage(),
        session_metadata=session_metadata,source_preprocessing_sha256=job["files"]["preprocessing.json"]["sha256"],
        preprocessing_result_hash=source_hash(processed),baseline_directory=job["directory"],
        numerical_schema_unchanged=True,session_metadata_scope="receipt only; never numerical records")
    single.write_json(output/"receipt.json",receipt);single.write_json(output/"stats.json",dict(pairs=stats))
    return dict(pairs=stats,receipt=receipt)


def compare_records(job, output, *, validate_models=False):
    """Full record equality, including every observation/provenance/linkage field."""
    output=Path(output);matched=0;mismatches=[]
    for side in ("left","right"):
        old=read_verified(Path(job["directory"])/(side+".jsonl"),job["files"][side+".jsonl"])
        new=(output/(side+".jsonl")).read_bytes()
        old_rows=old.splitlines();new_rows=new.splitlines()
        if len(old_rows)!=len(new_rows) or len(old_rows)!=job["pairs"][0]["total_frames"]:
            raise ValueError("Comparison cannot drop or add observations")
        for a,b in zip(old_rows,new_rows):
            reference,optimized=json.loads(a),json.loads(b)
            if validate_models:
                for row in (reference,optimized):
                    obs=GeometryFeatureObservation.model_validate_json(json.dumps(row["observation"]),strict=True)
                    if content_hash(obs)!=row["source_linkage"]["observation_hash"]:
                        raise ValueError("Observation hash mismatch during strict readback")
            # Canonical comparison includes all fields, values and numerical
            # JSON representations. No exclusions/tolerances or status-only test.
            encode=lambda row:json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
            if encode(reference)==encode(optimized):matched+=1
            else:mismatches.append(dict(side=side,frame_index=reference.get("frame_index"),
                reference_hash=digest_bytes(encode(reference).encode()),optimized_hash=digest_bytes(encode(optimized).encode())))
    result=dict(matched=matched,mismatch_count=len(mismatches),mismatches=mismatches,
        comparison="complete canonical JSON record; no excluded fields or tolerance",
        strict_numeric_readback=validate_models)
    single.write_json(output/"comparison.json",result)
    if mismatches:raise ValueError(f"Baseline mismatch: {len(mismatches)} observations; preserved separately")
    return result


def verified_jobs(root, plan):
    base=single.scoped(root,Path(plan["baseline"]),Path("outputs"))
    if single.sha(base/"artifact-inventory.json")!=plan["baseline_inventory_sha256"]:
        raise ValueError("Baseline inventory identity changed")
    inv=json.loads((base/"artifact-inventory.json").read_bytes())
    read_verified(base/"batch/manifest.json",inv["run_files"]["batch/manifest.json"])
    if single.sha(base/"batch/manifest.json")!=plan["batch_manifest_sha256"]:raise ValueError("Baseline manifest changed")
    manifest=json.loads((base/"batch/manifest.json").read_bytes())
    for name,expected in manifest["files"].items():
        p=single.scoped(root,(base/"batch"/name).relative_to(root),base.relative_to(root))
        read_verified(p,expected)
    selection=json.loads((base/"batch/selection.json").read_bytes())
    if single.sha(base/"batch/selection.json")!=manifest["selection_sha256"]:raise ValueError("Baseline selection changed")
    profile=load_preprocessing_profile(root/"configs/preprocessing_s3_v1.toml")
    if selection["registry_hash"]!=GEOMETRY_REGISTRY.content_hash() or selection["profile_hash"]!=source_hash(profile):
        raise ValueError("Current registry/profile differs from frozen baseline")
    if any(selection[k]!=v for k,v in single.STATUS.items()):raise ValueError("Disabled gates changed")
    pairs=selection["pairs"];jobs=[]
    if len(pairs)!=80 or len({p['path'] for p in pairs})!=40:raise ValueError("Expected frozen40/80 scope")
    for source_file in sorted((base/"batch/clips").glob("*/source.json")):
        info=json.loads(source_file.read_bytes());folder=source_file.parent;path=info["candidate"]["path"]
        if selection["source_hashes"][path]!=info["candidate"]["sha256"]:raise ValueError("Source video hash linkage changed")
        local={n:manifest['files'][(folder/n).relative_to(base/'batch').as_posix()] for n in
            ('source.json','raw-pose-unknown.jsonl','preprocessing.json','left.jsonl','right.jsonl')}
        jobs.append(dict(directory=str(folder),path=path,review_id=folder.name,
                         source_video_sha256=info["candidate"]["sha256"],files=local,
                         pairs=sorted([p for p in pairs if p["path"]==path],key=lambda p:p['side'])))
    if len(jobs)!=40 or len({j["path"] for j in jobs})!=40:raise ValueError("Missing/duplicate baseline clip")
    return jobs,pairs,profile,base


def benchmark_case(job, dest, profile, warmups, repeats):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);results=[]
    for mode in ("reference","session"):
        for n in range(warmups):
            out=dest/f"warmup-{n}-{mode}";replay_clip(job,out,mode,profile);compare_records(job,out)
            print(json.dumps(dict(case=job['path'],warmup=mode,index=n)),flush=True)
    for n in range(repeats):
        for mode in (("reference","session") if n%2==0 else ("session","reference")):
            out=dest/f"repeat-{n}-{mode}";run=replay_clip(job,out,mode,profile);compare_records(job,out)
            results.append(run['receipt'])
            print(json.dumps(dict(case=job['path'],repeat=n,mode=mode,seconds=run['receipt']['timing']['total_wall_seconds'])),flush=True)
    medians={m:{key:statistics.median(r['timing'][key] for r in results if r['mode']==m)
                for key in results[0]['timing']} for m in ('reference','session')}
    result=dict(path=job['path'],frame_count=job['pairs'][0]['total_frames'],worker_count=1,
        warmups_per_mode=warmups,repeats_per_mode=repeats,order='alternating after warmups',
        medians=medians,speedup=medians['reference']['total_wall_seconds']/medians['session']['total_wall_seconds'],runs=results)
    single.write_json(dest/'summary.json',result)
    return result


def batch_worker(job,dest,profile):
    run=replay_clip(job,dest,'session',profile)
    t=time.perf_counter();comparison=compare_records(job,dest,validate_models=True)
    return dict(**run,comparison=comparison,comparison_seconds=time.perf_counter()-t,path=job['path'])


def run(plan_path,output_path,action,*,workspace=None):
    root=Path(workspace or Path.cwd()).resolve();plan_path=single.scoped(root,plan_path,Path('outputs'))
    out=single.scoped(root,output_path,Path('outputs'))
    if out.exists():raise FileExistsError(out)
    plan=json.loads(plan_path.read_bytes());started=time.perf_counter()
    jobs,pairs,profile,base=verified_jobs(root,plan);verification_seconds=time.perf_counter()-started
    out.mkdir(parents=True,exist_ok=False)
    single.write_json(out/'input-verification.json',dict(status='PASSED',manifest_files=452,
        plan_sha256=single.sha(plan_path),baseline_manifest_sha256=single.sha(base/'batch/manifest.json'),
        verification_seconds=verification_seconds,**single.STATUS))
    if action=='benchmark':
        results={}
        with ProcessPoolExecutor(max_workers=plan['benchmark_workers']) as pool:
            for label,case in plan['cases'].items():
                job=next(j for j in jobs if j['path']==case['path'])
                if job['pairs'][0]['total_frames']!=case['total_frames']:raise ValueError('Frozen benchmark frame count mismatch')
                results[label]=pool.submit(benchmark_case,job,str(out/label),profile,
                    plan['warmups_per_case_mode'],plan['measured_repeats']).result()
        result=dict(version=VERSION,**single.STATUS,benchmark=results,pytest_overlap=False,
                    wall_seconds=time.perf_counter()-started,old_767_seconds_used_as_control=False)
    elif action=='batch':
        measured=[];receipts=[];comparisons=[];failures=[];t=time.perf_counter()
        with ProcessPoolExecutor(max_workers=plan['batch_workers']) as pool:
            futures={pool.submit(batch_worker,j,str(out/'clips'/j['review_id']),profile):j for j in jobs}
            for n,future in enumerate(as_completed(futures),1):
                job=futures[future]
                try:
                    done=future.result();measured.extend(done['pairs']);receipts.append(done['receipt']);comparisons.append(done['comparison'])
                    print(json.dumps(dict(progress=f'{n}/40',path=job['path'],matched=done['comparison']['matched'])),flush=True)
                except Exception as exc:
                    failures.append(dict(path=job['path'],type=type(exc).__name__,message=str(exc)))
                    for pair in job['pairs']:
                        stat=batch.base_stat(pair);stat.update(processing_status='processing_error',error=str(exc));measured.append(stat)
                    print(json.dumps(dict(progress=f'{n}/40',error=str(exc),path=job['path'])),flush=True)
        replay_compare_wall=time.perf_counter()-t
        summary=batch.aggregate(pairs,measured);single.write_json(out/'coverage.json',summary)
        original=json.loads((base/'batch/summary.json').read_bytes())
        counts=Counter()
        for r in receipts:counts.update(r['validation_counts'])
        result=dict(version=VERSION,**single.STATUS,worker_count=plan['batch_workers'],receipts=receipts,
            failures=failures,comparison=dict(matched=sum(c['matched'] for c in comparisons),
                mismatches=sum(c['mismatch_count'] for c in comparisons),complete_records=True,no_excluded_fields=True,
                summary_equal=summary==original),full_source_validations=dict(counts),
            replay_and_strict_compare_wall_seconds=replay_compare_wall,total_wall_seconds=time.perf_counter()-started,
            inference_run=False,render_run=False)
        single.write_json(out/'summary.json',result)
        if failures or result['comparison']['matched']!=10628 or result['comparison']['mismatches'] or summary!=original:
            raise ValueError('Full batch equivalence incomplete; see preserved failures/receipts')
    else:raise ValueError('Unknown action')
    if action=='benchmark':single.write_json(out/'summary.json',result)
    files={p.relative_to(out).as_posix():dict(sha256=single.sha(p),size_bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file()}
    single.write_json(out/'manifest.json',dict(version=VERSION,**single.STATUS,files=files,
        baseline_manifest_sha256=single.sha(base/'batch/manifest.json'),inference_run=False,render_run=False))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--action',choices=('benchmark','batch'),required=True)
    args=p.parse_args(argv);result=run(args.plan,args.output_dir,args.action)
    print(json.dumps(dict(status='COMPLETED',action=args.action,comparison=result.get('comparison'))),flush=True)


if __name__=='__main__':main()
