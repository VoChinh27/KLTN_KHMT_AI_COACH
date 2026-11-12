"""Coverage of the frozen 40-clip development pool, independently per knee.

Reuses saved poses/quality, current Stage4/factory/geometry adapters and the
existing video renderer. No MediaPipe, temporal analysis, or threshold changes.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
import os
from pathlib import Path
import time
import traceback

from ..features import FeatureContextSession, FeatureInputContext, GeometryFeatureObservation, build_feature_input_context, observe_geometry_feature
from ..features.models import content_hash
from ..preprocessing import load_preprocessing_profile, preprocess_pose, read_preprocessing_result, write_preprocessing_result
from ..preprocessing.models import content_hash as preprocessing_hash
from . import real_descriptive_demo as single

VERSION = "squat-real-batch-coverage-v1"
SAMPLE_RULE = "For each existing metric action allow/degrade/skip, first relative video path then left/right; whole clip; no reselection from numeric results or rendered appearance."


def _frame_input(processed, index):
    """Same frame/report and identity projection; session execution is separate.

    Reference behavior stays unchanged. A session is owned by this clip/worker,
    never supplied as a caller-declared validation flag or serialized cache.
    """
    if isinstance(processed, FeatureContextSession):
        selected = processed.select(frame_index=index)
        return selected.input_context(), selected.reference_linkage()
    linked = build_feature_input_context(processed, frame_index=index)
    context = FeatureInputContext(frame=linked.frame, quality=linked.quality)
    link = linked.model_dump(mode="json", exclude={"frame", "quality", "preprocessing_result"})
    link.update(frame_index=index, timestamp_seconds=linked.frame.timestamp_seconds,
                measurement_input_hash=linked.measurement_input_hash)
    return context, link


def select_examples(pairs):
    return {action: next((p for p in sorted(pairs, key=lambda p: (p["path"], ("left", "right").index(p["side"])))
                          if p["metric"]["action"] == action), None)
            for action in ("allow", "degrade", "skip")}


def base_stat(pair):
    return dict(path=pair["path"], side=pair["side"], feature_id=pair["feature_id"],
        total_frames=pair["total_frames"], metric_action=pair["metric"]["action"],
        metric_reasons=pair["metric"]["reasons"], sequence_action=pair["sequence"]["action"],
        raw_observed_in_frame=pair["observed_valid_frames"], raw_ratio=pair["observed_valid_ratio"],
        evaluated_frames=0, available_frames=0, unavailable_frames=0, processing_status="pending",
        reasons={}, input_status={}, input_source_status={}, available_input_status={})


def account(stat, observation):
    stat["evaluated_frames"] += 1
    stat[observation.status + "_frames"] += 1
    for name, values in (("reasons", tuple(dict.fromkeys(code.value for code in observation.reasons))),
                         ("input_status", (str(j.status) for j in observation.joint_inputs)),
                         ("input_source_status", (str(j.source_status) for j in observation.joint_inputs)),
                         ("available_input_status", (str(j.status) for j in observation.joint_inputs
                                                     if observation.status == "available"))):
        counts = Counter(stat[name]); counts.update(values); stat[name] = dict(counts)


def aggregate(expected_pairs, measured, *, expected_clips=40):
    """Full denominators remain, including processing failures/unmeasured frames."""
    expected = {(p["path"], p["side"]): p for p in expected_pairs}
    paths = {p["path"] for p in expected_pairs}
    if len(expected) != 2 * expected_clips or len(paths) != expected_clips:
        raise ValueError("Frozen batch must contain two unique sides for every expected clip")
    if any({s for path, s in expected if path == p} != {"left", "right"} for p in paths):
        raise ValueError("Missing fixed-side pair")
    by_key = {(p["path"], p["side"]): p for p in measured}
    if len(by_key) != len(measured) or set(by_key) - set(expected):
        raise ValueError("Duplicate/unexpected measured pair")
    complete = []
    for key, pair in expected.items():
        stat = by_key.get(key)
        if stat is None:
            stat = base_stat(pair)
            stat.update(processing_status="processing_error", error="Pair result absent; retained as unevaluated")
        if (stat["total_frames"] != pair["total_frames"] or stat["feature_id"] != pair["feature_id"]
                or stat["metric_action"] != pair["metric"]["action"]
                or stat["available_frames"] + stat["unavailable_frames"] != stat["evaluated_frames"]
                or min(stat["available_frames"], stat["unavailable_frames"]) < 0
                or not 0 <= stat["evaluated_frames"] <= stat["total_frames"]
                or (stat["processing_status"] == "completed" and stat["evaluated_frames"] != stat["total_frames"])):
            raise ValueError("Measured pair counts/identity contradict frozen denominator")
        complete.append(stat)
    def metrics(items):
        n = sum(p["total_frames"] for p in items)
        available = sum(p["available_frames"] for p in items)
        raw = sum(p["raw_observed_in_frame"] for p in items)
        evaluated = sum(p["evaluated_frames"] for p in items)
        def sum_counts(key):
            counts = Counter()
            for p in items: counts.update(p[key])
            return dict(counts)
        return dict(pair_count=len(items), frame_pair_denominator=n, evaluated_frame_pairs=evaluated,
            unevaluated_frame_pairs=n - evaluated, available=available,
            unavailable=sum(p["unavailable_frames"] for p in items), raw_observed_in_frame=raw,
            numeric_micro=available/n if n else None, raw_micro=raw/n if n else None,
            numeric_macro=sum(p["available_frames"]/p["total_frames"] for p in items)/len(items),
            raw_macro=sum(p["raw_ratio"] for p in items)/len(items),
            metric_actions=dict(Counter(p["metric_action"] for p in items)),
            processing_errors=sum(p["processing_status"] != "completed" for p in items),
            unavailable_reasons=sum_counts("reasons"), input_status=sum_counts("input_status"),
            input_source_status=sum_counts("input_source_status"),
            available_input_status=sum_counts("available_input_status"))
    clips = []
    for path in sorted(paths):
        rows = [next(p for p in complete if p["path"] == path and p["side"] == side) for side in ("left", "right")]
        if rows[0]["total_frames"] != rows[1]["total_frames"]:
            raise ValueError("Sides must use the same whole-clip frame denominator")
        clips.append(dict(path=path, total_frames=rows[0]["total_frames"], left=rows[0], right=rows[1],
                          mean_side_numeric_ratio=sum(p["available_frames"]/p["total_frames"] for p in rows)/2,
                          mean_side_raw_ratio=sum(p["raw_ratio"] for p in rows)/2))
    return dict(batch_version=VERSION, **single.STATUS, camera_view="UNKNOWN",
        viewpoint_evidence="visually_assessed_unverified", documented=False, clip_count=len(clips),
        source_frames=sum(c["total_frames"] for c in clips), total=metrics(complete),
        by_side={side: metrics([p for p in complete if p["side"] == side]) for side in ("left", "right")},
        clip_macro_numeric=sum(c["mean_side_numeric_ratio"] for c in clips)/len(clips),
        clip_macro_raw=sum(c["mean_side_raw_ratio"] for c in clips)/len(clips),
        clips_with_any_numeric=sum(c["left"]["available_frames"] > 0 or c["right"]["available_frames"] > 0 for c in clips),
        sequence_actions=dict(Counter(expected[(c["path"], "left")]["sequence"]["action"] for c in clips)),
        clips=clips, pairs=complete, numeric_scope="independent fixed side, framewise projected2D",
        gate_scope="per-clip metric decisions, never promoted to per-frame gates",
        denominator_note="source_frames counts each frame once; frame_pair_denominator counts it once per side. Macro averages all40 clips equally, mean of the two side ratios; no angle fusion. Unprocessed frames stay in denominator, separately unevaluated.")


def _process_clip(root, output, source, pairs, profile, prior_preprocessing=None):
    """One worker/clip; one validated factory context per frame, two adapters."""
    root, output = Path(root), Path(output)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    stats = [base_stat(p) for p in pairs]
    timing = {}
    try:
        for name in ("raw-pose-unknown.jsonl", "pose-assumed-side.jsonl", "quality-assumed-side.json", "quality-unknown.json", "source.json"):
            with (output / name).open("xb") as stream: stream.write(source["payloads"][name])
        t = time.perf_counter()
        if prior_preprocessing is not None:
            processed = read_preprocessing_result(prior_preprocessing)
            if (processed.raw_frames != source["assumed"] or processed.quality != source["quality"]
                    or processed.metadata != source["metadata"] or processed.profile != profile):
                raise ValueError("Prior preprocessing differs from validated source/profile")
            with (output / "preprocessing.json").open("xb") as stream: stream.write(Path(prior_preprocessing).read_bytes())
            preprocessing_mode = "reused_validated_prior_demo"
        else:
            processed = preprocess_pose(source["assumed"], source["quality"], source["metadata"], profile=profile)
            write_preprocessing_result(processed, output / "preprocessing.json")
            preprocessing_mode = "new_once_for_both_sides"
        timing["preprocessing_seconds"] = time.perf_counter() - t
        before = preprocessing_hash(processed)
        factory_seconds = adapter_seconds = 0.
        streams = [(output / f"{side}.jsonl").open("x", encoding="utf-8", newline="\n") for side in ("left", "right")]
        try:
            for frame in processed.frames:
                t = time.perf_counter()
                context, link = _frame_input(processed, frame.frame_index)
                input_hash = link["measurement_input_hash"]
                factory_seconds += time.perf_counter() - t
                for stream, stat in zip(streams, stats):
                    t = time.perf_counter()
                    obs = observe_geometry_feature(stat["feature_id"], context)
                    if obs.availability.input_hash != input_hash:
                        raise ValueError("Factory/observation input linkage mismatch")
                    adapter_seconds += time.perf_counter() - t
                    record = dict(video_id=frame.video_id, frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds, side=stat["side"],
                        source_video_sha256=source["candidate"]["sha256"],
                        viewpoint_evidence="visually_assessed_unverified", camera_view="UNKNOWN",
                        source_linkage=dict(link, observation_hash=content_hash(obs)),
                        observation=obs.model_dump(mode="json"))
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    account(stat, obs)
        finally:
            for stream in streams: stream.close()
        if preprocessing_hash(processed) != before:
            raise ValueError("Batch mutated source preprocessing/quality")
        timing.update(factory_seconds=factory_seconds, adapter_seconds=adapter_seconds,
                      factory_calls=len(processed.frames), adapter_calls=2*len(processed.frames),
                      preprocessing_mode=preprocessing_mode)
        for stat in stats: stat["processing_status"] = "completed"
    except Exception as exc:  # Isolated worker boundary; errors remain in all denominators.
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        single.write_json(output / "processing-error.json", error)
        for stat in stats: stat.update(processing_status="processing_error", error=error)
    timing["worker_elapsed_seconds"] = time.perf_counter() - started
    single.write_json(output / "stats.json", dict(pairs=stats, timing=timing))
    hashes = {p.name: dict(sha256=single.sha(p), size_bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    single.write_json(output / "manifest.json", dict(files=hashes, **single.STATUS,
        source_sha256=source["candidate"]["sha256"], inference_run=False))
    return dict(pairs=stats, timing=timing, review_id=source["candidate"]["review_id"])


def validate_prior_preprocessing(root, prior_relative, loaded):
    prior = single.scoped(root, prior_relative, Path("outputs")) / "demo"
    manifest = json.loads((prior / "manifest.json").read_bytes())
    result_entry = manifest["files"]["preprocessing.json"]
    path = prior / "preprocessing.json"
    if single.sha(path) != result_entry["sha256"] or path.stat().st_size != result_entry["size_bytes"]:
        raise ValueError("Prior demo preprocessing hash mismatch")
    selected_entry = manifest["files"]["selection.json"]
    if single.sha(prior / "selection.json") != selected_entry["sha256"]:
        raise ValueError("Prior demo selection hash mismatch")
    selected = json.loads((prior / "selection.json").read_bytes())
    if selected["path"] not in loaded or selected["sha256"] != loaded[selected["path"]]["candidate"]["sha256"]:
        raise ValueError("Prior demo is not linked to the frozen batch pool")
    return {selected["path"]: str(path)}


def materialize_example(root, batch, pair, source, selection_hash):
    """Use the existing v1 artifact/renderer; only reads batch numeric outputs."""
    clip = batch / "clips" / source["candidate"]["review_id"]
    processed = read_preprocessing_result(clip / "preprocessing.json")
    records = [json.loads(line) for line in (clip / (pair["side"] + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    if len(records) != pair["total_frames"]:
        raise ValueError("Cannot render incomplete pair; no reselection allowed")
    observations = []; links = []
    for i, record in enumerate(records):
        obs = GeometryFeatureObservation.model_validate_json(json.dumps(record["observation"]), strict=True)
        if (record["side"] != pair["side"] or obs.feature_id != pair["feature_id"]
                or obs.availability.input.frame != processed.frames[i]
                or obs.availability.input.quality != processed.quality
                or content_hash(obs) != record["source_linkage"]["observation_hash"]):
            raise ValueError("Saved pair/frame linkage mismatch")
        observations.append(obs); links.append(record["source_linkage"])
    action = pair["metric"]["action"]
    dest = batch / "examples" / action
    dest.mkdir(parents=True, exist_ok=False)
    receipt = dict(**pair, video_id=source["metadata"].video_id, sha256=source["candidate"]["sha256"],
        selection_version="squat-batch-representative-v1", rule=SAMPLE_RULE,
        frozen_selection_sha256=selection_hash, geometry_started_at_selection=False,
        batch_values_reused=True, camera_view="UNKNOWN", viewpoint_evidence="visually_assessed_unverified",
        reused_artifacts={name: dict(path=(clip / name).relative_to(root).as_posix(), sha256=single.sha(clip/name))
                          for name in ("raw-pose-unknown.jsonl", "pose-assumed-side.jsonl", "quality-assumed-side.json", "quality-unknown.json", "source.json")},
        **single.STATUS)
    single.write_json(dest / "selection.json", receipt)
    for name in ("raw-pose-unknown.jsonl", "pose-assumed-side.jsonl", "quality-assumed-side.json", "quality-unknown.json", "source.json", "preprocessing.json"):
        with (dest / name).open("xb") as stream: stream.write((clip / name).read_bytes())
    with (dest / "original.avi").open("xb") as stream: stream.write((root / pair["path"]).read_bytes())
    result = single.make_result(receipt, source["raw"], processed, observations, links)
    single.write_json(dest / "result.json", result)
    single.write_json(dest / "coverage.json", result["coverage"])
    from .real_descriptive_render import render_video_bundle
    render_video_bundle(root, dest, localized_unavailable=True)
    single.write_json(dest / "manifest.json", dict(files={p.name: dict(sha256=single.sha(p), size_bytes=p.stat().st_size)
                       for p in dest.iterdir() if p.is_file()}, **single.STATUS, result_hash=result["result_hash"]))
    return dict(action=action, path=pair["path"], side=pair["side"], coverage=result["coverage"],
                link=f"examples/{action}/START_HERE.html")


def opening_page(summary, examples):
    esc = html.escape
    t = summary["total"]
    cards = "".join(f'<li><a href="{esc(e["link"],quote=True)}">{esc(e["action"])} · {esc(Path(e["path"]).name)} · {esc(e["side"])}</a></li>'
                    if e.get("link") else f'<li>{esc(e["action"])}: {esc(e["status"])}</li>' for e in examples)
    rows = []
    for clip in summary["clips"]:
        left, right = clip["left"], clip["right"]
        cells = [Path(clip["path"]).name, str(clip["total_frames"]), left["sequence_action"],
                 left["metric_action"], f'{left["available_frames"]}/{left["total_frames"]}', f'{left["raw_observed_in_frame"]}/{left["total_frames"]}',
                 right["metric_action"], f'{right["available_frames"]}/{right["total_frames"]}', f'{right["raw_observed_in_frame"]}/{right["total_frames"]}']
        rows.append('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in cells)+'</tr>')
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Squat · batch coverage</title><style>body{{font:16px/1.6 system-ui;max-width:1280px;margin:24px auto;padding:0 20px;background:#0c1522;color:#eef6fc}}a{{color:#72e2d2}}a:focus-visible{{outline:3px solid #ffbe4d}}th,td{{border:1px solid #536378;padding:7px}}table{{border-collapse:collapse}}.scroll{{overflow:auto}}.notice{{color:#ffbe4d}}</style>
<h1>BodyWeightSquats · độ phủ đo góc gối</h1><p class="notice">EXPLORATORY / NOT_VALIDATED · development set đã chọn, không đo accuracy</p>
<p>40 clip ·80 cặp clip–bên ·{summary['source_frames']} source frames ·{t['frame_pair_denominator']} frame–bên.</p>
<p>Numeric: {t['available']}/{t['frame_pair_denominator']} ({t['numeric_micro']:.0%}); raw observed/in-frame: {t['raw_observed_in_frame']}/{t['frame_pair_denominator']} ({t['raw_micro']:.0%}).
Macro numeric: {summary['clip_macro_numeric']:.0%} (40 clip có trọng số bằng nhau, trung bình hai tỷ lệ bên; không gộp góc).</p>
<h2>Ba mẫu cố định trước render</h2><ul>{cards}</ul>
<p>Degrade/skip: “Không đủ điều kiện đo”, null ngay cả ở frame nhìn rõ nếu gate toàn clip khóa metric.
Đây là chất lượng dữ liệu, không lỗi kỹ thuật người tập. Không đổi bên, không cắt clip.</p>
<p>Góc lấy từ processed coordinates; skeleton từ raw pose cùng frame. Camera_view=UNKNOWN;
side là visually_assessed_unverified/documented=false. Không rep/phase/pause/score/tư vấn.</p>
<p>official_eligibility=false · assessment=false · G5.7 blocked · G6 IN_PROGRESS · browser/playback QA NOT_RUN.</p>
<h2>Đủ40 clip — numeric và raw tách riêng</h2><div class="scroll"><table><thead><tr><th>Clip</th><th>Frames</th><th>Sequence</th><th>L gate</th><th>L numeric</th><th>L raw</th><th>R gate</th><th>R numeric</th><th>R raw</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><a href="summary.json">Summary/reasons/provenance</a> · <a href="pairs.csv">80 pairs CSV</a> · <a href="selection.json">Frozen selection</a> · <a href="manifest.json">Hashes/timing</a> · <a href="HUONG_DAN_MO_DEMO.txt">Hướng dẫn</a></p></html>'''


def run_batch(audit_relative, prior_relative, output_relative, *, workspace=None, workers=4):
    root = Path(workspace or Path.cwd()).resolve()
    output = single.scoped(root, output_relative, Path("outputs"))
    if output.exists(): raise FileExistsError(output)
    start = time.perf_counter()
    audit, pairs, loaded = single.load_pool(root, audit_relative)
    if len(loaded) != 40 or len(pairs) != 80: raise ValueError("Expected exactly the frozen40 clips/80 pairs")
    prior = validate_prior_preprocessing(root, prior_relative, loaded)
    profile = load_preprocessing_profile(root / "configs/preprocessing_s3_v1.toml")
    validation_seconds = time.perf_counter() - start
    examples = select_examples(pairs)
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(batch_version=VERSION, **single.STATUS, rule=SAMPLE_RULE, examples=examples, pairs=pairs,
        audit_run=audit.relative_to(root).as_posix(), audit_inventory_sha256=single.sha(audit/"artifact-inventory.json"),
        profile=profile.model_dump(mode="json"), profile_hash=preprocessing_hash(profile),
        registry_hash=single.GEOMETRY_REGISTRY.content_hash(), render_started=False, geometry_started=False,
        source_hashes={p:s["candidate"]["sha256"] for p,s in loaded.items()}, worker_count=workers)
    single.write_json(output / "selection.json", frozen)
    selection_hash = single.sha(output / "selection.json")
    measured = []; timings = []; futures = {}
    batch_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path, source in loaded.items():
            selected_pairs = [p for p in pairs if p["path"] == path]
            future = executor.submit(_process_clip, str(root), str(output/"clips"/source["candidate"]["review_id"]),
                                     source, selected_pairs, profile, prior.get(path))
            futures[future] = selected_pairs
        for i, future in enumerate(as_completed(futures), 1):
            try:
                done = future.result(); measured.extend(done["pairs"]);timings.append(done["timing"])
                message = dict(progress=f"{i}/40", clip=futures[future][0]["path"],
                               status=[p["processing_status"] for p in done["pairs"]],
                               available=[p["available_frames"] for p in done["pairs"]])
            except Exception as exc:  # Failed worker still has both full denominators.
                for pair in futures[future]:
                    stat=base_stat(pair);stat.update(processing_status="processing_error",error=str(exc));measured.append(stat)
                message=dict(progress=f"{i}/40",error=str(exc),clip=futures[future][0]["path"])
            print(json.dumps(message), flush=True)
    batch_seconds = time.perf_counter() - batch_start
    summary = aggregate(pairs, measured)
    single.write_json(output / "summary.json", summary)
    columns=("path","side","feature_id","total_frames","metric_action","raw_observed_in_frame","available_frames","unavailable_frames","evaluated_frames","processing_status")
    with (output/"pairs.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(summary["pairs"])
    rendered=[];render_start=time.perf_counter()
    for action, pair in examples.items():
        if pair is None:
            rendered.append(dict(action=action,status="GROUP_ABSENT"));continue
        try:
            rendered.append(materialize_example(root, output, pair, loaded[pair["path"]], selection_hash))
        except Exception as exc:
            rendered.append(dict(action=action,status="RENDER_ERROR",error=str(exc),path=pair["path"],side=pair["side"]))
        print(json.dumps(dict(rendered=action,result=rendered[-1])),flush=True)
    single.write_json(output/"examples.json",rendered)
    with (output/"START_HERE.html").open("x",encoding="utf-8") as stream:stream.write(opening_page(summary,rendered))
    with (output/"HUONG_DAN_MO_DEMO.txt").open("x",encoding="utf-8-sig") as stream:
        stream.write("Mở START_HERE.html để chọn allow/degrade/skip và xem đủ40 clips.\nGiữ nguyên thư mục/JSON/video. Không cần server/mạng/Python để xem.\nCác metric gate áp dụng toàn clip; raw coverage không là numeric availability/accuracy.\nSource cameraUNKNOWN, side chỉ unverified; không chấm kỹ thuật.\nNếu file/codec bị chặn, không tắt bảo vệ hoặc đi vòng security policy.\nNOT_VALIDATED, official=false, assessment=false, G5.7 blocked, G6 IN_PROGRESS.\nBrowser/playback QA NOT_RUN; xem receipt sampled-frame visual QA riêng.\n")
    timing=dict(validation_seconds=validation_seconds,batch_wall_seconds=batch_seconds,
        render_seconds=time.perf_counter()-render_start,total_wall_seconds=time.perf_counter()-start,
        worker_count=workers,clips=timings,preprocessing_reused=sum(t.get("preprocessing_mode")=="reused_validated_prior_demo" for t in timings),
        factory_calls=sum(t.get("factory_calls",0) for t in timings),adapter_calls=sum(t.get("adapter_calls",0) for t in timings))
    single.write_json(output/"timing.json",timing)
    files={p.relative_to(output).as_posix():dict(sha256=single.sha(p),size_bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()}
    single.write_json(output/"manifest.json",dict(batch_version=VERSION,**single.STATUS,files=files,
        selection_sha256=selection_hash,inference_run=False,browser_qa="NOT_RUN",processing_errors=summary["total"]["processing_errors"]))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run',required=True,type=Path)
    parser.add_argument('--prior-demo',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv)
    if not 1 <= args.workers <= min(6,os.cpu_count() or 1):parser.error('workers must be1..min(6,CPU count)')
    summary=run_batch(args.audit_run,args.prior_demo,args.output_dir,workers=args.workers)
    print(json.dumps(dict(total=summary['total'],status='COMPLETED' if not summary['total']['processing_errors'] else 'COMPLETED_WITH_ERRORS')),flush=True)


if __name__=='__main__':main()
