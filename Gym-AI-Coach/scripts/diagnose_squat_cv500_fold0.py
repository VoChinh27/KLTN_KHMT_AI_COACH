"""Saved-prediction diagnosis only; no forward, fitting, training or source writes."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.readiness import file_info, write_json
from gym_coach.counting.research.data import read, identity, local, encode
from gym_coach.counting.research.contract import Config, JOINTS
from gym_coach.counting.inner_cv import manifest as cv, runtime as old_runtime
from gym_coach.counting.inner_cv_500 import manifest as m500, runtime, evaluation, reproduction
from gym_coach.counting.inner_cv_500.safety import no_learning
from gym_coach.counting.ai_annotations import load_pass, verify_evidence, conservative_pair
from gym_coach.pose.models import PoseFrame


def previous_tool():
    spec = importlib.util.spec_from_file_location('saved_cv_diagnosis', Path(__file__).with_name('diagnose_squat_count_inner_cv.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Sources:
    """Byte inventory before use; verify again after work. Does not edit sources."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.files = {}

    def add(self, path, expected=None):
        path = local(self.root, path)
        if 'KLTN_AI_COACH' in path.parts or '_archive' in path.parts:
            raise ValueError('Out of scope source')
        item = identity(self.root, path)
        if expected and any(item[k] != expected[k] for k in ('sha256', 'size_bytes') if k in expected):
            raise ValueError('Source hash mismatch: '+str(path))
        if item['path'] in self.files and self.files[item['path']] != item:
            raise ValueError('Source changed during analysis')
        self.files[item['path']] = item
        return path

    def link(self, link): return self.add(link['path'], link)

    def json(self, path, sealed=False):
        d = read(self.add(path)); return cv.unseal(d) if sealed else d

    def inventory(self, folder):
        folder = local(self.root, folder)
        for name, expected in self.json(folder/'inventory.json')['files'].items():
            if not (folder/name).resolve().is_relative_to(folder): raise ValueError('Escaping inventory')
            self.add(folder/name, expected)

    def preserve(self):
        for item in list(self.files.values()): self.link(item)
        return dict(status='UNCHANGED', files=len(self.files), sources=sorted(self.files.values(), key=lambda x:x['path']))


def exact_ids(records, expected):
    ids = [r['video_id'] for r in records]
    if len(ids) != len(set(ids)) or len(expected) != len(set(expected)) or set(ids) != set(expected):
        raise ValueError('Missing/duplicate/unexpected video; fixed denominator')


def intervals(mask, value):
    output = []; start = None
    for i, v in enumerate([*mask, not value]):
        if v == value and start is None: start = i
        elif v != value and start is not None:
            output.append(dict(start=start, end=i-1, frames=i-start)); start = None
    return output


def measure(rows):
    if not rows: raise ValueError('Empty analysis subset')
    stats = previous_tool().describe
    result = dict(n=len(rows), groups=len({r['group_id'] for r in rows}),
        count_histogram=dict(sorted(Counter(str(r['target']) for r in rows).items())),
        duration=stats([r['duration_seconds'] for r in rows]), models={})
    for name in ('gru50', 'gru500', 'duration'):
        p = [r[name] for r in rows]; errors = [v-r['target'] for v,r in zip(p,rows)]
        stats(errors)
        result['models'][name] = dict(predictions=stats(p), MAE=statistics.mean(map(abs,errors)),
            total_AE=math.fsum(map(abs,errors)), signed_mean=statistics.mean(errors),
            under=sum(e<0 for e in errors), over=sum(e>0 for e in errors), exact=sum(e==0 for e in errors))
    a = result['models']; delta = [abs(r['gru500']-r['target'])-abs(r['gru50']-r['target']) for r in rows]
    result.update(delta_MAE_500_minus_50=a['gru500']['MAE']-a['gru50']['MAE'],
        improved=sum(d<0 for d in delta), worsened=sum(d>0 for d in delta), unchanged=sum(d==0 for d in delta),
        duration_better=sum(abs(r['duration']-r['target'])<abs(r['gru500']-r['target']) for r in rows))
    return result


def analyze(rows, expected):
    exact_ids(rows, expected)
    overall=measure(rows); total=overall['models']['gru500']['total_AE']
    result=[]
    for r in sorted(rows,key=lambda r:r['video_id']):
        r=dict(r)
        for k in ('gru50','gru500','duration'):
            r[k+'_signed_error']=r[k]-r['target']; r[k+'_AE']=abs(r[k+'_signed_error'])
        r.update(delta_AE_500_minus_50=r['gru500_AE']-r['gru50_AE'],
            delta_AE_500_minus_duration=r['gru500_AE']-r['duration_AE'],
            fold_AE_fraction=r['gru500_AE']/total if total else None,
            fold_MAE_contribution=r['gru500_AE']/len(rows))
        result.append(r)
    return dict(overall=overall, records=result,
        by_group={g:measure([r for r in rows if r['group_id']==g]) for g in sorted({r['group_id'] for r in rows})},
        by_count={str(c):measure([r for r in rows if r['target']==c]) for c in sorted({r['target'] for r in rows})},
        visual_priority=[r['video_id'] for r in sorted(result,key=lambda r:(-r['gru500_AE'],r['video_id']))])


def support(root, row, sources, tool):
    cache=sources.json(sources.link(row['input_artifact']))
    for link in row['source_links']+row['label_links']: sources.link(link)
    raw_links=[l for l in row['source_links'] if l['path'].endswith('.jsonl')]
    if len(raw_links)!=1: raise ValueError('Ambiguous raw pose linkage')
    raw=sources.link(raw_links[0]); frames=[PoseFrame.model_validate_json(s) for s in raw.read_text(encoding='utf8').splitlines()]
    encoded=encode(frames,row['source'],Config(),cache['original_quality_report']['flags'])
    for key in ('source','values','complete','duration_seconds','stats','status','reasons','config_hash'):
        if encoded[key]!=cache[key]: raise ValueError('Raw/cache mismatch: '+row['source']['video_id']+'/'+key)
    stats=tool.support_stats(row,cache); mask=cache['complete']; fps=row['source']['fps']
    stats.update(gaps=intervals(mask,False), complete_intervals=intervals(mask,True),
        hidden_zero_samples=sum(not v for v in mask),
        internal_reset_transitions=sum(a and not b for a,b in zip(mask,mask[1:])),
        temporal_restarts_after_gap=sum(not a and b for a,b in zip(mask,mask[1:])),
        raw_without_pose=sum(not f.has_pose for f in frames), raw_cache_exact=True,
        joint_visibility={}, original_quality=cache['original_quality_report']['eligibility'],
        quality_profile_hash=cache['original_quality_report']['profile_hash'],
        timeline='all raw/cache samples index i and timestamp i/fps exactly; CFR assumption, not capture PTS',
        source_links=row['source_links'], input_artifact=row['input_artifact'])
    for name in JOINTS:
        js=[next(j for j in f.joints if j.joint.value==name) for f in frames]
        stats['joint_visibility'][name]=dict(provenance=dict(Counter(j.status.value for j in js)),
            visibility=tool.describe([j.visibility for j in js if j.visibility is not None]),
            invalid_frame_indices=[i for i,j in enumerate(js) if not (j.status.value=='observed' and 0<=j.x<=1 and 0<=j.y<=1)])
    for item in stats['gaps']+stats['complete_intervals']:
        item.update(start_seconds=item['start']/fps,end_seconds=item['end']/fps, sample_duration_seconds=item['frames']/fps)
    return stats


def support_summary(rows):
    stats=previous_tool().describe; n=sum(r['support']['frames'] for r in rows); complete=sum(r['support']['complete_frames'] for r in rows)
    return dict(clips=len(rows),frames=n,complete_frames=complete,frame_weighted_complete_fraction=complete/n,
        clip_mean_complete_fraction=statistics.mean(r['support']['complete_fraction'] for r in rows),
        with_gaps=sum(r['support']['incomplete_frames']>0 for r in rows),
        longest_gap_frames=stats([r['support']['longest_gap_frames'] for r in rows]),
        valid_runs=stats([r['support']['valid_runs'] for r in rows]),
        total_hidden_zero_samples=sum(r['support']['hidden_zero_samples'] for r in rows),
        total_internal_reset_transitions=sum(r['support']['internal_reset_transitions'] for r in rows),
        raw_without_pose=sum(r['support']['raw_without_pose'] for r in rows))


def verify(root, manifest_path, comparison, review_path, sources):
    man=sources.json(manifest_path,True); comp=local(root,comparison); sources.inventory(comp)
    saved=sources.json(comp/'result.json',True); review=sources.json(review_path)
    if saved['manifest_sha256']!=canonical_hash(man) or review['result_sha256']!=canonical_hash(saved): raise ValueError('Result/manifest/review hash mismatch')
    if (local(root,review['manifest'])!=local(root,manifest_path) or local(root,review['comparison'])!=comp
            or local(root,review['run'])!=comp.parent):raise ValueError('Review path mismatch')
    receipt=sources.json(comp/'receipt.json')
    if receipt['result']!=saved: raise ValueError('Receipt differs')
    for directory, files in (('research',man['core_identity']['code']),('inner_cv',man['cv_code']),('inner_cv_500',man['code'])):
        for name,info in files.items(): sources.add('src/gym_coach/counting/'+directory+'/'+name,info)
    rows,_,folds=cv.partition(man['rows'])
    if rows!=man['rows'] or folds!=man['folds']:raise ValueError('Fold drift')
    base=sources.json(sources.link(man['controls']['original_manifest']),True)
    if base['rows']!=rows or base['folds']!=folds or {k for k in base['config'] if base['config'][k]!=man['config'][k]}!={'epochs'}: raise ValueError('Design drift')
    dataset=sources.json(sources.link(man['dataset']),True)
    if rows!=[r for r in dataset['rows'] if r['role']=='train' and r['supervised_eligible']]:raise ValueError('Dataset rows differ')
    # Do not load/evaluate outer validation/development-test caches or predictions.
    for path,info in dataset['frozen_inputs'].items():sources.add(path,info)
    diag=sources.link(man['controls']['diagnostic_tool']); tool=m500.diagnostic_tool(root,man['controls']['diagnostic_tool'])
    controls=sources.json(sources.link(man['controls']['diagnostic_files']['per-video-errors.json']),True)
    controls=[r for r in controls if r['method'] in ('gru','duration','constant')]
    baseline=sources.json(sources.link(man['controls']['verified_lineage']['baseline']),True)
    if baseline['manifest_sha256']!=canonical_hash(base):raise ValueError('Baseline wrong manifest')
    for method in ('constant','duration'):
        held=[r for r in controls if r['method']==method and r['side']=='held_out']
        for a,b in zip(sorted(held,key=lambda r:r['video_id']),sorted(baseline['predictions'][method],key=lambda r:r['video_id'])):
            if any(a[k]!=b[k] for k in ('video_id','fold','target','source_sha256','prediction')):raise ValueError('Baseline saved prediction mismatch')
        exact_ids(held,[r['source']['video_id'] for r in rows])
    loaded={}; receipts={(r['epoch'],r['fold']):r for r in saved['inference_receipts']}
    if len(receipts)!=6:raise ValueError('Missing/duplicate inference receipts')
    for s in saved['checkpoint_sources']:
        path=sources.link(s); e=s['epoch']; f=s['fold']
        if (e,f) in loaded or e not in (49,499) or path!=comp.parent/f'fold-{f}'/f'epoch-{e:03d}.pt':raise ValueError('Wrong/duplicate checkpoint')
        sources.add(str(path)+'.sha256.json'); cp=runtime.load_checkpoint(path,man,f); loaded[(e,f)]=cp
        rec=receipts[e,f]
        if (rec['model_before_sha256']!=cp['model_sha256'] or rec['model_after_sha256']!=cp['model_sha256']
                or rec['train_ids']!=folds[f]['train_ids'] or not rec['inference_mode'] or not rec['model_eval']):raise ValueError('Train prediction checkpoint receipt mismatch')
        if e==499:
            run=sources.json(path.parent/'run.json'); complete=sources.json(path.parent/'complete.json')
            if run['identity']!=cp['identity'] or complete['identity']!=cp['identity'] or complete['checkpoint']!=file_info(path):raise ValueError('Run/completion mismatch')
            for i,envelope in enumerate(cp['epoch_reports']):
                if sources.json(path.parent/f'epoch-{i:03d}.json')!=envelope:raise ValueError('Epoch log mismatch')
                pt=path.parent/f'epoch-{i:03d}.pt'; sources.add(pt,sources.json(str(pt)+'.sha256.json'))
    if set(loaded)!={(e,f) for e in (49,499) for f in range(3)}:raise ValueError('Need all six checkpoints')
    reproduced=[]
    for f in range(3):
        path=sources.link(man['controls']['verified_lineage']['checkpoint_links']['gru'][str(f)])
        sources.add(str(path)+'.sha256.json'); old=old_runtime.load_checkpoint(path,cv.fold_identity(base,f))
        reproduced.append(reproduction.compare(old,loaded[49,f],loaded[499,f]))
    if dict(identical=all(r['identical'] for r in reproduced),folds=reproduced)!=saved['reproducibility']:raise ValueError('Reproduction replay mismatch')
    for e in (49,499):
        ep=saved['endpoints'][str(e)]; train={}; held={}
        for f in range(3):
            train[f]=[r for r in ep['records'] if r['method']=='cv500' and r['fold']==f and r['side']=='train']
            # Strip descriptive fields before replay because original predictions did not contain them.
            train[f]=[{k:v for k,v in r.items() if k not in ('method','side','signed_error','absolute_error')} for r in train[f]]
            held[f]=[dict(r,fold=f) for r in cv.unseal(loaded[e,f]['epoch_reports'][e])['records']]
        replay=evaluation.compare_endpoint(man,train,held,controls,tool,e)
        if canonical_hash(ep)!=canonical_hash(replay):raise ValueError('Saved prediction/metric replay mismatch')
        for method in evaluation.METHODS:
            for key in ('train_video_equal_MAE','held_out_pooled_MAE'):
                if review['metrics'][str(e)][method][key]!=ep['results'][method]['aggregate'][key]:raise ValueError('Review metric mismatch')
    if sources.json(comp/'per-video.json',True)!=[dict(r,epoch_index=int(e)) for e in ('49','499') for r in saved['endpoints'][e]['records']]:raise ValueError('Per-video sidecar mismatch')
    return man,dataset,saved,dict(checkpoints_loaded=9,cv500_checkpoint_byte_hashes=1500,epoch_logs=1500,
        reproduction=['EXACT_MATCH' if r['identical'] else r['status'] for r in reproduced],
        metrics_replayed=True,model_forwards=0,training_steps=0,baseline_fits=0,
        train_evidence='saved predictions + unchanged-weight inference receipts; not regenerated',
        manifest_sha256=canonical_hash(man),comparison_sha256=canonical_hash(saved),review_sha256=file_info(review_path)['sha256'])


def labels(row, references, sources):
    passes=[load_pass(sources.link(l)) for l in row['label_links']]
    for p in passes:
        verify_evidence(p,sources.root)
        if p.source.model_dump(mode='json')!=row['source']:raise ValueError('Label/source mismatch')
        for e in p.evidence:
            sources.add(e.path,dict(sha256=e.sha256));sources.add(Path(e.path).parent/'manifest.json')
    ref=references[row['source']['video_id']]
    if ref['source']!=row['source'] or canonical_hash(ref)!=row['label_record_hash']:raise ValueError('Reference hash/source mismatch')
    pair=conservative_pair(*passes)
    if any(ref[k]!=v for k,v in pair.items()) or pair['whole_clip_count']!=row['target']:raise ValueError('Conservative reconciliation mismatch')
    return dict(pass_links=row['label_links'],reference_record_sha256=row['label_record_hash'],
        passes=[p.model_dump(mode='json') for p in passes],reconciliation=pair)


def decode_sheet(root,row,out,sources):
    """Decode every frame; sample selection independent of labels/pose/errors."""
    import cv2
    from PIL import Image, ImageDraw
    src=row['source']; path=sources.add(src['path'],dict(sha256=src['sha256']))
    n=src['frame_count']; indices=sorted({round(i*(n-1)/23) for i in range(24)})
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS)
    metadata=dict(fps=fps,frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps!=src['fps']:raise ValueError('Decoder/FPS mismatch')
    sheet=Image.new('RGB',(1280,6*268+32),'#161b22'); draw=ImageDraw.Draw(sheet)
    draw.text((8,8),src['video_id']+' | sampled frames; no continuous playback',fill='white')
    decoded=0; sampled=[]; time_errors=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape[:2]!=(src['height'],src['width']):raise ValueError('Resolution changed')
        pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000; time_errors.append(abs(pts-decoded/fps))
        if decoded in indices:
            k=len(sampled);x=(k%4)*320;y=32+(k//4)*268
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));image.thumbnail((320,240))
            sheet.paste(image,(x,y));draw.text((x+4,y+242),f'frame {decoded} | {decoded/fps:.3f}s',fill='white')
            sampled.append(dict(frame_index=decoded,timestamp_seconds=decoded/fps,decoder_pos_msec_seconds=pts))
        decoded+=1
    cap.release()
    if decoded!=n or metadata['frames']!=n or [r['frame_index'] for r in sampled]!=indices:raise ValueError('Decode/sample count mismatch')
    out.mkdir(exist_ok=True);sheet_path=out/(src['video_id']+'.jpg');sheet.save(sheet_path,quality=94)
    return dict(source=src,decoded_frames=decoded,decoder_metadata=metadata,decoder_timestamp_max_abs_error_seconds=max(time_errors),
        input_timestamp_policy=src['timestamp_policy'],samples=sampled,sheet=identity(root,sheet_path),
        visual_status='EXTRACTED_NOT_YET_VIEWED',continuous_playback=False)


def execute(args):
    root=Path.cwd().resolve();out=local(root,args.output_dir);out.mkdir(parents=True,exist_ok=False)
    sources=Sources(root);start=time.perf_counter();tool=previous_tool()
    sources.add(__file__);sources.add(Path(__file__).with_name('diagnose_squat_count_inner_cv.py'))
    with no_learning(),tool.no_learning_or_forward():
        man,data,saved,verification=verify(root,args.manifest,args.comparison,args.review,sources)
        refpath=next(p for p in data['frozen_inputs'] if p.endswith('/results-v1/ai-reference.json'))
        references={r['video_id']:r for r in sources.json(refpath)['records']}
        records=[];label_records={}
        for row in man['rows']:
            vid=row['source']['video_id']; record=dict(video_id=vid,group_id=row['source']['group_id'],
                source=row['source'],source_sha256=row['source']['sha256'],target=row['target'],
                label_record_hash=row['label_record_hash'],duration_seconds=row['source']['frame_count']/row['source']['fps'],
                support=support(root,row,sources,tool))
            for epoch,key in (('49','gru50'),('499','gru500')):
                matches=[r for r in saved['endpoints'][epoch]['records'] if r['method']=='cv500' and r['fold']==0 and r['video_id']==vid]
                if len(matches)!=1:raise ValueError('Fold0 pairing error')
                record[key]=matches[0]['prediction'];record['side']=matches[0]['side']
            duration=[r for r in saved['endpoints']['499']['records'] if r['method']=='duration' and r['fold']==0 and r['video_id']==vid]
            if len(duration)!=1:raise ValueError('Duration pairing error')
            record['duration']=duration[0]['prediction']
            record['other_fold_train_predictions']=[dict(fold=r['fold'],prediction=r['prediction'],absolute_error=r['absolute_error'])
                for r in saved['endpoints']['499']['records'] if r['method']=='cv500' and r['side']=='train' and r['video_id']==vid and r['fold']!=0]
            records.append(record);label_records[vid]=labels(row,references,sources)
        held=analyze([r for r in records if r['side']=='held_out'],man['folds'][0]['evaluation_ids'])
        train=analyze([r for r in records if r['side']=='train'],man['folds'][0]['train_ids'])
        context={e:{method:dict(aggregate=q['aggregate'],folds=q['folds']) for method,q in saved['endpoints'][e]['results'].items()} for e in ('49','499')}
        analysis=dict(**STATUS,fold=0,held_out=held,train=train,
            support_comparison=dict(train=support_summary(train['records']),held_out=support_summary(held['records'])),
            cohort_context=context,unit='fold0:13held-out vs20train; other-fold predictions are in-sample from other models')
        write_json(out/'analysis.json',analysis);write_json(out/'labels.json',label_records);write_json(out/'verification.json',verification)
        sheets=[];byid={r['source']['video_id']:r for r in man['rows']}
        for vid in held['visual_priority']:sheets.append(decode_sheet(root,byid[vid],out/'visual',sources))
        write_json(out/'visual-index.json',dict(selection='24 uniformly spaced indices including endpoints per clip, all13held-out clips',records=sheets))
        write_json(out/'preservation.json',sources.preserve())
        write_json(out/'receipt.json',dict(command=[sys.executable,*sys.argv],seconds=time.perf_counter()-start,
            model_forwards=0,pose_inference=0,training_steps=0,baseline_fits=0,decoded_clips=13,
            decoded_frames=sum(s['decoded_frames'] for s in sheets),held_out_clips=13,train_clips=20,
            visual_status='EXTRACTED_NOT_YET_VIEWED',python=sys.version))
    print(json.dumps(dict(output=str(out),verification=verification,held=held['overall'],train=train['overall']),ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','comparison','review','output-dir'):parser.add_argument('--'+key,required=True)
    execute(parser.parse_args())
