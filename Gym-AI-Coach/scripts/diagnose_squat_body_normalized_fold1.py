"""Read-only scientific diagnosis. Only held-out forward inference; never trains or fits."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
from gym_coach.counting import body_normalized_diagnosis as d
from gym_coach.counting.body_normalized import manifest as bm,runtime as br,evaluation as be
from gym_coach.counting.body_normalized.contract import SEEDS,config_type
from gym_coach.counting.body_normalized.safety import no_learning
from gym_coach.counting.inner_cv import manifest as cv,runtime as original
from gym_coach.counting.models import STATUS,canonical_hash
from gym_coach.counting.research.data import read,local,identity
from gym_coach.counting.research.network import CountGRU,setup,collate
from gym_coach.counting.research.cli import environment
from gym_coach.counting.readiness import write_json,file_info

MANIFEST='outputs/squat_count_body_normalized/prepare-5b23ef3e8ed54cc5bacfe226db03f2c6/manifest.json'
RUN='outputs/squat_count_body_normalized/user-53db8b286ba74091b0566de79115c34f'
COMPARISON=RUN+'/comparison-d1301d8897af49a9aa94e59dad78a8f7'
REVIEW='outputs/squat_count_body_normalized/result-review-44c5c17ad0344fe395b0cb27215cde36/verification.json'


def helper():
    path=Path(__file__).with_name('diagnose_squat_cv500_fold0.py')
    spec=importlib.util.spec_from_file_location('previous_fold_diagnostic',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def strip(r):
    return {k:v for k,v in r.items() if k not in ('method','side','signed_error','absolute_error','seed')}


def verify(root,sources):
    man,data,controls=bm.load(root,MANIFEST)
    for link in man['sources']['sources']:sources.link(link)
    sources.add(MANIFEST);sources.inventory(COMPARISON)
    saved=sources.json(COMPARISON+'/result.json',True)
    receipt=sources.json(COMPARISON+'/receipt.json');review=sources.json(REVIEW)
    d.require(receipt['result']==saved and saved['manifest_sha256']==canonical_hash(man)==review['manifest_sha256'],'Comparison/manifest/review mismatch')
    d.require(saved['status']=='COMPLETE_DEVELOPMENT_ONLY' and saved['endpoint_index']==499,'Incorrect result status/endpoint')
    tool=bm.sources.diagnostic_tool(root,man['sources']['diagnostic_tool'])
    loaded={};log_count=0
    for link in saved['checkpoints']:
        path=sources.link(link);seed=link['seed'];fold=link['fold']
        d.require((seed,fold) not in loaded and path==local(root,RUN)/f'seed-{seed}'/f'fold-{fold}'/'epoch-499.pt','Duplicate/wrong checkpoint path')
        engine,selected,_=br.engine(man,seed)
        cp=engine.load_checkpoint(path,selected,fold);loaded[seed,fold]=cp
        sources.add(str(path)+'.sha256.json')
        run=sources.json(path.parent/'run.json');complete=sources.json(path.parent/'complete.json')
        d.require(cp['epoch']==499 and cp['identity']==run['identity']==complete['identity'],'Run identity drift')
        d.require(complete['checkpoint']==file_info(path),'Completion byte hash mismatch')
        d.require(run['initial_fresh_state_sha256']==man['sources']['controls'][str(seed)]['initializations'][str(fold)],'Initialization mismatch')
        for e,envelope in enumerate(cp['epoch_reports']):
            d.require(sources.json(path.parent/f'epoch-{e:03d}.json')==envelope,'Saved epoch log mismatch');log_count+=1
        rr=[r for r in saved['inference_receipts'] if r['seed']==seed and r['fold']==fold]
        d.require(len(rr)==1 and rr[0]['before_model_sha256']==rr[0]['after_model_sha256']==cp['model_sha256']
                  and rr[0]['train_ids']==man['folds'][fold]['train_ids'] and not rr[0]['optimizer_restored'],'Train receipt linkage')
        checked=[r for r in review['endpoints'] if r['seed']==seed and r['fold']==fold]
        d.require(len(checked)==1 and checked[0]['checkpoint_sha256']==file_info(path)['sha256'],'Previous review checkpoint mismatch')
    d.require(set(loaded)=={(s,f) for s in SEEDS for f in range(3)},'Six endpoints required')
    for seed in SEEDS:
        q=saved['seeds'][str(seed)];train={};held={}
        for f in range(3):
            train[f]=[strip(r) for r in q['records'] if r['method']=='body_normalized' and r['side']=='train' and r['fold']==f]
            held[f]=[dict(r,fold=f) for r in cv.unseal(loaded[seed,f]['epoch_reports'][499])['records']]
        recomputed=be.compare(man,seed,train,held,controls[str(seed)],tool)
        d.require(recomputed==q,'Saved comparison differs from exact recomputed statistics')
        for method,metrics in q['results'].items():
            expected=review['metrics'][str(seed)][method]
            a=metrics['aggregate']
            d.require(expected['train_MAE']==a['train_video_equal_MAE'] and expected['OOF_MAE']==a['held_out_pooled_MAE'],'Previous review metric mismatch')
    d.require(be.combine(saved['seeds'])==saved['combined'],'Two-seed aggregation mismatch')
    d.require(sources.json(COMPARISON+'/per-video.json',True)==[r for q in saved['seeds'].values() for r in q['records']],'Per-video sidecar mismatch')
    d.require(sources.json(COMPARISON+'/paired-errors.json',True)==[r for q in saved['seeds'].values() for r in q['paired_records']],'Paired sidecar mismatch')
    return man,data,saved,loaded,dict(manifest_sha256=canonical_hash(man),comparison_sha256=canonical_hash(saved),
        epoch_logs_verified=log_count,checkpoint_endpoints_verified=6,comparison_recomputed_exact=True)


def replay(model,dataset,expected,checkpoint_hash):
    """Hook observes the actual batch at model entry, with frozen collate parity."""
    before=original.state_hash(model.state_dict());model.eval();observations=[];seen=[]
    d.require(before==checkpoint_hash,'Loaded model hash differs from selected checkpoint')
    expected_batches={dataset[i]['video_id']:collate([dataset[i]]) for i in range(len(dataset))}
    def hook(module,args):
        batch=args[0];v=batch['video_ids'][0];ref=expected_batches[v]
        d.require(not module.training and torch.is_inference_mode_enabled(),'Replay is not eval/inference_mode')
        for key in ('x','complete','lengths','duration','target','target_mask'):
            d.require(torch.equal(batch[key],ref[key]),'Actual model tensor drift: '+key)
        seen.append(v);observations.append(dict(video_id=v,tensor_sha256=original.state_hash({k:batch[k] for k in ('x','complete','lengths','duration')}),
                                             input_shape=list(batch['x'].shape),finite=bool(torch.isfinite(batch['x']).all())))
    handle=model.register_forward_pre_hook(hook)
    try:predicted=original.predict(model,dataset)
    finally:handle.remove()
    d.exact_ids(predicted,list(expected));d.require(seen==[r['source']['video_id'] for r in dataset.rows],'Model not called once per expected input')
    differences=[]
    for r in predicted:
        p=expected[r['video_id']]
        if r['status']!='estimated' or r['prediction']!=p:
            differences.append(dict(video_id=r['video_id'],prediction=r['prediction'],saved_prediction=p,
                                    delta=None if r['prediction'] is None else r['prediction']-p,status=r['status']))
    after=original.state_hash(model.state_dict())
    d.require(before==after and all(p.grad is None for p in model.parameters()),'Weights/gradients changed')
    return dict(checkpoint_model_sha256=checkpoint_hash,before=before,after=after,exact=not differences,
        differences=differences,predictions=predicted,actual_inputs=observations,model_eval=True,inference_mode=True)


def replay_all(root,data,man,saved,loaded,sources):
    checks=[];inputs=[]
    for seed in SEEDS:
        engine,selected,ns=br.engine(man,seed)
        d.require(ns['subset'] is engine.subset and engine.train.__code__ is br.TEMPLATE.train.__code__,'Different train transform seam')
        for fold in man['folds']:
            f=fold['fold']
            for side in ('train','evaluation'):
                dataset=engine.subset(root,data,fold[side+'_ids'])
                for i in range(len(dataset)):
                    sample=dataset[i];h=canonical_hash(sample)
                    d.require(h==man['input_audit'][sample['video_id']]['normalized_sample_sha256'],'Train/eval input hash differs')
                    inputs.append(dict(seed=seed,fold=f,side=side,video_id=sample['video_id'],input_sha256=h))
                if side=='evaluation':
                    cp=loaded[seed,f];setup(config_type(seed)());model=CountGRU(config_type(seed)());model.load_state_dict(cp['model'])
                    expected={r['video_id']:r['prediction'] for r in saved['seeds'][str(seed)]['records'] if r['method']=='body_normalized' and r['side']=='held_out' and r['fold']==f}
                    checks.append(dict(condition='body_normalized',seed=seed,fold=f,**replay(model,dataset,expected,cp['model_sha256'])))
        # Additional raw58 replay only for the13 fold1 focus videos; uses same frozen source.
        link=man['sources']['controls'][str(seed)]
        rawman=sources.json(sources.link(link['manifest']),True)
        if seed==SEEDS[0]:
            from gym_coach.counting.inner_cv_500 import runtime as rr
        else:
            from gym_coach.counting.inner_cv_500_seed import runtime as rr
        cp=rr.load_checkpoint(sources.link(link['checkpoint_sources']['1']),rawman,1)
        setup(rr.Config());model=CountGRU(rr.Config());model.load_state_dict(cp['model'])
        dataset=original.subset(root,data,man['folds'][1]['evaluation_ids'])
        expected={r['video_id']:r['prediction'] for r in saved['seeds'][str(seed)]['records'] if r['method']=='raw58' and r['side']=='held_out' and r['fold']==1}
        checks.append(dict(condition='raw58',seed=seed,fold=1,**replay(model,dataset,expected,cp['model_sha256'])))
    return dict(exact=all(r['exact'] for r in checks),checks=checks,input_checks=inputs,
        body_held_forwards=66,raw_fold1_forwards=26,train_forwards=0,weights_unchanged=all(r['before']==r['after'] for r in checks))


def visual(root,row,sample,norm,out,sources,all_frames):
    import cv2
    from PIL import Image,ImageDraw
    src=row['source'];path=sources.add(src['path'],dict(sha256=src['sha256']))
    require=d.require;n=src['frame_count'];fps=src['fps']
    chosen=list(range(n)) if all_frames else sorted({round(i*(n-1)/11) for i in range(12)})
    cap=cv2.VideoCapture(str(path));require(cap.isOpened(),'Video not opened')
    require(cap.get(cv2.CAP_PROP_FPS)==fps and cap.get(cv2.CAP_PROP_FRAME_COUNT)==n,'Decoder timeline mismatch')
    folder=out/src['video_id'];folder.mkdir(parents=True,exist_ok=False)
    sheet=None;sheets=[];decoded=0;selected=[];pts_errors=[];saved_images={}
    perpage=36 if all_frames else 12;cols=6 if all_frames else 4
    while True:
        ok,frame=cap.read()
        if not ok:break
        require(frame.shape[:2]==(src['height'],src['width']),'Decoder resolution mismatch')
        pts_errors.append(abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-decoded/fps))
        if decoded in chosen:
            k=len(selected)%perpage
            if k==0:
                sheet=Image.new('RGB',(cols*320,32+math.ceil(min(perpage,len(chosen)-len(selected))/cols)*266),'#101820')
                ImageDraw.Draw(sheet).text((5,8),src['video_id']+' | '+('ALL consecutive frames' if all_frames else '12 sampled frames')+' | original images, no inference',fill='white')
                page=[]
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            sheet.paste(im,((k%cols)*320,32+(k//cols)*266))
            ImageDraw.Draw(sheet).text(((k%cols)*320+4,32+(k//cols)*266+242),f'f{decoded}  t={decoded/fps:.2f}s',fill='white')
            selected.append(decoded);page.append(decoded)
            if all_frames:saved_images[decoded]=im
            if k==perpage-1 or len(selected)==len(chosen):
                dest=folder/f'frames-{page[0]:03d}-{page[-1]:03d}.jpg';sheet.save(dest,quality=94)
                sheets.append(dict(path=dest.relative_to(root).as_posix(),sha256=file_info(dest)['sha256'],frame_indices=page,
                    first_seconds=page[0]/fps,last_seconds=page[-1]/fps,status='EXTRACTED_NOT_YET_VIEWED'))
        decoded+=1
    cap.release();require(decoded==n and selected==chosen,'Decode/visual coverage mismatch')
    overlays=[]
    if all_frames:
        # Independent fixed uniform selection, no error/pose-threshold-based cherry picking.
        indices=sorted({round(i*(n-1)/7) for i in range(8)})
        board=Image.new('RGB',(880,len(indices)*266+30),'#14202a');draw=ImageDraw.Draw(board)
        draw.text((5,8),'Original video | raw cached skeleton | normalized torso-unit chart (not image overlay)',fill='white')
        edges=((0,1),(0,2),(1,3),(2,3),(2,4),(4,6),(3,5),(5,7))
        for k,i in enumerate(indices):
            y=30+k*266;board.paste(saved_images[i],(0,y));board.paste(saved_images[i],(320,y))
            raw=sample['values'][i];new=norm['values'][i];points=[]
            for j in range(8):points.append((320+(raw[7*j]/(src['width']/src['height'])+.5)*src['width'],y+(raw[7*j+1]+.5)*src['height']))
            for a,b in edges:
                if raw[7*a+3] and raw[7*b+3]:draw.line((*points[a],*points[b]),fill='#00ff88',width=2)
            for j,(x,z) in enumerate(points):
                if raw[7*j+3]:draw.ellipse((x-3,z-3,x+3,z+3),fill='#ffbb00')
            draw.rectangle((640,y,879,y+239),fill='#202a33')
            draw.line((760,y,760,y+239),fill='#5c6875');draw.line((640,y+80,879,y+80),fill='#5c6875')
            if sample['complete'][i]:
                ps=[(760+new[j*7]*50,y+80+new[j*7+1]*50) for j in range(8)]
                for a,b in edges:draw.line((*ps[a],*ps[b]),fill='#58c8ff',width=2)
            draw.text((4,y+243),f'frame {i}, {i/fps:.2f}s | complete={sample["complete"][i]} | right: 50px per torso unit; x right, y down',fill='white')
        dest=folder/'raw-normalized-check.jpg';board.save(dest,quality=95)
        overlays.append(dict(path=dest.relative_to(root).as_posix(),sha256=file_info(dest)['sha256'],frame_indices=indices,status='EXTRACTED_NOT_YET_VIEWED'))
    return dict(source=src,decoded_frames=decoded,decoder_timestamp_max_abs_error_seconds=max(pts_errors),
        timestamp_policy='i/fps CFR assumption, not verified physical capture PTS',sheets=sheets,overlays=overlays,
        continuous_playback=False,all_frames_extracted=all_frames)


def execute(root,out):
    out=local(root,out);out.mkdir(parents=True,exist_ok=False);h=helper();sources=h.Sources(root);start=time.perf_counter()
    sources.add(__file__);sources.add(Path(d.__file__));sources.add(Path(__file__).with_name('diagnose_squat_cv500_fold0.py'))
    sources.add('scripts/diagnose_squat_count_inner_cv.py')
    with no_learning():
        man,data,saved,loaded,verification=verify(root,sources)
        replay_result=replay_all(root,data,man,saved,loaded,sources)
        write_json(out/'replay.json',cv.sealed(replay_result))
        d.require(replay_result['exact'],'Replay mismatch saved; stop before interpreting differences')
        samples=original.subset(root,data,[r['source']['video_id'] for r in man['rows']]);metadata={};normalized={};input_checks=[]
        engine,_,_=br.engine(man,SEEDS[0]);body=engine.subset(root,data,[r['source']['video_id'] for r in man['rows']])
        tool=h.previous_tool()
        for i,row in enumerate(samples.rows):
            v=row['source']['video_id'];raw=samples[i];norm=body[i];normalized[v]=norm
            # Raw pose re-encode is shared production code: linkage check, not an independent accuracy reference.
            support=h.support(root,row,sources,tool)
            check=d.check_transform(raw,norm,man['transform']['epsilon'],man['transform']['numeric_tolerance'])
            expected,_=d.independent_geometry(raw,man['transform']['epsilon'])
            a,b=collate([expected]),collate([norm]);tensor_delta=float(torch.max(torch.abs(a['x']-b['x'])))
            d.require(tensor_delta<=1e-6,'Independent float32 tensor mismatch')
            check.update(video_id=v,raw_cache_exact=True,independent_float32_tensor_exact=torch.equal(a['x'],b['x']),
                independent_float32_max_abs_difference=tensor_delta,raw_sample_sha256=canonical_hash(raw),normalized_sample_sha256=canonical_hash(norm))
            input_checks.append(check)
            pose=d.measure(raw,norm,man['input_audit'][v]['scale'],row['source']['fps'])
            metadata[v]=dict(video_id=v,group_id=row['source']['group_id'],target=row['target'],source=row['source'],
                duration_seconds=raw['duration_seconds'],side='held_out' if v in man['folds'][1]['evaluation_ids'] else 'train',
                input_artifact=row['input_artifact'],pose=pose,support=support)
        records=list(metadata.values());paired={}
        for seed in SEEDS:
            rows=[r for r in saved['seeds'][str(seed)]['records'] if r['fold']==1 and r['side']=='held_out' and r['method'] in ('body_normalized','raw58','duration')]
            paired[str(seed)]=d.paired_table(rows,metadata,man['folds'][1]['evaluation_ids'])
        train=[r for r in records if r['side']=='train'];held=[r for r in records if r['side']=='held_out']
        distribution=d.compare_distribution(train,held)
        # The source verifier intentionally returns only the33 selected rows as data.
        # Read the locked manifest for label lineage; never load excluded input caches.
        full_dataset=sources.json(sources.link(man['dataset']),True)
        refpath=next(p for p in full_dataset['frozen_inputs'] if p.endswith('/results-v1/ai-reference.json'))
        refs={r['video_id']:r for r in sources.json(refpath)['records']};labels={}
        for row in man['rows']:
            if row['source']['group_id'] in ('g17','g19'):labels[row['source']['video_id']]=h.labels(row,refs,sources)
        for p in ('docs/SQUAT_AI_ANNOTATION_PROTOCOL.md','docs/SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md'):sources.add(p)
        for record in labels.values():
            for p in record['passes']:
                d.require(p['protocol_sha256']==file_info(root/'docs/SQUAT_AI_ANNOTATION_PROTOCOL.md')['sha256']
                    and p['semantics_protocol_sha256']==file_info(root/'docs/SQUAT_TEMPORAL_ANNOTATION_PROTOCOL.md')['sha256'],
                    'Label protocol hash mismatch')
        analysis=dict(**STATUS,version='body-normalization-fold1-diagnosis-v1',metadata=metadata,paired=paired,
            train_vs_held=distribution,input_checks=input_checks,saved_metrics={s:q['results'] for s,q in saved['seeds'].items()},
            definitions=dict(steps='Euclidean joint displacement between adjacent complete samples only, no bridging gaps, units/frame',
                largest_steps='top3 by raw max joint displacement, tie frame index; descriptive, not a failure threshold',
                torso_cv='population SD / mean of per-complete-frame shoulder-midpoint to hip-midpoint lengths',
                hip_motion='complete-frame range and sum adjacent-complete Euclidean hip displacement; excludes gaps',
                independent_float32_tolerance=1e-6,range_overlap='fraction of held-out clip summaries within empirical train min/max, not a multivariate support test'),
            scientific_status='AI weak reference; adaptive development, not independent validation or causal attribution')
        write_json(out/'analysis.json',cv.sealed(analysis));write_json(out/'labels.json',cv.sealed(labels))
        write_json(out/'fold1-per-video.json',cv.sealed(paired));write_json(out/'verification.json',cv.sealed(verification))
        visuals=[];lookup={r['source']['video_id']:r for r in man['rows']};samplemap={samples[i]['video_id']:samples[i] for i in range(len(samples))}
        for v in man['folds'][1]['evaluation_ids']:
            visuals.append(visual(root,lookup[v],samplemap[v],normalized[v],out/'visual',sources,lookup[v]['source']['group_id'] in ('g17','g19')))
        write_json(out/'visual-index.json',dict(records=visuals,view_status='EXTRACTED_NOT_YET_VIEWED',continuous_playback=False))
        write_json(out/'preservation.json',sources.preserve())
        write_json(out/'receipt.json',dict(command=[sys.executable,*sys.argv],seconds=time.perf_counter()-start,
            environment=environment(),body_held_out_forwards=66,raw_fold1_forwards=26,train_forwards=0,
            pose_inference=0,training_steps=0,baseline_fits=0,status='COMPUTATION_COMPLETE_VISUAL_REVIEW_PENDING'))
    print(json.dumps(dict(output=str(out),replay_exact=replay_result['exact'],forward_calls=92,epoch_logs=verification['epoch_logs_verified'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    try:execute(Path.cwd().resolve(),args.output_dir)
    except Exception as exc:
        out=local(Path.cwd(),args.output_dir)
        # Only a previously unused directory is accepted by execute; preserve any existing files.
        if out.exists() and not (out/'failure.json').exists() and not isinstance(exc,FileExistsError):
            write_json(out/'failure.json',dict(type=type(exc).__name__,detail=str(exc),comparison_interpretation='STOP'))
        raise
