"""Saved-evidence cohort audit. No model execution, extraction, fitting or source edits."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys

from gym_coach.counting.inner_cv import manifest as cv, runtime
from gym_coach.counting.fixed_origin import manifest as fm, evaluation as fe
from gym_coach.counting.models import STATUS, canonical_hash
from gym_coach.counting.research.data import read, identity
from gym_coach.counting.readiness import write_json, file_info

RUN='outputs/squat_count_fixed_origin/user-b11ae5d244df4ebd985123f2a106739c'
COMPARISON=RUN+'/comparison-4acd5fdf4d104f6c9037e55d4f855bda'
REVIEW='outputs/squat_count_fixed_origin/result-review-9b4fb3a3edb54d69a1dcc6cbe196ad0f'
OLD='outputs/squat_body_normalized_fold1_diagnosis/run-c18372526a494a48868f3c57f9771cfc'
POLICY=dict(version='cohort33-pose-review-v1',
    ranking='Per video: largest max-joint Euclidean step and largest count of strict pixel-x pair sign flips; ties: largest step then earliest frame.',
    eligibility='Only adjacent complete samples. No crossing gaps. Pixel-x changes are candidates, never automatic errors.',
    windows='Each selected edge [i-1,i], context i-3..i+2 clipped to video. Union of two windows; no new threshold.',
    labels_used=False,predictions_used=False,coordinate_units='image height',
    scope='All33 locked videos, all edges ranked; two windows per clip merged by sample index.')

def require(ok,message):
    if not ok: raise ValueError(message)

def helper():
    path=Path(__file__).with_name('diagnose_squat_cv500_fold0.py')
    spec=importlib.util.spec_from_file_location('cohort_previous_audit',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def timestamp(index,source):
    require(type(index) is int and 0<=index<source['frame_count'],'Invalid frame index')
    require(math.isfinite(source['fps']) and source['fps']>0,'Invalid FPS')
    return index/source['fps']

def pixels(row,joint,source):
    require(len(row)==58 and type(joint) is int and 0<=joint<8,'Invalid raw58 point')
    if not row[joint*7+3]:return None
    x,y=row[joint*7:joint*7+2]
    require(math.isfinite(x) and math.isfinite(y),'Nonfinite usable coordinate')
    return x*source['height']+source['width']/2,(y+.5)*source['height']

def rank_windows(cache,source):
    values=cache['values'];mask=cache['complete'];n=source['frame_count']
    require(cache['source']==source and len(values)==len(mask)==n,'Cache/source/frame mismatch')
    require(cache['duration_seconds']==n/source['fps'],'Duration mismatch')
    for i,r in enumerate(values):
        require(len(r)==58 and all(math.isfinite(v) for v in r),'Invalid raw58')
        require(r[56]==timestamp(i,source) and r[57]==(0 if i==0 else 1/source['fps']),'Timeline mismatch')
        require(bool(mask[i])==all(bool(r[j*7+3]) for j in range(8)),'Mask/usable mismatch')
    edges=[]
    for i in range(1,n):
        if not(mask[i-1] and mask[i]):continue
        a,b=values[i-1:i+1]
        steps=[math.hypot(b[j*7]-a[j*7],b[j*7+1]-a[j*7+1]) for j in range(8)]
        flips=[j//2 for j in (0,2,4,6) if (a[j*7]-a[(j+1)*7])*(b[j*7]-b[(j+1)*7])<0]
        edges.append(dict(previous_frame=i-1,frame_index=i,timestamp_seconds=i/source['fps'],
            max_step=max(steps),joint_steps=steps,pixel_x_flips=flips,flip_count=len(flips),status='suspected_candidate_only'))
    if not edges:return dict(edges=[],selected_edges=[],frame_indices=[])
    step=min(edges,key=lambda r:(-r['max_step'],r['frame_index']))
    flip=min(edges,key=lambda r:(-r['flip_count'],-r['max_step'],r['frame_index']))
    selected=sorted({step['frame_index'],flip['frame_index']})
    indices=sorted({j for i in selected for j in range(max(0,i-3),min(n,i+3))})
    return dict(edges=edges,selected_edges=selected,frame_indices=indices)

def validate_review(records,expected,issues,evidence_inventory):
    helper().exact_ids(records,list(expected))
    lookup={r['video_id']:r for r in records};seen=set()
    for v,source in expected.items():
        r=lookup[v];require(r['source_sha256']==source['sha256'],'Review source mismatch')
        require(r['independent_ground_truth'] is False and r['human_reviewed'] is False,'AI is not independent/human ground truth')
        require(r['count_status'] in ('unverified','AI_semantics_consistent'),'Invalid count verification claim')
        require(r['viewed_evidence'] and r['notes'],'Missing actual visual evidence/notes')
        original_frames=[]
        for e in r['viewed_evidence']:
            key=(v,e['path']);require(key in evidence_inventory,'Unlinked evidence')
            linked=evidence_inventory[key]
            require(e['sha256']==linked['sha256'] and e['frame_indices']==linked['frame_indices'],'Evidence hash/frame mismatch')
            require(e['actually_viewed'] is True and e['frame_indices'],'Unviewed evidence')
            require(len(e['frame_indices'])==len(set(e['frame_indices'])),'Duplicate frames')
            require(e['timestamps_seconds']==[timestamp(i,source) for i in e['frame_indices']],'Frame/time mismatch')
            if e['kind']=='original_contact_sheet':original_frames.extend(e['frame_indices'])
        require(original_frames==list(range(source['frame_count'])),'Incomplete/duplicate original visual coverage')
    for x in issues:
        require(x['issue_id'] not in seen,'Duplicate issue');seen.add(x['issue_id'])
        v=x['video_id'];require(v in expected and x['source_sha256']==expected[v]['sha256'],'Issue source mismatch')
        require(x['status'] in ('confirmed','suspected','unresolved'),'Invalid issue status')
        require(x['timestamps_seconds']==[timestamp(i,expected[v]) for i in x['frame_indices']],'Issue timeline mismatch')
        for k in ('type','detection','verification_evidence','known_impact','proposed_handling','risks','previous_issue'):
            require(k in x,'Missing issue field '+k)
        if x['status']=='confirmed':
            require(x['verification_method'] in ('image_pose_comparison','source_record_comparison'),'Numeric ranking cannot confirm')
            require(x['verification_evidence'],'Confirmed issue without evidence')
            if x['verification_method']=='image_pose_comparison':
                viewed={i for e in lookup[v]['viewed_evidence'] for i in e['frame_indices']}
                require(set(x['frame_indices'])<=viewed,'Confirmed frames not viewed')
                paths={e['path'] for e in lookup[v]['viewed_evidence'] if e['kind'] in ('raw_pose_overlay','prior_raw_pose_overlay')}
                require(set(x['verification_evidence'])<=paths,'Confirmed issue without linked viewed overlay')
    return dict(videos=len(records),confirmed=sum(x['status']=='confirmed' for x in issues),
        suspected=sum(x['status']=='suspected' for x in issues),unresolved=sum(x['status']=='unresolved' for x in issues),
        independent_count_labels=0,AI_semantics_consistent=sum(r['count_status']=='AI_semantics_consistent' for r in records))

def verify(root,sources):
    import torch
    h=helper();rr=sources.json(RUN+'/run-root.json',True)
    man=sources.json(sources.link(rr['manifest']),True)
    require(man['code']==fm.code_identity(),'Frozen fixed code changed')
    # Rehash prior evidence; do not repeat 3000-history parsing, normalization or inference.
    for link in man['sources']['sources']:sources.link(link)
    sources.inventory(OLD)
    done=sources.json(RUN+'/completed-22f622e33a2343928b4fc72c840e9c89.json',True)
    require(done['comparison']==COMPARISON,'Completion comparison mismatch')
    for link in done['artifacts']+done['checkpoints']:sources.link(link)
    saved=sources.json(COMPARISON+'/result.json',True)
    review=sources.json(REVIEW+'/verification.json')
    require(sources.json(REVIEW+'/result.json',True)==saved,'Review/comparison drift')
    require(saved['manifest_sha256']==canonical_hash(man) and saved['endpoint_index']==499,'Manifest/endpoint mismatch')
    require(review['aggregate_exact'] and review['train_replay']==132 and review['held_replay']==66,'Missing prior exact replay')
    parent=sources.json(sources.link(man['sources']['body_comparison']),True)
    dataset=sources.json(sources.link(man['dataset']),True)
    from gym_coach.counting.mse.manifest import diagnostic_tool
    tool=diagnostic_tool(root,man['sources']['diagnostic_tool'])
    loaded={};endpoints=[]
    for link in saved['checkpoints']:
        path=sources.link(link);cp=torch.load(path,map_location='cpu',weights_only=True)
        s,f=link['seed'],link['fold'];require((s,f) not in loaded,'Duplicate endpoint')
        require(cp['identity']==fm.fold_identity(fm.select(man,s),f) and cp['epoch']==499,'Wrong endpoint identity')
        require(runtime.state_hash(cp['model'])==cp['model_sha256'],'Weight hash mismatch')
        require(sources.json(str(path)+'.sha256.json')==file_info(path),'Checkpoint sidecar mismatch')
        r=sources.json(path.parent/'run.json');require(r['initial_fresh_state_sha256']==man['sources']['controls'][str(s)]['initializations'][str(f)],'Init changed')
        loaded[s,f]=cp;endpoints.append(dict(seed=s,fold=f,**identity(root,path),model_sha256=cp['model_sha256']))
    require(set(loaded)=={(s,f) for s in man['seeds'] for f in range(3)},'Incomplete endpoint grid')
    strip=lambda r:{k:v for k,v in r.items() if k not in ('method','side','signed_error','absolute_error','seed')}
    for seed,q in saved['seeds'].items():
        p=parent['seeds'][seed]
        controls=dict(raw=[dict(r,method='gru' if r['method']=='raw58' else r['method']) for r in p['records'] if r['method']!='body_normalized'],body=[r for r in p['records'] if r['method']=='body_normalized'])
        train={f:[strip(r) for r in q['records'] if r['method']=='fixed_origin' and r['side']=='train' and r['fold']==f] for f in range(3)}
        held={f:[dict(r,fold=f) for r in cv.unseal(loaded[int(seed),f]['epoch_reports'][499])['records']] for f in range(3)}
        require(fe.compare(man,int(seed),train,held,controls,tool)==q,'Metric/paired recomputation mismatch')
        for method,r in q['results'].items():
            expected=review['metrics'][seed][method];a=r['aggregate']
            require(a['train_video_equal_MAE']==expected['train_MAE'] and a['held_out_pooled_MAE']==expected['OOF_MAE'] and r['macro_group_MAE']==expected['macro'],'Metric review mismatch')
    require(fe.combine(saved['seeds'])==saved['combined'],'Combined mismatch')
    old=sources.json(OLD+'/analysis.json',True)
    require(set(old['metadata'])=={r['source']['video_id'] for r in man['rows']} and len(man['rows'])==33,'Cohort mismatch')
    return man,dataset,saved,old,dict(endpoints=endpoints,metrics_exact=True,prior_exact_replay_reused=review,
        new_forward_calls=0,training_steps=0,baseline_fits=0,history_reparsed=False)

def overlay(root,source,cache,ranking,out,sources,reuse=None):
    import cv2
    from PIL import Image,ImageDraw
    indices=ranking['frame_indices'];require(indices,'No candidate window')
    cap=cv2.VideoCapture(str(sources.add(source['path'],source)))
    require(cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==source['fps'] and cap.get(cv2.CAP_PROP_FRAME_COUNT)==source['frame_count'],'Decode metadata mismatch')
    w,h=source['width'],source['height'];board=Image.new('RGB',(w*4,30+math.ceil(len(indices)/2)*(h+25)),'#16202a');d=ImageDraw.Draw(board)
    d.text((5,7),source['video_id']+'  Original | RAW pose. L=cyan R=orange. Ranked windows; not automatic errors.',fill='white')
    decoded=0;errors=[];points={};decoder_times=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        require(frame.shape[:2]==(h,w),'Resolution mismatch')
        decoder_times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        errors.append(abs(decoder_times[-1]-decoded/source['fps']))
        if decoded in indices:
            k=indices.index(decoded);x=(k%2)*2*w;y=30+(k//2)*(h+25)
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));board.paste(im,(x,y));board.paste(im,(x+w,y))
            r=cache['values'][decoded];ps=[pixels(r,j,source) for j in range(8)];points[str(decoded)]=ps
            for a,b in ((0,2),(2,4),(4,6),(1,3),(3,5),(5,7),(0,1),(2,3)):
                if ps[a] and ps[b]:d.line((x+w+ps[a][0],y+ps[a][1],x+w+ps[b][0],y+ps[b][1]),fill='#00ddff' if a%2==0 else '#ff9900',width=2)
            for j,p in enumerate(ps):
                if p:
                    xx,yy=x+w+p[0],y+p[1];color='#00ddff' if j%2==0 else '#ff9900'
                    d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color);d.text((xx+3,yy-8),str(j),fill=color)
            d.text((x+4,y+h+4),f'f{decoded} t={decoded/source["fps"]:.2f}s complete={cache["complete"][decoded]}',fill='white')
        decoded+=1
    cap.release();require(decoded==source['frame_count'],'Decode frame count mismatch')
    path=(reuse if reuse else out)/(source['video_id']+'.jpg')
    if reuse:
        # Reuse this audit's completed export after serialization failure, byte-exact.
        import io
        buf=io.BytesIO();board.save(buf,format='JPEG',quality=96)
        require(path.read_bytes()==buf.getvalue(),'Reused overlay differs from current source/renderer')
        sources.add(path)
    else:
        require(not path.exists(),'No overlay overwrite');board.save(path,quality=96)
    return dict(**identity(root,path),frame_indices=indices,timestamps_seconds=[timestamp(i,source) for i in indices],
        decode_frames=decoded,max_decoder_pos_msec_difference_seconds=max(errors),decoder_pos_msec_seconds=decoder_times,
        timeline='Sequential decode index / FPS, matching frozen CFR policy. CAP_PROP_POS_MSEC is diagnostic only; not claimed as capture PTS.',
        points_pixels=points,actually_viewed=False)

def run(root,out,reuse=None):
    root=Path(root).resolve();out=root/out;out.mkdir(parents=True,exist_ok=False)
    h=helper();sources=h.Sources(root)
    write_json(out/'review-policy.json',POLICY)
    man,data,saved,old,verification=verify(root,sources)
    refpath=next(p for p in data['frozen_inputs'] if p.endswith('/results-v1/ai-reference.json'))
    sources.add(refpath,data['frozen_inputs'][refpath]);references={r['video_id']:r for r in sources.json(refpath)['records']}
    records=[];labels={};visual=[];ranks={};(out/'overlays').mkdir()
    for row in man['rows']:
        src=row['source'];v=src['video_id'];meta=old['metadata'][v]
        require(meta['source']==src and meta['input_artifact']==row['input_artifact'],'Prior pose summary/source mismatch')
        cache=sources.json(sources.link(row['input_artifact']));rank=rank_windows(cache,src);ranks[v]=rank
        labels[v]=h.labels(row,references,sources)
        require(sum(cache['complete'])==meta['pose']['complete_frames'],'Reused numerical coverage mismatch')
        sources.add(src['path'],src)
        sheets=labels[v]['passes'][0]['evidence']
        require([i for e in sheets for i in e['frame_indices']]==list(range(src['frame_count'])),'All-frame original sheets missing/duplicate')
        original=[dict(e,timestamps_seconds=[timestamp(i,src) for i in e['frame_indices']],actually_viewed=False) for e in sheets]
        detail=overlay(root,src,cache,rank,out/'overlays',sources,root/reuse if reuse else None)
        visual.append(dict(video_id=v,source_sha256=src['sha256'],original=original,overlay=detail))
        records.append(dict(video_id=v,source=src,group_id=src['group_id'],fold=next(f['fold'] for f in man['folds'] if v in f['evaluation_ids']),
            duration_seconds=meta['duration_seconds'],input_artifact=row['input_artifact'],pose=meta['pose'],support=meta['support'],rank=rank,
            numerical_verification='Reused hash-matched old all33 statistics; raw58 timeline/mask/edge ranking freshly checked',visual_status='EXPORTED_NOT_YET_VIEWED'))
    for name,value in [('numerical-audit.json',records),('label-audit.json',labels),('visual-index-unreviewed.json',visual),('verification.json',verification),('experiment-result.json',saved)]:
        write_json(out/name,cv.sealed(value))
    (out/'EXPERIMENT_SUMMARY.md').write_text(fe.report(saved),encoding='utf8')
    sources.add(Path(__file__));write_json(out/'preservation.json',sources.preserve())
    write_json(out/'manifest.json',cv.sealed(dict(version='representation-cohort33-audit-v1',**STATUS,source_run=RUN,comparison=COMPARISON,
        source_manifest=identity(root,sources.link(read(root/RUN/'run-root.json')['payload']['manifest'])),
        rule=POLICY,rule_sha256=canonical_hash(POLICY),cohort=[r['source'] for r in man['rows']],new_forward_calls=0,training_steps=0,baseline_fits=0)))
    print(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reuse-overlays',type=Path);a=parser.parse_args()
    with helper().previous_tool().no_learning_or_forward():run(a.root,a.out,a.reuse_overlays)
