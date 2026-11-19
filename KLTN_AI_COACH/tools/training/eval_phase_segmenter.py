"""Leave-one-person-out Ä‘Ã¡nh giÃ¡ heuristic/model trÃªn cÃ¹ng cÃ¡c clip."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,csv,json
from pathlib import Path
import numpy as np
from core.phase_schema import Rep,PhaseWindow,PHASE_ORDER
from stage3_features.phase_model.core.phase_dataset import load_all
from stage3_features.phase_model.core.phase_features import build_feature_matrix
from stage3_features.phase_model.core.phase_classifier import FramePhaseClassifier,boundary_proximity_weights
from stage3_features.phase_model.core.phase_labels import IGNORE_LABEL,rep_to_frame_labels
from stage3_features.phase_model.core.phase_decoder import DurationPriors,PhaseDecoder
from stage3_features.phase_model.eval.phase_metrics import boundary_summary,per_clip_within_tol,bootstrap_ci,frame_macro_f1,segment_f1

def pose_array(clip):
    return np.stack([np.concatenate((p.xy,np.asarray(p.conf).reshape(-1,1)),axis=1) for p in clip.poses]).astype(np.float32)
def boundaries_to_rep(b):
    r=Rep()
    for i,n in enumerate(PHASE_ORDER):
        if b[i+1]>b[i]: r.phases[n]=PhaseWindow(int(b[i]),int(b[i+1]-1))
    return r
def duration_data(clips):
    d={n:[] for n in PHASE_ORDER}
    for c in clips:
        for n,w in c.rep.present(): d[n].append(w.n_frames/c.fps)
    return d
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phases-dir',type=Path,default=Path('data/phases'));ap.add_argument('--pose-cache-dir',type=Path,default=Path('data/pose_cache'));ap.add_argument('--method',choices=('heuristic','model'),required=True);ap.add_argument('--group',choices=('all','coach','trainee'),default='all');ap.add_argument('--tol',type=int,default=3);ap.add_argument('--backend',choices=('lightgbm','logreg'),default='logreg');ap.add_argument('--out-dir',type=Path,default=Path('reports'));ap.add_argument('--seed',type=int,default=0);args=ap.parse_args()
    clips=load_all(args.phases_dir,args.pose_cache_dir,group=args.group)
    if not clips: raise SystemExit('KhÃ´ng cÃ³ clip Ä‘á»§ cáº·p phase/pose cache.')
    people=sorted({c.person for c in clips});truth=[];pred=[];frame_y=[];frame_p=[];fold_rows=[]
    for person in people:
        train=[c for c in clips if c.person!=person]; test=[c for c in clips if c.person==person]
        if not train: continue
        if args.method=='model':
            Xs=[];ys=[];ws=[];names=None
            for c in train:
                X,names=build_feature_matrix(pose_array(c),c.fps,c.front_side_override,is_coach=c.group=='coach')
                y=c.pose_labels(); ok=y!=IGNORE_LABEL
                if ok.any(): Xs.append(X[ok]);ys.append(y[ok]);ws.append(boundary_proximity_weights(y[ok],c.fps))
            if not Xs: continue
            model=FramePhaseClassifier(args.backend,random_state=args.seed).fit(np.concatenate(Xs),np.concatenate(ys),np.concatenate(ws),names)
            priors=DurationPriors.fit(duration_data(train)); decoder=PhaseDecoder(priors=priors)
            for c in test:
                X,_=build_feature_matrix(pose_array(c),c.fps,c.front_side_override,is_coach=c.group=='coach')
                out=decoder.decode(model.predict_proba(X),fps=c.fps).to_rep()
                pred.append(out);truth.append(c.rep);frame_y.append(c.pose_labels());frame_p.append(rep_to_frame_labels(out,c.total_frames)[:len(c.poses)])
        else:
            from make_heuristic_baseline import detect_heuristic
            for c in test:
                pred.append(detect_heuristic(c.poses,c.fps,c.front_side_override));truth.append(c.rep)
        if len(truth)>len(fold_rows): fold_rows.append({'held_out_person':person,'n_test':len(test)})
    if not truth: raise SystemExit('KhÃ´ng táº¡o Ä‘Æ°á»£c fold há»£p lá»‡.')
    summary=boundary_summary(truth,pred,args.tol);per=per_clip_within_tol(truth,pred,args.tol);mean,lo,hi=bootstrap_ci(per,seed=args.seed)
    report={'method':args.method,'group':args.group,'tolerance_frames':args.tol,'n_clips':len(truth),'n_people':len(people),'within_tol_clip_mean':mean,'within_tol_clip_ci95':[lo,hi],'boundary_summary':summary,'segment_f1':segment_f1(truth,pred),'folds':fold_rows}
    if frame_y: report['frame_macro_f1']=frame_macro_f1(np.concatenate(frame_y),np.concatenate(frame_p))
    args.out_dir.mkdir(parents=True,exist_ok=True);stem=f'phase_eval_{args.method}';(args.out_dir/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    with (args.out_dir/f'{stem}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['boundary','n','tol','within_tol','within_tol_all','bias','mae','median_abs','p90_abs','missed','spurious']);
        for name,row in summary.items():w.writerow([name]+[row.get(k) for k in ('n','tol','within_tol','within_tol_all','bias','mae','median_abs','p90_abs','missed','spurious')])
    print(f"method={args.method}; clips={len(truth)}; people={len(people)}; within-tol={mean:.3f} CI95=[{lo:.3f},{hi:.3f}]; report={args.out_dir}")
if __name__=='__main__':main()

