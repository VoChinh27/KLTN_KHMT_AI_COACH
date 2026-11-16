"""Táº¡o model challenger cÃ³ feedback weighting vÃ  chá»‰ promote sau cá»•ng Ä‘Ã¡nh giÃ¡."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse,json,shutil,subprocess,sys,re
from pathlib import Path
from stage3_features.phase_model.inference.phase_segmenter import load_registry,write_registry

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def compare_reports(current,candidate,current_trainee,candidate_trainee,min_gain=0.0,max_trainee_drop=0.0):
    old,new=read(current),read(candidate)
    old_score=float(old.get('within_tol_clip_mean',float('nan')));new_score=float(new.get('within_tol_clip_mean',float('nan')))
    if not (new_score>=old_score+min_gain):return False,f'overall gate failed: {old_score:.4f} -> {new_score:.4f}'
    old_group,new_group=read(current_trainee),read(candidate_trainee)
    old_t=float(old_group.get('within_tol_clip_mean',float('nan')));new_t=float(new_group.get('within_tol_clip_mean',float('nan')))
    if not (new_t>=old_t-max_trainee_drop):return False,f'trainee gate failed: {old_t:.4f} -> {new_t:.4f}'
    return True,f'overall {old_score:.4f}->{new_score:.4f}; trainee {old_t:.4f}->{new_t:.4f}'
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phases-dir',type=Path,default=Path('data/phases'));ap.add_argument('--pose-cache-dir',type=Path,default=Path('data/pose_cache'));ap.add_argument('--production-models-dir',type=Path,default=Path('models/phase_segmenter'));ap.add_argument('--candidate-dir',type=Path,default=Path('models/phase_segmenter/candidates/pending'));ap.add_argument('--feedback-log',type=Path,default=Path('data/phase_feedback/log.jsonl'));ap.add_argument('--backend',choices=('lightgbm','logreg'),default='lightgbm');ap.add_argument('--current-report',type=Path);ap.add_argument('--candidate-report',type=Path);ap.add_argument('--current-trainee-report',type=Path);ap.add_argument('--candidate-trainee-report',type=Path);ap.add_argument('--min-gain',type=float,default=0.0);ap.add_argument('--max-trainee-drop',type=float,default=0.0);ap.add_argument('--promote',action='store_true');args=ap.parse_args()
    if args.promote and any(x is None for x in (args.current_report,args.candidate_report,args.current_trainee_report,args.candidate_trainee_report)):raise SystemExit('--promote cáº§n bá»‘n report overall/trainee cá»§a current vÃ  candidate, cÃ¹ng fold/táº­p Ä‘Ã¡nh giÃ¡.')
    if not args.candidate_dir.exists():
        cmd=[sys.executable,'tools/train_phase_model.py','--phases-dir',str(args.phases_dir),'--pose-cache-dir',str(args.pose_cache_dir),'--models-dir',str(args.candidate_dir),'--backend',args.backend,'--feedback-log',str(args.feedback_log)]
        subprocess.run(cmd,check=True)
    if not args.promote:print(f'Candidate sáºµn sÃ ng táº¡i {args.candidate_dir}; production registry chÆ°a Ä‘á»•i. ÄÃ¡nh giÃ¡ candidate vÃ  gá»i láº¡i vá»›i --promote cÃ¹ng hai report.');return
    ok,reason=compare_reports(args.current_report,args.candidate_report,args.current_trainee_report,args.candidate_trainee_report,args.min_gain,args.max_trainee_drop)
    if not ok:raise SystemExit('Promotion bá»‹ tá»« chá»‘i: '+reason)
    reg=load_registry(args.production_models_dir);candidate_reg=load_registry(args.candidate_dir);cv=candidate_reg.get('current')
    if not cv:raise SystemExit('Candidate registry khÃ´ng cÃ³ current version')
    entry=candidate_reg['versions'][cv];src=args.candidate_dir/entry['model'];n=1
    for v in reg.get('versions',{}):
        m=re.fullmatch(r'v(\d+)',v)
        if m:n=max(n,int(m.group(1))+1)
    ver=f'v{n}';args.production_models_dir.mkdir(parents=True,exist_ok=True);dst=args.production_models_dir/f'{ver}.joblib';shutil.copy2(src,dst)
    meta=read(args.candidate_dir/entry['meta']) if entry.get('meta') else {};meta['promotion_gate']={'reason':reason,'current_report':str(args.current_report),'candidate_report':str(args.candidate_report),'current_trainee_report':str(args.current_trainee_report),'candidate_trainee_report':str(args.candidate_trainee_report)}
    (args.production_models_dir/f'{ver}.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    reg.setdefault('versions',{})[ver]={'model':dst.name,'meta':f'{ver}.meta.json'};reg['current']=ver;write_registry(args.production_models_dir,reg);print(f'Promoted {ver}: {reason}')
if __name__=='__main__':main()

